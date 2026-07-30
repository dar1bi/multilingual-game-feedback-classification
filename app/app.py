"""Streamlit-демо: класифікація технічних скарг користувачів хмарного гейм-сервісу.

Запуск з кореня проєкту:  streamlit run app/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Модель береться локально, якщо вона вже збережена розділом 15 нотбука,
# інакше — тягнеться з HuggingFace Hub (transformers кешує її після першого разу)
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models' / 'bert_multilabel'
HUB_MODEL_ID = 'dar1bi/cloud-gaming-feedback-multilabel'

# Людські назви класів — у моделі вони технічні
LABEL_UA = {
    'ping_latency': 'Мережева затримка (пінг)',
    'frames_drop': 'Падіння FPS, фрізи',
    'unable_launch': 'Гра або сервіс не запускається',
    'mouse_keyboard_headset': 'Периферія: миша, клавіатура, гарнітура',
    'game_bug': 'Баг усередині гри',
    'failed_save': 'Не зберігається прогрес',
}

# Приклади різними мовами — щоб було видно, що модель мультимовна
EXAMPLES = [
    'Latence en jeu, impossible de jouer correctement',
    'Constant micro stutter and very low fps in The Finals',
    'Não consigo iniciar nenhum jogo, fica carregando para sempre',
    'My controller is not detected and the game keeps crashing',
    'Дуже великий пінг, грати неможливо',
    'Save files disappear every time I log out',
]

st.set_page_config(page_title='Класифікація скарг користувачів', page_icon='🎮', layout='centered')


@st.cache_resource(show_spinner='Завантажую модель…')
def load_model():
    """Вантажить модель один раз на всі сесії (інакше — на кожен клік)."""
    if MODEL_DIR.exists():
        source = str(MODEL_DIR)
        config_path = MODEL_DIR / 'inference_config.json'
    else:
        source = HUB_MODEL_ID
        config_path = Path(hf_hub_download(HUB_MODEL_ID, 'inference_config.json'))

    config = json.loads(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source).eval()
    return tokenizer, model, config, source


def predict(text, tokenizer, model, config):
    """Повертає ймовірність кожного з шести класів для одного коментаря."""
    encoded = tokenizer(text, truncation=True, padding='max_length',
                        max_length=config['max_len'], return_tensors='pt')
    with torch.no_grad():
        logits = model(**encoded).logits
    return torch.sigmoid(logits).numpy()[0]


tokenizer, model, config, model_source = load_model()
labels = config['labels']
thresholds = config['thresholds']

st.title('🎮 Класифікація технічних скарг')
st.caption(
    'Модель визначає тип технічної проблеми з коментаря, який користувач хмарного '
    'гейм-сервісу залишає при відписці. Коментар може стосуватись кількох проблем одночасно.'
)

demo_tab, model_tab = st.tabs(['Демо', 'Про модель'])

with demo_tab:
    if 'text' not in st.session_state:
        st.session_state.text = EXAMPLES[0]

    with st.expander('Приклади коментарів різними мовами'):
        columns = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if columns[i % 2].button(example, key=f'example_{i}', use_container_width=True):
                st.session_state.text = example
        st.caption('Клік підставляє приклад у поле — далі натисни «Класифікувати».')

    # Форма: класифікація запускається лише по кліку на кнопку
    with st.form('classify'):
        text = st.text_area('Коментар користувача (будь-якою мовою):', key='text', height=120)
        submitted = st.form_submit_button('Класифікувати', type='primary', use_container_width=True)

    if submitted:
        if text.strip():
            st.session_state.result = predict(text, tokenizer, model, config)
        else:
            st.session_state.pop('result', None)
            st.warning('Спочатку введи коментар або обери приклад вище.')

    if 'result' in st.session_state:
        probabilities = st.session_state.result
        predicted = [LABEL_UA[label] for label, probability, threshold
                     in zip(labels, probabilities, thresholds) if probability >= threshold]

        st.subheader('Передбачені проблеми')
        if predicted:
            for name in predicted:
                st.success(name)
        else:
            st.info('Жодна проблема не визначена впевнено — усі ймовірності нижчі за пороги.')

        st.subheader('Ймовірності по класах')
        table = pd.DataFrame({
            'Клас': [LABEL_UA[label] for label in labels],
            'Ймовірність': probabilities,
            'Поріг': thresholds,
            'Спрацював': ['так' if p >= t else '—' for p, t in zip(probabilities, thresholds)],
        }).sort_values('Ймовірність', ascending=False)

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Ймовірність': st.column_config.ProgressColumn(
                    'Ймовірність', min_value=0.0, max_value=1.0, format='%.2f'),
                'Поріг': st.column_config.NumberColumn('Поріг', format='%.3f'),
            },
        )
        st.caption(
            'Мітка ставиться, коли ймовірність перевищує поріг цього класу. '
            'Пороги різні, бо підбирались окремо для кожного класу на validation-вибірці.'
        )

with model_tab:
    st.subheader('Модель')
    left, right = st.columns(2)
    left.metric('Validation macro-F1', f'{config["val_macro_f1"]:.3f}')
    right.metric('Базова модель', config['base_checkpoint'].replace('distilbert-base-', ''))

    hub_link = f'[{HUB_MODEL_ID}](https://huggingface.co/{HUB_MODEL_ID})'
    st.caption(
        f'Модель завантажена з HuggingFace Hub: {hub_link}' if model_source == HUB_MODEL_ID
        else f'Модель завантажена локально з `models/bert_multilabel/` · на Hub: {hub_link}'
    )

    st.markdown(
        f"""
Це **fine-tuned {config['base_checkpoint']}** — мультимовний трансформер, донавчений на
коментарях користувачів при відписці. Задача — multi-label класифікація: один коментар
може одночасно належати до кількох із шести класів технічних проблем.

Модель обрана як фінальна за результатом на validation-вибірці серед семи підходів:
від TF-IDF з класичними класифікаторами до zero-shot LLM. Деталі експериментів,
аналіз помилок і висновки — у нотбуці проєкту.

**Про пороги.** Модель повертає шість незалежних ймовірностей. Поріг для кожного класу
підбирався окремо на validation під максимізацію F1 — саме тому вони різні. Це дає
помітно кращий результат, ніж стандартний поріг 0.5 для всіх класів.

**Чесно про якість.** Найкраще розпізнаються часті проблеми — мережева затримка
та неможливість запустити гру. Рідкісні класи (`game_bug`, `failed_save`) модель
пропускає частіше: у даних їх мало, а межі між ними й рештою класів розмиті.
        """
    )

    st.subheader('Пороги по класах')
    st.dataframe(
        pd.DataFrame({
            'Клас': [LABEL_UA[label] for label in labels],
            'Технічна назва': labels,
            'Поріг': thresholds,
        }),
        hide_index=True,
        use_container_width=True,
        column_config={'Поріг': st.column_config.NumberColumn('Поріг', format='%.3f')},
    )
