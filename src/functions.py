"""Helper functions for the Multi-label Classification of Cloud-Gaming Churn Feedback project."""

import json

import numpy as np
from langdetect import detect
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# List of the 6 class names. Set from the notebook: functions.LABELS = LABELS
LABELS = None

# evaluate() appends the result of every experiment here.
# To reset it use functions.RESULTS.clear() instead of RESULTS = [],
# otherwise the module keeps appending to the old list.
RESULTS = []


# ---------------------------------------------------------------------------
# Loading and building the dataset
# ---------------------------------------------------------------------------

# Parses a single JSON record: returns the comment text and the labels it contains
def parse_answer(js):
    """Parses a JSON string from the `answer` field.

    Args:
        js: JSON string where the "other" key holds the comment text
            and the remaining keys are the names of the assigned labels.

    Returns:
        A tuple (comment text, dict {label name: 1}).
    """
    d = json.loads(js)
    text = str(d.get('other', '')).strip()
    labels = {key: 1 for key in d.keys() if key != "other"}
    return text, labels


# ---------------------------------------------------------------------------
# Exploratory data analysis (EDA)
# ---------------------------------------------------------------------------

def safe_detect(text):
    """Detects the language of a text with langdetect without failing on tricky rows.

    Args:
        text: comment text.

    Returns:
        A language code ('en', 'pt', ...) or 'unknown' if detection failed.
    """
    try:
        return detect(text)
    except Exception:
        return 'unknown'   # short / mixed-language texts sometimes cannot be detected


# ---------------------------------------------------------------------------
# Metrics and evaluation
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_prob, thresholds=0.5, params='', train_time=None, split='val'):
    """Computes multi-label metrics and appends the result to RESULTS.

    Args:
        name: model name for the experiments table.
        y_true: matrix of true labels [n_samples, n_labels].
        y_prob: matrix of model scores [n_samples, n_labels].
        thresholds: a single value or a per-class array of thresholds.
        params: description of the model hyperparameters (string for the table).
        train_time: training time in seconds.
        split: 'train', 'val' or 'test'.

    Returns:
        Matrix of binary predictions after applying the thresholds.
    """
    if np.isscalar(thresholds):
        thresholds = np.full(y_true.shape[1], thresholds)
    y_pred = (y_prob >= thresholds).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, average='macro')
        mapr = average_precision_score(y_true, y_prob, average='macro')
    except ValueError:
        # The metric can be undefined if a class has only positive or only negative examples in the subset
        auc = mapr = np.nan

    RESULTS.append({'model': name, 'params': params, 'split': split,
                     'macro_f1': macro_f1, 'micro_f1': micro_f1,
                     'macro_auc': auc, 'mAP': mapr, 'train_time_s': train_time})
    print(f'[{name} | {split}] macro-F1={macro_f1:.3f}  micro-F1={micro_f1:.3f}  '
          f'AUC={auc:.3f}  mAP={mapr:.3f}  train_time:{train_time}')
    return y_pred


def tune_thresholds(y_true, y_prob):
    """Picks a separate F1-optimal threshold for every class.

    Scans 33 threshold values in the 0.1-0.9 range and keeps, for each class,
    the one that yields the highest F1.

    Args:
        y_true: matrix of true labels [n_samples, n_labels].
        y_prob: matrix of model scores [n_samples, n_labels].

    Returns:
        Array of thresholds of length n_labels.
    """
    thresholds = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 33):
            f = f1_score(y_true[:, j], (y_prob[:, j] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        thresholds[j] = best_t
    return thresholds


# ---------------------------------------------------------------------------
# LLM classification via the OpenAI API
# ---------------------------------------------------------------------------

# Converts lists of label names into a binary matrix - so the same metrics can be computed as for the other models
def labels_to_matrix(list_of_labels):
    """Converts lists of label names into a binary matrix.

    Args:
        list_of_labels: list where each element is the list of label names of one comment.

    Returns:
        A 0/1 matrix of shape [len(list_of_labels), len(LABELS)].
    """
    matrix = np.zeros((len(list_of_labels), len(LABELS)), dtype=int)
    for i, labs in enumerate(list_of_labels):
        for l in labs:
            matrix[i, LABELS.index(l)] = 1
    return matrix
