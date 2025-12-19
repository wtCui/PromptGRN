import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).flatten()
    y_score = np.asarray(y_score).astype(float).flatten()

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    aupr = average_precision_score(y_true=y_true, y_score=y_score)
    aupr_norm = aupr / (np.mean(y_true) + 1e-12)
    return auc, aupr, aupr_norm
