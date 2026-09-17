import numpy as np


def compute_binary_metrics(y_true, y_pred):
    """
    Computes one-vs-rest binary classification metrics for IDS evaluation.
    Assumes: 1 = target class (attack or the class under evaluation), 0 = rest.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    specificity = TN / (TN + FP + 1e-9)
    FAR = FP / (FP + TN + 1e-9)

    f1_score = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "FAR": FAR,
        "F1-score": f1_score,
    }
