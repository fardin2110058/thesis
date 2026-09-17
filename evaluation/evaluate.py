import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from training.episodic_sampler import sample_episode
from evaluation.metrics import compute_binary_metrics


def evaluate_fewshot(model, X, y, n_way=4, k_shot=5, q_query=10, num_episodes=3000,
                      random_state=42, minority_classes=None):
    """
    Episodic few-shot evaluation using prototype-based classification
    (encoder embeddings + nearest-prototype rule).

    `minority_classes`, if given, restricts episode sampling to that pool —
    pass only classes with at least (k_shot + q_query) samples in this split
    (see `main.py`'s `eligible_fewshot_classes` helper), otherwise
    `sample_episode` raises on any class too small for the episode.
    """
    np.random.seed(random_state)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    for _ in range(num_episodes):
        support_X, support_y, query_X, query_y = sample_episode(
            X, y, n_way, k_shot, q_query, minority_classes=minority_classes
        )

        support_X = support_X[..., np.newaxis]
        query_X = query_X[..., np.newaxis]

        support_embed = model.predict(support_X, verbose=0)
        query_embed = model.predict(query_X, verbose=0)

        classes = np.unique(support_y)
        prototypes = np.stack([
            support_embed[support_y == cls].mean(axis=0) for cls in classes
        ])

        preds = []
        for q in query_embed:
            dists = np.sum((prototypes - q) ** 2, axis=1)
            preds.append(classes[np.argmin(dists)])
        preds = np.array(preds)

        all_acc.append(accuracy_score(query_y, preds))
        all_prec.append(precision_score(query_y, preds, average="macro", zero_division=0))
        all_rec.append(recall_score(query_y, preds, average="macro", zero_division=0))
        all_f1.append(f1_score(query_y, preds, average="macro", zero_division=0))

    return {
        "Accuracy": np.mean(all_acc),
        "Precision": np.mean(all_prec),
        "Recall": np.mean(all_rec),
        "F1": np.mean(all_f1),
        "Std_Accuracy": np.std(all_acc),
        "Std_F1": np.std(all_f1),
    }


def evaluate_classifier(model, X_test_seq, y_test, class_names=None):
    """
    Full multiclass (non-episodic) evaluation across ALL classes, producing
    a Table 6/7-style per-class report via one-vs-rest metrics.
    Not present in the official reference repo.
    """
    probs = model.predict(X_test_seq, verbose=0)
    preds = np.argmax(probs, axis=1)

    classes = np.unique(y_test)
    rows = []
    for c in classes:
        y_true_bin = (y_test == c).astype(int)
        y_pred_bin = (preds == c).astype(int)
        m = compute_binary_metrics(y_true_bin, y_pred_bin)
        name = class_names[c] if class_names is not None else str(c)
        rows.append({"Class": name, **m})

    return pd.DataFrame(rows), preds
