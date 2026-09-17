import numpy as np


def sample_episode(X, y, n_classes, k_shot, q_query, minority_classes=None):
    """
    Samples one N-way K-shot episode. If minority_classes is given, episode
    classes are drawn exclusively from that pool (this is how the paper
    constructs episodic few-shot tasks: support samples come from minority
    attack classes only). Ensures disjoint support and query sets.
    """
    unique_classes = np.unique(y)

    if minority_classes is not None:
        candidate_classes = np.array(
            [c for c in minority_classes if c in unique_classes]
        )
        if len(candidate_classes) < n_classes:
            raise ValueError("Not enough minority classes to form an episode.")
    else:
        candidate_classes = unique_classes

    classes = np.random.choice(candidate_classes, n_classes, replace=False)

    support_X, support_y = [], []
    query_X, query_y = [], []

    for cls in classes:
        idx = np.where(y == cls)[0]

        if len(idx) < (k_shot + q_query):
            raise ValueError(
                f"Class {cls} does not have enough samples "
                f"for {k_shot}-shot {q_query}-query episode."
            )

        selected = np.random.choice(idx, k_shot + q_query, replace=False)

        support_X.extend(X[selected[:k_shot]])
        support_y.extend(y[selected[:k_shot]])

        query_X.extend(X[selected[k_shot:]])
        query_y.extend(y[selected[k_shot:]])

    return (
        np.array(support_X),
        np.array(support_y),
        np.array(query_X),
        np.array(query_y),
    )
