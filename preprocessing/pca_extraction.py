import numpy as np
from sklearn.decomposition import PCA


def apply_pca(X, n_components=None, variance_threshold=None, pca=None):
    """
    If pca is None: fit a new PCA (on train).
    Else: transform using an existing fitted PCA (on test).

    If variance_threshold is set (e.g. 0.932, matching the paper's 93.2%
    retained variance), the number of components is chosen automatically as
    the smallest count that reaches that threshold — this is more robust
    than a fixed n_components, since the "right" component count depends on
    exactly how many features you end up with after encoding, which varies
    by dataset. n_components (if given) is used as a fallback/override when
    variance_threshold is None.
    """
    if pca is None:
        if variance_threshold is not None:
            probe = PCA(random_state=42)
            probe.fit(X)
            cumvar = np.cumsum(probe.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
            n_components = min(n_components, X.shape[1])

        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = pca.transform(X)

    return X_pca, pca
