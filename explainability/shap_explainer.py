import shap
import numpy as np
import matplotlib.pyplot as plt


def explain_model(model, X_background, X_explain, feature_names=None, save_path=None):
    """
    SHAP explanation for the LSTM-based IDS model, following the paper's
    global SHAP summary plot (Fig. 6). Model input is flattened for
    KernelExplainer, then reshaped back to (batch, timesteps, 1) inside
    the wrapped predict function.

    - X_background: small background sample (e.g. 100 samples)
    - X_explain: samples to explain (e.g. 10-50; KernelExplainer is slow)
    """
    X_background_flat = X_background.reshape(X_background.shape[0], X_background.shape[1])
    X_explain_flat = X_explain.reshape(X_explain.shape[0], X_explain.shape[1])

    def model_predict(X):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return model.predict(X, verbose=0)

    explainer = shap.KernelExplainer(model_predict, X_background_flat)
    shap_values = explainer.shap_values(X_explain_flat, nsamples=100)

    shap.summary_plot(
        shap_values,
        X_explain_flat,
        feature_names=feature_names,
        show=False,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return shap_values
