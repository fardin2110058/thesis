import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from reproducibility import set_seed
from preprocessing.load_data import load_ton_iot
from preprocessing.normalize import minmax_normalize
from preprocessing.pca_extraction import apply_pca
from training.train_model import train_model
from training.train_classifier import train_classifier
from evaluation.evaluate import evaluate_fewshot, evaluate_classifier
from explainability.shap_explainer import explain_model


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(42)

    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test, feature_names, label_encoder = load_ton_iot(
        args.csv, label_col=args.label_col, test_size=0.2, random_state=42
    )
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {list(label_encoder.classes_)}")

    print("Min-max normalizing...")
    X_train, scaler = minmax_normalize(X_train)
    X_test, _ = minmax_normalize(X_test, scaler=scaler)

    print(f"Applying PCA (variance_threshold={config.get('pca_variance_threshold')}, "
          f"fallback n_components={config['pca_components']})...")
    X_train, pca = apply_pca(
        X_train,
        n_components=config["pca_components"],
        variance_threshold=config.get("pca_variance_threshold"),
    )
    X_test, _ = apply_pca(X_test, pca=pca)
    print(f"Selected {pca.n_components_} components, retained variance: "
          f"{pca.explained_variance_ratio_.sum():.4f}")

    X_train_seq = X_train[..., np.newaxis]
    X_test_seq = X_test[..., np.newaxis]
    input_shape = (X_train_seq.shape[1], 1)

    counts = np.bincount(y_train, minlength=num_classes)
    min_needed = config["k_shot"] + config["q_query"]

    # Rank classes by count (ascending) and take enough of the smallest ones
    # to comfortably exceed n_way, so episode sampling never starves — a
    # fixed "< median" cutoff can leave too few classes if the distribution
    # is lopsided (e.g. one huge majority class, everything else similar).
    eligible = [c for c in np.argsort(counts) if counts[c] >= min_needed]
    n_minority = max(config["n_way"] + 1, len(eligible) // 2)
    minority_classes = eligible[:n_minority]

    if len(minority_classes) < config["n_way"]:
        raise ValueError(
            f"Only {len(minority_classes)} classes have >= {min_needed} samples "
            f"(k_shot={config['k_shot']} + q_query={config['q_query']}), but "
            f"n_way={config['n_way']} needs at least that many. Lower k_shot/"
            f"q_query/n_way in config/config.yml, or check your class balance."
        )
    print(f"Minority classes (few-shot pool): {minority_classes}")

    print("\n=== Phase 1: episodic few-shot training of attention-LSTM encoder ===")
    encoder = train_model(X_train, y_train, input_shape, config, minority_classes=minority_classes)

    print("\n=== Phase 2: full multiclass classifier head (PCA+AM+LSTM+FL) ===")
    classifier = train_classifier(X_train_seq, y_train, input_shape, num_classes, config, encoder=encoder)

    print("\n=== Evaluation: full multiclass report ===")
    report_df, preds = evaluate_classifier(classifier, X_test_seq, y_test, class_names=label_encoder.classes_)
    print(report_df.to_string(index=False))
    report_df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

    print("\n=== Confusion matrix (full) ===")
    class_names = list(label_encoder.classes_)
    cm = confusion_matrix(y_test, preds, labels=range(num_classes))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())
    cm_df.to_csv(args.confusion_csv)
    print(f"Saved: {args.confusion_csv}")

    watch_classes = [c for c in ["injection", "mitm", "password"] if c in class_names]
    if watch_classes:
        print("\n=== Confusion matrix: injection/mitm/password rows only "
              "(row = true class, columns = what it got predicted as) ===")
        print(cm_df.loc[watch_classes].to_string())
        print(
            "\nRead this as: for each of these rows, which OTHER class is stealing "
            "its predictions? A few wrong columns dominating = feature separability "
            "issue between those specific classes. Errors spread thinly across many "
            "columns = general imbalance/noise, not a specific class confusion."
        )

    print("\n=== Evaluation: episodic few-shot metrics ===")
    min_needed = config["k_shot"] + config["q_query"]
    test_counts = np.bincount(y_test, minlength=num_classes)
    eligible_fewshot_classes = [c for c, n in enumerate(test_counts) if n >= min_needed]
    if len(eligible_fewshot_classes) < config["n_way"]:
        print(
            f"Skipping few-shot eval: only {len(eligible_fewshot_classes)} test classes "
            f"have >= {min_needed} samples (need >= n_way={config['n_way']})."
        )
    else:
        fs_metrics = evaluate_fewshot(
            encoder, X_test, y_test,
            n_way=config["n_way"], k_shot=config["k_shot"], q_query=config["q_query"],
            num_episodes=args.fs_eval_episodes,
            minority_classes=eligible_fewshot_classes,
        )
        for k, v in fs_metrics.items():
            print(f"  {k}: {v:.4f}")

    if not args.skip_shap:
        print("\n=== SHAP explainability (this is slow) ===")
        bg_idx = np.random.choice(len(X_train_seq), size=min(100, len(X_train_seq)), replace=False)
        ex_idx = np.random.choice(len(X_test_seq), size=min(args.shap_samples, len(X_test_seq)), replace=False)
        explain_model(
            classifier,
            X_train_seq[bg_idx],
            X_test_seq[ex_idx],
            feature_names=[f"PC{i+1}" for i in range(X_train.shape[1])],
            save_path="shap_summary.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeXAI-AttentionCPS pipeline")
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV")
    parser.add_argument("--label_col", default="label", help="Name of the label column")
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--output_csv", default="results_multiclass.csv")
    parser.add_argument("--confusion_csv", default="confusion_matrix.csv")
    parser.add_argument("--fs_eval_episodes", type=int, default=200,
                         help="Number of episodes for few-shot evaluation (paper uses 3000; slow)")
    parser.add_argument("--skip_shap", action="store_true", help="Skip SHAP (fastest to iterate)")
    parser.add_argument("--shap_samples", type=int, default=30)
    args = parser.parse_args()
    main(args)
