import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DROP_CANDIDATES = ["ts", "id", "index", "Unnamed: 0"]

# Common alternate label columns in ToN_IoT-style datasets. Whichever one
# ISN'T the chosen target gets dropped automatically — e.g. if you train on
# `type` (multiclass), the binary `label` column must not stay in as a
# feature, or it leaks a near-perfect "attack or not" shortcut without
# helping distinguish attack subtypes, and wastes PCA capacity that should
# go toward genuinely discriminating features.
KNOWN_LABEL_COLS = ["label", "type", "attack_cat", "attack_type"]

# Categorical columns with more unique values than this are frequency-encoded
# (mapped to a single numeric column = how common that value is) instead of
# one-hot encoded, to avoid exploding the feature matrix — e.g. src_ip/dst_ip
# can have thousands of distinct values, and one-hot encoding those would
# drown PCA in mostly-empty columns instead of letting it concentrate on the
# informative flow features (port, duration, bytes, conn_state, etc.) the
# paper's own SHAP analysis identifies as most important.
MAX_ONEHOT_CARDINALITY = 20


def _encode_categoricals(X_df, max_onehot_cardinality=MAX_ONEHOT_CARDINALITY):
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        return X_df

    low_card = [c for c in cat_cols if X_df[c].nunique() <= max_onehot_cardinality]
    high_card = [c for c in cat_cols if c not in low_card]

    if low_card:
        X_df = pd.get_dummies(X_df, columns=low_card)

    for c in high_card:
        freq = X_df[c].value_counts(normalize=True)
        X_df[c] = X_df[c].map(freq).astype(np.float32)

    return X_df


def load_ton_iot(csv_path, label_col="label", test_size=0.2, random_state=42):
    """
    Loads a ToN_IoT2020-style CSV, encodes categorical columns
    (one-hot for low-cardinality, frequency-encoding for high-cardinality
    columns like src_ip/dst_ip/dns_query), label-encodes the multiclass
    target, and produces a stratified 80/20 train/test split.
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    y_raw = df[label_col]
    X_df = df.drop(columns=[label_col])

    other_label_cols = [c for c in KNOWN_LABEL_COLS if c != label_col and c in X_df.columns]
    X_df = X_df.drop(columns=other_label_cols + [c for c in DROP_CANDIDATES if c in X_df.columns])
    if other_label_cols:
        print(f"Dropped alternate label column(s) from features: {other_label_cols}")

    X_df = _encode_categoricals(X_df)

    X_df = X_df.fillna(0)
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, label_encoder

