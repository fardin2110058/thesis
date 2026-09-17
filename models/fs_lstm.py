import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Input, Dense, BatchNormalization
from models.attention_layer import AttentionLayer


def build_fs_lstm(input_shape):
    """
    Few-shot encoder network (attention-enhanced LSTM).
    Outputs a fixed-length embedding vector used for prototype computation.
    """
    inputs = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)

    x = LSTM(64, return_sequences=True)(x)

    x = AttentionLayer()(x)

    model = Model(inputs, x)
    return model


def build_classifier(input_shape, num_classes, encoder=None,
                      head_dense_units=64, head_dropout=0.3):
    """
    Full multiclass classifier reusing the fs_lstm encoder as a feature
    extractor. This is the "deployable" IDS head used for standard
    (non-episodic) evaluation across ALL classes — i.e. Table 6/7-style
    per-class accuracy/precision/recall/F1/specificity/FAR reporting.

    Not present in the official reference repo, which only evaluates via
    episodic few-shot classification.

    Adds a small Dense+BatchNorm+Dropout head between the (shared, few-shot
    pretrained) embedding and the final softmax, rather than putting
    softmax directly on the 64-dim attention output. This gives the
    classifier extra capacity to separate classes that are easily
    confused at the raw embedding level (e.g. injection/mitm/password)
    without disturbing the embedding space the episodic encoder learned.
    """
    inputs = Input(shape=input_shape)
    if encoder is None:
        encoder = build_fs_lstm(input_shape)
    embedding = encoder(inputs)
    x = Dense(head_dense_units, activation="relu")(embedding)
    x = BatchNormalization()(x)
    x = Dropout(head_dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model
