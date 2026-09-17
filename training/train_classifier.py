import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from models.fs_lstm import build_classifier
from models.focal_loss import focal_loss


def train_classifier(X_train_seq, y_train, input_shape, num_classes, config, encoder=None):
    """
    Batch-based training of the full multiclass IDS head (PCA+AM+LSTM+FL),
    reusing the episodically pretrained encoder as feature extractor.
    This is what produces the per-class Table 6/7-style metrics.
    """
    model = build_classifier(
        input_shape, num_classes, encoder=encoder,
        head_dense_units=config.get("head_dense_units", 64),
        head_dropout=config.get("head_dropout", 0.3),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            config["learning_rate"], clipnorm=config.get("clipnorm", 1.0)
        ),
        loss=focal_loss(config["alpha"], config["gamma"]),
        metrics=["accuracy"],
    )

    y_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    class_weight = None
    if config.get("use_class_weight", True):
        classes = np.arange(num_classes)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        class_weight = dict(zip(classes, weights))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.get("early_stopping_patience", 15),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.get("reduce_lr_factor", 0.5),
            patience=config.get("reduce_lr_patience", 8),
            min_lr=config.get("min_lr", 1e-5),
            verbose=1,
        ),
    ]

    model.fit(
        X_train_seq,
        y_onehot,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    return model
