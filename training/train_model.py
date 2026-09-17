import tensorflow as tf
import numpy as np
from models.fs_lstm import build_fs_lstm
from models.focal_loss import focal_loss
from training.episodic_sampler import sample_episode


def compute_prototypes(embeddings, labels):
    labels = tf.convert_to_tensor(labels)
    prototypes = []
    unique_labels = tf.unique(labels)[0]

    for cls in unique_labels:
        mask = tf.equal(labels, cls)
        cls_embeddings = tf.boolean_mask(embeddings, mask)
        prototypes.append(tf.reduce_mean(cls_embeddings, axis=0))

    return tf.stack(prototypes), unique_labels


def train_model(X_train, y_train, input_shape, config, minority_classes=None, encoder=None):
    """
    Episodic prototypical training of the attention-enhanced LSTM encoder
    with focal loss on the query set, following Algorithm 1 of the paper.
    `minority_classes`, if given, restricts episode sampling to those
    classes (the paper's few-shot regime is defined at the minority-class
    level, K << N_minority).
    """
    if encoder is None:
        encoder = build_fs_lstm(input_shape)

    optimizer = tf.keras.optimizers.Adam(
        config["learning_rate"], clipnorm=config.get("clipnorm", 1.0)
    )
    loss_fn = focal_loss(config["alpha"], config["gamma"])

    n_way = config["n_way"]
    k_shot = config["k_shot"]
    q_query = config["q_query"]
    episodes = config["episodes"]

    for episode in range(episodes):
        support_X, support_y, query_X, query_y = sample_episode(
            X_train, y_train, n_way, k_shot, q_query, minority_classes=minority_classes
        )

        support_X = support_X[..., np.newaxis]
        query_X = query_X[..., np.newaxis]

        with tf.GradientTape() as tape:
            support_embed = encoder(support_X, training=True)
            query_embed = encoder(query_X, training=True)

            prototypes, proto_labels = compute_prototypes(support_embed, support_y)

            # Squared Euclidean distance
            dists = tf.reduce_sum(
                (tf.expand_dims(query_embed, 1) - prototypes) ** 2, axis=2
            )

            logits = -dists
            probs = tf.nn.softmax(logits, axis=1)

            proto_labels = tf.cast(proto_labels, tf.int32)
            query_y = tf.cast(query_y, tf.int32)

            mapped_labels = tf.argmax(
                tf.equal(
                    tf.expand_dims(query_y, 1),
                    tf.expand_dims(proto_labels, 0),
                ),
                axis=1,
            )

            loss = loss_fn(tf.one_hot(mapped_labels, depth=n_way), probs)

        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

        if episode % 100 == 0:
            print(f"[Episode {episode}] Loss: {loss.numpy():.4f}")

    return encoder
