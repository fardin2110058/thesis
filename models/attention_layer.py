import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class AttentionLayer(Layer):
    """
    Bahdanau-style additive attention layer.
    Collapses (batch, timesteps, hidden_dim) -> (batch, hidden_dim)
    by weighting each timestep's hidden state.
    """

    def __init__(self):
        super().__init__()
        self.W = Dense(1)

    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector
