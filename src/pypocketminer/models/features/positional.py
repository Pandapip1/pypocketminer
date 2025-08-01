import numpy as np
import tensorflow as tf

from tensorflow.keras import Model


class PositionalEncodings(Model):
    def __init__(self, num_embeddings, period_range=[2, 1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def call(self, E_idx):
        # i-j
        # N_batch = tf.shape(E_idx)[0]
        N_nodes = tf.shape(E_idx)[1]
        # N_neighbors = tf.shape(E_idx)[2]
        ii = tf.reshape(tf.cast(tf.range(N_nodes), tf.float32), (1, -1, 1))
        d = tf.expand_dims((tf.cast(E_idx, tf.float32) - ii), -1)
        # Original Transformer frequencies
        frequency = tf.math.exp(
            tf.cast(tf.range(0, self.num_embeddings, 2), tf.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * tf.reshape(frequency, (1, 1, 1, -1))
        E = tf.concat((tf.math.cos(angles), tf.math.sin(angles)), -1)
        return E
