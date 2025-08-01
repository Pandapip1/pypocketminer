import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Sequential

from pypocketminer.models.layers.gvp import GVPLayerNorm, GVPDropout
from pypocketminer.models.gvp import GVP


class MPNNLayer(Model):
    def __init__(self, vec_in, num_hidden, dropout=0.1):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.vec_in = vec_in
        self.vo, self.so = vo, so = num_hidden
        self.norm = [GVPLayerNorm(vo) for _ in range(2)]
        self.dropout = GVPDropout(dropout, vo)

        # this receives the vec_in message AND the receiver node
        self.W_EV = Sequential(
            [
                GVP(vi=vec_in + vo, vo=vo, so=so),
                GVP(vi=vo, vo=vo, so=so),
                GVP(vi=vo, vo=vo, so=so, nls=None, nlv=None),
            ]
        )

        self.W_dh = Sequential(
            [
                GVP(vi=vo, vo=2 * vo, so=4 * so),
                GVP(vi=2 * vo, vo=vo, so=so, nls=None, nlv=None),
            ]
        )

    def call(self, h_V, h_M, mask_V=None, mask_attend=None, train=False):
        # Concatenate h_V_i to h_E_ij
        h_V_expand = tf.tile(tf.expand_dims(h_V, -2), [1, 1, tf.shape(h_M)[-2], 1])
        h_EV = vs_concat(h_V_expand, h_M, self.vo, self.vec_in)
        h_message = self.W_EV(h_EV)
        if mask_attend is not None:
            h_message = tf.cast(tf.expand_dims(mask_attend, -1), tf.float32) * h_message
        dh = tf.math.reduce_mean(h_message, -2)
        h_V = self.norm[0](h_V + self.dropout(dh, training=train))

        # Position-wise feedforward
        dh = self.W_dh(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh, training=train))

        if mask_V is not None:
            mask_V = tf.cast(tf.expand_dims(mask_V, -1), tf.float32)
            h_V = mask_V * h_V

        return h_V
