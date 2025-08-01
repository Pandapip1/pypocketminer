import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LayerNormalization

from pypocketminer.models.base import Encoder
from pypocketminer.models.gvp import GVP
from pypocketminer.models.features.structural import StructuralFeatures
from pypocketminer.models.utils import vs_concat


class MQAModel(Model):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_layers=3,
        k_neighbors=30,
        dropout=0.1,
        regression=False,
        multiclass=False,
        ablate_aa_type=False,
        ablate_sidechain_vectors=True,
        ablate_rbf=False,
        use_lm=False,
        squeeze_lm=False,
    ):

        super(MQAModel, self).__init__()

        # Model type
        self.multiclass = multiclass
        self.ablate_aa_type = ablate_aa_type
        self.use_lm = use_lm
        self.squeeze_lm = squeeze_lm

        # Hyperparameters
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features

        # Featurization layers
        self.features = StructuralFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            ablate_sidechain_vectors=ablate_sidechain_vectors,
            ablate_rbf=ablate_rbf,
        )

        # Sequence embedding layers
        if not use_lm:
            self.W_s = Embedding(20, self.hs)
        if use_lm and squeeze_lm:
            self.W_s = Sequential(
                [Dense(100, activation="relu"), Dropout(rate=dropout)]
            )

        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs, nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs, nls=None, nlv=None)

        self.encoder = Encoder(
            hidden_dim, edge_features, num_layers=num_layers, dropout=dropout
        )

        self.W_V_out = GVP(vi=self.hv, vo=0, so=self.hs, nls=None, nlv=None)

        if regression:
            self.dense = Sequential(
                [
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    LayerNormalization(),
                    Dense(1, activation=None),
                ]
            )
        elif multiclass:
            self.multiclass = True
            self.dense = Sequential(
                [
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    LayerNormalization(),
                    Dense(3, activation="softmax"),
                ]
            )
        else:
            self.dense = Sequential(
                [
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    LayerNormalization(),
                    Dense(1, activation="sigmoid"),
                ]
            )

    def call(self, X, S, mask, train=False, res_level=False, ablate_aa_type=False):
        # X [B, N, 4, 3], S [B, N], mask [B, N]

        V, E, E_idx = self.features(X, mask)
        if self.ablate_aa_type:
            h_V = self.W_v(V)
        else:
            if self.use_lm and not self.squeeze_lm:
                h_S = S
            elif self.use_lm and self.squeeze_lm:
                h_S = self.W_s(S)
            else:
                h_S = self.W_s(S)
            V = vs_concat(V, h_S, self.nv, 0)
            h_V = self.W_v(V)

        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)

        h_V_out = self.W_V_out(h_V)
        mask = tf.expand_dims(mask, -1)  # [B, N, 1]

        if not res_level:
            if train:
                h_V_out = tf.math.reduce_mean(h_V_out * mask, -2)  # [B, N, D] -> [B, D]
            else:
                h_V_out = tf.math.reduce_sum(h_V_out * mask, -2)  # [B, N, D] -> [B, D]
                h_V_out = tf.math.divide_no_nan(
                    h_V_out, tf.math.reduce_sum(mask, -2)
                )  # [B, D]
        out = h_V_out
        # out = self.dense(out, training=train)
        if self.multiclass:
            out = self.dense(out, training=train)  # [B, N, 3]
        else:
            out = tf.squeeze(self.dense(out, training=train), -1)  # + 0.5 # [B, N]

        return out
