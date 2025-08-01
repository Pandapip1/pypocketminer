import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import LayerNormalization

from pypocketminer.models.features.positional import PositionalEncodings
from pypocketminer.models.gvp import GVP


class StructuralFeatures(Model):
    def __init__(
        self,
        node_features,
        edge_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        ablate_sidechain_vectors=True,
        ablate_rbf=False,
    ):
        super(StructuralFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)

        # Normalization and embedding
        vo, so = node_features
        ve, se = edge_features
        if ablate_sidechain_vectors:
            vi_v = 3
        else:
            vi_v = 4
        self.node_embedding = GVP(vi=vi_v, vo=vo, so=so, nlv=None, nls=None)
        if ablate_sidechain_vectors:
            vi_e = 1
        else:
            vi_e = 2
        self.edge_embedding = GVP(vi=vi_e, vo=ve, so=se, nlv=None, nls=None)
        self.norm_nodes = LayerNormalization()
        self.norm_edges = LayerNormalization()

        # ablation settings
        self.ablate_sidechain_vectors = ablate_sidechain_vectors
        self.ablate_rbf = ablate_rbf

    def _dist(self, X, mask, eps=1e-6):  # [B, N, 3]
        """Pairwise euclidean distances"""
        # Convolutional network on NCHW
        mask = tf.cast(mask, tf.float32)
        mask_2D = tf.expand_dims(mask, 1) * tf.expand_dims(mask, 2)
        dX = tf.expand_dims(X, 1) - tf.expand_dims(X, 2)
        D = mask_2D * tf.math.sqrt(tf.math.reduce_sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max = tf.math.reduce_max(D, -1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = tf.math.top_k(-D_adjust, k=min(self.top_k, tf.shape(X)[1]))
        D_neighbors = -D_neighbors
        mask_neighbors = gather_edges(tf.expand_dims(mask_2D, -1), E_idx)

        return D_neighbors, E_idx, mask_neighbors

    def _directions(self, X, E_idx):
        # X: B, N, 3
        dX = X[:, 1:, :] - X[:, :-1, :]
        X_neighbors = gather_nodes(X, E_idx)
        dX = X_neighbors - tf.expand_dims(X, -2)
        dX = normalize(dX, axis=-1)
        return dX

    def _terminal_sidechain_direction(self, X, E_idx):
        # ['N', 'CA', 'C', 'O', 'T']
        # X: B, N, 5, 3
        ca, t = X[:, :, 1, :], X[:, :, 4, :]
        # convert t from B, N, 3 to B, N, K, 3
        X_neighbors = gather_nodes(t, E_idx)
        # subtract C-alpha positions from t positions
        dX = X_neighbors - tf.expand_dims(ca, -2)
        # normalize to unit vector
        dx = normalize(dX)
        return dX

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0.0, 20.0, self.num_rbf
        D_mu = tf.linspace(D_min, D_max, D_count)
        D_mu = tf.reshape(D_mu, [1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = tf.expand_dims(D, -1)
        RBF = tf.math.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

        return RBF

    def _orientations(self, X):
        # X: B, N, 3
        forward = normalize(X[:, 1:] - X[:, :-1])
        backward = normalize(X[:, :-1] - X[:, 1:])
        forward = tf.pad(forward, [[0, 0], [0, 1], [0, 0]])
        backward = tf.pad(backward, [[0, 0], [1, 0], [0, 0]])
        return tf.concat(
            [tf.expand_dims(forward, -1), tf.expand_dims(backward, -1)], -1
        )  # B, N, 3, 2

    def _sidechains(self, X):
        # ['N', 'CA', 'C', 'O']
        # X: B, N, 4, 3
        n, origin, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perp = normalize(tf.linalg.cross(c, n))
        vec = -bisector * tf.math.sqrt(1 / 3) - perp * tf.math.sqrt(2 / 3)
        return vec  # B, N, 3

    def _sidechain_terminal_vector(self, X):
        # ['N', 'CA', 'C', 'O', 'T']
        # X: B, N, 5, 3
        ca, t = X[:, :, 1, :], X[:, :, 4, :]
        return normalize(t - ca)  # B, N, 3

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = tf.reshape(X[:, :, :3, :], [tf.shape(X)[0], 3 * tf.shape(X)[1], 3])

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = normalize(dX, axis=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]

        # Backbone normals
        n_2 = normalize(tf.linalg.cross(u_2, u_1), axis=-1)
        n_1 = normalize(tf.linalg.cross(u_1, u_0), axis=-1)

        # Angle between normals
        cosD = tf.math.reduce_sum(n_2 * n_1, -1)
        cosD = tf.clip_by_value(cosD, -1 + eps, 1 - eps)
        D = tf.math.sign(tf.math.reduce_sum(u_2 * n_1, -1)) * tf.math.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = tf.pad(D, [[0, 0], [1, 2]])  # what dims!
        D = tf.reshape(D, [tf.shape(D)[0], int(tf.shape(D)[1] / 3), 3])

        # Lift angle representations to the circle
        D_features = tf.concat([tf.math.cos(D), tf.math.sin(D)], 2)
        return D_features

    def call(self, X, mask):
        """Featurize coordinates as an attributed graph"""

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        # Pairwise features
        E_directions = self._directions(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)
        E_positional = self.embeddings(E_idx)

        if not self.ablate_sidechain_vectors:
            E_sidechain_directions = self._terminal_sidechain_direction(X, E_idx)

        # Full backbone angles
        # V_sidechains is a scalar
        V_dihedrals = self._dihedrals(X)
        V_orientations = self._orientations(X_ca)
        V_sidechains = self._sidechains(X)

        if self.ablate_sidechain_vectors:
            V_vec = tf.concat([tf.expand_dims(V_sidechains, -1), V_orientations], -1)
        else:
            # C-alpha-terminal sidechain atom vector
            V_sidechain_terminal_vector = self._sidechain_terminal_vector(X)
            V_vec = tf.concat(
                [
                    tf.expand_dims(V_sidechains, -1),
                    tf.expand_dims(V_sidechain_terminal_vector, -1),
                    V_orientations,
                ],
                -1,
            )

        V = merge(V_vec, V_dihedrals)

        # allow ablation of RBF or sidechain vectors
        if self.ablate_sidechain_vectors:
            if self.ablate_rbf:
                E = tf.concat([E_directions, E_positional], -1)
            else:
                E = tf.concat([E_directions, RBF, E_positional], -1)
        else:
            if self.ablate_rbf:
                E = tf.concat([E_directions, E_sidechain_directions, E_positional], -1)
            else:
                E = tf.concat(
                    [E_directions, E_sidechain_directions, RBF, E_positional], -1
                )

        # Embed the nodes
        Vv, Vs = self.node_embedding(V, return_split=True)
        V = merge(Vv, self.norm_nodes(Vs))

        Ev, Es = self.edge_embedding(E, return_split=True)
        E = merge(Ev, self.norm_edges(Es))

        return V, E, E_idx
