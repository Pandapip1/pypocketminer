class CPDModel(Model):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_layers=3,
        num_letters=20,
        k_neighbors=30,
        dropout=0.1,
    ):
        super(CPDModel, self).__init__()

        # Hyperparameters
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features

        # Featurization layers
        self.features = StructuralFeatures(
            node_features, edge_features, top_k=k_neighbors
        )

        # Embedding layers
        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs, nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs, nls=None, nlv=None)
        self.W_s = Embedding(num_letters, self.hs)
        self.encoder = Encoder(hidden_dim, edge_features, num_layers=num_layers)
        self.decoder = Decoder(
            hidden_dim, edge_features, s_features=(0, self.hs), num_layers=num_layers
        )
        self.W_out = GVP(vi=self.hv, vo=0, so=num_letters, nls=None, nlv=None)

    def call(self, X, S, mask, train=False):
        # X [B, N, 4, 3], S [B, N], mask [B, N]

        V, E, E_idx = self.features(X, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)
        h_S = self.W_s(S)
        h_V = self.decoder(h_V, h_S, h_E, E_idx, mask, train=train)
        logits = self.W_out(h_V)

        return logits

    def sample(self, X, mask=None, temperature=0.1):
        V, E, E_idx = self.features(X, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=False)
        return self.decoder.sample(
            h_V, h_E, E_idx, mask, W_s=self.W_s, W_out=self.W_out, temperature=0.1
        )
