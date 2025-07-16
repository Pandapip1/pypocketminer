class Encoder(Model):
    def __init__(self, node_features, edge_features, num_layers=3, dropout=0.1):
        super(Encoder, self).__init__()

        # Hyperparameters
        self.nv, ns = node_features
        self.ev, _ = edge_features

        # Encoder layers
        self.vglayers = [
            MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
            for _ in range(num_layers)
        ]

    def call(self, h_V, h_E, E_idx, mask, train=False):
        # Encoder is unmasked self-attention

        mask_attend = tf.squeeze(gather_nodes(tf.expand_dims(mask, -1), E_idx), -1)
        # [B, N] => [B, N, 1] => [B, N, K, 1] => [B, N, K]
        mask_attend = tf.expand_dims(mask, -1) * mask_attend

        for layer in self.vglayers:
            h_M = cat_neighbors_nodes(
                h_V, h_E, E_idx, self.nv, self.ev
            )  # nv = self.hv + 1
            h_V = layer(h_V, h_M, mask_V=mask, mask_attend=mask_attend, train=train)

        return h_V


class Decoder(Model):  # DECODER
    def __init__(
        self, node_features, edge_features, s_features, num_layers=3, dropout=0.1
    ):
        super(Decoder, self).__init__()

        # Hyperparameters
        self.nv, self.ns = node_features
        self.ev, self.es = edge_features
        self.sv, self.ss = s_features

        # Decoder layers
        self.vglayers = [
            MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
            for _ in range(num_layers)
        ]

    def call(self, h_V, h_S, h_E, E_idx, mask, train=False):
        # h_V [B, N, *], h_S [B, N, *], mask [B, N]
        # h_E [B, N, K, *], E_idx [B, N, K]

        # Concatenate sequence embeddings for autoregressive decoder
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx, 0, self.ev)  # nv = 1

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(
            tf.zeros_like(h_S), h_E, E_idx, self.sv, self.ev
        )  # nv = 1
        h_ESV_encoder = cat_neighbors_nodes(
            h_V, h_ES_encoder, E_idx, self.nv, self.sv + self.ev
        )  # nv = self.nv+1

        # Decoder uses masked self-attention
        mask_attend = tf.expand_dims(autoregressive_mask(E_idx), -1)
        mask_1D = tf.cast(tf.expand_dims(tf.expand_dims(mask, -1), -1), tf.float32)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_ESV_encoder_fw = mask_fw * h_ESV_encoder

        for layer in self.vglayers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(
                h_V, h_ES, E_idx, self.nv, self.ev
            )  # nv = self.nv + 1
            h_M = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_M, mask_V=mask, train=train)

        return h_V

    # This is slow because TensorFlow doesn't allow indexed tensor writes
    # at runtime, so we have to move between CPU/GPU at every step.
    # If you can find a way around this, it will run a lot faster
    def sample(self, h_V, h_E, E_idx, mask, W_s, W_out, temperature=0.1):
        mask_attend = tf.expand_dims(autoregressive_mask(E_idx), -1)
        mask_1D = tf.reshape(mask, [mask.shape[0], mask.shape[1], 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = h_V.shape[0], h_V.shape[1]

        h_S = np.zeros((N_batch, N_nodes, self.ss), dtype=np.float32)
        S = np.zeros((N_batch, N_nodes), dtype=np.int32)
        h_V_stack = [tf.split(h_V, N_nodes, 1)] + [
            tf.split(tf.zeros_like(h_V), N_nodes, 1) for _ in range(len(self.vglayers))
        ]
        for t in tqdm.trange(N_nodes):
            # Hidden layers
            E_idx_t = E_idx[:, t : t + 1, :]
            h_E_t = h_E[:, t : t + 1, :, :]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t, 0, self.ev)
            # Stale relational features for future states
            h_ESV_encoder_t = mask_fw[:, t : t + 1, :, :] * cat_neighbors_nodes(
                h_V, h_ES_t, E_idx_t, self.nv, self.ev
            )
            for l, layer in enumerate(self.vglayers):
                # Updated relational features for future states
                h_ESV_decoder_t = cat_neighbors_nodes(
                    tf.stack(h_V_stack[l], 1), h_ES_t, E_idx_t, self.nv, self.ev
                )
                h_V_t = h_V_stack[l][t]  # [:,t:t+1,:]
                h_ESV_t = (
                    mask_bw[:, t : t + 1, :, :] * h_ESV_decoder_t + h_ESV_encoder_t
                )
                mask_to_pass = mask[:, t : t + 1]
                tmp = layer(h_V_t, h_ESV_t, mask_V=mask_to_pass)
                h_V_stack[l + 1][t] = tmp
            # Sampling step
            h_V_t = tf.squeeze(h_V_stack[-1][t], 1)  # [:,t,:]
            logits = (
                W_out(h_V_t) / temperature
            )  # this is the main issue, where to get W_out?
            # probs = F.softmax(logits, dim=-1)
            S_t = tf.squeeze(tf.random.categorical(logits, 1), -1)

            # Update
            h_S[:, t, :] = W_s(S_t)  # where to get W_S?
            S[:, t] = S_t
        return S