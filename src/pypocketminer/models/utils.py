def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = tf.maximum(tf.math.reduce_sum(tf.math.square(x), axis, keepdims), eps)
    return (tf.sqrt(out) if sqrt else out)

# [..., 3*nv + ns] -> [..., 3, nv], [..., ns]
# nv = number of vector channels
# ns = number of scalar channels
# vector channels are ALWAYS at the top!
def split(x, nv):
    v = tf.reshape(x[..., :3*nv], x.shape[:-1] + [3, nv])
    s = x[..., 3*nv:]
    return v, s

# [..., 3, nv], [..., ns] -> [..., 3*nv + ns]
def merge(v, s):
    v = tf.reshape(v, v.shape[:-2] + [3*v.shape[-1]])
    return tf.concat([v, s], -1)

# Concat in a way that keeps vector channels at the top
def vs_concat(x1, x2, nv1, nv2):
    
    v1, s1 = split(x1, nv1)
    v2, s2 = split(x2, nv2)
    
    v = tf.concat([v1, v2], -1)
    s = tf.concat([s1, s2], -1)
    return merge(v, s)

def autoregressive_mask(E_idx):
    N_nodes = tf.shape(E_idx)[1]
    ii = tf.range(N_nodes)
    ii = tf.reshape(ii, [1, -1, 1])
    mask = E_idx - ii < 0
    mask = tf.cast(mask, tf.float32)
    return mask


def normalize(tensor, axis=-1):
    return tf.math.divide_no_nan(
        tensor, tf.linalg.norm(tensor, axis=axis, keepdims=True)
    )


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    edge_features = tf.gather(edges, neighbor_idx, axis=2, batch_dims=2)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK]
    neighbors_flat = tf.reshape(neighbor_idx, [neighbor_idx.shape[0], -1])

    # Gather and re-pack
    # nodes [B, N, C], neighbors_flat [B, NK, C] => [B, NK, C]
    # tf: nf[i][j][k] = nodes[i][nf[i][j]][k]
    neighbor_features = tf.gather(nodes, neighbors_flat, axis=1, batch_dims=1)
    neighbor_features = tf.reshape(
        neighbor_features, list(neighbor_idx.shape)[:3] + [-1]
    )  # => [B, N, K, C]
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx, nv_nodes, nv_neighbors):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return vs_concat(h_neighbors, h_nodes, nv_neighbors, nv_nodes)
