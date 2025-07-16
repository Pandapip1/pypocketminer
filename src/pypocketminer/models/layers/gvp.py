from ..utils import merge, split, norm_no_nan

# Dropout that drops vector and scalar channels separately
class GVPDropout(Layer):
    def __init__(self, rate, nv):
        super(GVPDropout, self).__init__()
        self.nv = nv
        self.vdropout = Dropout(rate, noise_shape=[1, nv])
        self.sdropout = Dropout(rate)
    def call(self, x, training):
        if not training: return x
        v, s = split(x, self.nv)
        v, s = self.vdropout(v), self.sdropout(s)
        return merge(v, s)

# Normal layer norm for scalars, nontrainable norm for vectors
class GVPLayerNorm(Layer):
    def __init__(self, nv):
        super(GVPLayerNorm, self).__init__()
        self.nv = nv
        self.snorm = LayerNormalization()
    def call(self, x):
        v, s = split(x, self.nv)
        vn = norm_no_nan(v, axis=-2, keepdims=True, sqrt=False) # [..,1, nv]
        vn = tf.sqrt(tf.math.reduce_mean(vn, axis=-1, keepdims=True))
        return merge(v/vn, self.snorm(s))
