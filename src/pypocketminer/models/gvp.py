import tensorflow as tf
import tqdm
from tensorflow.keras import Model
from tensorflow.keras.layers import *

from .utils import merge, split, norm_no_nan


class GVP(Model):
    def __init__(self, vi, vo, so, nlv=tf.math.sigmoid, nls=tf.nn.relu):
        """[v/s][i/o] = number of [vector/scalar] channels [in/out]"""
        super(GVP, self).__init__()
        if vi:
            self.wh = Dense(max(vi, vo))
        self.ws = Dense(so, activation=nls)
        if vo:
            self.wv = Dense(vo)
        self.vi, self.vo, self.so, self.nlv = vi, vo, so, nlv

    def call(self, x, return_split=False):
        # X: [..., 3*vi + si]
        # if split, returns: [..., 3, vo], [..., so[
        # if not split, returns [..., 3*vo + so]
        v, s = split(x, self.vi)
        if self.vi:
            vh = self.wh(v)
            vn = norm_no_nan(vh, axis=-2)
            out = self.ws(tf.concat([s, vn], -1))
        else:
            out = self.ws(s)
        if self.vo:
            vo = self.wv(vh)
            if self.nlv:
                vo *= self.nlv(norm_no_nan(vo, axis=-2, keepdims=True))
            out = (vo, out) if return_split else merge(vo, out)
        return out
