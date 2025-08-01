from pypocketminer.models.base import Encoder, Decoder
from pypocketminer.models.layers.gvp import GVPDropout, GVPLayerNorm
from pypocketminer.models.gvp import GVP

VGEncoder = Encoder
VGDecoder = Decoder

Velu = GVP
VGDropout = GVPDropout
VGLayerNorm = GVPLayerNorm
