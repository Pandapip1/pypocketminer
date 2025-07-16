from .base import Encoder, Decoder
from .layers.gvp import GVPDropout, GVPLayerNorm

VGEncoder = Encoder
VGDecoder = Decoder

Velu = GVP
VGDropout = GVPDropout
VGLayerNorm = GVPLayerNorm
