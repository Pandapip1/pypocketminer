import pytest
import tensorflow as tf
from pypocketminer.models.mqa_model import MQAModel
from pypocketminer.util import load_checkpoint

def test_model_loads_checkpoint(tmp_path):
    from pypocketminer.models.pretrained import pocketminer_v1, pocketminer_v1_status

    pocketminer_v1_status.assert_consumed()
