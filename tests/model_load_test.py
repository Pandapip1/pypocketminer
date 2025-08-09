
def test_model_loads_checkpoint(tmp_path):
    from pypocketminer.models.pretrained import pocketminer_v1_status

    pocketminer_v1_status.assert_consumed()
