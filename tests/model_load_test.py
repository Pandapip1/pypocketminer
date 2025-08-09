import os
from pypocketminer.validate_performance_on_xtals import process_strucs
import mdtraj as md
import numpy as np

def test_model_loads_checkpoint():
    from pypocketminer.models.pretrained import pocketminer_v1_status

    pocketminer_v1_status.assert_consumed()

def test_model_runs():
    from pypocketminer.models.pretrained import pocketminer_v1 as model

    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1JWP.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(model(X, S, mask, train=False, res_level=True))

    print(preds)

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
