import os
from pypocketminer.validate_performance_on_xtals import process_strucs
from pypocketminer.models.pretrained import pocketminer_v1, pocketminer_v1_status
import mdtraj as md
import numpy as np

def test_model_loads_checkpoint():
    pocketminer_v1_status.assert_consumed()

def test_model_runs_1JWP():
    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1JWP.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(pocketminer_v1(X, S, mask, train=False, res_level=True))[0]

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
    assert len(preds) == 263

def test_model_runs_1EXM():
    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1EXM.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(pocketminer_v1(X, S, mask, train=False, res_level=True))[0]

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
    assert len(preds) == 263

def test_model_runs_1NEP():
    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1NEP.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(pocketminer_v1(X, S, mask, train=False, res_level=True))[0]

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
    assert len(preds) == 263
