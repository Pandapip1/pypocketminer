import os
from pypocketminer.validate_performance_on_xtals import process_strucs
from pypocketminer.models.pretrained import pocketminer_v1
import mdtraj as md
import numpy as np
import tempfile
import pickle

def test_model_runs_1JWP():
    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1JWP.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(pocketminer_v1(X, S, mask, train=False, res_level=True))[0]

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
    assert len(preds) == 263

def test_pretrained_model_serializes_deserializes_runs_1JWP():
    with tempfile.TemporaryFile("wb+") as temppkl:
        pkl = pickle.dump(pocketminer_v1, temppkl)
        temppkl.flush()
        temppkl.seek(0)
        pocketminer_v1_restored = pickle.load(temppkl)

    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1JWP.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(pocketminer_v1_restored(X, S, mask, train=False, res_level=True))[0]

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
    assert len(preds) == 263

def test_model_runs_1EXM():
    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1EXM.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(pocketminer_v1(X, S, mask, train=False, res_level=True))[0]

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
    assert len(preds) == 403

def test_model_runs_1NEP():
    pdb = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1NEP.pdb"))

    X, S, mask = process_strucs([pdb])

    preds = np.array(pocketminer_v1(X, S, mask, train=False, res_level=True))[0]

    assert np.all(0 <= preds)
    assert np.all(preds <= 1)
    assert len(preds) == 130
