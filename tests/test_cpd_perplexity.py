import tensorflow as tf
import sys

import numpy as np

from pypocketminer import utils
from pypocketminer.models.cpd_model import CPDModel
from pypocketminer.datasets import cath_dataset

model = CPDModel(node_features=(8, 100), edge_features=(1, 32), hidden_dim=(16, 100))
optimizer = tf.keras.optimizers.Adam()

utils.load_checkpoint(model, optimizer, sys.argv[1])

_, _, testset = cath_dataset(3000)  # fix this to only give individual amino acids
loss, acc, confusion = utils.loop(testset, model, train=False)
print("ALL TEST PERPLEXITY {}, ACCURACY {}".format(np.exp(loss), acc))
utils.save_confusion(confusion)

_, _, testset = cath_dataset(3000, filter_file="../data/test_split_L100.json")
loss, acc, confusion = utils.loop(testset, model, train=False)
print("SHORT TEST PERPLEXITY {}, ACCURACY {}".format(np.exp(loss), acc))
utils.save_confusion(confusion)

_, _, testset = cath_dataset(3000, filter_file="../data/test_split_sc.json")
loss, acc, confusion = utils.loop(testset, model, train=False)
print("SINGLE CHAIN TEST PERPLEXITY {}, ACCURACY {}".format(np.exp(loss), acc))
utils.save_confusion(confusion)
