from pypocketminer.models.mqa_model import MQAModel
import tensorflow as tf
import importlib.util
import os

pname = "pypocketminer"

spec = importlib.util.find_spec(pname)
if spec is None or spec.origin is None:
    raise Exception(f"Package '{pname}' not installed correctly.")

package_path = os.path.dirname(spec.origin)
checkpoints_dir = os.path.join(package_path, "models", "pretrained", "checkpoints")

pocketminer_v1_path = os.path.join(checkpoints_dir, "pocketminer.index")
pocketminer_v1 = MQAModel(
    node_features=(8, 50),
    edge_features=(1, 32),
    hidden_dim=(16, 100),
    num_layers=4,
    dropout=0.1
)
pocketminer_v1_ckpt = tf.train.Checkpoint(
    model=pocketminer_v1,
)
pocketminer_v1_status = pocketminer_v1_ckpt.restore(pocketminer_v1_path)
