import os
import sys

# Same runtime fixes as the scripts (ptxas for XLA, TF32 off); must run before TF is imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nnads_ic2 import runtime  # noqa: E402,F401

import tensorflow as tf  # noqa: E402

tf.config.experimental.enable_tensor_float_32_execution(False)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# IC1 repository (sibling folder) used to cross-check the isotherm and the example case.
IC1_ROOT = os.environ.get("NNADS_IC1_ROOT", os.path.join(REPO_ROOT, "..", "NNAdsorption_IC1"))
