import os
import sys

# Same runtime fixes as the scripts (ptxas for XLA, TF32 off); must run before TF is imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nnads_ic2 import runtime  # noqa: E402,F401

import tensorflow as tf  # noqa: E402

tf.config.experimental.enable_tensor_float_32_execution(False)

import pytest  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# IC1 repository (sibling folder) used to cross-check the isotherm and the example case.
IC1_ROOT = os.environ.get("NNADS_IC1_ROOT", os.path.join(REPO_ROOT, "..", "NNAdsorption_IC1"))
EXAMPLE_CASE = os.path.join(REPO_ROOT, "configs", "cases", "ic1_example_h2_co2.json")

# Small settings for tests (the real values live in configs/*.yaml).
TEST_SCALES = dict(q_ref_floor=1e-6, dT_ref_floor=1.0, t_final_factor=2.0)
TEST_SOLVER = dict(nx=61, scheme="vanleer", fp_iters=8, n_out=401, rtol=1e-6, atol_rel=1e-8, method="BDF")


@pytest.fixture(scope="session")
def example_case():
    from nnads_ic2.physics.params import CaseParams
    from nnads_ic2.physics.scales import compute_scales

    p = CaseParams.from_ic1_json(EXAMPLE_CASE)
    return p, compute_scales(p, **TEST_SCALES)


@pytest.fixture(scope="session")
def mol_example(example_case):
    """Reference MOL solution of the IC1 example case (computed once, ~4 s)."""
    from nnads_ic2.physics.mol_solver import solve_case

    p, s = example_case
    return solve_case(p, s.t_final, TEST_SOLVER)
