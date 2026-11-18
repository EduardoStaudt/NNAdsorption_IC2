import json
import os

import numpy as np
import pytest

from conftest import IC1_ROOT, REPO_ROOT
from ic1_reference import cfg_from_json, load_ic1_module, vetor_X_v3
from nnads_ic2.physics.params import (
    CaseParams,
    ThetaScaler,
    theta_dim,
    theta_log_mask,
    theta_names,
)

EXAMPLE = os.path.join(REPO_ROOT, "configs", "cases", "ic1_example_h2_co2.json")


def load_example():
    with open(EXAMPLE, encoding="utf-8") as f:
        return json.load(f)


def test_example_case_is_the_ic1_one():
    original = os.path.join(IC1_ROOT, "src", "example_input.json")
    if not os.path.exists(original):
        pytest.skip("NNAdsorption_IC1 folder not found")
    with open(original, encoding="utf-8") as f:
        assert json.load(f) == load_example()


def test_theta_matches_ic1_vector_for_two_components():
    e = load_example()
    p = CaseParams.from_ic1_json(e)
    ref = vetor_X_v3(cfg_from_json(e))  # IC1 stores float32
    theta = p.to_theta()
    assert theta.shape == (31,)
    np.testing.assert_allclose(theta, ref.astype(np.float64), rtol=1e-6)
    # the costly detail of IC1: Dt is column 29, y0 is column 30
    names = theta_names(2)
    assert names[29] == "Dt" and names[30] == "y0"
    assert theta[29] == pytest.approx(e["Dt"]) and theta[30] == pytest.approx(e["y0"])


def test_theta_names_match_live_ic1_contract():
    cio = load_ic1_module(IC1_ROOT, "src/contrato/contract_io.py", "ic1_contract_io")
    if cio is None:
        pytest.skip("NNAdsorption_IC1 folder not found")
    assert theta_names(2) == cio.x_column_names()
    assert theta_dim(2) == cio.X_DIM == 31


def test_ic1_cfg_and_json_give_the_same_case():
    e = load_example()
    a = CaseParams.from_ic1_json(e)
    b = CaseParams.from_ic1_cfg(cfg_from_json(e))
    np.testing.assert_allclose(a.to_theta(), b.to_theta(), rtol=1e-14)
    np.testing.assert_array_equal(a.y_init, [1.0, 0.0])  # bed starts with pure carrier


@pytest.mark.parametrize("n", [2, 3, 5])
def test_theta_round_trip(n):
    rng = np.random.default_rng(n)
    y = rng.dirichlet(np.ones(n))
    p = CaseParams(
        k1=rng.uniform(1, 15, n), k2=rng.uniform(-0.1, 0, n), k3=rng.uniform(1e-6, 1e-3, n),
        k4=rng.uniform(0, 3200, n), k5=rng.uniform(-3, 3, n), k6=rng.uniform(-2200, 2200, n),
        kL=rng.uniform(0.01, 1, n), dH=rng.uniform(5e3, 5e4, n), Cpg=rng.uniform(25, 45, n),
        eb=0.4, rho_b=600.0, Cps=1000.0, vs=0.02, Tin=300.0, P=1e6, L=1.0, hw=50.0, lam=0.5,
        dp=2e-3, Dm=1e-5, Dt=0.03, y_feed=y, y_init=np.eye(n)[0],
    )
    theta = p.to_theta()
    assert theta.shape == (theta_dim(n),) == (len(theta_names(n)),)
    q = CaseParams.from_theta(theta, n)
    np.testing.assert_allclose(q.to_theta(), theta, rtol=1e-13)
    for name in ["k1", "k2", "k3", "k4", "k5", "k6", "y_feed"]:
        np.testing.assert_allclose(getattr(q, name), getattr(p, name), rtol=1e-12, atol=1e-12)


def test_log_mask_and_scaler():
    mask = theta_log_mask(2, ["B_ref", "kL", "dp", "Dm"])
    names = np.array(theta_names(2))
    assert set(names[mask]) == {"L0_c0_B_ref", "L0_c1_B_ref", "L0_c0_kL", "L0_c1_kL", "dp", "Dm"}
    theta = CaseParams.from_ic1_json(load_example()).to_theta()[None, :]
    sc = ThetaScaler.fit(theta, mask, sd_floor=1e-8)  # one case: every sd -> 1
    np.testing.assert_allclose(sc.transform(theta), 0.0, atol=1e-12)
    sc2 = ThetaScaler.from_dict(sc.to_dict())
    np.testing.assert_allclose(sc2.transform(theta), sc.transform(theta))
