import os

import numpy as np
import pytest

from conftest import IC1_ROOT, REPO_ROOT
from ic1_reference import X_RANGES, load_ic1_module, sample_cfg_22, vetor_X_v3
from nnads_ic2.config import load_config
from nnads_ic2.data.cases import prefilter, sample_case_ic1, sample_cases, severity_ic1
from nnads_ic2.data.loader import load_recorder_h5, write_recorder_h5
from nnads_ic2.data.sampling import CollocationPool, select_largest

CFG = load_config(os.path.join(REPO_ROOT, "configs", "pinn_base.yaml"))
CASES = CFG["cases"]
SEEDS = [0, 1, 2, 17, 12345, 69000000]


def test_config_ranges_are_the_ic1_contract():
    assert {k: tuple(v) for k, v in CASES["x_ranges"].items()} == X_RANGES
    cio = load_ic1_module(IC1_ROOT, "src/contrato/contract_io.py", "ic1_contract_io")
    if cio is not None:
        assert {k: tuple(v) for k, v in CASES["x_ranges"].items()} == cio.X_RANGES


@pytest.mark.parametrize("seed", SEEDS)
def test_sampler_reproduces_ic1_recorder(seed):
    lo, hi = CASES["y0_range"]
    p = sample_case_ic1(seed, CASES["x_ranges"], CASES["y0_range"], CASES["log_params"])
    ref = vetor_X_v3(sample_cfg_22(seed, lo, hi))
    np.testing.assert_allclose(p.to_theta(), ref.astype(np.float64), rtol=1e-6)


def test_severity_matches_live_ic1_features():
    feat = load_ic1_module(IC1_ROOT, "src/contrato/features.py", "ic1_features")
    if feat is None:
        pytest.skip("NNAdsorption_IC1 folder not found")
    for seed in SEEDS:
        p = sample_case_ic1(seed, CASES["x_ranges"], CASES["y0_range"], CASES["log_params"])
        ref = feat.severidade_regime(p.to_theta()[None, :])[0]
        assert severity_ic1(p) == pytest.approx(ref, abs=1e-9)


def test_prefilter_keeps_only_fast_cases_inside_the_domain():
    kept = sample_cases(range(200), CASES)
    assert 0 < len(kept) < 200
    for _, p in kept:
        ok, info = prefilter(p, CASES["max_tst_s"], CASES["sev_max"])
        assert ok and info["tst_max_s"] <= CASES["max_tst_s"] and info["severity"] < CASES["sev_max"]


def fake_solution(p, nt, nx, rng):
    t = np.linspace(0, 100.0, nt)
    c = rng.uniform(1, 10, (nt, p.n_comp, nx))
    return dict(
        z=(np.arange(nx) + 0.5) / nx * p.L, t=t, c=c, q=rng.uniform(0, 1, (nt, p.n_comp, nx)),
        T=rng.uniform(300, 310, (nt, nx)), v_cell=rng.uniform(0.01, 0.02, (nt, nx)),
        c_out=c[:, :, -1], y_out=c[:, :, -1] / c[:, :, -1].sum(1, keepdims=True),
        T_out=rng.uniform(300, 310, nt), v_out=rng.uniform(0.01, 0.02, nt),
    )


@pytest.mark.parametrize("profiles", [False, True])
def test_recorder_h5_round_trip(tmp_path, profiles):
    rng = np.random.default_rng(0)
    items = [(s, p, fake_solution(p, 41, 11, rng)) for s, p in sample_cases(range(30), CASES)[:3]]
    path = os.path.join(tmp_path, "ds.h5")
    write_recorder_h5(path, items, write_profiles=profiles)
    cases = load_recorder_h5(path, n_comp=2, use_internal_profiles=True)
    assert len(cases) == 3
    for case, (seed, p, sol) in zip(cases, items):
        assert case.seed == seed
        np.testing.assert_allclose(case.theta, p.to_theta(), rtol=1e-6)  # stored as float32 like IC1
        np.testing.assert_allclose(case.t, sol["t"][1:], rtol=1e-6)  # t_k = (k + 1) TF / NT
        np.testing.assert_allclose(case.y_out, sol["y_out"][1:], rtol=1e-6)
        np.testing.assert_allclose(case.v_out, sol["v_out"][1:], rtol=1e-6)
        assert case.final["q"].shape == (2, 11)
        assert (case.profiles is not None) == profiles
        if profiles:
            assert case.profiles["c"].shape == (41, 2, 11)


def test_rar_pool_keeps_its_size_and_the_new_points():
    pool = CollocationPool(n_init=1000, n_cases=3, t_final=2.0, max_size=1200, seed=0)
    z, t, case = pool.candidates(500)
    idx = select_largest(-np.abs(z - 0.5), 400)  # the 400 points closest to z = 0.5
    pool.add(z[idx], t[idx], case[idx])
    assert pool.size == 1200 and pool.from_rar.sum() == 400
    assert np.all(np.abs(pool.z[pool.from_rar] - 0.5) <= np.sort(np.abs(z - 0.5))[399] + 1e-15)
