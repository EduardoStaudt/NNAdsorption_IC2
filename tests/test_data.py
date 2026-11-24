import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from conftest import IC1_ROOT, REPO_ROOT
from ic1_reference import X_RANGES, load_ic1_module, sample_cfg_22, vetor_X_v3
from nnads_ic2.config import load_config
from nnads_ic2.data.cases import prefilter, sample_case_ic1, sample_cases, severity_ic1
from nnads_ic2.data.loader import dataset_info, load_recorder_h5, write_recorder_h5
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


def test_dataset_tag_and_solver_statistics_are_stored(tmp_path):
    rng = np.random.default_rng(1)
    items = [(s, p, fake_solution(p, 21, 11, rng)) for s, p in sample_cases(range(30), CASES)[:2]]
    path = os.path.join(tmp_path, "provisorio_MOL_test.h5")
    write_recorder_h5(path, items, attrs=dict(dataset_tag="provisório_MOL", solver_cfg='{"nx": 11}'),
                      row_data=dict(solve_wall_s=[1.5, 2.5]))
    info = dataset_info(path)
    assert info["dataset_tag"] == "provisório_MOL"
    assert info["source"] == "MOL reference solver"
    assert info["n_rows"] == 2 and info["n_valid"] == 2
    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["solve_info/solve_wall_s"][:], [1.5, 2.5])
    assert len(load_recorder_h5(path, n_comp=2)) == 2  # extra groups do not disturb the loader


def test_mol_dataset_script_tags_the_file_and_logs_the_cost(tmp_path):
    out = os.path.join(tmp_path, "provisorio_MOL_tiny.h5")
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "make_mol_dataset.py"), "--n-cases", "2", "--workers", "2",
           "--out", out, "--set", "solver.nx=21", "solver.n_out=41"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    info = dataset_info(out)
    assert info["dataset_tag"] == "provisório_MOL" and "NOT the v23b" in info["source"]
    assert info["n_valid"] == 2
    cases = load_recorder_h5(out, n_comp=2, use_internal_profiles=True)
    assert [c.seed for c in cases] == sorted(c.seed for c in cases)
    assert cases[0].profiles["T"].shape == (11, 21)  # 41 output times, every 4th kept
    with open(os.path.splitext(out)[0] + ".cost.csv", encoding="utf-8") as f:
        assert f.readline().startswith("seed,tst_max_s,severity,t_final_s,wall_s")


def test_rar_pool_keeps_its_size_and_the_new_points():
    pool = CollocationPool(n_init=1000, n_cases=3, t_final=2.0, max_size=1200, seed=0)
    z, t, case = pool.candidates(500)
    idx = select_largest(-np.abs(z - 0.5), 400)  # the 400 points closest to z = 0.5
    pool.add(z[idx], t[idx], case[idx])
    assert pool.size == 1200 and pool.from_rar.sum() == 400
    assert np.all(np.abs(pool.z[pool.from_rar] - 0.5) <= np.sort(np.abs(z - 0.5))[399] + 1e-15)


def three_gas_case():
    """IC1 example case plus a third gas (a copy of CO2 with lower affinity), feed 50/30/20."""
    from nnads_ic2.physics.params import CaseParams

    p = CaseParams.from_ic1_json(os.path.join(REPO_ROOT, "configs", "cases", "ic1_example_h2_co2.json"))
    arrays = {k: np.append(getattr(p, k), getattr(p, k)[1]) for k in ("k1", "k2", "k3", "k4", "k5", "k6", "kL", "dH", "Cpg")}
    arrays["k3"][2] *= 0.5
    scalars = {k: getattr(p, k) for k in ("eb", "rho_b", "Cps", "vs", "Tin", "P", "L", "hw", "lam", "dp", "Dm", "Dt")}
    return CaseParams(**arrays, **scalars, y_feed=np.array([0.5, 0.3, 0.2]), y_init=np.array([1.0, 0.0, 0.0]))


def test_loader_reads_a_v23b_like_file_without_the_optional_series(tmp_path):
    # three gases (theta width 41), theta under another name, no c_out, no v_out, profiles without v
    p = three_gas_case()
    nt, nx, k = 50, 11, 4
    rng = np.random.default_rng(0)
    y = rng.dirichlet(np.ones(3), nt)
    T = 300.0 + rng.uniform(0, 5, nt)
    path = os.path.join(tmp_path, "v23b_like.h5")
    vlf = h5py.special_dtype(vlen=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=p.to_theta()[None, :])
        ys = f.create_dataset("y_out", (1, 3), dtype=vlf)
        for i in range(3):
            ys[0, i] = y[:, i]
        ts = f.create_dataset("T_out", (1,), dtype=vlf)
        ts[0] = T
        f.create_dataset("TF", data=[100.0])
        f.create_dataset("NT", data=[nt])
        g = f.create_group("profiles/0")
        g.create_dataset("t", data=np.linspace(0, 100, k))
        g.create_dataset("c", data=rng.uniform(0, 1, (k, 3, nx)))
        g.create_dataset("q", data=rng.uniform(0, 1, (k, 3, nx)))
        g.create_dataset("T", data=rng.uniform(300, 305, (k, nx)))
        g.attrs["z"] = np.linspace(0, p.L, nx)
    keys = {"theta": "X"}
    (case,) = load_recorder_h5(path, keys=keys, use_internal_profiles=True)
    assert case.params.n_comp == 3 and case.y_out.shape == (nt, 3)
    np.testing.assert_allclose(case.params.y_feed, [0.5, 0.3, 0.2])
    assert case.derived == ("c_out",) and case.v_out is None
    from nnads_ic2.physics.constants import R

    np.testing.assert_allclose(case.c_out, y.astype(np.float32) * p.P / (R * T.astype(np.float32)[:, None]), rtol=1e-6)
    assert set(case.profiles) == {"t", "z", "c", "q", "T"}
    np.testing.assert_allclose(case.t, (np.arange(nt) + 1) * 100.0 / nt)
    info = dataset_info(path, keys)
    assert info["n_comp"] == 3 and info["n_valid"] == 1 and not info["has"]["v_out"] and not info["has"]["c_out"]
    with pytest.raises(ValueError):
        load_recorder_h5(path, n_comp=2, keys=keys)
