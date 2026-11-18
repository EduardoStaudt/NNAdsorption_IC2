import os

import numpy as np
import pytest

from conftest import IC1_ROOT, REPO_ROOT
from ic1_reference import load_ic1_module
from nnads_ic2.experiment import resolve_dT_ratio
from nnads_ic2.physics.params import CaseParams
from nnads_ic2.physics.scales import (
    compute_scales,
    dimensionless_groups,
    dT_ratio_from_data,
    feed_temperature_rise,
    initial_holdup_times,
    measured_temperature_rise,
    stoich_times,
    to_dimensional,
    to_dimless,
)

EXAMPLE = os.path.join(REPO_ROOT, "configs", "cases", "ic1_example_h2_co2.json")
SCALE_CFG = dict(q_ref_floor=1e-6, dT_ref_floor=1.0, t_final_factor=2.0)


@pytest.fixture
def case():
    return CaseParams.from_ic1_json(EXAMPLE)


def test_round_trip_recovers_dimensional_values(case):
    s = compute_scales(case, **SCALE_CFG)
    rng = np.random.default_rng(0)
    n = case.n_comp
    vals = dict(
        z=rng.uniform(0, case.L, 50),
        t=rng.uniform(0, s.t_final, 50),
        c=rng.uniform(0, 2 * s.C_F, (50, n)),
        q=rng.uniform(0, 5, (50, n)),
        T=rng.uniform(280, 380, 50),
        v=rng.uniform(0, 2 * case.vs, 50),
    )
    back = to_dimensional(s, **to_dimless(s, **vals))
    for k in vals:
        np.testing.assert_allclose(back[k], vals[k], rtol=1e-14, atol=1e-12)


def test_reference_time_matches_ic1_solver_setup(case):
    sj = load_ic1_module(IC1_ROOT, "src/contrato/solver_jee.py", "ic1_solver_jee")
    if sj is None:
        pytest.skip("NNAdsorption_IC1 folder not found")
    S = sj.setup(case.to_ic1_cfg(), 71)  # IC1: TF = 2 max_i t_st,i
    s = compute_scales(case, **SCALE_CFG)
    assert s.t_final == pytest.approx(S["TF"], rel=1e-12)
    assert s.C_F == pytest.approx(S["c_tot"], rel=1e-14)


def test_example_case_numbers(case):
    # hand check of Eq. (7) for the H2/CO2 example (values in docs/nondim.md)
    tst = stoich_times(case)
    assert tst[1] > tst[0]  # CO2 (strong) is the slower front
    s = compute_scales(case, **SCALE_CFG)
    assert s.t_ref == pytest.approx(tst[1])
    t_init = initial_holdup_times(case)
    assert t_init[1] == 0.0  # CO2 is not in the bed at t = 0
    assert t_init[0] > 0.0


def test_groups_are_consistent(case):
    s = compute_scales(case, **SCALE_CFG)
    g = dimensionless_groups(case, s)
    np.testing.assert_allclose(g["beta"], g["alpha"] * g["K"], rtol=1e-14)
    assert 0 < g["alpha"] <= 1.0  # t_ref >= eb L / vs by construction
    np.testing.assert_allclose(g["tst"][np.nanargmax(g["tst"])], 1.0)
    np.testing.assert_allclose(g["c_init"], case.y_init)
    for k in ["Pe", "Pe_T", "diff_T", "St_w", "eps_T"]:
        assert np.isfinite(g[k]) and g[k] > 0
    for k in ["K", "Da", "gamma", "conv_T", "heat", "scale_mass", "scale_ldf"]:
        assert np.all(np.isfinite(g[k])) and np.all(g[k] > 0)


def test_floors(case):
    p = CaseParams.from_ic1_json(EXAMPLE)
    p.dH = np.zeros(2)  # no heat of adsorption -> dT_ref falls back to its floor
    s = compute_scales(p, **SCALE_CFG)
    assert s.dT_ref == 1.0
    # a component absent from feed and bed gets the q_ref floor
    p3 = CaseParams(
        **{k: np.append(getattr(case, k), getattr(case, k)[1]) for k in
           ["k1", "k2", "k3", "k4", "k5", "k6", "kL", "dH", "Cpg"]},
        **{k: getattr(case, k) for k in
           ["eb", "rho_b", "Cps", "vs", "Tin", "P", "L", "hw", "lam", "dp", "Dm", "Dt"]},
        y_feed=np.append(case.y_feed, 0.0), y_init=np.array([1.0, 0.0, 0.0]),
    )
    s3 = compute_scales(p3, **SCALE_CFG)
    assert s3.q_ref[2] == 1e-6
    assert np.isnan(stoich_times(p3)[2])


def test_measured_rise_becomes_dT_ref(case):
    s0 = compute_scales(case, **SCALE_CFG)  # no data: feed estimate
    assert s0.dT_ref == pytest.approx(feed_temperature_rise(case, s0.q_ref))
    rise = measured_temperature_rise(case.Tin + np.array([0.0, 2.0, 4.8, -1.0]), case.Tin)
    assert rise == pytest.approx(4.8)
    ratio = dT_ratio_from_data([case], [rise], SCALE_CFG["q_ref_floor"])
    assert compute_scales(case, **SCALE_CFG, dT_ratio=ratio).dT_ref == pytest.approx(4.8, rel=1e-12)
    # a rise below the floor gives dT_ref = 1 K
    small = dT_ratio_from_data([case], [0.3], SCALE_CFG["q_ref_floor"])
    assert compute_scales(case, **SCALE_CFG, dT_ratio=small).dT_ref == 1.0


def test_cooling_counts_as_a_temperature_change():
    assert measured_temperature_rise([300.0, 297.0, 301.0], 300.0) == pytest.approx(3.0)


def test_dT_ratio_is_the_median_over_cases_with_data(case):
    est = feed_temperature_rise(case, compute_scales(case, **SCALE_CFG).q_ref)
    rises = [0.1 * est, 0.2 * est, None, 0.4 * est]
    assert dT_ratio_from_data([case] * 4, rises, SCALE_CFG["q_ref_floor"]) == pytest.approx(0.2)
    assert dT_ratio_from_data([case], [None], SCALE_CFG["q_ref_floor"]) is None


def test_no_data_falls_back_to_the_feed_estimate_with_a_warning(case, capsys):
    cfg = {"scales": dict(SCALE_CFG, dT_ratio=None)}
    assert resolve_dT_ratio(cfg, [case], [None]) == 1.0
    assert "WARNING" in capsys.readouterr().out
    assert cfg["scales"]["dT_ratio"] == 1.0  # saved with the run config
    cfg = {"scales": dict(SCALE_CFG, dT_ratio=0.25)}  # a fixed value is kept
    assert resolve_dT_ratio(cfg, [case], [3.0]) == 0.25


def test_example_rise_is_far_below_the_feed_estimate(example_case, mol_example):
    # Phase 0 finding: the real rise (~5 K) is several times smaller than the feed estimate (33.6 K)
    p, s = example_case
    rise = measured_temperature_rise(mol_example["T"], p.Tin)
    assert 1.0 < rise < 0.25 * s.dT_ref
