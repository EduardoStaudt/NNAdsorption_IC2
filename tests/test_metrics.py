import numpy as np
import pytest

from nnads_ic2.eval.mass_balance import compare_mole_balance, outlet_moment
from nnads_ic2.eval.metrics import level_time, regression_metrics
from nnads_ic2.physics.scales import feed_and_initial_state


def test_perfect_prediction():
    r = np.linspace(0, 1, 50)
    m = regression_metrics(r, r, aard_min=1e-3)
    assert m["rmse"] == 0 and m["mbe"] == 0 and m["r2"] == 1 and m["d"] == 1 and m["aard"] == 0


def test_bias_and_aard_definition():
    r = np.array([0.0, 0.5, 1.0, 2.0])
    p = r + 0.1
    m = regression_metrics(p, r, aard_min=1e-3)
    assert m["mbe"] == pytest.approx(0.1) and m["rmse"] == pytest.approx(0.1)
    # AARD ignores the reference value 0 (|ref| <= 1e-3), as in IC1
    assert m["aard"] == pytest.approx(100 * np.mean([0.1 / 0.5, 0.1 / 1.0, 0.1 / 2.0]))


def test_level_time_interpolates():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    s = np.array([0.0, 0.2, 0.6, 1.0])
    assert level_time(t, s, 0.5) == pytest.approx(1.75)
    assert level_time(t, s, 0.5, interpolate=False) == 2.0
    assert np.isnan(level_time(t, s, 1.5))


def test_outlet_moment_of_a_sharp_front():
    # outlet jumps from 0 to the feed concentration at t0 with v = vs: the moment is t0
    t = np.linspace(0, 100, 100001)
    c = np.where(t >= 40.0, 5.0, 0.0)
    assert outlet_moment(t, np.full_like(t, 0.02), c, 0.02, 5.0) == pytest.approx(40.0, abs=1e-2)


def test_reference_solution_closes_the_mole_balance(example_case, mol_example):
    p, s = example_case
    sol = mol_example
    rows = compare_mole_balance(p, s.t_ref, sol["t"], sol, sol)
    co2 = rows[1]
    assert co2["pinn_vs_ref_pct_tref"] == 0.0
    # truncated at t_F the bed is still a little warm, so the identity holds within ~1 % of t_ref
    assert abs(co2["ref_vs_theory_pct_tref"]) < 2.0
    # exact closure on the same horizon: moment from the integrated outflow
    cf = feed_and_initial_state(p)["c_feed"]
    exact = sol["t"][-1] - sol["n_out"][-1] / (p.vs * cf)
    np.testing.assert_allclose([r["ref_moment_s"] for r in rows], exact, rtol=1e-3)
