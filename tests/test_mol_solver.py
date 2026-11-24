import numpy as np

from nnads_ic2.physics.constants import R
from nnads_ic2.physics.scales import feed_and_initial_state


def test_mole_balance_closes(example_case, mol_example):
    sol = mol_example
    # moles now in the bed - moles at t=0  ==  moles fed - moles that left (per bed area)
    err = (sol["holdup"] - sol["holdup"][0]) - (sol["n_in"] - sol["n_out"])
    assert np.max(np.abs(err) / sol["n_in"][-1]) < 1e-8


def test_gas_total_follows_ideal_gas_law(example_case, mol_example):
    p, s = example_case
    sol = mol_example
    drift = np.abs(sol["c"].sum(axis=1) - p.P / (R * sol["T"])).max() / s.C_F
    assert drift < 1e-6


def test_initial_state_is_steady_with_feed_velocity(example_case, mol_example):
    p, _ = example_case
    np.testing.assert_allclose(mol_example["v_faces"][0], p.vs, rtol=1e-12)


def test_physical_behaviour(example_case, mol_example):
    p, s = example_case
    sol = mol_example
    assert sol["c"].min() > -1e-8 * s.C_F  # no negative concentrations (beyond solver tolerance)
    # the strong component (CO2) breaks through after its gas residence time and before t_F
    cf = feed_and_initial_state(p)["c_feed"]
    ratio = sol["c_out"][:, 1] / cf[1]
    t_b = sol["t"][np.argmax(ratio >= 0.05)]
    assert p.eb * p.L / p.vs < t_b < s.t_final
    # at t_F the outlet is close to the feed composition
    np.testing.assert_allclose(sol["y_out"][-1], p.y_feed, atol=0.01)
    # adsorption slows the gas down: outlet velocity drops below the feed velocity
    assert sol["v_out"].min() < 0.99 * p.vs
