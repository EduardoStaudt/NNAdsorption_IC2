import numpy as np
import pytest
import tensorflow as tf

from conftest import IC1_ROOT
from ic1_reference import TREF, load_ic1_module, qstar_slope
from nnads_ic2.physics.isotherm import qstar_np, qstar_tf


def random_isotherms(rng, n, m):
    """m random isotherm sets for n components, drawn over the IC1 X_RANGES."""
    qm_ref = rng.uniform(1.0, 15.0, (m, n))
    k2 = rng.uniform(-0.1, 0.0, (m, n))
    B_ref = 10 ** rng.uniform(-4, 0, (m, n))
    k4 = rng.uniform(0.0, 3200.0, (m, n))
    n_ref = rng.uniform(0.5, 1.5, (m, n))
    k6 = rng.uniform(-2200.0, 2200.0, (m, n))
    return dict(
        k1=qm_ref - k2 * TREF,
        k2=k2,
        k3=B_ref * np.exp(-k4 / TREF),
        k4=k4,
        k5=n_ref - k6 / TREF,
        k6=k6,
    )


def random_states(rng, n, m):
    c = rng.uniform(0.0, 1500.0, (m, n))  # mol/m3 (up to ~30 bar at 300 K)
    c[rng.random((m, n)) < 0.15] = 0.0  # absent components, as at t = 0
    T = rng.uniform(280.0, 380.0, m)
    return c, T


def ic1_qstar(qfun, c, T, iso):
    """Call an IC1-style qstar_slope (component on axis 0) one state at a time."""
    out = np.empty_like(c)
    for k in range(c.shape[0]):
        K = [iso[name][k][:, None] for name in ["k1", "k2", "k3", "k4", "k5", "k6"]]
        qs, _ = qfun(c[k][:, None], np.array([T[k]]), *K)
        out[k] = qs[:, 0]
    return out


@pytest.mark.parametrize("n", [2, 3, 5])
def test_isotherm_matches_ic1(n):
    rng = np.random.default_rng(n)
    m = 400
    iso = random_isotherms(rng, n, m)
    c, T = random_states(rng, n, m)
    ref = ic1_qstar(qstar_slope, c, T, iso)

    q_np = qstar_np(c, T, iso)
    iso_tf = {k: tf.constant(v, tf.float64) for k, v in iso.items()}
    q_tf = qstar_tf(tf.constant(c, tf.float64), tf.constant(T[:, None], tf.float64), iso_tf).numpy()

    assert np.max(np.abs(q_np - ref)) < 1e-10
    assert np.max(np.abs(q_tf - ref)) < 1e-10


def test_isotherm_matches_live_ic1_file():
    sj = load_ic1_module(IC1_ROOT, "src/contrato/solver_jee.py", "ic1_solver_jee")
    if sj is None:
        pytest.skip("NNAdsorption_IC1 folder not found")
    rng = np.random.default_rng(7)
    iso = random_isotherms(rng, 2, 300)
    c, T = random_states(rng, 2, 300)
    ref = ic1_qstar(sj.qstar_slope, c, T, iso)
    assert np.max(np.abs(qstar_np(c, T, iso) - ref)) < 1e-10


def test_pressure_floor_gives_finite_gradients_at_zero_concentration():
    # n < 1 at this temperature, so d(p^n)/dp is infinite at p = 0 without the floor
    iso = {k: tf.constant(v, tf.float64) for k, v in dict(
        k1=[5.0, 5.0], k2=[0.0, 0.0], k3=[0.01, 0.01], k4=[0.0, 0.0], k5=[0.6, 0.6], k6=[0.0, 0.0]
    ).items()}
    c = tf.Variable([[0.0, 0.0], [100.0, 0.0]], dtype=tf.float64)
    T = tf.constant([[300.0], [300.0]], tf.float64)
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(qstar_tf(c, T, iso, p_min=1e-6))
    g = tape.gradient(total, c)
    assert np.all(np.isfinite(g.numpy()))
