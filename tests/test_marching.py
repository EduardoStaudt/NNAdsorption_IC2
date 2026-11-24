"""Time marching (Phase 3 proof of concept): time windows of the collocation, the causal slices
and the window models."""

import numpy as np
import tensorflow as tf
from test_theta_colloc import make_colloc, small_cfg

from conftest import REPO_ROOT
from nnads_ic2.data.domain import Domain, ThetaSampler
from nnads_ic2.data.sampling import CollocationPool, boundary_points
from nnads_ic2.experiment import case_table, fit_theta_scaler, make_model
from nnads_ic2.models.pinn import MarchedPINN, MarchingWindow
from nnads_ic2.physics.residuals import gather_cases
from nnads_ic2.training.losses import causal_weights, model_inputs
from nnads_ic2.training.theta_colloc import ThetaCollocation
from nnads_ic2.training.trainer import Trainer


def test_pool_and_boundary_times_stay_inside_the_window():
    pool = CollocationPool(n_init=512, n_cases=3, t_final=1.5, max_size=600, seed=0, t_start=1.0)
    assert pool.t.min() >= 1.0 and pool.t.max() < 1.5 and pool.t.max() - pool.t.min() > 0.45
    _, t, _ = pool.candidates(256)
    assert t.min() >= 1.0 and t.max() < 1.5
    pool.renew(seed=3, keep=np.zeros(pool.size, dtype=bool))
    assert pool.t.min() >= 1.0 and pool.t.max() < 1.5
    _, tb, _ = boundary_points(np.random.default_rng(0), 256, 0.0, 1.5, 3, t_start=1.0)
    assert tb.min() >= 1.0 and tb.max() < 1.5
    # default: the whole time range, as before
    assert CollocationPool(64, 1, 2.0, 64, 0).t_start == 0.0


def test_causal_slices_span_the_window():
    """Residual only in the second half of the window [1, 1.5]: with 2 slices the first one has
    weight 1 and the second exp(-eps * 0) = 1 too; with the residual in the first half the second
    slice is damped."""
    t = tf.constant(np.linspace(1.0, 1.5, 100, endpoint=False)[:, None], tf.float64)
    t_end = tf.fill([100, 1], tf.constant(1.5, tf.float64))
    first_half = tf.constant(np.r_[np.ones(50), np.zeros(50)], tf.float64)
    w, w_bins = causal_weights(t, t_end, first_half, n_bins=2, eps=1.0, t_start=1.0)
    np.testing.assert_allclose(w_bins.numpy(), [1.0, np.exp(-1.0)])
    np.testing.assert_allclose(w.numpy()[:50], 1.0)
    second_half = tf.constant(np.r_[np.zeros(50), np.ones(50)], tf.float64)
    _, w_bins = causal_weights(t, t_end, second_half, n_bins=2, eps=1.0, t_start=1.0)
    np.testing.assert_allclose(w_bins.numpy(), [1.0, 1.0])


def window_setup():
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    dom = Domain(cfg, REPO_ROOT)
    scaler = fit_theta_scaler(ThetaSampler(dom, seed=5).draw(64) + [dom.base], cfg)
    table, _, _, _ = case_table([dom.base], cfg, "float32", scaler=scaler)
    first = make_model(cfg, 2)
    net = make_model(cfg, 2)
    return cfg, dom, scaler, table, first, net


def inputs_at(table, z, t):
    g = gather_cases(table, tf.zeros([z.size], tf.int32))
    return model_inputs(tf.constant(z[:, None], tf.float32), tf.constant(t[:, None], tf.float32), g)


def test_window_starts_exactly_from_the_previous_state():
    cfg, _, _, table, first, net = window_setup()
    window = MarchingWindow(first, net, t_start=0.5)
    z = np.linspace(0.0, 1.0, 11)
    at_start = window(inputs_at(table, z, np.full(11, 0.5)))
    prev = first(inputs_at(table, z, np.full(11, 0.5)))
    for k in ("c", "q", "T", "v"):
        np.testing.assert_allclose(at_start[k].numpy(), prev[k].numpy(), rtol=0, atol=1e-6)
    later = window(inputs_at(table, z, np.full(11, 1.2)))
    assert np.all(later["c"].numpy() >= 0) and np.all(later["q"].numpy() >= 0)
    np.testing.assert_allclose(later["v"].numpy()[0], 1.0, atol=1e-6)  # v = v_s at the inlet
    # only the new network trains; the earlier window is fixed
    assert {id(v) for v in window.trainable_variables} == {id(v) for v in net.trainable_variables}
    assert not first.trainable_variables


def test_window_state_at_its_start_does_not_depend_on_time():
    cfg, _, _, table, first, net = window_setup()
    window = MarchingWindow(first, net, t_start=0.5)
    g = gather_cases(table, tf.zeros([5], tf.int32))
    z = tf.constant(np.linspace(0.1, 0.9, 5)[:, None], tf.float32)
    t = tf.constant(np.full((5, 1), 0.5), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(t)
        c = window(model_inputs(z, t, g))["c"]
    # d c/dt at t_start = (c_net - c_prev)/delta: only the new network's part moves in time
    dc = tape.gradient(c, t).numpy()
    net_c = tf.nn.softplus(net.subnets[0](z, t, g["theta"])[0]).numpy()
    prev_c = first(model_inputs(z, t, g))["c"].numpy()
    np.testing.assert_allclose(dc[:, 0], ((net_c - prev_c) / window.delta).sum(axis=1), rtol=1e-4, atol=1e-4)


def test_marched_model_answers_each_point_with_its_window():
    cfg, _, _, table, first, net = window_setup()
    window = MarchingWindow(first, net, t_start=0.5)
    marched = MarchedPINN([first, window], [0.0, 0.5])
    z = np.full(4, 0.4)
    t = np.array([0.1, 0.49, 0.5, 1.7])
    out = marched(inputs_at(table, z, t))["c"].numpy()
    np.testing.assert_allclose(out[:2], first(inputs_at(table, z[:2], t[:2]))["c"].numpy(), atol=1e-6)
    np.testing.assert_allclose(out[2:], window(inputs_at(table, z[2:], t[2:]))["c"].numpy(), atol=1e-6)


def test_trainer_runs_a_window_and_keeps_its_points_inside(tmp_path):
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    cfg["train"]["steps"] = 6
    dom = Domain(cfg, REPO_ROOT)
    scaler = fit_theta_scaler(ThetaSampler(dom, seed=5).draw(64) + [dom.base], cfg)
    col = ThetaCollocation(cfg, [dom.base], ThetaSampler(dom, seed=1), scaler, "float32", 6, t_range=(1.0, 1.5))
    first = make_model(cfg, 2)
    net = make_model(cfg, 2)
    net.set_weights(first.get_weights())
    window = MarchingWindow(first, net, t_start=1.0)
    before = [v.numpy().copy() for v in first.weights]
    trainer = Trainer(cfg, window, col.table, col.pool, str(tmp_path), resampler=col)
    assert trainer.losses.t_window == (1.0, 1.5)
    b = trainer.make_batch()
    for key in ("t", "t_in", "t_out"):
        assert b[key].numpy().min() >= 1.0 and b[key].numpy().max() < 1.5
    trainer.run()
    assert col.pool.t.min() >= 1.0 and col.pool.t.max() < 1.5  # RAR and renewals stay in the window
    for a, v in zip(before, first.weights):
        np.testing.assert_array_equal(a, v.numpy())  # the earlier window did not move
    # a pool over the whole time range keeps the per-case causal slices (no window)
    full = make_colloc(small_cfg())
    assert Trainer(small_cfg(), make_model(cfg, 2), full.table, full.pool, str(tmp_path)).losses.t_window is None
