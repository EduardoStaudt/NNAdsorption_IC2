import copy
import os

import numpy as np
import tensorflow as tf

from conftest import REPO_ROOT
from nnads_ic2.config import load_config
from nnads_ic2.data.domain import Domain, ThetaSampler
from nnads_ic2.data.sampling import CollocationPool
from nnads_ic2.experiment import fit_theta_scaler, make_model
from nnads_ic2.physics.residuals import stack_cases
from nnads_ic2.training.theta_colloc import ThetaCollocation
from nnads_ic2.training.trainer import Trainer


def small_cfg():
    cfg = copy.deepcopy(load_config(os.path.join(REPO_ROOT, "configs", "phase2_stage_a.yaml")))
    cfg["runtime"]["dtype"] = "float32"
    cfg["scales"]["dT_ratio"] = 0.2
    cfg["theta_colloc"].update(n_theta=16, points_per_theta=8, resample_every_epochs=1)
    cfg["sampling"].update(n_interior=64, n_boundary=16, n_data=16)
    cfg["sampling"]["rar"].update(every=3, n_candidates=128, n_add=8)
    cfg["model"].update(width=16, depth=2, head_width=8)
    cfg["model"]["fourier"]["n_features"] = 8
    cfg["train"].update(steps=4, jit_compile=False)
    cfg["logging"].update(every=2, ckpt_every=100, tensorboard=False)
    cfg["loss"]["weights"]["data"] = 0.0
    return cfg


def make_colloc(cfg, steps=4):
    dom = Domain(cfg, REPO_ROOT)
    scaler = fit_theta_scaler(ThetaSampler(dom, seed=5).draw(64) + [dom.base], cfg)
    return ThetaCollocation(cfg, [dom.base], ThetaSampler(dom, seed=1), scaler, "float32", steps)


def test_variable_table_can_be_reassigned():
    rows = [dict(a=np.array([1.0]), b=np.array([1.0, 2.0])), dict(a=np.array([3.0]), b=np.array([4.0, 5.0]))]
    table = stack_cases(rows, tf.float64, variable=True)
    assert isinstance(table["a"], tf.Variable) and table["b"].shape == (2, 2)
    table["a"].assign(np.array([[7.0], [8.0]]))
    np.testing.assert_array_equal(table["a"].numpy()[:, 0], [7.0, 8.0])


def test_pool_renew_draws_new_points_and_keeps_the_selected_ones():
    pool = CollocationPool(n_init=64, n_cases=4, t_final=2.0, max_size=100, seed=0)
    pool.add(np.array([0.5, 0.25]), np.array([0.1, 0.2]), np.array([0, 3]))
    z_old = pool.z[:64].copy()
    pool.renew(seed=7, keep=pool.from_rar & (pool.case < 1))
    assert pool.size == 65 and pool.from_rar.sum() == 1
    assert pool.z[-1] == 0.5 and pool.case[-1] == 0
    assert not np.allclose(pool.z[:64], z_old)
    assert pool.t.max() < 2.0


def test_resample_replaces_only_the_sobol_rows():
    cfg = small_cfg()
    col = make_colloc(cfg)
    assert col.table["theta"].shape == (17, 31) and col.pool.size == 17 * 8
    assert col.every_steps == round(17 * 8 / 64)
    theta0 = col.table["theta"].numpy().copy()
    vs0 = col.table["Pe"].numpy().copy()
    col.pool.add(np.array([0.5, 0.5]), np.array([0.1, 0.1]), np.array([0, 5]))  # RAR on the base case and on a Sobol row
    col.resample()
    theta1 = col.table["theta"].numpy()
    np.testing.assert_array_equal(theta1[0], theta0[0])  # base case row unchanged
    assert np.abs(theta1[1:] - theta0[1:]).max() > 0.1  # new theta in the Sobol rows
    assert not np.allclose(col.table["Pe"].numpy()[1:], vs0[1:])  # and their constants follow
    assert col.pool.from_rar.sum() == 1 and col.pool.case[col.pool.from_rar][0] == 0
    assert col.n_theta_seen == 32 and col.info()["n_resamples"] == 1
    # only the varying columns change (normalized fixed columns stay at 0)
    changed = np.flatnonzero(np.abs(theta1[1:] - theta0[1:]).max(axis=0) > 0)
    assert len(changed) == 4


def test_trainer_resamples_during_training(tmp_path):
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    col = make_colloc(cfg, steps=4)
    col.every_steps = 2
    model = make_model(cfg, 2)
    trainer = Trainer(cfg, model, col.table, col.pool, str(tmp_path), resampler=col)
    trainer.run()
    assert col.n_resamples == 1  # at step 2 (never at the last step)
    with open(os.path.join(tmp_path, "train_log.csv"), encoding="utf-8") as f:
        header = f.readline()
    assert "theta_resamples" in header


def test_trainer_skips_steps_with_a_non_finite_loss(tmp_path):
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    cfg["sampling"]["rar"]["enabled"] = False
    col = make_colloc(cfg, steps=4)
    col.table["Pe"].assign(tf.fill(tf.shape(col.table["Pe"]), float("nan")))  # every residual is NaN
    model = make_model(cfg, 2)
    trainer = Trainer(cfg, model, col.table, col.pool, str(tmp_path))
    before = [w.numpy().copy() for w in model.trainable_variables]
    trainer.run()
    assert int(trainer.skipped.numpy()) == 4
    for w0, w in zip(before, model.trainable_variables):
        np.testing.assert_array_equal(w.numpy(), w0)  # zero gradients from the start: Adam does not move
    with open(os.path.join(tmp_path, "train_log.csv"), encoding="utf-8") as f:
        assert "skipped_steps" in f.readline()


def test_trainer_calls_the_evaluator_every_eval_every_steps(tmp_path):
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    cfg["train"]["steps"] = 6
    cfg["sampling"]["rar"]["enabled"] = False
    col = make_colloc(cfg, steps=6)
    model = make_model(cfg, 2)
    seen = []
    trainer = Trainer(cfg, model, col.table, col.pool, str(tmp_path), evaluator=seen.append, eval_every=3)
    timing = trainer.run()
    assert seen == [3, 6] and timing["periodic_eval_s"] >= 0.0


def test_zero_resample_epochs_keeps_the_theta_for_the_whole_run(tmp_path):
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    cfg["theta_colloc"]["resample_every_epochs"] = 0
    cfg["sampling"]["rar"]["enabled"] = False
    col = make_colloc(cfg, steps=4)
    assert col.every_steps == 0
    theta0 = col.table["theta"].numpy().copy()
    Trainer(cfg, make_model(cfg, 2), col.table, col.pool, str(tmp_path), resampler=col).run()
    assert col.n_resamples == 0
    np.testing.assert_array_equal(col.table["theta"].numpy(), theta0)
