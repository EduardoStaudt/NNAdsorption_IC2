import numpy as np
import pytest
import tensorflow as tf

from nnads_ic2.models.pinn import PINN
from nnads_ic2.physics.params import theta_dim
from nnads_ic2.physics.residuals import case_arrays, gather_cases, stack_cases
from nnads_ic2.physics.scales import dimensionless_groups
from nnads_ic2.runtime import set_float_dtype
from nnads_ic2.training.losses import causal_weights, compute_boundary_fields, compute_fields, model_inputs
from nnads_ic2.training.weighting import LossWeights

MODEL_CFG = dict(
    width=32, depth=3, activation="tanh", head_width=16, head_hidden=1, ic_delta=0.05, hard_ic=True,
    fourier=dict(n_features=8, sigma_z=1.0, sigma_t=1.0, seed=0, trainable=False),
)


@pytest.fixture
def float64():
    old = tf.keras.backend.floatx()
    set_float_dtype("float64")
    yield
    set_float_dtype(old)


def test_autodiff_derivatives_match_finite_differences(example_case, float64):
    p, s = example_case
    arr = case_arrays(p, s, dimensionless_groups(p, s))
    arr["theta"] = np.zeros(theta_dim(2))
    table = stack_cases([arr], tf.float64)
    model = PINN(2, theta_dim(2), MODEL_CFG, t_final=2.0).build_with_dummy()
    rng = np.random.default_rng(0)
    B = 64
    z = rng.uniform(0.1, 0.9, (B, 1))
    t = rng.uniform(0.2, 1.8, (B, 1))
    g = gather_cases(table, tf.zeros(B, tf.int32))
    f = compute_fields(model, tf.constant(z), tf.constant(t), g)

    def out(zz, tt):
        o = model(model_inputs(tf.constant(zz), tf.constant(tt), g))
        return {k: v.numpy() for k, v in o.items()}

    h = 1e-4
    zp, zm, tp, tm, o0 = out(z + h, t), out(z - h, t), out(z, t + h), out(z, t - h), out(z, t)
    checks = dict(
        c_z=(zp["c"] - zm["c"]) / (2 * h), c_t=(tp["c"] - tm["c"]) / (2 * h),
        c_zz=(zp["c"] - 2 * o0["c"] + zm["c"]) / h**2, q_t=(tp["q"] - tm["q"]) / (2 * h),
        T_z=(zp["T"] - zm["T"]) / (2 * h), T_t=(tp["T"] - tm["T"]) / (2 * h),
        T_zz=(zp["T"] - 2 * o0["T"] + zm["T"]) / h**2, v_z=(zp["v"] - zm["v"]) / (2 * h),
    )
    for k, fd in checks.items():
        tol = 1e-4 if k.endswith("zz") else 1e-6
        np.testing.assert_allclose(f[k].numpy(), fd, rtol=tol, atol=tol, err_msg=k)
    fb = compute_boundary_fields(model, tf.constant(z), tf.constant(t), g)
    np.testing.assert_allclose(fb["c_z"].numpy(), f["c_z"].numpy(), rtol=1e-12)


def test_causal_weights_respect_time_order():
    t = tf.constant(np.linspace(0, 2, 400)[:, None])
    r2 = tf.constant(np.full(400, 0.5))
    w, w_bins = causal_weights(t, tf.fill([400, 1], tf.constant(2.0, tf.float64)), r2, n_bins=8, eps=1.0)
    wb = w_bins.numpy()
    assert wb[0] == 1.0 and np.all(np.diff(wb) < 0)  # later slices wait for earlier ones
    np.testing.assert_allclose(wb, np.exp(-0.5 * np.arange(8)))


def test_grad_norm_annealing_raises_weight_of_small_gradient_group():
    cfg = dict(weights=dict(data=0.0, pde=1.0, bc=1.0, ic=0.0),
               adaptive=dict(method="grad_norm", every=10, alpha=0.5, min=1e-3, max=1e4))
    w = LossWeights(cfg, "float64")
    w.update({"pde": np.array([10.0, 1.0]), "bc": np.array([1.0, 0.1])})  # lambda_hat_bc = 10 / 0.1
    assert w.as_dict()["bc"] == pytest.approx(0.5 * 1.0 + 0.5 * 100.0)
    assert w.as_dict()["data"] == 0.0 and w.as_dict()["pde"] == 1.0


def test_data_weight_has_its_own_cap_and_can_be_fixed():
    base = dict(weights=dict(data=1.0, pde=1.0, bc=1.0, ic=0.0),
                adaptive=dict(method="grad_norm", every=10, alpha=1.0, min=1e-3, max=100.0, max_data=10.0))
    stats = {"pde": np.array([10.0, 1.0]), "bc": np.array([1.0, 1e-3]), "data": np.array([1.0, 1e-3])}
    w = LossWeights(base, "float64")
    w.update(stats)  # lambda_hat = 1e4 for both groups
    assert w.as_dict()["bc"] == 100.0 and w.as_dict()["data"] == 10.0
    fixed = dict(base, adaptive=dict(base["adaptive"], fixed=["data"]))
    w = LossWeights(fixed, "float64")
    w.update(stats)
    assert w.as_dict()["data"] == 1.0 and w.as_dict()["bc"] == 100.0
    with pytest.raises(ValueError):
        LossWeights(dict(base, adaptive=dict(base["adaptive"], fixed=["dta"])), "float64")



def test_loss_weights_ignore_a_non_finite_batch():
    cfg = dict(weights=dict(data=1.0, pde=1.0, bc=1.0, ic=0.0),
               adaptive=dict(method="grad_norm", every=10, alpha=0.5, min=1e-3, max=1e4))
    w = LossWeights(cfg, "float64")
    before = w.as_dict()
    w.update({"pde": np.array([10.0, 1.0]), "bc": np.array([np.nan, np.nan]), "data": np.array([1.0, 0.1])})
    assert w.as_dict() == before  # one NaN in the batch: no weight changes (it would stay NaN forever)


def test_robust_residual_loss_has_the_gradient_of_the_cauchy_loss(example_case, float64):
    from nnads_ic2.training.losses import PDE_TERMS, LossBuilder

    p, s = example_case
    arr = case_arrays(p, s, dimensionless_groups(p, s))
    arr["theta"] = np.zeros(theta_dim(2))
    table = stack_cases([arr], tf.float64)
    model = PINN(2, theta_dim(2), MODEL_CFG, t_final=2.0).build_with_dummy()
    rng = np.random.default_rng(1)
    B, c = 256, 0.5
    z = tf.constant(rng.uniform(0.0, 1.0, (B, 1)))
    t = tf.constant(rng.uniform(0.0, 2.0, (B, 1)))
    case = tf.zeros(B, tf.int32)
    cfg = dict(pde_terms=dict(mass=1.0, ldf=1.0, energy=10.0, global_=1.0), causal=dict(enabled=False, n_bins=8))
    cfg["pde_terms"]["global"] = cfg["pde_terms"].pop("global_")
    plain = LossBuilder(model, table, cfg, p_min=1e-6)
    robust = LossBuilder(model, table, dict(cfg, robust_c=c), p_min=1e-6)
    eps = tf.constant(0.0, tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        terms, _, s_pt = robust.interior(z, t, case, eps)
        loss = tf.add_n([robust.pde_w[k] * terms[k] for k in PDE_TERMS])
        cauchy = tf.reduce_mean(c * tf.math.log1p(plain.interior(z, t, case, eps)[2] / c))
    g1 = tape.gradient(loss, model.trainable_variables)
    g2 = tape.gradient(cauchy, model.trainable_variables)
    for a, b in zip(g1, g2):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-8, atol=1e-12)
    assert float(tf.reduce_max(s_pt)) > 10 * c  # the test covers the damped region (s >> c)
    # robust_c missing or null: the plain mean squares of Phases 0-1
    assert plain.robust_c == 0.0 and LossBuilder(model, table, dict(cfg, robust_c=None), 1e-6).robust_c == 0.0


def test_front_weight_is_the_sum_of_absolute_time_derivatives(example_case, float64):
    from nnads_ic2.training.losses import LossBuilder, compute_fields

    p, s = example_case
    arr = case_arrays(p, s, dimensionless_groups(p, s))
    arr["theta"] = np.zeros(theta_dim(2))
    table = stack_cases([arr], tf.float64)
    model = PINN(2, theta_dim(2), MODEL_CFG, t_final=2.0).build_with_dummy()
    rng = np.random.default_rng(3)
    z = tf.constant(rng.uniform(0.0, 1.0, (64, 1)))
    t = tf.constant(rng.uniform(0.0, 2.0, (64, 1)))
    case = tf.zeros(64, tf.int32)
    cfg = dict(pde_terms={"mass": 1.0, "ldf": 1.0, "energy": 10.0, "global": 1.0}, causal=dict(enabled=False, n_bins=8))
    w = LossBuilder(model, table, cfg, p_min=1e-6).front_weight(z, t, case).numpy()
    c_t = compute_fields(model, z, t, gather_cases(table, case))["c_t"].numpy()
    np.testing.assert_allclose(w, np.abs(c_t).sum(axis=1), rtol=1e-10)
