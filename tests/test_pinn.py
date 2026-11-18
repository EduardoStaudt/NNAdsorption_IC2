import numpy as np
import pytest
import tensorflow as tf

from nnads_ic2.models.pinn import PINN, Domain, outputs_as_matrix
from nnads_ic2.physics.params import theta_dim
from nnads_ic2.runtime import set_float_dtype

MODEL_CFG = dict(
    width=32, depth=3, activation="tanh", head_width=16, head_hidden=1, ic_delta=0.02, hard_ic=True,
    fourier=dict(n_features=16, sigma_z=2.0, sigma_t=2.0, seed=0, trainable=False),
)


def make_inputs(rng, n, d, B, dtype, z=None, t=None):
    return dict(
        z=tf.constant(rng.uniform(0, 1, (B, 1)) if z is None else np.full((B, 1), z), dtype),
        t=tf.constant(rng.uniform(0, 2, (B, 1)) if t is None else np.full((B, 1), t), dtype),
        theta=tf.constant(rng.standard_normal((B, d)), dtype),
        c_ic=tf.constant(rng.uniform(0, 1, (B, n)), dtype),
        q_ic=tf.constant(rng.uniform(0, 1, (B, n)), dtype),
    )


@pytest.fixture(params=["float32", "float64"])
def floatx(request):
    old = tf.keras.backend.floatx()
    set_float_dtype(request.param)
    yield request.param
    set_float_dtype(old)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_output_shape_is_2n_plus_2(n, floatx):
    model = PINN(n, theta_dim(n), MODEL_CFG, t_final=2.0).build_with_dummy()
    out = model(make_inputs(np.random.default_rng(0), n, theta_dim(n), 64, floatx))
    assert model.n_outputs == 2 * n + 2
    assert outputs_as_matrix(out).shape == (64, 2 * n + 2)
    assert out["c"].shape == (64, n) and out["q"].shape == (64, n)
    assert out["T"].shape == (64, 1) and out["v"].shape == (64, 1)


def test_inlet_velocity_is_exactly_one(floatx):
    n = 2
    model = PINN(n, theta_dim(n), MODEL_CFG, t_final=2.0).build_with_dummy()
    out = model(make_inputs(np.random.default_rng(1), n, theta_dim(n), 256, floatx, z=0.0))
    np.testing.assert_array_equal(out["v"].numpy(), 1.0)


def test_initial_state_is_exact(floatx):
    n = 3
    rng = np.random.default_rng(2)
    model = PINN(n, theta_dim(n), MODEL_CFG, t_final=2.0).build_with_dummy()
    inp = make_inputs(rng, n, theta_dim(n), 256, floatx, t=0.0)
    out = model(inp)
    np.testing.assert_array_equal(out["c"].numpy(), inp["c_ic"].numpy())
    np.testing.assert_array_equal(out["q"].numpy(), inp["q_ic"].numpy())
    np.testing.assert_array_equal(out["T"].numpy(), 0.0)
    np.testing.assert_array_equal(out["v"].numpy(), 1.0)


def test_concentrations_and_loadings_are_non_negative(floatx):
    n = 5
    model = PINN(n, theta_dim(n), MODEL_CFG, t_final=2.0).build_with_dummy()
    out = model(make_inputs(np.random.default_rng(3), n, theta_dim(n), 2048, floatx))
    assert out["c"].numpy().min() >= 0.0 and out["q"].numpy().min() >= 0.0


def test_two_domains_is_a_later_step():
    with pytest.raises(NotImplementedError):
        PINN(2, theta_dim(2), MODEL_CFG, t_final=2.0, domains=[Domain(0.0, 0.5), Domain(0.5, 1.0)])
