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


@pytest.mark.parametrize("activation, n_features", [("silu", 16), ("tanh", 0), ("gelu", 16), ("softplus", 16),
                                                     ("mish", 16), ("sin", 16)])
def test_every_activation_gives_finite_second_derivatives(activation, n_features):
    """Options of the Optuna search spaces: every activation, and no Fourier features. The PDE
    residuals need d2/dz2 of the outputs, so it must exist and be finite."""
    n = 2
    cfg = dict(MODEL_CFG, activation=activation, fourier=dict(MODEL_CFG["fourier"], n_features=n_features))
    model = PINN(n, theta_dim(n), cfg, t_final=2.0).build_with_dummy()
    inp = make_inputs(np.random.default_rng(4), n, theta_dim(n), 128, "float32")
    with tf.GradientTape() as outer:
        outer.watch(inp["z"])
        with tf.GradientTape() as inner:
            inner.watch(inp["z"])
            c = model(inp)["c"][:, :1]
        dc_dz = inner.gradient(c, inp["z"])
    d2c_dz2 = outer.gradient(dc_dz, inp["z"]).numpy()
    assert np.all(np.isfinite(d2c_dz2)) and np.abs(d2c_dz2).max() > 0.0


@pytest.mark.parametrize("activation", ["relu", "elu"])
def test_activations_without_a_smooth_second_derivative_are_refused(activation):
    with pytest.raises(ValueError, match="smooth"):
        PINN(2, theta_dim(2), dict(MODEL_CFG, activation=activation), t_final=2.0)


@pytest.mark.parametrize("activation", ["tanh", "silu", "gelu", "softplus", "mish"])
def test_heads_use_the_activation_of_the_trunk(activation):
    model = PINN(2, theta_dim(2), dict(MODEL_CFG, activation=activation), t_final=2.0).build_with_dummy()
    net = model.subnets[0]
    expected = tf.keras.activations.get(activation)
    hidden = net.trunk + [h[k] for h in (net.head_c, net.head_q, net.head_T, net.head_v) for k in range(len(h) - 1)]
    assert len(hidden) == MODEL_CFG["depth"] + 4 * MODEL_CFG["head_hidden"]
    assert all(layer.activation is expected for layer in hidden)
    outputs = [h[-1] for h in (net.head_c, net.head_q, net.head_T, net.head_v)]
    assert all(layer.activation is tf.keras.activations.linear for layer in outputs)


def test_siren_initialization():
    """sin layers: W ~ U(-1/n, 1/n) in the first layer, U(-sqrt(6/n)/w0, sqrt(6/n)/w0) in the others
    (heads included), b ~ U(-1/sqrt(n), 1/sqrt(n)); the output layers stay linear."""
    w0 = 30.0
    cfg = dict(MODEL_CFG, activation="sin", siren_w0=w0)
    model = PINN(2, theta_dim(2), cfg, t_final=2.0).build_with_dummy()
    net = model.subnets[0]
    n_in = 2 + 2 * cfg["fourier"]["n_features"] + theta_dim(2)
    hidden = [(net.trunk[0], n_in, 1.0 / n_in)] + [
        (layer, cfg["width"], np.sqrt(6.0 / cfg["width"]) / w0)
        for layer in net.trunk[1:] + [h[0] for h in (net.head_c, net.head_q, net.head_T, net.head_v)]]
    for layer, n, lim in hidden:
        W, b = layer.kernel.numpy(), layer.bias.numpy()
        assert W.shape[0] == n
        assert np.abs(W).max() <= lim and np.abs(W).max() > 0.9 * lim  # uses the whole interval
        assert np.abs(b).max() <= 1.0 / np.sqrt(n)
    assert all(h[-1].activation is tf.keras.activations.linear for h in (net.head_c, net.head_q, net.head_T, net.head_v))


def test_sine_layer_is_sin_of_w0_times_the_affine_map():
    from nnads_ic2.models.pinn import SineDense

    layer = SineDense(8, 30.0, first=False)
    x = tf.constant(np.random.default_rng(5).uniform(-1, 1, (16, 5)), "float32")
    y = layer(x).numpy()
    expected = np.sin(30.0 * (x.numpy() @ layer.kernel.numpy() + layer.bias.numpy()))
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)


def test_two_domains_is_a_later_step():
    with pytest.raises(NotImplementedError):
        PINN(2, theta_dim(2), MODEL_CFG, t_final=2.0, domains=[Domain(0.0, 0.5), Domain(0.5, 1.0)])
