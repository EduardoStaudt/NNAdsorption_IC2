"""PINN for the fixed bed: Fourier features in (z~, t~) + MLP trunk + one head per output group.

Inputs per point: z~, t~, theta^ (normalized case parameters) and the initial state of the case
(c~_0, q~_0). Outputs (all dimensionless, docs/nondim.md):
    c~_i (n values), q~_i (n values), T~, v~      ->  2 n + 2 values
Every component has its own output (no y1 = 1 - y0 shortcut).

Hard constraints (exact by construction, docs/nondim.md Section 4), with s = 1 - exp(-t~/delta):
    c~ = (1 - s) c~_0 + s softplus(N_c)      initial state exact and c~ >= 0
    q~ = (1 - s) q~_0 + s softplus(N_q)      initial state exact and q~ >= 0
    T~ = s N_T                               T = Tin at t = 0
    v~ = 1 + z~ s N_v                        v = vs at the inlet and at t = 0
The Danckwerts inlet and the zero-gradient outlet stay as soft (loss) terms.
"""

import math
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

# Hidden-layer activations. The PDE residuals need d2/dz2 of the outputs (dispersion, conduction),
# so only smooth activations are allowed: ReLU has a zero second derivative (except at 0, where it
# is undefined) and ELU a second derivative that jumps at 0 (docs/optuna_search_space.md).
# "sin" is a SIREN layer with its own initialization (SineDense).
ACTIVATIONS = ("tanh", "silu", "gelu", "softplus", "mish", "sin")


@dataclass(frozen=True)
class Domain:
    """Axial interval [z_lo, z_hi] (dimensionless) handled by one sub-network.

    One domain = one adsorbent layer. Two layers (XPINN) will use one sub-network per domain
    plus an interface loss: continuity of c_i and T and of the convective + diffusive fluxes.
    """

    z_lo: float = 0.0
    z_hi: float = 1.0


class FourierFeatures(tf.keras.layers.Layer):
    """gamma(x) = [sin(2 pi x B), cos(2 pi x B)] with a fixed random matrix B (Tancik et al., 2020).

    Plain MLPs learn smooth functions first and struggle with steep fronts ("spectral bias").
    Feeding sines and cosines of several frequencies lets the network build sharp fronts.
    Row 0 of B multiplies z~, row 1 multiplies t~/t~_F; sigma sets the typical frequency.
    """

    def __init__(self, n_features, sigma_z, sigma_t, seed, trainable_B=False, **kwargs):
        super().__init__(**kwargs)
        rng = np.random.default_rng(seed)
        self._B_init = rng.standard_normal((2, n_features)) * np.array([[sigma_z], [sigma_t]])
        self.B = self.add_weight(name="B", shape=(2, n_features), initializer="zeros", trainable=trainable_B)
        self.B.assign(self._B_init.astype(self.B.dtype))

    def call(self, x):
        proj = (2.0 * np.pi) * tf.matmul(x, self.B)
        return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)


class SineDense(tf.keras.layers.Layer):
    """y = sin(w0 (x W + b)): one SIREN layer (Sitzmann et al., 2020) with its own initialization.

    With n inputs, the first layer of the network draws W ~ U(-1/n, 1/n) and the later ones
    W ~ U(-sqrt(6/n)/w0, sqrt(6/n)/w0), so that w0 (x W + b) has the same spread in every layer and
    the activations keep one distribution through the depth. b ~ U(-1/sqrt(n), 1/sqrt(n)), as in
    the reference code of the paper. w0 = 30 there; it also multiplies the gradients of W and b.
    """

    def __init__(self, units, w0, first, **kwargs):
        super().__init__(**kwargs)
        self.units, self.w0, self.first = int(units), float(w0), bool(first)

    def build(self, input_shape):
        n = int(input_shape[-1])
        lim = 1.0 / n if self.first else math.sqrt(6.0 / n) / self.w0
        blim = 1.0 / math.sqrt(n)
        self.kernel = self.add_weight(name="kernel", shape=(n, self.units),
                                      initializer=tf.keras.initializers.RandomUniform(-lim, lim))
        self.bias = self.add_weight(name="bias", shape=(self.units,),
                                    initializer=tf.keras.initializers.RandomUniform(-blim, blim))

    def call(self, x):
        return tf.sin(self.w0 * (tf.matmul(x, self.kernel) + self.bias))


def _hidden_layer(units, model_cfg, name, first=False):
    """A hidden layer of the trunk or of a head with the activation of model_cfg. `first`: the first
    layer of the network (only SIREN initializes it differently)."""
    activation = model_cfg["activation"]
    if activation not in ACTIVATIONS:
        raise ValueError("activation %r: use one of %s (smooth, so that d2/dz2 exists)" % (activation, ACTIVATIONS))
    if activation == "sin":
        return SineDense(units, model_cfg.get("siren_w0", 30.0), first, name=name)
    return tf.keras.layers.Dense(units, activation=tf.keras.activations.get(activation), name=name)


def _head(n_out, model_cfg, name):
    """Hidden layers with the activation of the trunk, then a linear output layer."""
    layers = [_hidden_layer(model_cfg["head_width"], model_cfg, "%s_h%d" % (name, k))
              for k in range(model_cfg["head_hidden"])]
    layers.append(tf.keras.layers.Dense(n_out, name=name + "_out"))
    return layers


class SubNet(tf.keras.layers.Layer):
    """Trunk and heads of one domain. Returns the raw (unconstrained) outputs N_c, N_q, N_T, N_v.
    The trunk and the hidden layers of the heads all use model_cfg["activation"]."""

    def __init__(self, n_comp, model_cfg, t_final, **kwargs):
        super().__init__(**kwargs)
        ff = model_cfg["fourier"]
        self.t_final = float(t_final)
        self.ff = (
            FourierFeatures(ff["n_features"], ff["sigma_z"], ff["sigma_t"], ff["seed"], ff["trainable"], name="fourier")
            if ff["n_features"] > 0
            else None
        )
        self.trunk = [
            _hidden_layer(model_cfg["width"], model_cfg, "trunk_%d" % k, first=(k == 0))
            for k in range(model_cfg["depth"])
        ]
        self.head_c = _head(n_comp, model_cfg, "head_c")
        self.head_q = _head(n_comp, model_cfg, "head_q")
        self.head_T = _head(1, model_cfg, "head_T")
        self.head_v = _head(1, model_cfg, "head_v")

    @staticmethod
    def _apply(layers, h):
        for layer in layers:
            h = layer(h)
        return h

    def call(self, z, t, theta):
        x = tf.concat([z, t / self.t_final], axis=-1)  # both in [0, 1]
        parts = [x] + ([self.ff(x)] if self.ff is not None else []) + [theta]
        h = self._apply(self.trunk, tf.concat(parts, axis=-1))
        return (
            self._apply(self.head_c, h),
            self._apply(self.head_q, h),
            self._apply(self.head_T, h),
            self._apply(self.head_v, h),
        )


class PINN(tf.keras.Model):
    """inputs: dict z [B,1], t [B,1], theta [B,d], c_ic [B,n], q_ic [B,n]  ->  dict c, q, T, v."""

    def __init__(self, n_comp, theta_dim, model_cfg, t_final, domains=None, **kwargs):
        super().__init__(**kwargs)
        self.domains = list(domains) if domains else [Domain()]
        if len(self.domains) != 1:
            raise NotImplementedError("several adsorbent layers (XPINN) are a later step")
        self.n_comp = int(n_comp)
        self.theta_dim = int(theta_dim)
        self.delta = float(model_cfg["ic_delta"])
        self.hard_ic = bool(model_cfg["hard_ic"])
        self.subnets = [SubNet(n_comp, model_cfg, t_final, name="subnet_%d" % k) for k in range(len(self.domains))]

    @property
    def n_outputs(self):
        return 2 * self.n_comp + 2

    def call(self, inputs):
        z, t = inputs["z"], inputs["t"]
        N_c, N_q, N_T, N_v = self.subnets[0](z, t, inputs["theta"])
        if self.hard_ic:
            s = 1.0 - tf.exp(-t / self.delta)  # 0 at t~ = 0, ~1 after a few delta
            c = (1.0 - s) * inputs["c_ic"] + s * tf.nn.softplus(N_c)
            q = (1.0 - s) * inputs["q_ic"] + s * tf.nn.softplus(N_q)
            T = s * N_T
            v = 1.0 + z * s * N_v
        else:  # initial state left to a soft loss term
            c = tf.nn.softplus(N_c)
            q = tf.nn.softplus(N_q)
            T = N_T
            v = 1.0 + z * N_v
        return dict(c=c, q=q, T=T, v=v)

    def build_with_dummy(self):
        """Create the weights by one call (Keras builds layers lazily)."""
        dt = tf.keras.backend.floatx()
        one = tf.zeros((1, 1), dt)
        self(dict(z=one, t=one, theta=tf.zeros((1, self.theta_dim), dt),
                  c_ic=tf.zeros((1, self.n_comp), dt), q_ic=tf.zeros((1, self.n_comp), dt)))
        return self

    def interface_loss(self, *args, **kwargs):
        """XPINN hook: continuity of c_i, T and of the total (convective + diffusive) fluxes at the
        boundary between two adsorbent layers. Not implemented yet."""
        raise NotImplementedError("XPINN interface loss is a later step")


class MarchingWindow(tf.keras.Model):
    """One window of time marching (Phase 3): the PINN from t~ = t_start on, started exactly from
    the state the previous window (frozen) reaches at t_start.

    With s = 1 - exp(-(t~ - t_start)/delta) and the previous window's state c_p, q_p, T_p, v_p at
    (z~, t_start, theta), the new network `net` only adds what changes after t_start:
        c~ = (1 - s) c_p + s softplus(N_c)      q~ = (1 - s) q_p + s softplus(N_q)
        T~ = (1 - s) T_p + s N_T                v~ = (1 - s) v_p + s (1 + z~ N_v)
    so c~, q~ >= 0 and v = v_s at the inlet stay exact. Before t_start (s clipped at 0) it returns
    the state at t_start. `prev` is a PINN (first window) or another MarchingWindow.
    """

    def __init__(self, prev, net, t_start, **kwargs):
        super().__init__(**kwargs)
        self.prev = prev
        self.prev.trainable = False  # earlier windows are fixed: only `net` trains
        self.net = net
        self.t_start = float(t_start)
        self.delta = float(net.delta)

    def call(self, inputs):
        z, t = inputs["z"], inputs["t"]
        p = self.prev(dict(inputs, t=tf.zeros_like(t) + self.t_start))  # state at t_start, no t-dependence
        N_c, N_q, N_T, N_v = self.net.subnets[0](z, t, inputs["theta"])
        s = 1.0 - tf.exp(-tf.maximum(t - self.t_start, 0.0) / self.delta)
        return dict(
            c=(1.0 - s) * p["c"] + s * tf.nn.softplus(N_c),
            q=(1.0 - s) * p["q"] + s * tf.nn.softplus(N_q),
            T=(1.0 - s) * p["T"] + s * N_T,
            v=(1.0 - s) * p["v"] + s * (1.0 + z * N_v),
        )


class MarchedPINN(tf.keras.Model):
    """The windows of time marching as one model: each point is answered by the window holding its
    t~ (window k covers [t_starts[k], t_starts[k+1])). For evaluation; training uses each window."""

    def __init__(self, windows, t_starts, **kwargs):
        super().__init__(**kwargs)
        if len(windows) != len(t_starts):
            raise ValueError("one start time per window")
        self.windows = list(windows)
        self.t_starts = [float(x) for x in t_starts]

    def call(self, inputs):
        t = inputs["t"]
        out = self.windows[0](inputs)
        for window, t0 in zip(self.windows[1:], self.t_starts[1:]):
            later = window(inputs)
            out = {k: tf.where(t >= t0, later[k], out[k]) for k in out}
        return out


def outputs_as_matrix(out):
    """dict c, q, T, v -> [B, 2n+2] in the order c_0..c_{n-1}, q_0..q_{n-1}, T, v."""
    return tf.concat([out["c"], out["q"], out["T"], out["v"]], axis=-1)
