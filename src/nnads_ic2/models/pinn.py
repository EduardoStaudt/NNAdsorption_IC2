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

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


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


def _head(n_out, hidden, width, activation, name):
    layers = [tf.keras.layers.Dense(width, activation=activation, name="%s_h%d" % (name, k)) for k in range(hidden)]
    layers.append(tf.keras.layers.Dense(n_out, name=name + "_out"))
    return layers


class SubNet(tf.keras.layers.Layer):
    """Trunk and heads of one domain. Returns the raw (unconstrained) outputs N_c, N_q, N_T, N_v."""

    def __init__(self, n_comp, model_cfg, t_final, **kwargs):
        super().__init__(**kwargs)
        ff = model_cfg["fourier"]
        act = tf.keras.activations.get(model_cfg["activation"])
        self.t_final = float(t_final)
        self.ff = (
            FourierFeatures(ff["n_features"], ff["sigma_z"], ff["sigma_t"], ff["seed"], ff["trainable"], name="fourier")
            if ff["n_features"] > 0
            else None
        )
        self.trunk = [
            tf.keras.layers.Dense(model_cfg["width"], activation=act, name="trunk_%d" % k)
            for k in range(model_cfg["depth"])
        ]
        hw, hh = model_cfg["head_width"], model_cfg["head_hidden"]
        self.head_c = _head(n_comp, hh, hw, act, "head_c")
        self.head_q = _head(n_comp, hh, hw, act, "head_q")
        self.head_T = _head(1, hh, hw, act, "head_T")
        self.head_v = _head(1, hh, hw, act, "head_v")

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


def outputs_as_matrix(out):
    """dict c, q, T, v -> [B, 2n+2] in the order c_0..c_{n-1}, q_0..q_{n-1}, T, v."""
    return tf.concat([out["c"], out["q"], out["T"], out["v"]], axis=-1)
