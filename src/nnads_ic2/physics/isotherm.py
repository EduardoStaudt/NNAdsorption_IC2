"""Multicomponent LRC isotherm (Section 2.2), identical to IC1 `solver_jee.qstar_slope`.

    q*_i = qm_i B_i p_i^n_i / (1 + sum_j B_j p_j^n_j)       equilibrium loading [mol/kg]
    qm_i = k1_i + k2_i T          saturation capacity, falls with temperature
    B_i  = k3_i exp(k4_i / T)     affinity [1/atm^n], falls with temperature (exothermic)
    n_i  = k5_i + k6_i / T        heterogeneity exponent
    p_i  = c_i R T / ATM          partial pressure in atm (IC1 convention, not Pa)

The sum in the denominator is the competition: every component takes adsorption sites
from the others. Arrays carry the component on the LAST axis: c [..., n], q* [..., n].
"""

import numpy as np

from .constants import ATM, R

# IC1 contract clips (solver_jee.qstar_slope). Keep them identical: tests compare with IC1 at 1e-10.
QM_FLOOR = 1e-6  # mol/kg, qm(T) can never become zero or negative
N_MIN = 0.3  # bounds of the exponent n(T)
N_MAX = 3.0


def qstar_np(c, T, iso):
    """Equilibrium loading (numpy).

    c: [..., n] gas concentrations [mol/m3]; T: [...] temperature [K];
    iso: dict with k1..k6, each [n]. Returns q* [..., n] in mol/kg.
    """
    T_ = np.asarray(T, dtype=float)[..., None]
    qm = np.maximum(iso["k1"] + iso["k2"] * T_, QM_FLOOR)
    B = iso["k3"] * np.exp(iso["k4"] / T_)
    n = np.clip(iso["k5"] + iso["k6"] / T_, N_MIN, N_MAX)
    p = np.maximum(c * R * T_ / ATM, 0.0)
    t = B * p**n  # "loading term" of each component
    return qm * t / (1.0 + t.sum(axis=-1, keepdims=True))


def qstar_tf(c, T, iso, p_min=0.0):
    """Equilibrium loading (TensorFlow, differentiable).

    c: [..., n]; T: [..., 1] (trailing axis of size 1, broadcast over components);
    iso: dict k1..k6, tensors broadcastable to [..., n].
    p_min > 0 puts a floor on the partial pressure. Use it only in training: the slope
    d(p^n)/dp = n p^(n-1) is infinite at p = 0 when n < 1, which would give NaN gradients.
    With p_min = 0 the function is exactly the IC1 isotherm.
    """
    import tensorflow as tf

    qm = tf.maximum(iso["k1"] + iso["k2"] * T, QM_FLOOR)
    B = iso["k3"] * tf.exp(iso["k4"] / T)
    n = tf.clip_by_value(iso["k5"] + iso["k6"] / T, N_MIN, N_MAX)
    p = tf.maximum(c * R * T / ATM, p_min)
    t = B * tf.pow(p, n)
    return qm * t / (1.0 + tf.reduce_sum(t, axis=-1, keepdims=True))
