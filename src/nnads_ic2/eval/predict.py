"""Predictions of a trained PINN, back in dimensional units."""

import time

import numpy as np
import tensorflow as tf

from ..physics.residuals import gather_cases
from ..physics.scales import to_dimensional
from ..training.losses import model_inputs


def predict_dimless(model, table, case_index, z, t, batch_size):
    """Dimensionless c, q, T, v at points (z~, t~) of one case. Returns numpy arrays."""
    dt = tf.keras.backend.floatx()
    z = np.asarray(z, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    parts = []
    for i in range(0, z.size, batch_size):
        zz = tf.constant(z[i : i + batch_size, None], dt)
        tt = tf.constant(t[i : i + batch_size, None], dt)
        g = gather_cases(table, tf.fill([zz.shape[0]], case_index))
        parts.append({k: v.numpy() for k, v in model(model_inputs(zz, tt, g)).items()})
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def predict_outlet(model, table, case_index, t_seconds, s, batch_size):
    """Outlet curves at z = L: c_i [mol/m3], y_i, T [K], v [m/s] on the given times."""
    t_dimless = np.asarray(t_seconds) / s.t_ref
    d = predict_dimless(model, table, case_index, np.ones_like(t_dimless), t_dimless, batch_size)
    dim = to_dimensional(s, c=d["c"], T=d["T"][:, 0], v=d["v"][:, 0])
    return dict(
        t=np.asarray(t_seconds),
        c_out=dim["c"],
        y_out=d["c"] / d["c"].sum(axis=1, keepdims=True),
        T_out=dim["T"],
        v_out=dim["v"],
    )


def inference_ms(model, table, case_index, n_points, repeats):
    """Mean wall time (ms) to predict one outlet curve of n_points times (after warm-up)."""
    dt = tf.keras.backend.floatx()
    t = tf.constant(np.linspace(0, 2, n_points)[:, None], dt)
    z = tf.ones_like(t)
    g = gather_cases(table, tf.fill([n_points], case_index))
    fn = tf.function(lambda zz, tt: model(model_inputs(zz, tt, g)))
    fn(z, t)["c"].numpy()  # warm-up (tracing)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(z, t)["c"].numpy()
    return 1000.0 * (time.perf_counter() - t0) / repeats
