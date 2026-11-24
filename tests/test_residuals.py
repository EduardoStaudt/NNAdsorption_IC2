"""Residuals vanish on a solution of the equations and do not vanish on an arbitrary field.

The solution comes from the MOL reference solver, interpolated with bicubic splines in
(t~, z~); the derivatives of the splines play the role of the automatic derivatives of the PINN.
"""

import numpy as np
import tensorflow as tf
from scipy.interpolate import RectBivariateSpline

from nnads_ic2.physics.residuals import (
    case_arrays,
    gather_cases,
    pde_residuals,
    stack_cases,
)
from nnads_ic2.physics.scales import dimensionless_groups, to_dimless

N_POINTS = 4000


def case_table(p, s):
    return stack_cases([case_arrays(p, s, dimensionless_groups(p, s))], tf.float64)


def solver_fields(sol, s, t_pts, z_pts):
    """Fields and derivatives of the solver solution at (t~, z~) points, by spline interpolation."""
    d = to_dimless(s, z=sol["z"], t=sol["t"])
    c = sol["c"] / s.C_F
    q = sol["q"] / s.q_ref[None, :, None]
    T = (sol["T"] - s.Tin) / s.dT_ref
    v = sol["v_cell"] / s.vs

    def spline(F):
        return RectBivariateSpline(d["t"], d["z"], F, kx=3, ky=3, s=0)

    def ev(F, dt=0, dz=0):
        return spline(F).ev(t_pts, z_pts, dx=dt, dy=dz)

    n = c.shape[1]
    comp = lambda F, dt=0, dz=0: np.stack([ev(F[:, i], dt, dz) for i in range(n)], axis=1)  # noqa: E731
    col = lambda F, dt=0, dz=0: ev(F, dt, dz)[:, None]  # noqa: E731
    return dict(
        c=comp(c), c_t=comp(c, 1, 0), c_z=comp(c, 0, 1), c_zz=comp(c, 0, 2),
        q=comp(q), q_t=comp(q, 1, 0),
        T=col(T), T_t=col(T, 1, 0), T_z=col(T, 0, 1), T_zz=col(T, 0, 2),
        v=col(v), v_z=col(v, 0, 1),
    )


def random_fields(rng, n, t_pts, z_pts):
    """Smooth but arbitrary fields (random sinusoids) with exact derivatives."""
    B = t_pts.size

    def wave(amp, mean):
        k, w, ph = rng.uniform(1, 6), rng.uniform(1, 6), rng.uniform(0, 2 * np.pi)
        arg = k * z_pts + w * t_pts + ph
        return (mean + amp * np.sin(arg), amp * w * np.cos(arg), amp * k * np.cos(arg), -amp * k**2 * np.sin(arg))

    cs = [wave(0.3, 0.5) for _ in range(n)]
    qs = [wave(0.3, 0.5) for _ in range(n)]
    Tw = wave(0.5, 0.0)
    vw = wave(0.1, 1.0)
    st = lambda items, j: np.stack([it[j] for it in items], axis=1)  # noqa: E731
    return dict(
        c=st(cs, 0), c_t=st(cs, 1), c_z=st(cs, 2), c_zz=st(cs, 3),
        q=st(qs, 0), q_t=st(qs, 1),
        T=Tw[0][:, None], T_t=Tw[1][:, None], T_z=Tw[2][:, None], T_zz=Tw[3][:, None],
        v=vw[0][:, None], v_z=vw[2][:, None],
    ), B


def rms_by_equation(fields, table):
    f = {k: tf.constant(v, tf.float64) for k, v in fields.items()}
    B = next(iter(fields.values())).shape[0]
    g = gather_cases(table, tf.zeros(B, tf.int32))
    res = pde_residuals(f, g, p_min=0.0)
    return {k: float(tf.sqrt(tf.reduce_mean(r**2))) for k, r in res.items()}


def test_residuals_small_on_solver_solution_and_large_on_random_field(example_case, mol_example):
    p, s = example_case
    sol = mol_example
    rng = np.random.default_rng(0)
    z_dim = to_dimless(s, z=sol["z"])["z"]
    # interior points: away from the first/last cells and from t~ = 0 (inlet/initial corner)
    z_pts = rng.uniform(z_dim[2], z_dim[-3], N_POINTS)
    t_pts = rng.uniform(0.05, 1.95, N_POINTS)
    table = case_table(p, s)

    rms_solver = rms_by_equation(solver_fields(sol, s, t_pts, z_pts), table)
    rms_random = rms_by_equation(random_fields(rng, p.n_comp, t_pts, z_pts)[0], table)
    print("\nRMS residual  solver vs random field")
    for k in rms_solver:
        print("  %-7s %.3e   %.3e" % (k, rms_solver[k], rms_random[k]))

    for k in rms_solver:
        assert rms_solver[k] < 0.01 * rms_random[k], k
    assert rms_solver["eos"] < 1e-5  # the solver keeps sum(c) = P/RT almost exactly


def test_residuals_stay_finite_when_the_network_temperature_is_unphysical(example_case):
    # T <= 0 K (T~ <= -1/eps_T) made 1/T and exp(k4/T) infinite and one such point turned every
    # weight into NaN (Phase 2, stage c); the closures now floor T at Tin/2. float32 as in training.
    p, s = example_case
    table = stack_cases([case_arrays(p, s, dimensionless_groups(p, s))], tf.float32)
    T_kelvin = np.array([-2.0 * s.Tin, 0.0, 10.0, 0.3 * s.Tin, s.Tin, 1.2 * s.Tin])
    n = T_kelvin.size
    rng = np.random.default_rng(1)
    fields, _ = random_fields(rng, p.n_comp, rng.uniform(0.05, 1.95, n), rng.uniform(0.05, 0.95, n))
    fields["T"] = ((T_kelvin - s.Tin) / s.dT_ref)[:, None]
    f = {k: tf.constant(v, tf.float32) for k, v in fields.items()}
    res = pde_residuals(f, gather_cases(table, tf.zeros(n, tf.int32)), p_min=1e-6)
    for k, r in res.items():
        assert np.all(np.isfinite(r.numpy())), k
    # physical temperatures (the last two points) are not touched by the floor
    eps_T = float(np.ravel(table["eps_T"].numpy())[0])
    np.testing.assert_allclose(res["eos"].numpy()[-2:, 0],
                               fields["c"][-2:].sum(axis=1) - 1.0 / (1.0 + eps_T * fields["T"][-2:, 0]), rtol=1e-5)
