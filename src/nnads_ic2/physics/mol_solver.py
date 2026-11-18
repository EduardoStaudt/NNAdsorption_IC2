"""Small reference solver for Eqs. (1)-(6) with variable velocity (method of lines).

ONLY for tests and Phase 0, while the v23b solver code is not available. It is not the
production solver: it follows the same model, but not the same numerics.

Discretization (finite volumes, nx uniform cells of width h = L/nx):
- convection: face values reconstructed upwind (flow in +z) with a MUSCL slope and the van
  Leer limiter (second order away from fronts, no new extrema at fronts);
- dispersion and conduction: central differences between neighbouring cells;
- inlet face: Danckwerts flux vs c_i,F for each gas and convective heat inflow at Tin, with no
  diffusive or conductive part; outlet face: no diffusive/conductive flux;
- velocity: from the overall mole balance (Eq. 5), marched from the inlet face by face, with
  C_T = P/(R T). The energy equation needs v and v needs dT/dt, so both are found by a few
  fixed-point (Picard) sweeps inside each right-hand-side evaluation;
- time: scipy `solve_ivp` with BDF (implicit, variable order and step).
Two extra states per gas integrate the moles that leave the column, for the mole balance check.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp

from .constants import R
from .isotherm import qstar_np
from .scales import feed_and_initial_state


class MOLSolver:
    def __init__(self, p, nx, scheme, fp_iters):
        if scheme not in ("vanleer", "upwind1"):
            raise ValueError("scheme must be 'vanleer' or 'upwind1'")
        self.p = p
        self.n = p.n_comp
        self.nx = nx
        self.h = p.L / nx
        self.z = (np.arange(nx) + 0.5) * self.h  # cell centres
        self.z_faces = np.arange(nx + 1) * self.h
        self.scheme = scheme
        self.fp_iters = fp_iters
        st = feed_and_initial_state(p)
        self.c_feed = st["c_feed"]
        self.G_in = p.vs * self.c_feed.sum()  # total molar flux entering [mol/(m2 s)]
        self.iso = p.isotherm_coeffs()
        self.DL = p.DL
        self.kap = 4.0 * p.hw / p.Dt  # wall heat loss per volume and kelvin [W/(m3 K)]
        c0 = np.repeat(st["c_init"][:, None], nx, axis=1)
        q0 = np.repeat(st["q_init"][:, None], nx, axis=1)
        T0 = np.full(nx, p.Tin)
        self.y0 = np.concatenate([c0.ravel(), q0.ravel(), T0, np.zeros(self.n)])
        self.q_scale = np.maximum(np.maximum(st["q_feed"], st["q_init"]), 1e-12)
        self.C_F = st["C_F"]

    # ---- helpers ------------------------------------------------------------------
    def unpack(self, y):
        n, nx = self.n, self.nx
        c = y[: n * nx].reshape(n, nx)
        q = y[n * nx : 2 * n * nx].reshape(n, nx)
        T = y[2 * n * nx : 2 * n * nx + nx]
        n_out = y[2 * n * nx + nx :]
        return c, q, T, n_out

    def faces(self, u):
        """Upwind face values of cell averages u [..., nx] -> [..., nx+1].
        Face 0 (inlet) is left as NaN for the caller; face nx (outlet) copies the last cell."""
        f = np.full(u.shape[:-1] + (self.nx + 1,), np.nan)
        slope = np.zeros_like(u)
        if self.scheme == "vanleer":
            dm = u[..., 1:-1] - u[..., :-2]
            dp = u[..., 2:] - u[..., 1:-1]
            prod = dm * dp
            den = np.where(prod > 0, dm + dp, 1.0)
            slope[..., 1:-1] = np.where(prod > 0, 2.0 * prod / den, 0.0)  # harmonic mean of the two slopes
        f[..., 1:-1] = u[..., :-1] + 0.5 * slope[..., :-1]
        f[..., -1] = u[..., -1]
        return f

    # ---- right-hand side ------------------------------------------------------------
    def core(self, c, q, T):
        """Time derivatives of c, q, T and the face velocities for one state."""
        p, h, nx = self.p, self.h, self.nx
        qdot = p.kL[:, None] * (qstar_np(c.T, T, self.iso).T - q)  # LDF, Eq. (3)

        c_f = self.faces(c)
        T_f = self.faces(T)
        T_f[0] = p.Tin  # convective heat inflow at the feed temperature
        sum_c = c.sum(axis=0)
        sum_c_f = c_f[:, 1:-1].sum(axis=0)
        disp_total = p.eb * self.DL * (sum_c[1:] - sum_c[:-1]) / h  # minus total dispersive flux

        S = p.Cpg @ c  # sum_i Cp_i c_i, heat capacity of the gas per volume [J/(m3 K)]
        A = p.eb * S + p.rho_b * p.Cps  # total heat capacity per bed volume
        heat = p.rho_b * (p.dH @ qdot)  # heat released by adsorption [W/m3]
        wall = self.kap * (T - p.Tin)  # heat lost through the wall (Tw = Tin)
        cond_f = np.zeros(nx + 1)
        cond_f[1:-1] = -p.lam * (T[1:] - T[:-1]) / h  # conductive heat flux, zero at both ends
        cond = -(cond_f[1:] - cond_f[:-1]) / h
        dTdz = (T_f[1:] - T_f[:-1]) / h
        sink = p.rho_b * qdot.sum(axis=0)  # moles leaving the gas into the solid [mol/(m3 s)]

        def velocity(Tdot):
            # Eq. (5): dG/dz = -(eb dC_T/dt + rho_b sum dq/dt), G = total molar flux, C_T = P/(R T)
            CTdot = -(p.P / (R * T**2)) * Tdot
            G = np.empty(nx + 1)
            G[0] = self.G_in
            G[1:] = self.G_in - h * np.cumsum(p.eb * CTdot + sink)
            v = np.empty(nx + 1)
            v[0] = p.vs
            v[1:-1] = (G[1:-1] + disp_total) / sum_c_f  # so that sum_i (component fluxes) = G
            v[-1] = G[-1] / sum_c[-1]
            return v

        Tdot = np.zeros(nx)
        for _ in range(self.fp_iters):
            v = velocity(Tdot)
            v_cell = 0.5 * (v[:-1] + v[1:])
            Tdot = (-v_cell * S * dTdz + cond + heat - wall) / A  # Eq. (4)
        v = velocity(Tdot)  # final velocity consistent with the returned dT/dt

        F = np.empty((self.n, nx + 1))  # component molar fluxes at the faces
        F[:, 0] = p.vs * self.c_feed  # Danckwerts: total inflow equals the feed
        F[:, 1:-1] = v[1:-1] * c_f[:, 1:-1] - p.eb * self.DL * (c[:, 1:] - c[:, :-1]) / h
        F[:, -1] = v[-1] * c[:, -1]
        cdot = (-(F[:, 1:] - F[:, :-1]) / h - p.rho_b * qdot) / p.eb  # Eq. (1)
        return cdot, qdot, Tdot, v, F[:, -1]

    def rhs(self, t, y):
        c, q, T, _ = self.unpack(y)
        cdot, qdot, Tdot, _, out_flux = self.core(c, q, T)
        return np.concatenate([cdot.ravel(), qdot.ravel(), Tdot, out_flux])

    # ---- driver ---------------------------------------------------------------------
    def solve(self, t_final, n_out, rtol, atol_rel, method="BDF"):
        n, nx, p = self.n, self.nx, self.p
        t_eval = np.linspace(0.0, t_final, n_out)
        atol = np.concatenate([
            np.full(n * nx, atol_rel * self.C_F),
            np.repeat(atol_rel * self.q_scale, nx),
            np.full(nx, atol_rel * p.Tin),
            np.full(n, atol_rel * self.C_F * p.vs * t_final),
        ])
        t0 = time.time()
        sol = solve_ivp(self.rhs, (0.0, t_final), self.y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
        wall = time.time() - t0
        if not sol.success:
            raise RuntimeError("MOL solver failed: %s" % sol.message)

        nt = sol.t.size
        c = np.empty((nt, n, nx))
        q = np.empty((nt, n, nx))
        T = np.empty((nt, nx))
        v = np.empty((nt, nx + 1))
        n_out = np.empty((nt, n))
        for k in range(nt):
            c[k], q[k], T[k], n_out[k] = self.unpack(sol.y[:, k])
            v[k] = self.core(c[k], q[k], T[k])[3]
        c_out = c[:, :, -1]
        holdup = self.h * (p.eb * c + p.rho_b * q).sum(axis=2)  # moles per bed area [mol/m2]
        n_in = p.vs * self.c_feed[None, :] * sol.t[:, None]
        return dict(
            z=self.z,
            z_faces=self.z_faces,
            t=sol.t,
            c=c,
            q=q,
            T=T,
            v_faces=v,
            v_cell=0.5 * (v[:, :-1] + v[:, 1:]),
            c_out=c_out,
            y_out=c_out / c_out.sum(axis=1, keepdims=True),
            T_out=T[:, -1],
            v_out=v[:, -1],
            n_in=n_in,
            n_out=n_out,
            holdup=holdup,
            info=dict(
                nx=nx, scheme=self.scheme, fp_iters=self.fp_iters, rtol=rtol, atol_rel=atol_rel,
                nfev=int(sol.nfev), njev=int(sol.njev), nlu=int(sol.nlu), wall_s=wall,
            ),
        )


def solve_case(p, t_final, solver_cfg):
    """Run the reference solver with the `solver:` section of the config."""
    s = MOLSolver(p, nx=int(solver_cfg["nx"]), scheme=solver_cfg["scheme"], fp_iters=int(solver_cfg["fp_iters"]))
    return s.solve(
        t_final=t_final,
        n_out=int(solver_cfg["n_out"]),
        rtol=float(solver_cfg["rtol"]),
        atol_rel=float(solver_cfg["atol_rel"]),
        method=solver_cfg["method"],
    )


def save_solution(sol, path):
    flat = {k: v for k, v in sol.items() if k != "info"}
    flat.update({"info_" + k: v for k, v in sol["info"].items()})
    np.savez_compressed(path, **flat)


def load_solution(path):
    d = np.load(path, allow_pickle=False)
    sol = {k: d[k] for k in d.files if not k.startswith("info_")}
    sol["info"] = {k[5:]: d[k].item() for k in d.files if k.startswith("info_")}
    return sol
