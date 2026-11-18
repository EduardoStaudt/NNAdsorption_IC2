"""Reference scales, stoichiometric times and dimensionless groups (derivation: docs/nondim.md).

Dimensionless variables (a tilde in the docs, plain names in code):
    z~ = z / L
    t~ = t / t_ref,             t_ref = max_i t_st,i   (stoichiometric time of the slowest front, Eq. 7)
    c~_i = c_i / C_F,           C_F = P / (R Tin)      (total gas concentration of the feed)
    q~_i = q_i / q_ref,i,       q_ref,i = max(q*_i at feed, q*_i at initial state, floor)
    T~ = (T - Tin) / dT_ref,    dT_ref = sum_i y_i,F (-dH_i) q_ref,i rho_b / (rho_b Cps), with a floor
    v~ = v / vs
Using the larger of the feed and initial loadings for q_ref avoids a fixed "carrier" index:
the gas that fills the bed at t = 0 gets its initial loading, the others their feed loading.
"""

from dataclasses import dataclass

import numpy as np

from .constants import R
from .isotherm import qstar_np


@dataclass(frozen=True)
class Scales:
    L: float  # m
    t_ref: float  # s
    t_final: float  # s, simulation horizon tF
    C_F: float  # mol/m3
    q_ref: np.ndarray  # mol/kg, one per component
    dT_ref: float  # K
    vs: float  # m/s
    Tin: float  # K


def feed_and_initial_state(p):
    """Gas concentrations and equilibrium loadings of the feed and of the initial bed (both at Tin)."""
    C_F = p.P / (R * p.Tin)
    c_feed = p.y_feed * C_F
    c_init = p.y_init * C_F
    iso = p.isotherm_coeffs()
    return dict(
        C_F=C_F,
        c_feed=c_feed,
        c_init=c_init,
        q_feed=qstar_np(c_feed, p.Tin, iso),
        q_init=qstar_np(c_init, p.Tin, iso),
    )


def stoich_times(p):
    """Eq. (7): t_st,i = (L/vs) (eb + rho_b q*_i,F / c_i,F). NaN for components absent from the feed.

    Time a front would need to cross the bed if the bed took up exactly the feed loading:
    the gas holdup (eb) plus the adsorbed holdup (rho_b q*/c) divided by the feed flow.
    """
    st = feed_and_initial_state(p)
    present = st["c_feed"] > 0
    tst = np.full(p.n_comp, np.nan)
    tst[present] = (p.L / p.vs) * (p.eb + p.rho_b * st["q_feed"][present] / st["c_feed"][present])
    return tst


def initial_holdup_times(p):
    """t_init,i = (L / (vs c_i,F)) (eb c_i,0 + rho_b q_i,0): moles of i in the bed at t = 0,
    written as a time of feed flow. Zero for components absent at t = 0.

    Mole balance of component i over the column, from t = 0 until the bed is back at the feed
    state:  integral of [1 - v(L,t) c_i(L,t) / (vs c_i,F)] dt  =  t_st,i - t_init,i.
    For a component that is not in the bed at t = 0 this is the identity of Section 2.4.
    """
    st = feed_and_initial_state(p)
    present = st["c_feed"] > 0
    out = np.full(p.n_comp, np.nan)
    hold = p.eb * st["c_init"] + p.rho_b * st["q_init"]
    out[present] = (p.L / (p.vs * st["c_feed"][present])) * hold[present]
    return out


def compute_scales(p, q_ref_floor, dT_ref_floor, t_final_factor):
    st = feed_and_initial_state(p)
    t_ref = float(np.nanmax(stoich_times(p)))
    q_ref = np.maximum(np.maximum(st["q_feed"], st["q_init"]), q_ref_floor)
    dT_est = np.sum(p.y_feed * p.dH * q_ref) * p.rho_b / (p.rho_b * p.Cps)
    return Scales(
        L=p.L,
        t_ref=t_ref,
        t_final=t_final_factor * t_ref,
        C_F=float(st["C_F"]),
        q_ref=q_ref,
        dT_ref=float(max(dT_est, dT_ref_floor)),
        vs=p.vs,
        Tin=p.Tin,
    )


def to_dimless(s, z=None, t=None, c=None, q=None, T=None, v=None):
    """Dimensional -> dimensionless. Components on the last axis of c and q."""
    out = {}
    if z is not None:
        out["z"] = np.asarray(z) / s.L
    if t is not None:
        out["t"] = np.asarray(t) / s.t_ref
    if c is not None:
        out["c"] = np.asarray(c) / s.C_F
    if q is not None:
        out["q"] = np.asarray(q) / s.q_ref
    if T is not None:
        out["T"] = (np.asarray(T) - s.Tin) / s.dT_ref
    if v is not None:
        out["v"] = np.asarray(v) / s.vs
    return out


def to_dimensional(s, z=None, t=None, c=None, q=None, T=None, v=None):
    """Dimensionless -> dimensional (inverse of to_dimless)."""
    out = {}
    if z is not None:
        out["z"] = np.asarray(z) * s.L
    if t is not None:
        out["t"] = np.asarray(t) * s.t_ref
    if c is not None:
        out["c"] = np.asarray(c) * s.C_F
    if q is not None:
        out["q"] = np.asarray(q) * s.q_ref
    if T is not None:
        out["T"] = s.Tin + np.asarray(T) * s.dT_ref
    if v is not None:
        out["v"] = np.asarray(v) * s.vs
    return out


def dimensionless_groups(p, s):
    """All dimensionless numbers that appear in the residuals (docs/nondim.md, Section 3).

    Scalars are floats, per-component groups are arrays [n].
    """
    st = feed_and_initial_state(p)
    Cp_F = float(np.sum(p.y_feed * p.Cpg))  # molar heat capacity of the feed mixture
    alpha = p.eb * p.L / (p.vs * s.t_ref)  # gas residence time / t_ref
    Pe = p.vs * p.L / (p.eb * p.DL)  # axial Peclet number (convection / dispersion)
    K = p.rho_b * s.q_ref / (p.eb * s.C_F)  # adsorbed / gas capacity ratio
    beta = alpha * K  # = rho_b q_ref L / (vs C_F t_ref)
    Da = p.kL * s.t_ref  # Damkohler: mass-transfer rate x reference time
    gamma = p.eb * s.C_F * p.Cpg / (p.rho_b * p.Cps)  # gas / solid heat capacity
    conv_T = p.vs * s.C_F * p.Cpg * s.t_ref / (p.L * p.rho_b * p.Cps)  # heat convection
    diff_T = p.lam * s.t_ref / (p.rho_b * p.Cps * p.L**2)  # axial conduction
    heat = p.dH * s.q_ref / (p.Cps * s.dT_ref)  # heat of adsorption
    St_w = 4.0 * p.hw * s.t_ref / (p.Dt * p.rho_b * p.Cps)  # wall heat loss
    Pe_T = p.L * p.vs * s.C_F * Cp_F / p.lam  # thermal Peclet number (inlet condition)
    tst = stoich_times(p) / s.t_ref
    groups = dict(
        alpha=alpha,
        Pe=Pe,
        K=K,
        beta=beta,
        Da=Da,
        gamma=gamma,
        conv_T=conv_T,
        diff_T=diff_T,
        heat=heat,
        St_w=St_w,
        Pe_T=Pe_T,
        cp_ratio=p.Cpg / Cp_F,
        eps_T=s.dT_ref / s.Tin,
        y_feed=p.y_feed.copy(),
        c_init=st["c_init"] / s.C_F,
        q_init=st["q_init"] / s.q_ref,
        q_feed=st["q_feed"] / s.q_ref,
        t_final=s.t_final / s.t_ref,
        tst=tst,
    )
    # Each residual is divided by its largest coefficient so all residuals are O(1).
    groups["scale_mass"] = np.maximum.reduce([np.ones_like(beta), np.full_like(beta, alpha), np.full_like(beta, 1.0 / Pe), beta])
    groups["scale_ldf"] = np.maximum(1.0, Da)
    groups["scale_energy"] = max(1.0 + gamma.sum(), conv_T.sum(), diff_T, heat.sum(), St_w)
    groups["scale_global"] = max(1.0, alpha, beta.sum(), 1.0 / Pe)
    return groups
