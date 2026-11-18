"""Dimensionless residuals of Eqs. (1)-(6) (derivation and notation: docs/nondim.md).

Every residual function receives two dicts of tensors (batch on axis 0):

  f: fields and their derivatives at the points
       c, c_t, c_z, c_zz   [B, n]  gas concentrations c~_i
       q, q_t              [B, n]  adsorbed loadings q~_i
       T, T_t, T_z, T_zz   [B, 1]  temperature T~
       v, v_z              [B, 1]  velocity v~
  g: constants of the case each point belongs to (see `case_arrays`, gathered per point)
       dimensionless groups [B, n] or [B, 1], plus what the isotherm needs.

The derivatives may come from automatic differentiation (PINN) or from a spline fitted to a
solver solution (tests); the residual algebra is the same. Residuals are already divided by
their largest coefficient (docs/nondim.md, Section 5), so all of them are O(1).
A zero residual means the equation is satisfied at that point.
"""

import numpy as np
import tensorflow as tf

from .isotherm import qstar_tf

# Case constants used by the residuals: one value per case, or one per component.
SCALAR_KEYS = ["alpha", "Pe", "diff_T", "St_w", "Pe_T", "eps_T", "t_final", "scale_energy", "scale_global"]
COMP_KEYS = [
    "beta", "Da", "gamma", "conv_T", "heat", "cp_ratio", "y_feed", "c_init", "q_init", "scale_mass", "scale_ldf",
]
ISO_KEYS = ["k1", "k2", "k3", "k4", "k5", "k6"]


def case_arrays(p, s, groups):
    """Flat dict of numpy arrays for one case: scalars as shape [1], per-component as [n]."""
    d = {k: np.atleast_1d(np.asarray(groups[k], dtype=np.float64)) for k in SCALAR_KEYS + COMP_KEYS}
    d.update({k: np.asarray(v, dtype=np.float64) for k, v in p.isotherm_coeffs().items()})
    d["q_ref"] = np.asarray(s.q_ref, dtype=np.float64)
    d["C_F"] = np.array([s.C_F])
    d["Tin"] = np.array([s.Tin])
    d["dT_ref"] = np.array([s.dT_ref])
    return d


def stack_arrays(cases):
    """List of `case_arrays` dicts -> dict of numpy arrays [n_cases, k]."""
    return {k: np.stack([c[k] for c in cases]) for k in cases[0]}


def stack_cases(cases, dtype, variable=False):
    """List of `case_arrays` dicts -> dict of tensors [n_cases, k].

    variable=True gives tf.Variables, so the rows can be replaced during training (new
    collocation theta) without retracing the compiled training step.
    """
    arrays = stack_arrays(cases)
    if variable:
        return {k: tf.Variable(v, dtype=dtype, trainable=False, name="case_" + k) for k, v in arrays.items()}
    return {k: tf.constant(v, dtype=dtype) for k, v in arrays.items()}


def gather_cases(table, case_idx):
    """Pick the constants of the case of each point: [n_cases, k] -> [B, k]."""
    return {k: tf.gather(v, case_idx) for k, v in table.items()}


# Floor of the temperature seen by the closures: T >= T_MIN_FRAC * Tin, i.e. 1 + eps_T T~ >= T_MIN_FRAC.
# Only unphysical network outputs reach it (real beds stay well above Tin/2; the coldest MOL
# solution seen, hard seed 21, went down to 0.74 Tin). Over the full domain eps_T = dT_ref/Tin
# reaches 0.8, so T~ = -1.3 already means T = 0 K, where 1/T and exp(k4/T) are infinite: one such
# point made every weight NaN at step 550 of the first stage (c) run (Phase 2).
T_MIN_FRAC = 0.5


def _sum(x):
    """Sum over components, keeping a trailing axis of size 1."""
    return tf.reduce_sum(x, axis=-1, keepdims=True)


def total_concentration(f, g):
    """C~_T = Tin/T = 1/(1 + eps_T T~) (ideal gas, constant P) and its derivatives (chain rule)."""
    e = g["eps_T"]
    D = tf.maximum(1.0 + e * f["T"], T_MIN_FRAC)  # = T/Tin, floored
    CT = 1.0 / D
    dCT_dT = -e / D**2
    CT_t = dCT_dT * f["T_t"]
    CT_z = dCT_dT * f["T_z"]
    CT_zz = dCT_dT * f["T_zz"] + 2.0 * e**2 / D**3 * f["T_z"] ** 2
    return CT, CT_t, CT_z, CT_zz


def qstar_dimless(f, g, p_min):
    """q~*_i: equilibrium loading at the local gas state, divided by q_ref,i."""
    c = g["C_F"] * f["c"]
    T = tf.maximum(g["Tin"] + g["dT_ref"] * f["T"], T_MIN_FRAC * g["Tin"])
    return qstar_tf(c, T, {k: g[k] for k in ISO_KEYS}, p_min=p_min) / g["q_ref"]


def mass_residual(f, g):
    """Eq. (1) per gas: accumulation + convection - dispersion + uptake by the solid."""
    convection = f["v"] * f["c_z"] + f["c"] * f["v_z"]  # d(v~ c~_i)/dz~ (product rule)
    r = g["alpha"] * f["c_t"] + convection - f["c_zz"] / g["Pe"] + g["beta"] * f["q_t"]
    return r / g["scale_mass"]


def ldf_residual(f, g, qstar):
    """Eq. (3): the solid loading relaxes towards equilibrium at rate Da."""
    return (f["q_t"] - g["Da"] * (qstar - f["q"])) / g["scale_ldf"]


def energy_residual(f, g):
    """Eq. (4): accumulation + convection - conduction - heat of adsorption + wall loss."""
    accumulation = (1.0 + _sum(g["gamma"] * f["c"])) * f["T_t"]  # gas + solid heat capacity
    convection = f["v"] * _sum(g["conv_T"] * f["c"]) * f["T_z"]  # heat carried by the flow
    conduction = g["diff_T"] * f["T_zz"]
    adsorption_heat = _sum(g["heat"] * f["q_t"])  # released when q increases (exothermic)
    wall = g["St_w"] * f["T"]  # loss to the wall at Tw = Tin
    return (accumulation + convection - conduction - adsorption_heat + wall) / g["scale_energy"]


def global_residual(f, g):
    """Eq. (5): overall mole balance with C_T = P/(RT); it is what determines the velocity."""
    CT, CT_t, CT_z, CT_zz = total_concentration(f, g)
    flux_change = f["v_z"] * CT + f["v"] * CT_z  # d(v~ C~_T)/dz~
    r = flux_change + g["alpha"] * CT_t + _sum(g["beta"] * f["q_t"]) - CT_zz / g["Pe"]
    return r / g["scale_global"]


def eos_residual(f, g):
    """Closure: the gas concentrations must add up to the ideal-gas total P/(RT)."""
    return _sum(f["c"]) - 1.0 / tf.maximum(1.0 + g["eps_T"] * f["T"], T_MIN_FRAC)


def inlet_residuals(f, g):
    """Danckwerts conditions at z~ = 0 (Eq. 6): dispersive + convective inflow equals the feed."""
    r_c = f["c_z"] / g["Pe"] - (f["c"] - g["y_feed"])
    r_T = f["T_z"] / g["Pe_T"] - _sum(g["cp_ratio"] * f["c"]) * f["T"]
    return r_c, r_T


def outlet_residuals(f, g):
    """Zero gradients at z~ = 1 (Eq. 6)."""
    return f["c_z"], f["T_z"]


def pde_residuals(f, g, p_min):
    """All interior residuals of one batch, as a dict of tensors."""
    return dict(
        mass=mass_residual(f, g),
        ldf=ldf_residual(f, g, qstar_dimless(f, g, p_min)),
        energy=energy_residual(f, g),
        global_=global_residual(f, g),
        eos=eos_residual(f, g),
    )
