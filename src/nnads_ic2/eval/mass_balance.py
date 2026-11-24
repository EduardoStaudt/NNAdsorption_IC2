"""Global mole balance check (docs/nondim.md, Section 6).

For each gas i, from t = 0 until the bed is back at the feed state:
    integral [1 - v(L,t) c_i(L,t) / (vs c_i,F)] dt = t_st,i - t_init,i
t_init,i = 0 for a gas absent from the bed at t = 0 (the identity of Section 2.4). The identity
holds whatever the dispersion, kinetics or velocity variation, so it tests whether the predicted
outlet velocity and concentrations conserve moles. The MLP has no velocity output and cannot be
checked this way.
"""

import numpy as np

from ..physics.scales import feed_and_initial_state, initial_holdup_times, stoich_times


def outlet_moment(t, v_out, c_out_i, vs, c_feed_i):
    """Trapezoidal integral of 1 - v c_i / (vs c_i,F) over the time grid [s]."""
    return float(np.trapezoid(1.0 - v_out * c_out_i / (vs * c_feed_i), t))


def mole_balance(p, t, v_out, c_out):
    """Outlet first moment of each fed gas and the exact value it must reach at long times."""
    c_feed = feed_and_initial_state(p)["c_feed"]
    theory = stoich_times(p) - initial_holdup_times(p)
    rows = []
    for i in range(p.n_comp):
        if c_feed[i] <= 0:
            continue
        rows.append(dict(component=i, moment_s=outlet_moment(t, v_out, c_out[:, i], p.vs, c_feed[i]),
                         theory_s=float(theory[i])))
    return rows


def compare_mole_balance(p, t_ref, t, pred, ref):
    """PINN and reference moments on the same horizon, and both against theory.

    Errors are given in % of t_ref (the moment of the initial gas is negative, so a relative
    error to its own value would be misleading).
    """
    mp = mole_balance(p, t, pred["v_out"], pred["c_out"])
    mr = mole_balance(p, t, ref["v_out"], ref["c_out"])
    out = []
    for a, b in zip(mp, mr):
        out.append(dict(
            component=a["component"],
            pinn_moment_s=a["moment_s"],
            ref_moment_s=b["moment_s"],
            theory_s=a["theory_s"],
            pinn_vs_ref_pct_tref=100.0 * (a["moment_s"] - b["moment_s"]) / t_ref,
            pinn_vs_theory_pct_tref=100.0 * (a["moment_s"] - a["theory_s"]) / t_ref,
            ref_vs_theory_pct_tref=100.0 * (b["moment_s"] - b["theory_s"]) / t_ref,
        ))
    return out
