"""Random cases over the parameter domain (Section 2.3) and the prefilter.

`sample_case_ic1` reproduces IC1 recorder.sample_cfg_22 draw by draw, so the same seed gives
the same case as in the IC1 datasets (n_comp = 2). Ranges and limits come from the `cases:`
section of the config (IC1 contract_io.X_RANGES, max t_st 1800 s, severity < 0.9), to be
confirmed with the advisor for v23b.
"""

import numpy as np

from ..physics.constants import TREF
from ..physics.params import CaseParams, one_hot
from ..physics.scales import stoich_times

COMP_DRAWS = ["qm_ref", "k2", "B_ref", "k4", "n_ref", "k6", "kL", "dH", "Cpg"]


def sample_case_ic1(seed, x_ranges, y0_range, log_params, n_comp=2, carrier=0):
    """One random case, identical to IC1 recorder.sample_cfg_22(seed) for n_comp = 2.

    Components are sorted by B_ref, so component 0 (the carrier that fills the bed) is the
    one with the lowest affinity, as in IC1.
    """
    if n_comp != 2:
        raise NotImplementedError("feed composition sampling for n_comp > 2 is not defined yet (Section 2.3)")
    rng = np.random.default_rng(seed)
    log_params = set(log_params)

    def draw(name, n=1):
        lo, hi = x_ranges[name]
        if name in log_params:
            return 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
        return rng.uniform(lo, hi, n)

    v = {name: draw(name, n_comp) for name in COMP_DRAWS}  # same draw order as IC1
    order = np.argsort(v["B_ref"])
    v = {k: a[order] for k, a in v.items()}
    eb, rho_b, Cps = (float(draw(k)[0]) for k in ("eb", "rho_b", "Cps"))
    yc = float(rng.uniform(y0_range[0], y0_range[1]))
    g = {k: float(draw(k)[0]) for k in ("vs", "Tin", "P", "L", "hw", "lam", "dp", "Dm", "Dt")}
    return CaseParams(
        k1=v["qm_ref"] - v["k2"] * TREF,
        k2=v["k2"],
        k3=v["B_ref"] * np.exp(-v["k4"] / TREF),
        k4=v["k4"],
        k5=v["n_ref"] - v["k6"] / TREF,
        k6=v["k6"],
        kL=v["kL"],
        dH=v["dH"],
        Cpg=v["Cpg"],
        eb=eb,
        rho_b=rho_b,
        Cps=Cps,
        y_feed=np.array([yc, 1.0 - yc]),
        y_init=one_hot(carrier, n_comp),
        **g,
    )


# IC1 severity index (features.severidade_regime). Its formula and weights are part of the IC1
# input contract (the application domain is s < 0.9), so they are fixed here, not tunable.
_SEV_TERMS = [  # (weight, low, high, log scale)
    (0.30, 1e-4, 1.0, True),  # B_ref of the strong component
    (0.15, 0.01, 1.0, True),  # kL of the strong component
    (0.15, 1e5, 3e6, False),  # pressure
    (0.15, 1.0, 15.0, False),  # qm_ref of the carrier
    (0.15, -2200.0, 2200.0, False),  # -k6 of the carrier
    (0.10, 0.0, 5.0, False),  # NTU of the strong component, kL L / vs
]


def _norm01(v, lo, hi, log):
    if log:
        v, lo, hi = np.log(max(v, 1e-30)), np.log(max(lo, 1e-30)), np.log(max(hi, 1e-30))
    return float(np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0))


def severity_ic1(p):
    """IC1 severity of a two-component case (0 = mild, 1 = hardest corner of the domain)."""
    if p.n_comp != 2:
        raise NotImplementedError("the IC1 severity index is defined for two components only")
    values = [p.B_ref[1], p.kL[1], p.P, p.qm_ref[0], -p.k6[0], p.kL[1] * p.L / max(p.vs, 1e-12)]
    return sum(w * _norm01(x, lo, hi, lg) for (w, lo, hi, lg), x in zip(_SEV_TERMS, values))


def prefilter(p, max_tst_s, sev_max):
    """Keep a case only if its slowest front is fast enough and it lies inside the IC1 domain."""
    tst_max = float(np.nanmax(stoich_times(p)))
    sev = severity_ic1(p) if p.n_comp == 2 else float("nan")
    ok = tst_max <= max_tst_s and (np.isnan(sev) or sev < sev_max)
    return ok, dict(tst_max_s=tst_max, severity=sev)


def sample_cases(seeds, cases_cfg, n_comp=2):
    """Sample the given seeds and keep the ones that pass the prefilter: [(seed, CaseParams)]."""
    out = []
    for seed in seeds:
        p = sample_case_ic1(seed, cases_cfg["x_ranges"], cases_cfg["y0_range"], cases_cfg["log_params"], n_comp)
        ok, _ = prefilter(p, float(cases_cfg["max_tst_s"]), float(cases_cfg["sev_max"]))
        if ok:
            out.append((int(seed), p))
    return out
