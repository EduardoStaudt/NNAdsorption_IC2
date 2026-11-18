"""Verbatim excerpts of IC1 (NNAdsorption_IC1 commit f58c2aa, src/contrato/), kept so the tests
run even without the IC1 folder. Do not edit: the tests compare the IC2 code against these.
When the IC1 folder exists, tests also load the live IC1 files and compare with them.
"""

import importlib.util
import os

import numpy as np

R = 8.314
ATM = 101325.0
TREF = 298.0


# --- solver_jee.py ------------------------------------------------------------------------
def qstar_slope(c, T, K1, K2, K3, K4, K5, K6):
    # LRC: qm=K1+K2*T ; B=K3*exp(K4/T) [1/atm] ; n=K5+K6/T ; p[atm]=c*R*T/ATM
    qm = np.maximum(K1 + K2 * T[None, :], 1e-6)
    B = K3 * np.exp(K4 / T[None, :])
    nn = np.clip(K5 + K6 / T[None, :], 0.3, 3.0)
    p = np.maximum(c * R * T[None, :] / ATM, 0.0)
    pf = np.maximum(p, 1e-6)
    t = B * p**nn
    S = t.sum(axis=0)
    den = 1.0 + S
    qs = qm * t / den
    dtdp = nn * B * pf ** (nn - 1.0)
    s = qm * (den - t) / den**2 * dtdp * (R * T[None, :] / ATM)
    return qs, np.clip(s, 0.0, 1e6)


# --- contract_io.py -----------------------------------------------------------------------
def vetor_X_v3(cfg, NMAX=2):
    out = []
    for L in cfg["layers"]:
        for i in range(NMAX):
            out += [
                L["k1"][i] + L["k2"][i] * TREF,  # qm_ref
                L["k2"][i],
                L["Bref"][i],
                L["k4"][i],
                L["k5"][i] + L["k6"][i] / TREF,  # n_ref
                L["k6"][i],
                L["om"][i],
                L["dH"][i],
                L["Cpg"][i],
            ]
    for L in cfg["layers"]:
        out += [L["eb"], L["rho_b"], L["Cps"]]
    out += [cfg["vs"], cfg["Tin"], cfg["P"], cfg["L"], cfg["hw"], cfg["lam"], cfg["dp"], cfg["Dm"], cfg["Dt"]]
    out += list(cfg["y"][: NMAX - 1])
    return np.array(out, dtype=np.float32)


# --- example_usage.py -------------------------------------------------------------------
def cfg_from_json(e):
    v = lambda k: np.asarray(e[k], np.float64)  # noqa: E731
    layer = dict(
        k1=v("qm_ref") - v("k2") * TREF,
        k2=v("k2"),
        k3=v("B_ref") * np.exp(-v("k4") / TREF),
        k4=v("k4"),
        k5=v("n_ref") - v("k6") / TREF,
        k6=v("k6"),
        Bref=v("B_ref"),
        om=v("kL"),
        dH=v("dH"),
        Cpg=v("Cpg"),
        eb=float(e["eb"]),
        rho_b=float(e["rho_b"]),
        Cps=float(e["Cps"]),
    )
    return dict(
        layers=[layer],
        y=np.array([float(e["y0"]), 1.0 - float(e["y0"])]),
        carrier=0,
        vs=float(e["vs"]),
        Tin=float(e["Tin"]),
        P=float(e["P"]),
        L=float(e["L"]),
        hw=float(e["hw"]),
        lam=float(e["lam"]),
        dp=float(e["dp"]),
        Dm=float(e["Dm"]),
        Dt=float(e["Dt"]),
    )


# --- contract_io.py X_RANGES and recorder.py sample_cfg_22 (recorder imports jax, so it is copied) ---
X_RANGES = {
    "qm_ref": (1.0, 15.0),
    "k2": (-0.1, 0.0),
    "B_ref": (1e-4, 1.0),
    "k4": (0.0, 3200.0),
    "n_ref": (0.5, 1.5),
    "k6": (-2200.0, 2200.0),
    "kL": (0.01, 1.0),
    "dH": (5e3, 50e3),
    "Cpg": (25.0, 45.0),
    "eb": (0.30, 0.50),
    "rho_b": (400.0, 900.0),
    "Cps": (800.0, 1200.0),
    "vs": (0.001, 0.05),
    "Tin": (288.0, 323.0),
    "P": (1e5, 3e6),
    "L": (0.3, 1.5),
    "hw": (20.0, 100.0),
    "lam": (0.1, 0.8),
    "dp": (0.5e-3, 5e-3),
    "Dm": (5e-6, 3e-5),
    "Dt": (0.01, 0.06),
}
LOG_PARAMS = {"B_ref", "kL", "dp", "Dm"}


def sample_cfg_22(seed, Y0_LO, Y0_HI, NMAX=2):
    rng = np.random.default_rng(seed)
    XR = X_RANGES

    def draw(name, n=1):
        lo, hi = XR[name]
        if name in LOG_PARAMS:
            v = 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
        else:
            v = rng.uniform(lo, hi, n)
        return v

    qm_ref = draw("qm_ref", NMAX)
    k2 = draw("k2", NMAX)
    B_ref = draw("B_ref", NMAX)
    k4 = draw("k4", NMAX)
    n_ref = draw("n_ref", NMAX)
    k6 = draw("k6", NMAX)
    kL = draw("kL", NMAX)
    dH = draw("dH", NMAX)
    Cpg = draw("Cpg", NMAX)
    k1 = qm_ref - k2 * TREF
    k3 = B_ref * np.exp(-k4 / TREF)
    k5 = n_ref - k6 / TREF
    order = np.argsort(B_ref)

    def ro(a):
        return a[order]

    qm_ref, k2, B_ref, k4, n_ref, k6, kL, dH, k1, k3, k5, Cpg = (
        ro(qm_ref), ro(k2), ro(B_ref), ro(k4), ro(n_ref), ro(k6), ro(kL), ro(dH), ro(k1), ro(k3), ro(k5), ro(Cpg),
    )
    layer = dict(
        k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, Bref=B_ref, om=kL, dH=dH, Cpg=Cpg,
        eb=float(draw("eb")[0]), rho_b=float(draw("rho_b")[0]), Cps=float(draw("Cps")[0]),
    )
    yc = float(rng.uniform(Y0_LO, Y0_HI))
    y = np.array([yc, 1.0 - yc])
    return dict(
        layers=[layer], y=y, carrier=0,
        vs=float(draw("vs")[0]), Tin=float(draw("Tin")[0]), P=float(draw("P")[0]), L=float(draw("L")[0]),
        hw=float(draw("hw")[0]), lam=float(draw("lam")[0]), dp=float(draw("dp")[0]), Dm=float(draw("Dm")[0]),
        Dt=float(draw("Dt")[0]),
    )


def load_ic1_module(ic1_root, relpath, name):
    """Load a live IC1 module from its file, or return None if the IC1 folder is missing."""
    path = os.path.join(ic1_root, relpath)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
