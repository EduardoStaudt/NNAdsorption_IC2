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


def load_ic1_module(ic1_root, relpath, name):
    """Load a live IC1 module from its file, or return None if the IC1 folder is missing."""
    path = os.path.join(ic1_root, relpath)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
