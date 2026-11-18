"""Physical parameters of one case and the theta vector (network input).

theta follows the IC1 contract (contract_io.vetor_X_v3), generalised to n_comp components:
    per component i (9 values): qm_ref, k2, B_ref, k4, n_ref, k6, kL, dH, Cpg
    adsorbent layer (3):        eb, rho_b, Cps
    global (9):                 vs, Tin, P, L, hw, lam, dp, Dm, Dt
    feed (n_comp - 1):          y_0 ... y_{n-2}  (the last fraction is 1 - sum of the others)
For n_comp = 2 this is exactly the 31-column IC1 vector (Dt at index 29, y0 at index 30).

Units (SI, as in IC1): P [Pa], kL [1/s], dH = (-DeltaH) > 0 [J/mol], Cpg [J/(mol K)],
qm [mol/kg], B [1/atm^n], lengths [m], vs [m/s], T [K].

The solver uses the isotherm coefficients k1..k6 (isotherm.py); theta stores their values
at TREF instead, like IC1: qm_ref = k1 + k2 TREF, B_ref = k3 exp(k4/TREF), n_ref = k5 + k6/TREF.

Only one adsorbent layer is implemented. Two layers (XPINN, one sub-network per layer) will
need per-layer isotherm/property blocks plus the layer split; see `theta_names(n_layer=...)`.
"""

import json
from dataclasses import dataclass

import numpy as np

from .constants import TREF

COMP_PARAM_NAMES = ["qm_ref", "k2", "B_ref", "k4", "n_ref", "k6", "kL", "dH", "Cpg"]
LAYER_PARAM_NAMES = ["eb", "rho_b", "Cps"]
GLOBAL_PARAM_NAMES = ["vs", "Tin", "P", "L", "hw", "lam", "dp", "Dm", "Dt"]

LOG_FLOOR = 1e-30  # same guard as IC1 before taking logs (np.log(np.maximum(x, 1e-30)))


@dataclass
class CaseParams:
    """All physical inputs of one simulation (one adsorbent layer, n_comp gases)."""

    k1: np.ndarray  # LRC coefficients per component (solver form, see isotherm.py)
    k2: np.ndarray
    k3: np.ndarray
    k4: np.ndarray
    k5: np.ndarray
    k6: np.ndarray
    kL: np.ndarray  # LDF mass-transfer coefficient [1/s] ("om" in IC1)
    dH: np.ndarray  # heat of adsorption (-DeltaH) > 0 [J/mol]
    Cpg: np.ndarray  # molar heat capacity of each gas [J/(mol K)]
    eb: float  # bed porosity [-]
    rho_b: float  # bed bulk density [kg/m3]
    Cps: float  # solid heat capacity [J/(kg K)]
    vs: float  # feed superficial velocity [m/s]
    Tin: float  # feed (and wall) temperature [K]
    P: float  # column pressure, uniform and constant [Pa]
    L: float  # bed length [m]
    hw: float  # wall heat transfer coefficient [W/(m2 K)]
    lam: float  # effective axial thermal conductivity [W/(m K)]
    dp: float  # particle diameter [m]
    Dm: float  # molecular diffusivity [m2/s]
    Dt: float  # column internal diameter [m]
    y_feed: np.ndarray  # feed mole fractions [n]
    y_init: np.ndarray  # mole fractions of the gas that fills the bed at t = 0 [n]

    def __post_init__(self):
        for name in ["k1", "k2", "k3", "k4", "k5", "k6", "kL", "dH", "Cpg", "y_feed", "y_init"]:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64).copy())
        for name in ["eb", "rho_b", "Cps", "vs", "Tin", "P", "L", "hw", "lam", "dp", "Dm", "Dt"]:
            setattr(self, name, float(getattr(self, name)))
        n = self.k1.shape[0]
        for name in ["k2", "k3", "k4", "k5", "k6", "kL", "dH", "Cpg", "y_feed", "y_init"]:
            if getattr(self, name).shape != (n,):
                raise ValueError("%s must have %d values (one per component)" % (name, n))
        for name in ["y_feed", "y_init"]:
            if abs(getattr(self, name).sum() - 1.0) > 1e-9:
                raise ValueError("%s must sum to 1, got %r" % (name, getattr(self, name)))

    # ---- derived quantities -------------------------------------------------------
    @property
    def n_comp(self):
        return self.k1.shape[0]

    @property
    def qm_ref(self):
        return self.k1 + self.k2 * TREF

    @property
    def B_ref(self):
        return self.k3 * np.exp(self.k4 / TREF)

    @property
    def n_ref(self):
        return self.k5 + self.k6 / TREF

    @property
    def DL(self):
        """Axial dispersion, Ruthven correlation (Eq. 2), evaluated at the feed velocity."""
        return 0.7 * self.Dm + 0.5 * (self.vs / self.eb) * self.dp

    def isotherm_coeffs(self):
        return dict(k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4, k5=self.k5, k6=self.k6)

    # ---- theta vector -------------------------------------------------------------
    def to_theta(self):
        """theta vector in the IC1 order (float64). For n_comp = 2 it equals vetor_X_v3."""
        out = []
        for i in range(self.n_comp):
            out += [
                self.qm_ref[i],
                self.k2[i],
                self.B_ref[i],
                self.k4[i],
                self.n_ref[i],
                self.k6[i],
                self.kL[i],
                self.dH[i],
                self.Cpg[i],
            ]
        out += [self.eb, self.rho_b, self.Cps]
        out += [self.vs, self.Tin, self.P, self.L, self.hw, self.lam, self.dp, self.Dm, self.Dt]
        out += list(self.y_feed[:-1])
        return np.asarray(out, dtype=np.float64)

    @classmethod
    def from_theta(cls, theta, n_comp, carrier=0):
        """Inverse of `to_theta`. The bed starts filled with pure `carrier` (IC1 convention)."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (theta_dim(n_comp),):
            raise ValueError("theta must have %d values for n_comp=%d" % (theta_dim(n_comp), n_comp))
        npc = len(COMP_PARAM_NAMES)
        block = theta[: n_comp * npc].reshape(n_comp, npc)
        col = {name: block[:, j] for j, name in enumerate(COMP_PARAM_NAMES)}
        k = n_comp * npc
        eb, rho_b, Cps = theta[k : k + 3]
        g = dict(zip(GLOBAL_PARAM_NAMES, theta[k + 3 : k + 3 + len(GLOBAL_PARAM_NAMES)]))
        y_head = theta[k + 3 + len(GLOBAL_PARAM_NAMES) :]
        y_feed = np.append(y_head, 1.0 - y_head.sum())
        return cls(
            k1=col["qm_ref"] - col["k2"] * TREF,
            k2=col["k2"],
            k3=col["B_ref"] * np.exp(-col["k4"] / TREF),
            k4=col["k4"],
            k5=col["n_ref"] - col["k6"] / TREF,
            k6=col["k6"],
            kL=col["kL"],
            dH=col["dH"],
            Cpg=col["Cpg"],
            eb=eb,
            rho_b=rho_b,
            Cps=Cps,
            y_feed=y_feed,
            y_init=one_hot(carrier, n_comp),
            **g,
        )

    # ---- IC1 formats ----------------------------------------------------------------
    @classmethod
    def from_ic1_cfg(cls, cfg):
        """From an IC1 solver cfg (recorder.sample_cfg_22 / example_usage.cfg_from_json)."""
        if len(cfg["layers"]) != 1:
            raise NotImplementedError("only one adsorbent layer is implemented")
        lay = cfg["layers"][0]
        n = len(lay["k1"])
        return cls(
            k1=lay["k1"],
            k2=lay["k2"],
            k3=lay["k3"],
            k4=lay["k4"],
            k5=lay["k5"],
            k6=lay["k6"],
            kL=lay["om"],
            dH=lay["dH"],
            Cpg=lay["Cpg"],
            eb=lay["eb"],
            rho_b=lay["rho_b"],
            Cps=lay["Cps"],
            vs=cfg["vs"],
            Tin=cfg["Tin"],
            P=cfg["P"],
            L=cfg["L"],
            hw=cfg["hw"],
            lam=cfg["lam"],
            dp=cfg["dp"],
            Dm=cfg["Dm"],
            Dt=cfg["Dt"],
            y_feed=cfg["y"],
            y_init=one_hot(int(cfg.get("carrier", 0)), n),
        )

    def to_ic1_cfg(self, carrier=0):
        """IC1 solver cfg (lets tests call the IC1 code on the same case)."""
        layer = dict(
            k1=self.k1.copy(),
            k2=self.k2.copy(),
            k3=self.k3.copy(),
            k4=self.k4.copy(),
            k5=self.k5.copy(),
            k6=self.k6.copy(),
            Bref=self.B_ref,
            om=self.kL.copy(),
            dH=self.dH.copy(),
            Cpg=self.Cpg.copy(),
            eb=self.eb,
            rho_b=self.rho_b,
            Cps=self.Cps,
        )
        return dict(
            layers=[layer],
            y=self.y_feed.copy(),
            carrier=carrier,
            vs=self.vs,
            Tin=self.Tin,
            P=self.P,
            L=self.L,
            hw=self.hw,
            lam=self.lam,
            dp=self.dp,
            Dm=self.Dm,
            Dt=self.Dt,
        )

    @classmethod
    def from_ic1_json(cls, source, carrier=0):
        """From the IC1 example_input.json format (values at TREF, like the theta vector).

        `source` is a path or an already loaded dict. Two components use the key `y0`;
        a general mixture may give the full list under `y`.
        """
        if isinstance(source, str):
            with open(source, encoding="utf-8") as f:
                e = json.load(f)
        else:
            e = source
        v = {k: np.asarray(e[k], dtype=np.float64) for k in COMP_PARAM_NAMES}
        n = v["qm_ref"].shape[0]
        y = np.asarray(e["y"], dtype=np.float64) if "y" in e else np.array([e["y0"], 1.0 - e["y0"]])
        return cls(
            k1=v["qm_ref"] - v["k2"] * TREF,
            k2=v["k2"],
            k3=v["B_ref"] * np.exp(-v["k4"] / TREF),
            k4=v["k4"],
            k5=v["n_ref"] - v["k6"] / TREF,
            k6=v["k6"],
            kL=v["kL"],
            dH=v["dH"],
            Cpg=v["Cpg"],
            eb=e["eb"],
            rho_b=e["rho_b"],
            Cps=e["Cps"],
            vs=e["vs"],
            Tin=e["Tin"],
            P=e["P"],
            L=e["L"],
            hw=e["hw"],
            lam=e["lam"],
            dp=e["dp"],
            Dm=e["Dm"],
            Dt=e["Dt"],
            y_feed=y,
            y_init=one_hot(carrier, n),
        )


def one_hot(index, n):
    out = np.zeros(n)
    out[index] = 1.0
    return out


def theta_dim(n_comp):
    return n_comp * len(COMP_PARAM_NAMES) + len(LAYER_PARAM_NAMES) + len(GLOBAL_PARAM_NAMES) + n_comp - 1


def theta_names(n_comp, n_layer=1):
    """Column names in IC1 style ('L0_c1_B_ref', 'L0_eb', 'Dt', 'y0', ...)."""
    if n_layer != 1:
        raise NotImplementedError("two adsorbent layers (XPINN) are a later step")
    names = ["L0_c%d_%s" % (i, p) for i in range(n_comp) for p in COMP_PARAM_NAMES]
    names += ["L0_%s" % p for p in LAYER_PARAM_NAMES]
    names += list(GLOBAL_PARAM_NAMES)
    names += ["y%d" % i for i in range(n_comp - 1)]
    return names


def param_key(column_name):
    """'L0_c1_B_ref' -> 'B_ref', 'L0_eb' -> 'eb', 'Dt' -> 'Dt', 'y0' -> 'y0'."""
    if column_name.startswith("L0_c"):
        return column_name.split("_", 2)[2]
    if column_name.startswith("L0_"):
        return column_name[3:]
    return column_name


def theta_log_mask(n_comp, log_params):
    """True for the columns that get a log transform (parameters with multiplicative range)."""
    return np.array([param_key(nm) in set(log_params) for nm in theta_names(n_comp)], dtype=bool)


@dataclass
class ThetaScaler:
    """log on the multiplicative parameters, then z-score (same recipe as IC1 fit_scaler)."""

    log_mask: np.ndarray
    mu: np.ndarray
    sd: np.ndarray

    @staticmethod
    def _log(Theta, log_mask):
        X = np.array(Theta, dtype=np.float64, copy=True)
        X[..., log_mask] = np.log(np.maximum(X[..., log_mask], LOG_FLOOR))
        return X

    @classmethod
    def fit(cls, Theta, log_mask, sd_floor):
        """Theta: [N, d]. Columns with sd < sd_floor (e.g. a single case) get sd = 1."""
        X = cls._log(np.atleast_2d(Theta), log_mask)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < sd_floor] = 1.0
        return cls(log_mask=np.asarray(log_mask, bool), mu=mu, sd=sd)

    def transform(self, Theta):
        return (self._log(Theta, self.log_mask) - self.mu) / self.sd

    def to_dict(self):
        return dict(log_mask=self.log_mask.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(log_mask=np.asarray(d["log_mask"], bool), mu=np.asarray(d["mu"]), sd=np.asarray(d["sd"]))
