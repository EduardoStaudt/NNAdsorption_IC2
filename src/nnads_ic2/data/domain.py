"""Parameter domain of the parametric PINN: the whole IC1 domain or a subspace of it.

mode 'full':     every theta column varies (cases.x_ranges, y0 over cases.y0_range). As in the
                 IC1 sampler, the components are sorted by B_ref, so component 0 (the carrier
                 that fills the bed at t = 0) is the least adsorbed.
mode 'subspace': theta = the base case (IC1 example_usage), except the columns listed in `vary`
                 (Phase 2 dimension curriculum: a few parameters first, then more).
Column names are those of physics/params.theta_names: 'L0_c1_kL' is kL of component 1 (the
strong gas), 'y0' the carrier feed fraction (the strong gas gets 1 - y0), 'vs', 'Tin', ...
A column varies log-uniformly when its parameter is in cases.log_params, uniformly otherwise.

Two ways to draw cases, both keeping only cases that pass the prefilter (slowest t_st <=
max_tst_s, IC1 severity < sev_max, carrier = least adsorbed component):
  Domain.sample(seed)  one case per integer seed (validation sets; 'full' = IC1 recorder seeds)
  ThetaSampler         scrambled Sobol points (evenly spread) for the collocation theta; the
                       residuals need no solver, so these cases cost nothing to add
"""

import numpy as np
from scipy.stats import qmc

from ..config import resolve_path
from ..physics.params import COMP_PARAM_NAMES, CaseParams, param_key, theta_dim, theta_names
from .cases import prefilter, sample_case_ic1


def sort_components_by_affinity(theta, n_comp):
    """Reorder the per-component blocks of theta by B_ref (IC1: component 0 = least adsorbed)."""
    theta = np.array(theta, dtype=np.float64)
    npc = len(COMP_PARAM_NAMES)
    block = theta[: n_comp * npc].reshape(n_comp, npc)
    theta[: n_comp * npc] = block[np.argsort(block[:, COMP_PARAM_NAMES.index("B_ref")])].ravel()
    return theta


class Domain:
    def __init__(self, cfg, root):
        self.cases_cfg = cfg["cases"]
        dc = cfg["domain"]
        self.mode = dc["mode"]
        self.n_comp = int(cfg["physics"]["n_comp"])
        self.carrier = int(cfg["data"]["carrier"])
        self.names = theta_names(self.n_comp)
        self.base_path = None
        if self.mode == "full":
            if self.n_comp != 2:
                raise NotImplementedError("feed composition sampling for n_comp > 2 is not defined yet (Section 2.3)")
            self.base = None
            self.vary = list(self.names)
        elif self.mode == "subspace":
            self.base_path = dc["base_case"]
            self.base = CaseParams.from_ic1_json(resolve_path(dc["base_case"], root), carrier=self.carrier)
            self.vary = list(dc["vary"])
            unknown = [v for v in self.vary if v not in self.names]
            if unknown or not self.vary:
                raise ValueError("domain.vary: unknown or no theta columns %s (names: %s)" % (unknown, self.names))
        else:
            raise ValueError("domain.mode must be 'full' or 'subspace'")
        self.cols = [self.names.index(v) for v in self.vary]
        self.ranges = [self._range(v) for v in self.vary]  # (low, high, log)

    def _range(self, name):
        key = param_key(name)
        if key[0] == "y" and key[1:].isdigit():
            lo, hi = self.cases_cfg["y0_range"]
            return float(lo), float(hi), False
        lo, hi = self.cases_cfg["x_ranges"][key]
        return float(lo), float(hi), key in set(self.cases_cfg["log_params"])

    @property
    def dim(self):
        """Number of varying columns."""
        return len(self.vary)

    def values_from_unit(self, u):
        """Point(s) u of the unit cube [..., dim] -> values of the varying columns."""
        u = np.asarray(u, dtype=np.float64)
        out = np.empty_like(u)
        for j, (lo, hi, log) in enumerate(self.ranges):
            if log:
                out[..., j] = 10 ** (np.log10(lo) + u[..., j] * (np.log10(hi) - np.log10(lo)))
            else:
                out[..., j] = lo + u[..., j] * (hi - lo)
        return out

    def case_from_unit(self, u):
        theta = self.base.to_theta() if self.mode == "subspace" else np.zeros(theta_dim(self.n_comp))
        theta[self.cols] = self.values_from_unit(u)
        if self.mode == "full":
            theta = sort_components_by_affinity(theta, self.n_comp)
        return CaseParams.from_theta(theta, self.n_comp, carrier=self.carrier)

    def accept(self, p):
        """Prefilter (t_st, severity) plus the IC1 order of the components. Returns (ok, info)."""
        ok, info = prefilter(p, float(self.cases_cfg["max_tst_s"]), float(self.cases_cfg["sev_max"]))
        ordered = bool(np.all(np.diff(p.B_ref) >= 0))
        info["carrier_least_adsorbed"] = ordered
        return ok and ordered, info

    def sample(self, seed):
        """(CaseParams, ok, prefilter info) of one integer seed. 'full': the IC1 recorder case."""
        if self.mode == "full":
            cc = self.cases_cfg
            p = sample_case_ic1(seed, cc["x_ranges"], cc["y0_range"], cc["log_params"], self.n_comp, self.carrier)
        else:
            p = self.case_from_unit(np.random.default_rng(seed).random(self.dim))
        ok, info = self.accept(p)
        return p, ok, info

    def snap(self, p):
        """Case with the fixed columns set back to the base case (subspace mode; else unchanged).

        Datasets store theta in float32 (IC1 recorder), so a case read from a file differs from
        the base case by round-off in the columns that should be identical.
        """
        if self.mode != "subspace":
            return p
        theta = p.to_theta()
        fixed = [j for j in range(theta.size) if j not in set(self.cols)]
        theta[fixed] = self.base.to_theta()[fixed]
        return CaseParams.from_theta(theta, self.n_comp, carrier=self.carrier)

    def describe(self):
        return dict(mode=self.mode, base_case=self.base_path, vary=self.vary,
                    ranges={v: list(r) for v, r in zip(self.vary, self.ranges)})


class ThetaSampler:
    """Collocation theta: scrambled Sobol points over the varying columns, prefiltered.

    Successive draws continue the same Sobol sequence, so each new draw fills the gaps left by
    the previous ones. Rejected points (prefilter) are skipped.
    """

    def __init__(self, domain, seed, chunk=1024):
        self.domain = domain
        self.engine = qmc.Sobol(d=domain.dim, scramble=True, seed=seed)
        self.chunk = int(chunk)  # power of 2 keeps the Sobol balance
        self.buffer = []
        self.n_drawn = 0
        self.n_rejected = 0

    def draw(self, n):
        while len(self.buffer) < n:
            for u in self.engine.random(self.chunk):
                p = self.domain.case_from_unit(u)
                self.n_drawn += 1
                if self.domain.accept(p)[0]:
                    self.buffer.append(p)
                else:
                    self.n_rejected += 1
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out
