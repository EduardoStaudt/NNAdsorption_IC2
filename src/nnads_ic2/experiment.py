"""Building blocks shared by the scripts: cases -> scales -> case table -> model."""

import os

import numpy as np

from .config import resolve_path
from .models.pinn import PINN
from .physics import mol_solver
from .physics.params import CaseParams, ThetaScaler, theta_dim, theta_log_mask
from .physics.residuals import case_arrays, stack_cases
from .physics.scales import compute_scales, dT_ratio_from_data, dimensionless_groups


def fit_theta_scaler(cases, cfg):
    """log + z-score normalization of theta, fitted on these cases."""
    mask = theta_log_mask(cases[0].n_comp, cfg["theta"]["log_params"])
    return ThetaScaler.fit(np.stack([p.to_theta() for p in cases]), mask, float(cfg["theta"]["sd_floor"]))


def case_rows(cases, cfg, scaler):
    """Per-case constant arrays (residual groups + normalized theta), scales and groups."""
    scales = [compute_scales(p, **cfg["scales"]) for p in cases]
    groups = [dimensionless_groups(p, s) for p, s in zip(cases, scales)]
    Theta_hat = scaler.transform(np.stack([p.to_theta() for p in cases]))
    rows = []
    for p, s, g, th in zip(cases, scales, groups, Theta_hat):
        a = case_arrays(p, s, g)
        a["theta"] = th
        rows.append(a)
    return rows, scales, groups


def case_table(cases, cfg, dtype, scaler=None):
    """Scales, groups and the stacked constant table of a list of CaseParams.

    theta is normalized with `scaler` (fitted on these cases when None).
    """
    if scaler is None:
        scaler = fit_theta_scaler(cases, cfg)
    rows, scales, groups = case_rows(cases, cfg, scaler)
    return stack_cases(rows, dtype), scales, groups, scaler


def resolve_dT_ratio(cfg, cases, rises):
    """Set cfg["scales"]["dT_ratio"] from the measured temperature rises of the reference data.

    A number already in the config is kept. Without data the feed estimate is used (ratio 1)
    and a warning goes to the log. The resolved value is saved with the run's config.yaml, so a
    re-evaluation rebuilds exactly the same scales.
    """
    sc = cfg["scales"]
    if sc.get("dT_ratio") is not None:
        print("dT_ref: dT_ratio = %.4g (fixed in the config)" % float(sc["dT_ratio"]), flush=True)
        return float(sc["dT_ratio"])
    ratio = dT_ratio_from_data(cases, rises, float(sc["q_ref_floor"]))
    if ratio is None:
        print("WARNING: no reference data for dT_ref -> fallback to the feed estimate "
              "sum_i y_i,F (-dH_i) q_ref,i / Cps, which can overestimate the temperature rise several times", flush=True)
        ratio = 1.0
    else:
        n = sum(r is not None for r in rises)
        print("dT_ref: measured rise / feed estimate = %.4g (%s of %d case(s))" % (ratio, "median" if n > 1 else "value", n), flush=True)
    sc["dT_ratio"] = float(ratio)
    return float(ratio)


def make_model(cfg, n_comp):
    t_final = float(cfg["scales"]["t_final_factor"])  # t~_F = t_F / t_ref, the same for every case
    return PINN(n_comp, theta_dim(n_comp), cfg["model"], t_final=t_final).build_with_dummy()


def single_case(cfg, root):
    return CaseParams.from_ic1_json(resolve_path(cfg["case"]["path"], root), carrier=int(cfg["case"]["carrier"]))


def reference_solution(p, s, cfg, case_name, root):
    """MOL solution of the case, cached in runs/cache/ (the solver takes tens of seconds)."""
    sc = cfg["solver"]
    cache_dir = os.path.join(root, cfg["logging"]["runs_dir"], "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "mol_%s_nx%d_nt%d.npz" % (case_name, sc["nx"], sc["n_out"]))
    if os.path.exists(path):
        print("reference: cached MOL solution %s" % path, flush=True)
        return mol_solver.load_solution(path)
    print("reference: running the MOL solver (nx=%d) ..." % sc["nx"], flush=True)
    sol = mol_solver.solve_case(p, s.t_final, sc)
    mol_solver.save_solution(sol, path)
    print("reference: done in %.1f s -> %s" % (sol["info"]["wall_s"], path), flush=True)
    return sol


def outlet_data(refs, scales, case_indices, n_per_case, rng):
    """Outlet curves of the solver as data points (z~ = 1, t~, y_i, T~), `n_per_case` per case."""
    z, t, case, y, T = [], [], [], [], []
    for ref, s, k in zip(refs, scales, case_indices):
        idx = np.arange(ref["t"].size) if n_per_case >= ref["t"].size else np.sort(rng.choice(ref["t"].size, n_per_case, replace=False))
        t.append(ref["t"][idx] / s.t_ref)
        z.append(np.ones(idx.size))
        case.append(np.full(idx.size, k, dtype=np.int64))
        y.append(ref["y_out"][idx])
        T.append((ref["T_out"][idx] - s.Tin) / s.dT_ref)
    return dict(z=np.concatenate(z), t=np.concatenate(t), case=np.concatenate(case), y=np.concatenate(y), T=np.concatenate(T))


def profile_data(profiles, scales, case_indices):
    """Internal solver fields as data points (z~, t~, c~, q~, T~, v~)."""
    out = {k: [] for k in ("z", "t", "case", "c", "q", "T", "v")}
    for pr, s, k in zip(profiles, scales, case_indices):
        if pr is None:
            continue
        missing = [name for name in ("c", "q", "T", "v") if pr.get(name) is None]
        if missing:
            raise ValueError("profile data loss needs c, q, T and v; the dataset lacks %s" % missing)
        TT, ZZ = np.meshgrid(pr["t"] / s.t_ref, pr["z"] / s.L, indexing="ij")
        n = pr["c"].shape[1]
        out["z"].append(ZZ.ravel())
        out["t"].append(TT.ravel())
        out["case"].append(np.full(ZZ.size, k, dtype=np.int64))
        out["c"].append(np.moveaxis(pr["c"], 1, 2).reshape(-1, n) / s.C_F)
        out["q"].append(np.moveaxis(pr["q"], 1, 2).reshape(-1, n) / s.q_ref)
        out["T"].append(((pr["T"] - s.Tin) / s.dT_ref).ravel())
        out["v"].append((pr["v"] / s.vs).ravel())
    if not out["z"]:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}
