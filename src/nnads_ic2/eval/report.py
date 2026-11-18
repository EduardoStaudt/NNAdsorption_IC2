"""Evaluation against reference solutions (MOL solver now, v23b dataset later).

A reference is a dict with t, c_out, y_out, T_out, optionally v_out (IC1 recorder files do not
store it) and, when available, the internal fields z, c, q, T, v_cell (physics/mol_solver.py).
"""

import csv
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..physics.scales import feed_and_initial_state
from . import plots
from .mass_balance import compare_mole_balance, mole_balance
from .metrics import characteristic_times, front_components, regression_metrics
from .predict import inference_ms, predict_dimless, predict_outlet


def _reference_fields_dimless(ref, s):
    return dict(
        c=ref["c"] / s.C_F,  # [nt, n, nx]
        q=ref["q"] / s.q_ref[None, :, None],
        T=(ref["T"] - s.Tin) / s.dT_ref,  # [nt, nx]
        v=ref["v_cell"] / s.vs,
    )


def field_errors(model, table, case_index, p, s, ref, ev, out_dir):
    """RMSE of each dimensionless field over a regular (z~, t~) grid, plus the error maps."""
    nz, nt = int(ev["field_grid"]["nz"]), int(ev["field_grid"]["nt"])
    z_cells = ref["z"] / s.L
    t_grid = ref["t"] / s.t_ref
    zg = np.linspace(z_cells[0], z_cells[-1], nz)
    tg = np.linspace(t_grid[0], t_grid[-1], nt)
    TT, ZZ = np.meshgrid(tg, zg, indexing="ij")
    pred = predict_dimless(model, table, case_index, ZZ.ravel(), TT.ravel(), int(ev["batch_size"]))
    rf = _reference_fields_dimless(ref, s)
    pts = np.stack([TT.ravel(), ZZ.ravel()], axis=1)

    def interp(F):
        return RegularGridInterpolator((t_grid, z_cells), F)(pts).reshape(nt, nz)

    ref_maps, pred_maps = {}, {}
    for i in range(p.n_comp):
        ref_maps["c%d" % i] = interp(rf["c"][:, i, :])
        pred_maps["c%d" % i] = pred["c"][:, i].reshape(nt, nz)
        ref_maps["q%d" % i] = interp(rf["q"][:, i, :])
        pred_maps["q%d" % i] = pred["q"][:, i].reshape(nt, nz)
    ref_maps["T"] = interp(rf["T"])
    pred_maps["T"] = pred["T"][:, 0].reshape(nt, nz)
    ref_maps["v"] = interp(rf["v"])
    pred_maps["v"] = pred["v"][:, 0].reshape(nt, nz)
    plots.plot_fields(zg, tg, ref_maps, pred_maps, os.path.join(out_dir, "fields_error_maps.png"))
    return {k: float(np.sqrt(np.mean((pred_maps[k] - ref_maps[k]) ** 2))) for k in ref_maps}


def _times(p, pred, ref, ev, names):
    c_feed = feed_and_initial_state(p)["c_feed"]
    comps = front_components(p.y_init, p.y_feed)
    args = (c_feed, p.y_feed, comps, ev["times"], bool(ev["interpolate_times"]))
    tp = characteristic_times(pred["t"], pred["c_out"], pred["y_out"], *args)
    tr = characteristic_times(ref["t"], ref["c_out"], ref["y_out"], *args)
    out = {}
    for name in tp:
        for j, i in enumerate(comps):
            out["%s_%s" % (name, names[i])] = dict(
                pinn_s=tp[name][j], ref_s=tr[name][j], rel_err_pct=100.0 * (tp[name][j] - tr[name][j]) / tr[name][j]
            )
    return out


def _mole_balance(p, s, pred, ref):
    if ref.get("v_out") is not None:
        return compare_mole_balance(p, s.t_ref, ref["t"], pred, ref)
    rows = mole_balance(p, pred["t"], pred["v_out"], pred["c_out"])
    return [dict(component=r["component"], pinn_moment_s=r["moment_s"], theory_s=r["theory_s"],
                 pinn_vs_theory_pct_tref=100.0 * (r["moment_s"] - r["theory_s"]) / s.t_ref) for r in rows]


def evaluate_case(model, table, case_index, p, s, ref, cfg, out_dir, names=None, pool=None, log_csv=None):
    """Full evaluation of one case: outlet metrics, characteristic times, mole balance, fields, figures."""
    ev = cfg["eval"]
    names = names or ["c%d" % i for i in range(p.n_comp)]
    pred = predict_outlet(model, table, case_index, ref["t"], s, int(ev["batch_size"]))
    res = dict(outlet={})
    for i in range(p.n_comp):
        res["outlet"]["y_%s" % names[i]] = regression_metrics(pred["y_out"][:, i], ref["y_out"][:, i], ev["aard_min"])
    res["outlet"]["T_out"] = regression_metrics(pred["T_out"], ref["T_out"], ev["aard_min"])
    if ref.get("v_out") is not None:
        res["outlet"]["v_out/vs"] = regression_metrics(pred["v_out"] / p.vs, ref["v_out"] / p.vs, ev["aard_min"])
    res["times"] = _times(p, pred, ref, ev, names)
    res["mole_balance"] = _mole_balance(p, s, pred, ref)
    if "c" in ref:
        res["fields_rmse_dimless"] = field_errors(model, table, case_index, p, s, ref, ev, out_dir)
    res["inference_ms_per_curve"] = inference_ms(model, table, case_index, len(ref["t"]), int(ev["inference_repeats"]))
    res["n_time_points"] = int(len(ref["t"]))

    plots.plot_outlet(ref, pred, names, os.path.join(out_dir, "outlet_curves.png"))
    if log_csv is not None and os.path.exists(log_csv):
        plots.plot_history(log_csv, os.path.join(out_dir, "training_history.png"))
    if pool is not None:
        plots.plot_collocation(pool, os.path.join(out_dir, "collocation_points.png"))
    np.savez_compressed(os.path.join(out_dir, "outlet_prediction.npz"), **pred)
    return res


def evaluate_cases(model, table, case_indices, params, scales, refs, cfg, out_dir, n_plots):
    """Several cases (Phase 1 validation/holdout): metrics pooled over all outlet points, as in IC1,
    plus per-case characteristic times and mole balance (CSV) and a few outlet figures."""
    ev = cfg["eval"]
    n = params[0].n_comp
    names = ["c%d" % i for i in range(n)]
    pooled = {k: ([], []) for k in ["y_%s" % nm for nm in names] + ["T_out"]}
    rows = []
    for j, (k, p, s, ref) in enumerate(zip(case_indices, params, scales, refs)):
        pred = predict_outlet(model, table, int(k), ref["t"], s, int(ev["batch_size"]))
        for i in range(n):
            pooled["y_%s" % names[i]][0].append(pred["y_out"][:, i])
            pooled["y_%s" % names[i]][1].append(ref["y_out"][:, i])
        pooled["T_out"][0].append(pred["T_out"])
        pooled["T_out"][1].append(ref["T_out"])
        row = dict(case=int(k))
        for key, v in _times(p, pred, ref, ev, names).items():
            row[key + "_rel_err_pct"] = v["rel_err_pct"]
        for mb in _mole_balance(p, s, pred, ref):
            row["moment_c%d_pinn_vs_theory_pct_tref" % mb["component"]] = mb["pinn_vs_theory_pct_tref"]
        rows.append(row)
        if j < n_plots:
            plots.plot_outlet(ref, pred, names, os.path.join(out_dir, "outlet_case%03d.png" % k))
    res = dict(outlet={k: regression_metrics(np.concatenate(a), np.concatenate(b), ev["aard_min"]) for k, (a, b) in pooled.items()})
    res["per_case_mean_abs"] = {key: float(np.nanmean(np.abs([r.get(key, np.nan) for r in rows]))) for key in rows[0] if key != "case"}
    with open(os.path.join(out_dir, "per_case_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return res
