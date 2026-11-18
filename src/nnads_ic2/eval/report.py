"""Evaluation of one case against a reference solution (MOL solver now, v23b dataset later).

The reference is a dict with t, c_out, y_out, T_out, v_out and, when available, the internal
fields z, c, q, T, v_cell (as returned by physics/mol_solver.py).
"""

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..physics.scales import feed_and_initial_state
from . import plots
from .mass_balance import compare_mole_balance
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
    t_ref_grid = ref["t"] / s.t_ref
    zg = np.linspace(z_cells[0], z_cells[-1], nz)
    tg = np.linspace(t_ref_grid[0], t_ref_grid[-1], nt)
    TT, ZZ = np.meshgrid(tg, zg, indexing="ij")
    pred = predict_dimless(model, table, case_index, ZZ.ravel(), TT.ravel(), int(ev["batch_size"]))
    rf = _reference_fields_dimless(ref, s)
    pts = np.stack([TT.ravel(), ZZ.ravel()], axis=1)

    def interp(F):
        return RegularGridInterpolator((t_ref_grid, z_cells), F)(pts).reshape(nt, nz)

    ref_maps, pred_maps, out = {}, {}, {}
    for i in range(p.n_comp):
        ref_maps["c%d" % i] = interp(rf["c"][:, i, :])
        pred_maps["c%d" % i] = pred["c"][:, i].reshape(nt, nz)
        ref_maps["q%d" % i] = interp(rf["q"][:, i, :])
        pred_maps["q%d" % i] = pred["q"][:, i].reshape(nt, nz)
    ref_maps["T"] = interp(rf["T"])
    pred_maps["T"] = pred["T"][:, 0].reshape(nt, nz)
    ref_maps["v"] = interp(rf["v"])
    pred_maps["v"] = pred["v"][:, 0].reshape(nt, nz)
    for k in ref_maps:
        out[k] = float(np.sqrt(np.mean((pred_maps[k] - ref_maps[k]) ** 2)))
    plots.plot_fields(zg, tg, ref_maps, pred_maps, os.path.join(out_dir, "fields_error_maps.png"))
    return out


def evaluate_case(model, table, case_index, p, s, ref, cfg, out_dir, names=None, pool=None, log_csv=None):
    ev = cfg["eval"]
    names = names or ["c%d" % i for i in range(p.n_comp)]
    pred = predict_outlet(model, table, case_index, ref["t"], s, int(ev["batch_size"]))
    res = dict(outlet={}, times={})
    for i in range(p.n_comp):
        res["outlet"]["y_%s" % names[i]] = regression_metrics(pred["y_out"][:, i], ref["y_out"][:, i], ev["aard_min"])
    res["outlet"]["T_out"] = regression_metrics(pred["T_out"], ref["T_out"], ev["aard_min"])
    res["outlet"]["v_out/vs"] = regression_metrics(pred["v_out"] / p.vs, ref["v_out"] / p.vs, ev["aard_min"])

    c_feed = feed_and_initial_state(p)["c_feed"]
    comps = front_components(p.y_init, p.y_feed)
    args = (c_feed, p.y_feed, comps, ev["times"], bool(ev["interpolate_times"]))
    tp = characteristic_times(pred["t"], pred["c_out"], pred["y_out"], *args)
    tr = characteristic_times(ref["t"], ref["c_out"], ref["y_out"], *args)
    for name in tp:
        for j, i in enumerate(comps):
            res["times"]["%s_%s" % (name, names[i])] = dict(
                pinn_s=tp[name][j], ref_s=tr[name][j], rel_err_pct=100.0 * (tp[name][j] - tr[name][j]) / tr[name][j]
            )

    res["mole_balance"] = compare_mole_balance(p, s.t_ref, ref["t"], pred, ref)
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
