import copy
import csv
import os

import numpy as np
import tensorflow as tf

from conftest import REPO_ROOT
from nnads_ic2.config import load_config
from nnads_ic2.data.cases import sample_cases
from nnads_ic2.eval.report import evaluate_cases, flat_metrics
from nnads_ic2.experiment import case_table, make_model
from nnads_ic2.physics.scales import compute_scales, feed_and_initial_state

CFG = load_config(os.path.join(REPO_ROOT, "configs", "pinn_base.yaml"))


def small_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["model"].update(width=16, depth=2, head_width=8)
    cfg["model"]["fourier"]["n_features"] = 8
    cfg["eval"].update(field_grid=dict(nz=11, nt=21), inference_repeats=2, batch_size=512)
    return cfg


def fake_reference(p, s, nt, nx, rng):
    """Outlet curves with a front of component 1 at t_ref, and random internal fields."""
    t = np.linspace(s.t_final / nt, s.t_final, nt)
    st = feed_and_initial_state(p)
    ratio = 1.0 / (1.0 + np.exp(-(t - s.t_ref) / (0.05 * s.t_ref)))
    c1 = ratio * st["c_feed"][1]
    c = np.stack([st["C_F"] - c1, c1], axis=1)
    ref = dict(t=t, c_out=c, y_out=c / c.sum(1, keepdims=True), T_out=p.Tin + 2.0 * np.sin(t / s.t_final),
               v_out=np.full(nt, p.vs))
    k = 9
    fields = dict(t=np.linspace(0.0, s.t_final, k), z=(np.arange(nx) + 0.5) * p.L / nx,
                  c=rng.uniform(0, st["C_F"], (k, 2, nx)), q=rng.uniform(0, 1, (k, 2, nx)),
                  T=p.Tin + rng.uniform(0, 3, (k, nx)), v=np.full((k, nx), p.vs))
    return ref, fields


def test_evaluate_cases_reports_every_phase0_metric(tmp_path):
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    cases = [p for _, p in sample_cases(range(40), cfg["cases"])[:2]]
    table, scales, _, _ = case_table(cases, cfg, "float32")
    model = make_model(cfg, 2)
    rng = np.random.default_rng(0)
    refs, fields = zip(*[fake_reference(p, compute_scales(p, **cfg["scales"]), 50, 7, rng) for p in cases])
    res = evaluate_cases(model, table, [0, 1], cases, scales, list(refs), cfg, str(tmp_path), n_plots=1, fields=list(fields))

    assert set(res["outlet"]) == {"y_c0", "y_c1", "T_out", "v_out/vs"}
    per_case = res["per_case_mean_abs"]
    for name in ("tb", "t10", "t50", "t90", "ts"):
        assert name + "_c1_rel_err_pct" in per_case
        assert res["per_case_n_finite"][name + "_c1_ref_s"] == 2  # the fake front reaches every level
    for key in ("moment_c1_pinn_vs_ref_pct_tref", "moment_c0_pinn_vs_theory_pct_tref", "field_rmse_T", "field_rmse_T_K",
                "field_rmse_q1"):
        assert np.isfinite(per_case[key])
    assert res["inference_ms_per_curve"] > 0 and res["n_cases"] == 2
    with open(os.path.join(tmp_path, "per_case_metrics.csv"), encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2
    assert os.path.exists(os.path.join(tmp_path, "fields_case000.png"))
    assert not os.path.exists(os.path.join(tmp_path, "fields_case001.png"))  # only n_plots figures
    row = flat_metrics(res)  # one row of the validation curve (train_phase2.py)
    assert row["y_c1_r2"] == res["outlet"]["y_c1"]["r2"] and row["T_out_rmse"] == res["outlet"]["T_out"]["rmse"]
    assert row["t50_c1_rel_err_pct"] == per_case["t50_c1_rel_err_pct"]
    assert not any(k.startswith(("moment", "field", "rmse_")) for k in row)


def test_evaluate_cases_without_velocity_data(tmp_path):
    # a dataset without v(L, t) and without internal v: the velocity metrics are skipped
    tf.keras.utils.set_random_seed(0)
    cfg = small_cfg()
    cases = [p for _, p in sample_cases(range(40), cfg["cases"])[:2]]
    table, scales, _, _ = case_table(cases, cfg, "float32")
    model = make_model(cfg, 2)
    rng = np.random.default_rng(1)
    refs, fields = zip(*[fake_reference(p, compute_scales(p, **cfg["scales"]), 50, 7, rng) for p in cases])
    for ref, fl in zip(refs, fields):
        ref["v_out"] = None
        del fl["v"]
    res = evaluate_cases(model, table, [0, 1], cases, scales, list(refs), cfg, str(tmp_path), n_plots=1, fields=list(fields))
    assert "v_out/vs" not in res["outlet"] and "field_rmse_v" not in res["per_case_mean_abs"]
    assert np.isfinite(res["per_case_mean_abs"]["field_rmse_T_K"])
