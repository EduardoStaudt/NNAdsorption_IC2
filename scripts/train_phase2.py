"""Phase 2: parametric PINN with collocation at theta drawn from the domain (dimension curriculum).

Run inside WSL2, from the repository root:
    ~/ic2env/bin/python scripts/train_phase2.py --config configs/phase2_stage_a.yaml \\
        --val data/provisorio_MOL_n2_stage_a_val20.h5 --run-name phase2a_physics --set loss.weights.data=0

Solver data for training (used only by the hybrid, loss.weights.data > 0; the physics run sees
no solver data except the scalar dT_ratio of the temperature scale):
    phase2.train_data = base_case   outlet of the MOL reference of domain.base_case (subspace stages)
    phase2.train_data = dataset     the Phase 1 training split of data.path (whole domain)
Fine-tuning: phase2.init_from = a trained run directory; the run starts from its weights, with its
network, dT_ratio and theta scaler (e.g. a physics run, then the outlet data at a small fixed
lambda_data: configs/phase3_finetune_c.yaml).
Collocation: the training cases plus theta_colloc.n_theta Sobol theta from the domain, renewed
every theta_colloc.resample_every_epochs passes over the pool (training/theta_colloc.py).
Validation: every valid case of --val (MOL cases of the stage, never seen in training). The
training cases are evaluated too (eval_train/: fit, not generalization). With
phase2.eval_every > 0 the outlet metrics on the validation cases are also computed during
training (val_curve.csv; per-case files in eval_curve/step_<k>/); that time is not counted in
the GPU hours of training.
Time marching (Phase 3 proof of concept, physics only): phase2.march_windows = K > 0 cuts
[0, t~_F] into K equal windows trained one after the other, steps/K steps each; window k starts
from the state of window k-1 at its start (models/pinn.py: MarchingWindow) and its network from
the weights of window k-1. Per-window logs and checkpoints in window_<k>/.
"""

import argparse
import copy
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2 import runtime  # noqa: E402  (must precede TensorFlow)
from nnads_ic2.config import apply_overrides, load_config, resolve_path, save_config  # noqa: E402
from nnads_ic2.data.domain import Domain, ThetaSampler  # noqa: E402
from nnads_ic2.data.loader import dataset_info, load_recorder_h5  # noqa: E402
from nnads_ic2.eval.report import evaluate_cases, flat_metrics, solution_fields  # noqa: E402
from nnads_ic2.experiment import (  # noqa: E402
    case_table, fit_theta_scaler, make_model, outlet_data, pretrained_setup, reference_solution, resolve_dT_ratio,
)
from nnads_ic2.models.pinn import MarchedPINN, MarchingWindow  # noqa: E402
from nnads_ic2.physics.scales import compute_scales, measured_temperature_rise  # noqa: E402
from nnads_ic2.training.theta_colloc import ThetaCollocation  # noqa: E402
from nnads_ic2.training.trainer import Trainer  # noqa: E402


def dataset_reference(case):
    return dict(t=case.t, c_out=case.c_out, y_out=case.y_out, T_out=case.T_out, v_out=case.v_out)


def mol_reference(sol):
    return dict(t=sol["t"], c_out=sol["c_out"], y_out=sol["y_out"], T_out=sol["T_out"], v_out=sol["v_out"])


def training_cases(cfg, domain):
    """Cases with solver data: dict(params, refs, fields, seeds, source) (+ Phase 1 validation cases)."""
    ph = cfg["phase2"]
    if ph["train_data"] == "base_case":
        p = domain.base
        s = compute_scales(p, **cfg["scales"])  # t_F only (does not depend on dT_ref)
        name = os.path.splitext(os.path.basename(domain.base_path))[0]
        sol = reference_solution(p, s, cfg, name, ROOT)  # Phase 0 MOL reference (cached)
        return dict(params=[p], refs=[mol_reference(sol)], fields=[solution_fields(sol)], seeds=[name],
                    source="MOL reference of %s (Phase 0)" % domain.base_path, extra=None)
    if ph["train_data"] == "dataset":
        path = resolve_path(cfg["data"]["path"], ROOT)
        cases = load_recorder_h5(path, int(cfg["physics"]["n_comp"]), carrier=int(cfg["data"]["carrier"]),
                                 use_internal_profiles=True, keys=cfg["data"].get("keys"))
        p1 = cfg["phase1"]
        perm = np.random.default_rng(int(p1["split_seed"])).permutation(len(cases))  # the Phase 1 split
        n_tr, n_val = int(p1["n_train_cases"]), int(p1["n_val_cases"])
        train = [cases[i] for i in perm[:n_tr]]
        val = [cases[i] for i in perm[n_tr : n_tr + n_val]]
        return dict(params=[c.params for c in train], refs=[dataset_reference(c) for c in train],
                    fields=[c.profiles for c in train], seeds=[c.seed for c in train],
                    source="training split of %s (%s)" % (path, dataset_info(path, cfg["data"].get("keys")).get("dataset_tag")),
                    extra=dict(params=[c.params for c in val], refs=[dataset_reference(c) for c in val],
                               fields=[c.profiles for c in val], seeds=[c.seed for c in val]))
    raise ValueError("phase2.train_data must be 'base_case' or 'dataset'")


def evaluate_set(model, cfg, scaler, dtype, cases, out_dir, n_plots):
    os.makedirs(out_dir, exist_ok=True)
    table, scales, _, _ = case_table(cases["params"], cfg, dtype, scaler=scaler)
    res = evaluate_cases(model, table, list(range(len(cases["params"]))), cases["params"], scales, cases["refs"], cfg,
                         out_dir, n_plots, fields=cases["fields"])
    res["seeds"] = [str(s) for s in cases["seeds"]]
    with open(os.path.join(out_dir, "summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    return res


def median_ms_per_step(log_path):
    with open(log_path, encoding="utf-8") as f:
        ms = [float(r["ms_per_step"]) for r in csv.DictReader(f)][1:]  # first row includes tracing
    return float(np.median(ms)) if ms else float("nan")


def train_marching(cfg, n_windows, train, domain, scaler, dtype, run_dir, n_comp):
    """Time marching: window k covers t~ in [k, k+1] t~_F/K, with its own collocation times,
    boundary times and causal slices. Returns the windows as one model and the last window's
    collocation and trainer."""
    tc = cfg["theta_colloc"]
    edges = np.linspace(0.0, float(cfg["scales"]["t_final_factor"]), n_windows + 1)
    steps_w = int(cfg["train"]["steps"]) // n_windows
    windows, nets, info = [], [], []
    timing = dict(train_s=0.0)
    for k in range(n_windows):
        cfg_k = copy.deepcopy(cfg)
        cfg_k["train"]["steps"] = steps_w
        cfg_k["logging"]["ckpt_every"] = min(int(cfg["logging"]["ckpt_every"]), steps_w)
        colloc = ThetaCollocation(cfg_k, train["params"], ThetaSampler(domain, seed=int(tc["seed"])), scaler, dtype,
                                  steps_w, t_range=(edges[k], edges[k + 1]))
        net = make_model(cfg_k, n_comp)
        if k == 0:
            model = net  # the plain PINN: hard initial condition at t~ = 0
        else:
            net.set_weights(nets[-1].get_weights())  # warm start from the previous window's network
            model = MarchingWindow(windows[-1], net, edges[k])
        nets.append(net)
        windows.append(model)
        win_dir = os.path.join(run_dir, "window_%d" % k)
        os.makedirs(win_dir, exist_ok=True)
        print("window %d: t~ in [%.3f, %.3f], %d steps" % (k, edges[k], edges[k + 1], steps_w), flush=True)
        trainer = Trainer(cfg_k, model, colloc.table, colloc.pool, win_dir, resampler=colloc)
        t = trainer.run()
        timing["train_s"] += t["train_s"]
        timing.setdefault("first_step_s", t.get("first_step_s"))
        info.append(dict(window=k, t_start=float(edges[k]), t_end=float(edges[k + 1]), steps=steps_w,
                         train_s=t["train_s"], median_ms_per_step=median_ms_per_step(os.path.join(win_dir, "train_log.csv")),
                         skipped_steps=int(trainer.skipped.numpy())))
    n_params = len(nets) * int(sum(v.numpy().size for v in nets[-1].trainable_variables))
    return MarchedPINN(windows, edges[:-1]), colloc, trainer, timing, dict(n_windows=n_windows, n_params=n_params, windows=info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--val", default=None, help="validation h5 of the stage (overrides phase2.val_data)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="config overrides, e.g. model.width=256")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.val:
        cfg["phase2"]["val_data"] = os.path.abspath(args.val)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if not cfg["phase2"]["val_data"]:
        sys.exit("no validation set: pass --val <file.h5>")
    run_name = args.run_name or cfg["logging"]["run_name"]
    dtype = runtime.setup_tf(cfg["runtime"])
    run_dir = os.path.join(ROOT, cfg["logging"]["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    n_comp = int(cfg["physics"]["n_comp"])
    domain = Domain(cfg, ROOT)
    hybrid = float(cfg["loss"]["weights"]["data"]) > 0
    print("device %s | domain %s | %s" % (runtime.device_name(), json.dumps(domain.describe()),
                                         "hybrid (outlet data)" if hybrid else "physics only"), flush=True)

    init_dir = resolve_path(cfg["phase2"]["init_from"], ROOT) if cfg["phase2"].get("init_from") else None
    pre_scaler = pretrained_setup(cfg, init_dir) if init_dir else None
    if init_dir:
        print("fine-tuning from %s (its network, dT_ratio %.4g and theta scaler)" % (init_dir, cfg["scales"]["dT_ratio"]), flush=True)
    train = training_cases(cfg, domain)
    print("training cases: %d (%s)" % (len(train["params"]), train["source"]), flush=True)
    val_path = resolve_path(cfg["phase2"]["val_data"], ROOT)
    val_info = dataset_info(val_path, cfg["data"].get("keys"))
    print("validation %s | tag %s | purpose %s" % (val_path, val_info.get("dataset_tag"), val_info.get("purpose")), flush=True)
    if "MOL" in str(val_info.get("dataset_tag", "")):
        print("WARNING: provisional MOL data (not v23b): keep these results apart from v23b runs", flush=True)

    # dT_ref from the outlet temperature of the training references (what a hybrid run may see)
    resolve_dT_ratio(cfg, train["params"], [measured_temperature_rise(r["T_out"], p.Tin) for p, r in zip(train["params"], train["refs"])])

    # theta normalization from the domain (a large Sobol sample plus the training cases), fixed for the run
    tc = cfg["theta_colloc"]
    scaler = pre_scaler or fit_theta_scaler(
        ThetaSampler(domain, seed=int(tc["seed"]) + 1000).draw(int(tc["scaler_n_theta"])) + train["params"], cfg)
    with open(os.path.join(run_dir, "theta_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f)

    steps = int(cfg["train"]["steps"])
    colloc = ThetaCollocation(cfg, train["params"], ThetaSampler(domain, seed=int(tc["seed"])), scaler, dtype, steps)
    print("collocation: %d fixed + %d Sobol theta, pool %d, new theta every %d steps"
          % (colloc.n_fixed, colloc.n_theta, colloc.pool.size, colloc.every_steps), flush=True)
    data = None
    if hybrid:
        rng = np.random.default_rng(int(cfg["phase2"]["data_seed"]))
        data = {"outlet": outlet_data(train["refs"], colloc.fixed_scales, range(colloc.n_fixed),
                                      int(cfg["phase2"]["n_time_per_case"]), rng)}
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    val_cases = load_recorder_h5(val_path, n_comp, carrier=int(cfg["data"]["carrier"]), use_internal_profiles=True,
                                 keys=cfg["data"].get("keys"))
    val = dict(params=[domain.snap(c.params) for c in val_cases], refs=[dataset_reference(c) for c in val_cases],
               fields=[c.profiles for c in val_cases], seeds=[c.seed for c in val_cases])

    n_windows = int(cfg["phase2"].get("march_windows") or 0)
    eval_every = int(cfg["phase2"].get("eval_every") or 0)
    if n_windows > 0 and (hybrid or init_dir or eval_every > 0):
        sys.exit("phase2.march_windows (time marching) is a physics-only proof of concept: no data, init_from or eval_every")
    model = make_model(cfg, n_comp)
    if init_dir:
        model.load_weights(os.path.join(init_dir, "model.weights.h5"))
    curve_path = os.path.join(run_dir, "val_curve.csv")

    def periodic_eval(step):
        """Outlet metrics on the validation cases at this step (no internal fields, no plots)."""
        res = evaluate_set(model, cfg, scaler, dtype, dict(val, fields=None),
                           os.path.join(run_dir, "eval_curve", "step_%06d" % step), 0)
        row = dict(step=step, **flat_metrics(res))
        new = not os.path.exists(curve_path)
        with open(curve_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print("validation at step %d: y R2 %.3f | T_out RMSE %.2f K" % (step, row["y_c1_r2"], row["T_out_rmse"]), flush=True)

    march = None
    if n_windows > 0:
        model, colloc, trainer, timing, march = train_marching(cfg, n_windows, train, domain, scaler, dtype, run_dir, n_comp)
    else:
        trainer = Trainer(cfg, model, colloc.table, colloc.pool, run_dir, data=data, resampler=colloc,
                          evaluator=periodic_eval if eval_every > 0 else None, eval_every=eval_every)
        timing = trainer.run()
    train_s = timing["train_s"] - timing.get("periodic_eval_s", 0.0)  # training only
    n_plots = int(cfg["phase2"]["n_plots"])
    summary = dict(
        mode="hybrid" if hybrid else "physics",
        init_from=init_dir,
        domain=domain.describe(),
        validation=evaluate_set(model, cfg, scaler, dtype, val, os.path.join(run_dir, "eval_val"), n_plots),
        train_fit=evaluate_set(model, cfg, scaler, dtype, train, os.path.join(run_dir, "eval_train"), 2),
        validation_data=dict(path=val_path, **{k: val_info.get(k) for k in ("dataset_tag", "purpose", "created_utc", "git_commit")}),
        train_data=dict(source=train["source"], seeds=[str(s) for s in train["seeds"]]),
        dT_ratio=float(cfg["scales"]["dT_ratio"]),
        theta_colloc=colloc.info(),
        training=dict(device=runtime.device_name(), steps=steps, train_s=train_s,
                      periodic_eval_s=timing.get("periodic_eval_s", 0.0),
                      gpu_hours=train_s / 3600.0 if runtime.device_name() == "GPU" else 0.0,
                      median_ms_per_step=(float(np.mean([w["median_ms_per_step"] for w in march["windows"]])) if march
                                          else median_ms_per_step(os.path.join(run_dir, "train_log.csv"))),
                      first_step_s=timing.get("first_step_s"), jit_compile=bool(cfg["train"]["jit_compile"]),
                      n_params=march["n_params"] if march else int(sum(v.numpy().size for v in model.trainable_variables)),
                      final_lambdas=trainer.weights.as_dict(), final_causal_eps=float(trainer.causal.eps.numpy()),
                      pool_size=colloc.pool.size),
        march=march,
    )
    if train["extra"] is not None:  # whole domain: also the Phase 1 validation seeds (comparison with Phase 1)
        summary["phase1_validation"] = evaluate_set(model, cfg, scaler, dtype, train["extra"],
                                                    os.path.join(run_dir, "eval_phase1_val"), 0)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    v = summary["validation"]
    print("validation: y R2 %.3f | T_out RMSE %.2f K | GPU h %.3f" % (v["outlet"]["y_c1"]["r2"], v["outlet"]["T_out"]["rmse"],
                                                                   summary["training"]["gpu_hours"]), flush=True)


if __name__ == "__main__":
    main()
