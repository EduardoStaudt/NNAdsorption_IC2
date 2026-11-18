"""Phase 1: parametric PINN over several cases (theta as input) with the hybrid loss.

Run inside WSL2, from the repository root:
    ~/ic2env/bin/python scripts/train_parametric.py --config configs/pinn_parametric.yaml --data <dataset.h5>
The dataset is a solver h5 file in the IC1 recorder format (data/loader.py). Cases are split at
random (seeded) into training and validation cases; the PINN sees solver data only from the
training cases, and the collocation points cover the training cases.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2 import runtime  # noqa: E402  (must precede TensorFlow)
from nnads_ic2.config import apply_overrides, load_config, resolve_path, save_config  # noqa: E402
from nnads_ic2.data.loader import load_recorder_h5  # noqa: E402
from nnads_ic2.data.sampling import CollocationPool  # noqa: E402
from nnads_ic2.eval.report import evaluate_cases  # noqa: E402
from nnads_ic2.experiment import case_table, make_model, outlet_data, profile_data  # noqa: E402
from nnads_ic2.training.trainer import Trainer  # noqa: E402


def reference(case):
    return dict(t=case.t, c_out=case.c_out, y_out=case.y_out, T_out=case.T_out, v_out=case.v_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "pinn_parametric.yaml"))
    ap.add_argument("--data", default=None, help="solver dataset (recorder h5); overrides data.path")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. train.lr=5e-4")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.data:
        cfg["data"]["path"] = os.path.abspath(args.data)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if not cfg["data"]["path"]:
        sys.exit("no dataset: pass --data <file.h5> or set data.path in the config")
    run_name = args.run_name or cfg["logging"]["run_name"]
    dtype = runtime.setup_tf(cfg["runtime"])
    run_dir = os.path.join(ROOT, cfg["logging"]["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    ph = cfg["phase1"]
    n_comp = int(cfg["physics"]["n_comp"])
    cases = load_recorder_h5(resolve_path(cfg["data"]["path"], ROOT), n_comp, carrier=int(cfg["data"]["carrier"]),
                             use_internal_profiles=bool(cfg["data"]["use_internal_profiles"]))
    rng = np.random.default_rng(int(ph["split_seed"]))
    perm = rng.permutation(len(cases))
    n_tr, n_val = int(ph["n_train_cases"]), int(ph["n_val_cases"])
    if n_tr + n_val > len(cases):
        sys.exit("dataset has %d valid cases, need %d" % (len(cases), n_tr + n_val))
    train = [cases[i] for i in perm[:n_tr]]
    val = [cases[i] for i in perm[n_tr : n_tr + n_val]]
    print("device %s | %d training cases, %d validation cases from %s" % (runtime.device_name(), n_tr, n_val, cfg["data"]["path"]))

    # theta normalized with the training cases only; table rows: training cases, then validation cases
    _, _, _, scaler = case_table([c.params for c in train], cfg, dtype)
    table, scales, groups, _ = case_table([c.params for c in train + val], cfg, dtype, scaler=scaler)
    with open(os.path.join(run_dir, "theta_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f)

    refs = [reference(c) for c in train + val]
    data = {"outlet": outlet_data(refs[:n_tr], scales[:n_tr], range(n_tr), int(ph["n_time_per_case"]), rng)}
    if cfg["data"]["use_internal_profiles"]:
        data["profiles"] = profile_data([c.profiles for c in train], scales[:n_tr], range(n_tr))

    model = make_model(cfg, n_comp)
    sc = cfg["sampling"]
    pool = CollocationPool(int(sc["pool_init"]), n_tr, float(cfg["scales"]["t_final_factor"]), int(sc["pool_max"]),
                           int(cfg["runtime"]["seed"]))
    trainer = Trainer(cfg, model, table, pool, run_dir, data=data)
    timing = trainer.run()

    val_idx = list(range(n_tr, n_tr + n_val))
    summary = dict(
        validation=evaluate_cases(model, table, val_idx, [c.params for c in val], scales[n_tr:], refs[n_tr:], cfg,
                                  run_dir, n_plots=int(ph["n_plots"])),
        train_seeds=[c.seed for c in train],
        val_seeds=[c.seed for c in val],
        training=dict(device=runtime.device_name(), steps=int(cfg["train"]["steps"]), train_s=timing["train_s"],
                      gpu_hours=timing["train_s"] / 3600.0 if runtime.device_name() == "GPU" else 0.0,
                      n_params=int(sum(v.numpy().size for v in model.trainable_variables)),
                      final_lambdas=trainer.weights.as_dict()),
    )
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary["validation"], indent=1, default=float))


if __name__ == "__main__":
    main()
