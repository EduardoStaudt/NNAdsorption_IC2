"""Phase 1: parametric PINN over several cases (theta as input) with the hybrid loss.

Run inside WSL2, from the repository root:
    ~/ic2env/bin/python scripts/train_parametric.py --config configs/pinn_parametric.yaml --data <dataset.h5>
The dataset is a solver h5 file in the IC1 recorder format (data/loader.py). Cases are split at
random (seeded) into training and validation cases; the PINN sees solver data only from the
training cases, and the collocation points cover the training cases. Internal profiles of the
validation cases (when the file has them) are used only to score the internal fields.
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
from nnads_ic2.data.loader import dataset_info, load_recorder_h5  # noqa: E402
from nnads_ic2.data.sampling import CollocationPool  # noqa: E402
from nnads_ic2.eval.report import evaluate_cases  # noqa: E402
from nnads_ic2.experiment import case_table, make_model, outlet_data, profile_data, resolve_dT_ratio  # noqa: E402
from nnads_ic2.physics.scales import measured_temperature_rise  # noqa: E402
from nnads_ic2.training.trainer import Trainer  # noqa: E402


def reference(case):
    return dict(t=case.t, c_out=case.c_out, y_out=case.y_out, T_out=case.T_out, v_out=case.v_out)


def data_temperatures(case, use_profiles):
    """Temperatures of a training case that the PINN may see as data (outlet, plus profiles if enabled)."""
    T = [case.T_out]
    if use_profiles and case.profiles is not None:
        T.append(case.profiles["T"].ravel())
    return np.concatenate(T)


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

    ph = cfg["phase1"]
    n_comp = int(cfg["physics"]["n_comp"])
    data_path = resolve_path(cfg["data"]["path"], ROOT)
    ds_info = dataset_info(data_path, cfg["data"].get("keys"))
    print("dataset %s | tag %s | source: %s" % (data_path, ds_info.get("dataset_tag", "-"), ds_info.get("source", "-")), flush=True)
    if "MOL" in str(ds_info.get("dataset_tag", "")):
        print("WARNING: provisional MOL dataset (not v23b): keep these results apart from v23b runs", flush=True)
    # profiles are always read: training cases use them only if data.use_internal_profiles is true,
    # validation cases only for the field errors
    cases = load_recorder_h5(data_path, n_comp, carrier=int(cfg["data"]["carrier"]), use_internal_profiles=True,
                             keys=cfg["data"].get("keys"))
    rng = np.random.default_rng(int(ph["split_seed"]))
    perm = rng.permutation(len(cases))
    n_tr, n_val = int(ph["n_train_cases"]), int(ph["n_val_cases"])
    if n_tr + n_val > len(cases):
        sys.exit("dataset has %d valid cases, need %d" % (len(cases), n_tr + n_val))
    train = [cases[i] for i in perm[:n_tr]]
    val = [cases[i] for i in perm[n_tr : n_tr + n_val]]
    print("device %s | %d training cases, %d validation cases from %s" % (runtime.device_name(), n_tr, n_val, cfg["data"]["path"]))

    # dT_ref: median of (measured rise / feed estimate) over the training cases, applied to every
    # case, so validation cases (no data at prediction time) follow the same rule as training cases
    use_prof = bool(cfg["data"]["use_internal_profiles"])
    resolve_dT_ratio(cfg, [c.params for c in train],
                     [measured_temperature_rise(data_temperatures(c, use_prof), c.params.Tin) for c in train])
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

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
                                  run_dir, n_plots=int(ph["n_plots"]), fields=[c.profiles for c in val]),
        dataset=dict(path=data_path, **{k: ds_info.get(k) for k in ("dataset_tag", "source", "created_utc", "git_commit")}),
        train_seeds=[c.seed for c in train],
        val_seeds=[c.seed for c in val],
        dT_ratio=float(cfg["scales"]["dT_ratio"]),
        dT_ref_val_K=[s.dT_ref for s in scales[n_tr:]],
        training=dict(device=runtime.device_name(), steps=int(cfg["train"]["steps"]), train_s=timing["train_s"],
                      gpu_hours=timing["train_s"] / 3600.0 if runtime.device_name() == "GPU" else 0.0,
                      first_step_s=timing.get("first_step_s"), lbfgs=timing.get("lbfgs"),
                      n_params=int(sum(v.numpy().size for v in model.trainable_variables)),
                      final_lambdas=trainer.weights.as_dict(), final_causal_eps=float(trainer.causal.eps.numpy()),
                      pool_size=pool.size),
    )
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary["validation"], indent=1, default=float))


if __name__ == "__main__":
    main()
