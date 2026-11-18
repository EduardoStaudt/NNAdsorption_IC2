"""Phase 0: PINN on one fixed case (forward problem), compared with the MOL reference solver.

Run inside WSL2 (TensorFlow GPU), from the repository root:
    ~/ic2env/bin/python scripts/train_single_case.py --config configs/pinn_single_case.yaml
Options override the config: --steps N, --run-name NAME, --set key.sub=value ...
Outputs go to runs/<run-name>/: config.yaml, train_log.csv, checkpoints, model.weights.h5,
summary.json and figures.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2 import runtime  # noqa: E402  (must precede TensorFlow)
from nnads_ic2.config import apply_overrides, load_config, save_config  # noqa: E402
from nnads_ic2.data.sampling import CollocationPool  # noqa: E402
from nnads_ic2.eval.report import evaluate_case  # noqa: E402
from nnads_ic2.experiment import case_table, make_model, outlet_data, reference_solution, single_case  # noqa: E402
from nnads_ic2.training.trainer import Trainer  # noqa: E402

COMPONENT_NAMES = ["H2", "CO2"]  # IC1 example case: [carrier, strong]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "pinn_single_case.yaml"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. loss.weights.data=1.0")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    run_name = args.run_name or cfg["logging"]["run_name"]
    dtype = runtime.setup_tf(cfg["runtime"])
    run_dir = os.path.join(ROOT, cfg["logging"]["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print("device: %s | dtype: %s | run: %s" % (runtime.device_name(), dtype, run_dir), flush=True)

    # 1. case, scales, case table (normalized theta + constants of the residuals)
    p = single_case(cfg, ROOT)
    table, (s,), (groups,), _ = case_table([p], cfg, dtype)
    print("t_ref %.2f s | t_F %.2f s | dT_ref %.2f K | Pe %.0f | Da %s" % (s.t_ref, s.t_final, s.dT_ref, groups["Pe"], groups["Da"]))

    # 2. reference solution
    sol = reference_solution(p, s, cfg, cfg["case"]["name"], ROOT)

    # 3. model, collocation pool, data (only for the hybrid variant)
    model = make_model(cfg, p.n_comp)
    n_params = int(sum(v.numpy().size for v in model.trainable_variables))
    sc = cfg["sampling"]
    pool = CollocationPool(int(sc["pool_init"]), 1, float(groups["t_final"]), int(sc["pool_max"]), int(cfg["runtime"]["seed"]))
    data = None
    if float(cfg["loss"]["weights"]["data"]) > 0:
        rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
        data = {"outlet": outlet_data([sol], [s], [0], sol["t"].size, rng)}

    # 4. training
    trainer = Trainer(cfg, model, table, pool, run_dir, data=data)
    timing = trainer.run()

    # 5. evaluation against the reference
    names = COMPONENT_NAMES if p.n_comp == len(COMPONENT_NAMES) else None
    summary = evaluate_case(model, table, 0, p, s, sol, cfg, run_dir, names=names, pool=pool,
                            log_csv=os.path.join(run_dir, "train_log.csv"))
    summary["training"] = dict(
        device=runtime.device_name(),
        dtype=dtype,
        steps=int(cfg["train"]["steps"]),
        n_params=n_params,
        train_s=timing["train_s"],
        gpu_hours=timing["train_s"] / 3600.0 if runtime.device_name() == "GPU" else 0.0,
        first_step_s=timing.get("first_step_s"),
        lbfgs=timing.get("lbfgs"),
        final_lambdas=trainer.weights.as_dict(),
        final_causal_eps=float(trainer.causal.eps.numpy()),
        pool_size=pool.size,
    )
    summary["reference"] = dict(solver="MOL (physics/mol_solver.py)", **sol["info"])
    summary["scales"] = dict(t_ref=s.t_ref, t_final=s.t_final, C_F=s.C_F, q_ref=s.q_ref.tolist(), dT_ref=s.dT_ref)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    keys = ["outlet", "times", "mole_balance", "fields_rmse_dimless"]
    print(json.dumps({k: summary[k] for k in keys if k in summary}, indent=1, default=float))
    print("training: %.1f min on %s | inference %.2f ms/curve" % (timing["train_s"] / 60, runtime.device_name(), summary["inference_ms_per_curve"]))


if __name__ == "__main__":
    main()
