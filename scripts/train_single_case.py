"""Phase 0: PINN on one fixed case (forward problem), compared with the MOL reference solver.

Run inside WSL2 (TensorFlow GPU), from the repository root:
    ~/ic2env/bin/python scripts/train_single_case.py --config configs/pinn_single_case.yaml
Options override the config: --steps N, --run-name NAME.
Outputs go to runs/<run-name>/: config.yaml, train_log.csv, TensorBoard logs, checkpoints,
model.weights.h5, summary.json and figures.
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2 import runtime  # noqa: E402  (must precede TensorFlow)
from nnads_ic2.config import load_config, resolve_path, save_config  # noqa: E402
from nnads_ic2.data.sampling import CollocationPool  # noqa: E402
from nnads_ic2.eval.report import evaluate_case  # noqa: E402
from nnads_ic2.models.pinn import PINN  # noqa: E402
from nnads_ic2.physics import mol_solver  # noqa: E402
from nnads_ic2.physics.params import CaseParams, ThetaScaler, theta_dim, theta_log_mask  # noqa: E402
from nnads_ic2.physics.residuals import case_arrays, stack_cases  # noqa: E402
from nnads_ic2.physics.scales import compute_scales, dimensionless_groups  # noqa: E402
from nnads_ic2.training.trainer import Trainer  # noqa: E402


def reference_solution(p, s, cfg, case_name):
    """MOL solution of the case, cached in runs/cache/ (the solver takes tens of seconds)."""
    sc = cfg["solver"]
    cache_dir = os.path.join(ROOT, cfg["logging"]["runs_dir"], "cache")
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


def outlet_data(sol, s):
    """Solver outlet curve as data points (used only when lambda_data > 0)."""
    import numpy as np

    t = sol["t"] / s.t_ref
    return dict(
        z=np.ones_like(t),
        t=t,
        case=np.zeros(t.size, dtype=np.int64),
        y=sol["y_out"],
        T=(sol["T_out"] - s.Tin) / s.dT_ref,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "pinn_single_case.yaml"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    run_name = args.run_name or cfg["logging"]["run_name"]
    dtype = runtime.setup_tf(cfg["runtime"])
    run_dir = os.path.join(ROOT, cfg["logging"]["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print("device: %s | dtype: %s | run: %s" % (runtime.device_name(), dtype, run_dir), flush=True)

    # 1. case, scales and dimensionless groups
    p = CaseParams.from_ic1_json(resolve_path(cfg["case"]["path"], ROOT), carrier=int(cfg["case"]["carrier"]))
    s = compute_scales(p, **cfg["scales"])
    groups = dimensionless_groups(p, s)
    print("t_ref %.2f s | t_F %.2f s | dT_ref %.2f K | Pe %.0f | Da %s" % (s.t_ref, s.t_final, s.dT_ref, groups["Pe"], groups["Da"]))

    # 2. reference solution
    sol = reference_solution(p, s, cfg, cfg["case"]["name"])

    # 3. case table: normalized theta + constants of the residuals
    theta = p.to_theta()[None, :]
    scaler = ThetaScaler.fit(theta, theta_log_mask(p.n_comp, cfg["theta"]["log_params"]), float(cfg["theta"]["sd_floor"]))
    arrays = case_arrays(p, s, groups)
    arrays["theta"] = scaler.transform(theta)[0]
    table = stack_cases([arrays], dtype)

    # 4. model, collocation pool, data
    model = PINN(p.n_comp, theta_dim(p.n_comp), cfg["model"], t_final=float(groups["t_final"])).build_with_dummy()
    n_params = int(sum(v.numpy().size for v in model.trainable_variables))
    sc = cfg["sampling"]
    pool = CollocationPool(int(sc["pool_init"]), 1, float(groups["t_final"]), int(sc["pool_max"]), int(cfg["runtime"]["seed"]))
    data = {"outlet": outlet_data(sol, s)} if float(cfg["loss"]["weights"]["data"]) > 0 else None

    # 5. training
    trainer = Trainer(cfg, model, table, pool, run_dir, data=data)
    timing = trainer.run()

    # 6. evaluation against the reference
    summary = evaluate_case(model, table, 0, p, s, sol, cfg, run_dir, names=["H2", "CO2"] if p.n_comp == 2 else None,
                            pool=pool, log_csv=os.path.join(run_dir, "train_log.csv"))
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
    print(json.dumps({k: summary[k] for k in ["outlet", "times", "mole_balance", "fields_rmse_dimless"]}, indent=1, default=float))
    print("training: %.1f min on %s | inference %.2f ms/curve" % (timing["train_s"] / 60, runtime.device_name(), summary["inference_ms_per_curve"]))


if __name__ == "__main__":
    main()
