"""Re-evaluate a trained run (config.yaml and model.weights.h5 saved in runs/<name>/).

    ~/ic2env/bin/python scripts/evaluate.py --run runs/phase0_single_case
    ~/ic2env/bin/python scripts/evaluate.py --run runs/phase1_parametric --data <holdout.h5>
    --set key.sub=value overrides the saved config (e.g. new eval.times entries for an old run).
Single-case runs are compared with the (cached) MOL reference of their case. Parametric runs are
evaluated on all valid cases of an h5 dataset (e.g. the 10k holdout), with the theta scaler saved
at training time; --split train|val keeps only the training or validation seeds of the run.
Results go to <run>/eval/ (or <run>/eval_train/, <run>/eval_val/).
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2 import runtime  # noqa: E402  (must precede TensorFlow)
from nnads_ic2.config import apply_overrides, load_config  # noqa: E402
from nnads_ic2.data.domain import Domain  # noqa: E402
from nnads_ic2.data.loader import load_recorder_h5  # noqa: E402
from nnads_ic2.eval.report import evaluate_case, evaluate_cases  # noqa: E402
from nnads_ic2.experiment import case_table, make_model, reference_solution, single_case  # noqa: E402
from nnads_ic2.physics.params import ThetaScaler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=None, help="h5 dataset for parametric runs")
    ap.add_argument("--n-plots", type=int, default=6)
    ap.add_argument("--split", choices=["all", "train", "val"], default="all",
                    help="parametric runs: evaluate only the training or validation seeds of the run (summary.json)")
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="overrides of the saved run config")
    ap.add_argument("--out-name", default=None, help="output folder inside the run (default eval/ or eval_<split>/)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    cfg = apply_overrides(load_config(os.path.join(run_dir, "config.yaml")), args.set)
    dtype = runtime.setup_tf(cfg["runtime"])
    out_dir = os.path.join(run_dir, args.out_name or ("eval" if args.split == "all" else "eval_" + args.split))
    os.makedirs(out_dir, exist_ok=True)
    n_comp = int(cfg["physics"]["n_comp"])

    if args.data is None:  # single-case run
        p = single_case(cfg, ROOT)
        table, (s,), _, _ = case_table([p], cfg, dtype)
        model = make_model(cfg, p.n_comp)
        model.load_weights(os.path.join(run_dir, "model.weights.h5"))
        sol = reference_solution(p, s, cfg, cfg["case"]["name"], ROOT)
        res = evaluate_case(model, table, 0, p, s, sol, cfg, out_dir)
    else:
        with open(os.path.join(run_dir, "theta_scaler.json"), encoding="utf-8") as f:
            scaler = ThetaScaler.from_dict(json.load(f))
        cases = load_recorder_h5(os.path.abspath(args.data), n_comp, carrier=int(cfg["data"]["carrier"]),
                                 use_internal_profiles=True, keys=cfg["data"].get("keys"))
        if args.split != "all":
            with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
                seeds = json.load(f)[args.split + "_seeds"]
            by_seed = {c.seed: c for c in cases}
            cases = [by_seed[sd] for sd in seeds]
        # subspace runs: fixed columns back to the base case values (the file stores float32)
        domain = Domain(cfg, ROOT) if cfg.get("domain", {}).get("mode") == "subspace" else None
        params = [domain.snap(c.params) if domain else c.params for c in cases]
        table, scales, _, _ = case_table(params, cfg, dtype, scaler=scaler)
        model = make_model(cfg, n_comp)
        model.load_weights(os.path.join(run_dir, "model.weights.h5"))
        refs = [dict(t=c.t, c_out=c.c_out, y_out=c.y_out, T_out=c.T_out, v_out=c.v_out) for c in cases]
        res = evaluate_cases(model, table, list(range(len(cases))), params, scales, refs, cfg, out_dir, args.n_plots,
                             fields=[c.profiles for c in cases])
    with open(os.path.join(out_dir, "summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
