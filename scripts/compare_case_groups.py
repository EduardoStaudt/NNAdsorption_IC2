"""Validation R2 of stage (c) runs by group of cases (Phase 6).

    python scripts/compare_case_groups.py runs/phase5c_long_seed0 runs/phase5c_long_seed1 ... [--json out.json]

Groups of the 20 stage (c) validation cases (indices of eval_val/per_case_metrics.csv):
  - hot (main metric): the 5 largest measured outlet temperature rises of the MOL reference,
    cases 6, 11, 2, 9, 16 (143, 167, 83, 54 and 43 K; the next one is 20 K). The group was approved
    as "rise above 50 K", but case 16 has 43 K, so the threshold that gives these 5 cases is 40 K
    (docs/phase6_results.md). A group defined by the physics, not by the error.
  - worst3 (secondary): cases 6, 11, 2, the 3 worst of the Phase 5 long run (seed 1). Chosen by
    looking at the errors of that run on this validation set, so a gain there is partly selection.
  - others: the 15 cases outside the hot group.
For each run: the pooled y R2 of summary.json, and the mean of the per-case y R2 of each group.
"""

import argparse
import csv
import json
import os
import statistics

HOT = (6, 11, 2, 9, 16)
WORST3 = (6, 11, 2)


def run_groups(run_dir):
    with open(os.path.join(run_dir, "eval_val", "per_case_metrics.csv"), encoding="utf-8") as f:
        r2 = {int(r["case"]): float(r["r2_y_c1"]) for r in csv.DictReader(f)}
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    others = [k for k in r2 if k not in HOT]
    return dict(run=run_dir, pooled=s["validation"]["outlet"]["y_c1"]["r2"],
                T_rmse=s["validation"]["outlet"]["T_out"]["rmse"],
                hot=statistics.fmean(r2[k] for k in HOT), worst3=statistics.fmean(r2[k] for k in WORST3),
                others=statistics.fmean(r2[k] for k in others), per_case={k: r2[k] for k in sorted(r2)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = [run_groups(r) for r in args.runs]
    print("%-36s %8s %8s %8s %8s %8s" % ("run", "pooled", "hot5", "worst3", "others15", "T RMSE"))
    for r in rows:
        print("%-36s %8.4f %8.4f %8.4f %8.4f %8.2f" % (os.path.relpath(r["run"]), r["pooled"], r["hot"], r["worst3"], r["others"], r["T_rmse"]))
    print("per case:", " ".join("%d" % k for k in HOT), "|", " ".join("%s" % os.path.basename(r["run"]) for r in rows))
    for k in HOT:
        print("  case %2d: %s" % (k, " ".join("%.3f" % r["per_case"][k] for r in rows)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
