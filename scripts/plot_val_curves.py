"""Validation curves of one or more runs (val_curve.csv, written when phase2.eval_every > 0).

Three panels against the training step: pooled outlet R2 of y_c1 (with the gate), outlet
temperature RMSE, and the mean |relative error| of t50. --final adds the final validation of
runs without a curve (summary.json) as single points at their number of steps.

    python scripts/plot_val_curves.py --runs runs/phase3c_long:uniform runs/phase3c_front:front \\
        --final runs/phase2c_physics_robust:"Phase 2 (50k)" --out docs/figures/phase3/val_curve_stage_c.png
"""

import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2.eval.report import flat_metrics  # noqa: E402

PANELS = [("y_c1_r2", "outlet y R² (20 validation cases, pooled)"),
          ("T_out_rmse", "outlet T RMSE (K)"),
          ("t50_c1_rel_err_pct", "t50: mean |relative error| (%)")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order


def split(arg):
    path, _, label = arg.partition(":")
    return path, label or os.path.basename(os.path.normpath(path))


def read_curve(run_dir):
    with open(os.path.join(run_dir, "val_curve.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [int(r["step"]) for r in rows], {k: [float(r[k]) for r in rows] for k, _ in PANELS}


def read_final(run_dir):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    return int(s["training"]["steps"]), flat_metrics(s["validation"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run_dir[:label] with a val_curve.csv")
    ap.add_argument("--final", nargs="*", default=[], help="run_dir[:label]: final validation only")
    ap.add_argument("--gate", type=float, default=0.95, help="R2 gate line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.2))
    series = [(label, *read_curve(path)) for path, label in map(split, args.runs)]
    finals = [(label, *read_final(path)) for path, label in map(split, args.final)]
    for ax, (key, title) in zip(axes, PANELS):
        for i, (label, steps, vals) in enumerate(series):
            ax.plot([s / 1000 for s in steps], vals[key], "-o", c=COLORS[i], lw=2, ms=6, label=label)
        for j, (label, steps, vals) in enumerate(finals):
            ax.plot(steps / 1000, vals[key], "D", c=COLORS[len(series) + j], ms=8, label=label)
        if key.endswith("_r2"):
            ax.axhline(args.gate, c="0.45", lw=1, ls="--")
            ax.annotate("gate %.2f" % args.gate, (0.01, args.gate), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points", fontsize=8, color="0.3")
        ax.set(xlabel="training step (thousands)", title=title)
        ax.grid(True, c="0.9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
