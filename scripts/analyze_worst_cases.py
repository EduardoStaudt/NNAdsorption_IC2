"""The worst validation cases of a trained run: outlet temperature rise, front speed and the theta
they share.

    ~/ic2env/bin/python scripts/analyze_worst_cases.py --run runs/phase5c_long_seed0 --n-worst 5 \
        --fig-dir docs/figures/phase5

Cases are ranked by the outlet y_c1 R2 of the run's final evaluation (eval_val/per_case_metrics.csv).
For each case, PINN and reference:
  - temperature rise of the outlet, max |T_out - Tin| [K];
  - front speed L / t50 of the strong gas [m/s] (t50: outlet y_c1 at half its feed value), and the
    PINN/reference ratio (> 1: the PINN front is too fast);
  - percentile of each theta column within the domain (a Sobol sample of the run's domain, the
    distribution the network was trained on).
A theta column is "shared" by the worst cases when all of them lie in the same quarter of the domain
(below the 25th or above the 75th percentile) and the other cases do not (at most half of them there).
Output: <run>/worst_cases/summary.json and figures (outlet curves of the worst cases, R2 against
the temperature rise and against the front speed error, theta percentiles).
"""

import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2 import runtime  # noqa: E402  (must precede TensorFlow)
from nnads_ic2.config import load_config, resolve_path  # noqa: E402
from nnads_ic2.data.domain import Domain, ThetaSampler  # noqa: E402
from nnads_ic2.data.loader import load_recorder_h5  # noqa: E402
from nnads_ic2.eval.predict import predict_outlet  # noqa: E402
from nnads_ic2.experiment import case_table, make_model  # noqa: E402
from nnads_ic2.physics.params import ThetaScaler, theta_names  # noqa: E402


def percentiles(sample, values):
    """Percentile (0-100) of each value within the sample, column by column."""
    import numpy as np

    srt = np.sort(sample, axis=0)
    return np.stack([100.0 * np.searchsorted(srt[:, j], values[:, j]) / len(srt) for j in range(sample.shape[1])], axis=1)


def shared_columns(pct, worst, names, varying, tail=25.0, others_max=0.5):
    """Columns where every worst case lies in the same tail of the domain and the others mostly do not."""
    import numpy as np

    others = np.setdiff1d(np.arange(pct.shape[0]), worst)
    out = []
    for j, nm in enumerate(names):
        if not varying[j]:
            continue
        for side, inside in (("low", pct[:, j] < tail), ("high", pct[:, j] > 100.0 - tail)):
            if inside[worst].all() and inside[others].mean() <= others_max:
                out.append(dict(column=nm, side=side, worst_pct=[float(x) for x in pct[worst, j]],
                                others_frac_in_tail=float(inside[others].mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-worst", type=int, default=5)
    ap.add_argument("--n-domain", type=int, default=4096, help="Sobol theta of the domain for the percentiles")
    ap.add_argument("--fig-dir", default=None, help="also copy the figures here")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    dtype = runtime.setup_tf(cfg["runtime"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = os.path.join(run_dir, "worst_cases")
    os.makedirs(out_dir, exist_ok=True)
    fig_dirs = [out_dir] + ([os.path.abspath(args.fig_dir)] if args.fig_dir else [])
    for d in fig_dirs:
        os.makedirs(d, exist_ok=True)

    def save(fig, name):
        for d in fig_dirs:
            fig.savefig(os.path.join(d, name), dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(run_dir, "eval_val", "per_case_metrics.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    r2 = np.array([float(r["r2_y_c1"]) for r in rows])
    t50_err = np.array([float(r["t50_c1_rel_err_pct"]) for r in rows])

    n_comp = int(cfg["physics"]["n_comp"])
    domain = Domain(cfg, ROOT)
    val_cases = load_recorder_h5(resolve_path(cfg["phase2"]["val_data"], ROOT), n_comp, carrier=int(cfg["data"]["carrier"]),
                                 keys=cfg["data"].get("keys"))
    params = [domain.snap(c.params) for c in val_cases]
    with open(os.path.join(run_dir, "theta_scaler.json"), encoding="utf-8") as f:
        scaler = ThetaScaler.from_dict(json.load(f))
    table, scales, _, _ = case_table(params, cfg, dtype, scaler=scaler)
    model = make_model(cfg, n_comp)
    model.load_weights(os.path.join(run_dir, "model.weights.h5"))
    bs = int(cfg["eval"]["batch_size"])

    cases = []
    preds = []
    for k, (c, p, s) in enumerate(zip(val_cases, params, scales)):
        pred = predict_outlet(model, table, k, c.t, s, bs)
        preds.append(pred)
        t50_ref, t50_pinn = float(rows[k]["t50_c1_ref_s"]), float(rows[k]["t50_c1_pinn_s"])
        cases.append(dict(
            case=k, seed=str(c.seed), r2_y_c1=float(r2[k]), r2_T_out=float(rows[k]["r2_T_out"]),
            rmse_T_out_K=float(rows[k]["rmse_T_out"]),
            dT_ref_K=float(np.max(np.abs(c.T_out - p.Tin))), dT_pinn_K=float(np.max(np.abs(pred["T_out"] - p.Tin))),
            t50_ref_s=t50_ref, t50_pinn_s=t50_pinn, t50_rel_err_pct=float(t50_err[k]),
            front_speed_ref_m_s=p.L / t50_ref if t50_ref > 0 else float("nan"),
            front_speed_ratio=t50_ref / t50_pinn if t50_pinn > 0 else float("nan"),
            v_out_ref_range=[float(c.v_out.min() / p.vs), float(c.v_out.max() / p.vs)],
        ))
    order = np.argsort(r2)
    worst = order[: args.n_worst]

    names = theta_names(n_comp)
    theta = np.stack([p.to_theta() for p in params])
    dom = np.stack([p.to_theta() for p in ThetaSampler(domain, seed=12345).draw(args.n_domain)])
    varying = dom.std(axis=0) > 1e-12 * (np.abs(dom).mean(axis=0) + 1e-30)
    pct = percentiles(dom, theta)
    shared = shared_columns(pct, worst, names, varying)
    # rank correlation of the per-case R2 with each theta column and with the temperature rise (20 cases: rough)
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    rank_corr = sorted(({"column": nm, "spearman_r2": spearman(theta[:, j], r2)} for j, nm in enumerate(names) if varying[j]),
                       key=lambda d: -abs(d["spearman_r2"]))
    for c in cases:
        c["theta_percentile"] = {nm: float(pct[c["case"], j]) for j, nm in enumerate(names) if varying[j]}

    # 1. outlet curves of the worst cases (y_c1 and T_out, reference and PINN)
    fig, axes = plt.subplots(2, len(worst), figsize=(3.2 * len(worst), 5.2), squeeze=False)
    for col, k in enumerate(worst):
        c, pr, p = val_cases[k], preds[k], params[k]
        axes[0, col].plot(c.t, c.y_out[:, 1], "k-", lw=1.6, label="MOL")
        axes[0, col].plot(pr["t"], pr["y_out"][:, 1], "C0--", lw=1.6, label="PINN")
        axes[0, col].set_title("case %d (seed %s)\ny R2 %.3f" % (k, c.seed, r2[k]), fontsize=9)
        axes[1, col].plot(c.t, c.T_out - p.Tin, "k-", lw=1.6)
        axes[1, col].plot(pr["t"], pr["T_out"] - p.Tin, "C3--", lw=1.6)
        axes[1, col].set_xlabel("t [s]")
    axes[0, 0].set_ylabel("outlet y_c1")
    axes[1, 0].set_ylabel("T_out - Tin [K]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Worst %d validation cases (stage c): MOL solid, PINN dashed" % len(worst), fontsize=10)
    save(fig, "worst_cases_outlet.png")

    # 2. R2 against the temperature rise and against the front speed error
    dT = np.array([c["dT_ref_K"] for c in cases])
    ratio = np.array([c["front_speed_ratio"] for c in cases])
    is_worst = np.isin(np.arange(len(cases)), worst)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, x, lab in ((axes[0], dT, "reference outlet temperature rise [K]"), (axes[1], ratio, "front speed PINN / MOL (L / t50)")):
        ax.scatter(x[~is_worst], r2[~is_worst], s=28, c="C7", label="other cases")
        ax.scatter(x[is_worst], r2[is_worst], s=40, c="C3", marker="D", label="worst %d" % len(worst))
        for k in worst:
            ax.annotate(str(k), (x[k], r2[k]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    axes[0].set_xscale("symlog", linthresh=1.0)
    axes[1].axvline(1.0, color="k", lw=0.8)
    axes[0].set_ylabel("validation y_c1 R2 (per case)")
    axes[0].legend(fontsize=8)
    save(fig, "worst_cases_scatter.png")

    # 3. theta percentiles of the worst cases against the others
    cols = [j for j in range(len(names)) if varying[j]]
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(cols)), 3.8))
    others = np.where(~is_worst)[0]
    xs = np.arange(len(cols))
    ax.scatter(np.repeat(xs, len(others)) + 0.12, pct[np.ix_(others, cols)].T.ravel(), s=10, c="C7", label="other cases")
    ax.scatter(np.repeat(xs, len(worst)) - 0.12, pct[np.ix_(worst, cols)].T.ravel(), s=16, c="C3", marker="D", label="worst %d" % len(worst))
    ax.axhspan(0, 25, color="C0", alpha=0.06)
    ax.axhspan(75, 100, color="C0", alpha=0.06)
    ax.set_xticks(xs)
    ax.set_xticklabels([names[j] for j in cols], rotation=90, fontsize=7)
    ax.set_ylabel("percentile in the domain")
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "worst_cases_theta.png")

    out = dict(run=os.path.relpath(run_dir, ROOT), n_worst=len(worst), worst=[int(k) for k in worst],
               shared_theta=shared, rank_corr_theta=rank_corr,
               rank_corr_dT_ref=spearman(np.array([c["dT_ref_K"] for c in cases]), r2), cases=cases,
               mean_worst=dict(dT_ref_K=float(dT[worst].mean()), front_speed_ratio=float(np.nanmean(ratio[worst]))),
               mean_others=dict(dT_ref_K=float(dT[others].mean()), front_speed_ratio=float(np.nanmean(ratio[others]))))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    for k in worst:
        c = cases[k]
        print("case %2d seed %s | y R2 %.3f | dT ref %.1f pinn %.1f K | t50 err %.1f %% | front ratio %.2f"
              % (k, c["seed"], c["r2_y_c1"], c["dT_ref_K"], c["dT_pinn_K"], c["t50_rel_err_pct"], c["front_speed_ratio"]))
    print("shared theta:", json.dumps(shared, indent=1))
    print("rank corr R2 vs dT_ref %.2f | top theta:" % out["rank_corr_dT_ref"],
          ", ".join("%s %.2f" % (d["column"], d["spearman_r2"]) for d in rank_corr[:6]))


if __name__ == "__main__":
    main()
