"""PINN prediction for cases without a solver reference (e.g. hard seeds the MOL solver cannot solve).

    ~/ic2env/bin/python scripts/predict_cases.py --run runs/phase2c_physics --seeds 21 38 62

The cases are drawn from the run's domain (Domain.sample: IC1 recorder seeds in 'full' mode).
Without a reference only internal checks are possible:
  - outlet first moment of each fed gas against its exact long-time value (mole balance), in % of
    t_ref: the 'pinn_vs_theory' column of the evaluation of cases with a reference;
  - mean squared PDE residual (training term weights, no causal weights) at random interior points,
    for early times (t~ < ic_delta, the layer of the hard initial condition) and for the rest;
  - characteristic times and outlet temperature rise of the PINN.
Seeds with a reference can be added to the list to calibrate these checks against known errors.
Results in <run>/<out-name>/ (default predict_seeds/): summary.json and outlet_seed<k>.png.
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
from nnads_ic2.eval import plots  # noqa: E402
from nnads_ic2.eval.mass_balance import mole_balance  # noqa: E402
from nnads_ic2.eval.metrics import characteristic_times, front_components  # noqa: E402
from nnads_ic2.eval.predict import predict_outlet  # noqa: E402
from nnads_ic2.experiment import case_table, make_model  # noqa: E402
from nnads_ic2.physics.params import ThetaScaler  # noqa: E402
from nnads_ic2.physics.scales import feed_and_initial_state  # noqa: E402


def mean_residual(lb, case_index, t_lo, t_hi, n, rng, dtype):
    """Mean squared PDE residual of one case at n random points, z~ in (0, 1), t~ in (t_lo, t_hi)."""
    import tensorflow as tf

    z = tf.constant(rng.uniform(0.0, 1.0, (n, 1)), dtype)
    t = tf.constant(rng.uniform(t_lo, t_hi, (n, 1)), dtype)
    _, _, total = lb.interior(z, t, tf.fill([n], case_index), tf.constant(0.0, dtype))
    return float(tf.reduce_mean(total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-times", type=int, default=801, help="output times over [0, t_F]")
    ap.add_argument("--n-resid", type=int, default=4096, help="random points per case and time window")
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="overrides of the saved run config")
    ap.add_argument("--out-name", default="predict_seeds")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    cfg = apply_overrides(load_config(os.path.join(run_dir, "config.yaml")), args.set)
    dtype = runtime.setup_tf(cfg["runtime"])
    import numpy as np

    from nnads_ic2.training.losses import LossBuilder

    out_dir = os.path.join(run_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(run_dir, "theta_scaler.json"), encoding="utf-8") as f:
        scaler = ThetaScaler.from_dict(json.load(f))
    domain = Domain(cfg, ROOT)
    drawn = [domain.sample(seed) for seed in args.seeds]
    params = [p for p, _, _ in drawn]
    table, scales, _, _ = case_table(params, cfg, dtype, scaler=scaler)
    model = make_model(cfg, params[0].n_comp)
    model.load_weights(os.path.join(run_dir, "model.weights.h5"))
    lb = LossBuilder(model, table, cfg["loss"], cfg["physics"]["isotherm_p_min_atm"])
    ev = cfg["eval"]
    delta = float(cfg["model"]["ic_delta"])
    rng = np.random.default_rng(0)

    rows = []
    for k, (seed, (p, ok, info), s) in enumerate(zip(args.seeds, drawn, scales)):
        t = np.linspace(0.0, s.t_final, args.n_times)
        pred = predict_outlet(model, table, k, t, s, int(ev["batch_size"]))
        c_feed = feed_and_initial_state(p)["c_feed"]
        comps = front_components(p.y_init, p.y_feed)
        times = characteristic_times(t, pred["c_out"], pred["y_out"], c_feed, p.y_feed, comps, ev["times"],
                                     bool(ev["interpolate_times"]))
        mole = [dict(component=r["component"], pinn_moment_s=r["moment_s"], theory_s=r["theory_s"],
                     pinn_vs_theory_pct_tref=100.0 * (r["moment_s"] - r["theory_s"]) / s.t_ref)
                for r in mole_balance(p, t, pred["v_out"], pred["c_out"])]
        t_end = s.t_final / s.t_ref
        rows.append(dict(
            seed=seed, prefilter_ok=bool(ok), tst_max_s=info.get("tst_max_s"), severity=info.get("severity"),
            t_ref_s=s.t_ref, t_final_s=s.t_final,
            pinn_times_s={"%s_c%d" % (name, i): v for name, vals in times.items() for i, v in zip(comps, vals)},
            pinn_T_out_rise_K=float(np.max(np.abs(pred["T_out"] - p.Tin))),
            pinn_v_out_over_vs=[float(pred["v_out"].min() / p.vs), float(pred["v_out"].max() / p.vs)],
            mole_balance=mole,
            pde_residual_early=mean_residual(lb, k, 0.0, delta, args.n_resid, rng, dtype),
            pde_residual_late=mean_residual(lb, k, delta, t_end, args.n_resid, rng, dtype),
        ))
        plots.plot_outlet(None, pred, ["c%d" % i for i in range(p.n_comp)], os.path.join(out_dir, "outlet_seed%d.png" % seed))
        np.savez_compressed(os.path.join(out_dir, "outlet_seed%d.npz" % seed), **pred)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(run=os.path.basename(run_dir), cases=rows), f, indent=2, default=float)
    print(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
