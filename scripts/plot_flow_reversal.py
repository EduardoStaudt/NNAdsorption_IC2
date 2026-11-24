"""Known limitation: flow reversal (v < 0) inside the bed, IC1 seed 21 (docs/known_limitations.md).

The MOL solver of this repository breaks on seed 21 at t ~ 86 s, right after the lowest velocity
in the bed falls below zero. This script
  1. integrates the MOL up to just before the break and keeps v(z, t) and T(z, t) (cached in
     data/, provisional MOL data: a diagnostic, not training data);
  2. predicts v(z, t)/v_s and T(z, t) of the same case with a trained PINN;
  3. plots the PINN velocity field, and the lowest velocity and the coldest/hottest temperature
     in the bed of both.

Run inside WSL2, from the repository root (single-threaded BLAS for the MOL part):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/ic2env/bin/python \\
        scripts/plot_flow_reversal.py --run runs/phase3c_long --seed 21
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2 import runtime  # noqa: E402  (must precede TensorFlow)
from nnads_ic2.config import load_config  # noqa: E402
from nnads_ic2.data.domain import Domain  # noqa: E402


def mol_until(p, cfg, t_stop, dt_out):
    """MOL solution (dataset settings) on [0, t_stop]: v at the cell faces and T at the cell centres."""
    from scipy.integrate import solve_ivp

    from nnads_ic2.physics.mol_solver import MOLSolver
    from nnads_ic2.physics.scales import compute_scales

    sc = cfg["solver"]
    t_final = compute_scales(p, **dict(cfg["scales"], dT_ratio=None)).t_final
    s = MOLSolver(p, nx=int(sc["nx"]), scheme=sc["scheme"], fp_iters=int(sc["fp_iters"]))
    a = float(sc["atol_rel"])
    atol = np.concatenate([np.full(s.n * s.nx, a * s.C_F), np.repeat(a * s.q_scale, s.nx), np.full(s.nx, a * p.Tin),
                           np.full(s.n, a * s.C_F * p.vs * t_final)])
    t_eval = np.arange(0.0, t_stop + 1e-9, dt_out)
    sol = solve_ivp(s.rhs, (0.0, t_stop), s.y0, method=sc["method"], rtol=float(sc["rtol"]), atol=atol, t_eval=t_eval)
    v, T = [], []
    for k in range(sol.t.size):
        c, q, Tk, _ = s.unpack(sol.y[:, k])
        v.append(s.core(c, q, Tk)[3])  # velocity consistent with dT/dt (Eq. 5), at the faces
        T.append(Tk.copy())
    return dict(t=sol.t, z_faces=s.z_faces / p.L, z_cells=s.z / p.L, v_over_vs=np.array(v) / p.vs, T=np.array(T),
                success=np.array(bool(sol.success)), message=np.array(str(sol.message)),
                dataset_tag=np.array("provisório_MOL"))


def pinn_fields(run_dir, p, nz, nt):
    """PINN v/v_s and T [K] on a regular (z/L, t) grid over [0, t_F]."""
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    dtype = runtime.setup_tf(cfg["runtime"])
    from nnads_ic2.eval.predict import predict_dimless
    from nnads_ic2.experiment import case_table, make_model
    from nnads_ic2.physics.params import ThetaScaler

    with open(os.path.join(run_dir, "theta_scaler.json"), encoding="utf-8") as f:
        scaler = ThetaScaler.from_dict(json.load(f))
    table, scales, _, _ = case_table([p], cfg, dtype, scaler=scaler)
    s = scales[0]
    model = make_model(cfg, p.n_comp)
    model.load_weights(os.path.join(run_dir, "model.weights.h5"))
    z = np.linspace(0.0, 1.0, nz)
    t = np.linspace(0.0, s.t_final / s.t_ref, nt)
    TT, ZZ = np.meshgrid(t, z, indexing="ij")
    d = predict_dimless(model, table, 0, ZZ.ravel(), TT.ravel(), int(cfg["eval"]["batch_size"]))
    return dict(t=t * s.t_ref, z=z, v_over_vs=d["v"][:, 0].reshape(nt, nz),
                T=s.Tin + s.dT_ref * d["T"][:, 0].reshape(nt, nz), t_final=s.t_final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trained stage (c) run (whole domain)")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--t-mol", type=float, default=85.5, help="MOL end time [s] (seed 21 breaks at 85.7 s)")
    ap.add_argument("--dt-mol", type=float, default=0.5, help="MOL output interval [s]")
    ap.add_argument("--t-zoom", type=float, default=150.0, help="time range of the line plots [s]")
    ap.add_argument("--out", default=None, help="figure path (default docs/figures/limitations/flow_reversal_seed<k>.png)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    p, ok, info = Domain(cfg, ROOT).sample(args.seed)
    cache = os.path.join(ROOT, "data", "provisorio_MOL_n2_hard_%d_until_%gs.npz" % (args.seed, args.t_mol))
    if not os.path.exists(cache):
        print("MOL up to %.1f s (cached in %s) ..." % (args.t_mol, cache), flush=True)
        np.savez_compressed(cache, **mol_until(p, cfg, args.t_mol, args.dt_mol))
    mol = dict(np.load(cache))
    pin = pinn_fields(run_dir, p, nz=101, nt=1201)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    im = ax[0].pcolormesh(pin["t"], pin["z"], pin["v_over_vs"].T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax[0], label="PINN v / v_s")
    t_end = float(mol["t"][-1])
    ax[0].axvline(t_end, ls="--", c="w", lw=1.0)
    ax[0].set(xlabel="t (s)", ylabel="z / L", title="PINN velocity field, seed %d (dashed: MOL breaks)" % args.seed)

    tz = pin["t"] <= args.t_zoom
    ax[1].plot(pin["t"][tz], pin["v_over_vs"][tz].min(axis=1), c="C3", label="PINN")
    ax[1].plot(mol["t"], mol["v_over_vs"].min(axis=1), "k.", ms=3, label="MOL (until %.1f s)" % t_end)
    ax[1].axhline(0.0, c="0.5", lw=0.8)
    ax[1].set_ylim(-2.4, 1.3)
    ax[1].set(xlabel="t (s)", ylabel="lowest v / v_s in the bed", title="lowest velocity in the bed")
    ax[1].legend()

    for name, fn, ls in (("coldest", np.min, "-"), ("hottest", np.max, "--")):
        ax[2].plot(pin["t"][tz], fn(pin["T"][tz], axis=1), c="C3", ls=ls, label="PINN, %s" % name)
        ax[2].plot(mol["t"], fn(mol["T"], axis=1), c="k", ls=ls, label="MOL, %s" % name)
    ax[2].axhline(p.Tin, c="0.5", lw=0.8)
    ax[2].set(xlabel="t (s)", ylabel="T (K)", title="coldest and hottest point of the bed")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(ROOT, "docs", "figures", "limitations", "flow_reversal_seed%d.png" % args.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)

    def at(t_s):
        i = int(np.argmin(np.abs(pin["t"] - t_s)))
        j = int(np.argmin(np.abs(mol["t"] - t_s)))
        return dict(t_s=t_s, pinn_v_min=float(pin["v_over_vs"][i].min()), mol_v_min=float(mol["v_over_vs"][j].min()),
                    pinn_T_min=float(pin["T"][i].min()), mol_T_min=float(mol["T"][j].min()),
                    pinn_T_max=float(pin["T"][i].max()), mol_T_max=float(mol["T"][j].max()))

    summary = dict(run=os.path.basename(run_dir), seed=args.seed, Tin=p.Tin, t_final_s=pin["t_final"],
                   mol_until_s=t_end, mol_success=bool(mol["success"]), mol_message=str(mol["message"]),
                   pinn_v_min_whole_run=float(pin["v_over_vs"].min()),
                   pinn_v_min_until_mol_end=float(pin["v_over_vs"][pin["t"] <= t_end].min()),
                   mol_v_min=float(mol["v_over_vs"].min()),
                   mol_first_negative_v_s=float(mol["t"][np.argmax(mol["v_over_vs"].min(axis=1) < 0)])
                   if np.any(mol["v_over_vs"] < 0) else None,
                   snapshots=[at(x) for x in (20.0, 60.0, 80.0, 83.0, 85.0)])
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
