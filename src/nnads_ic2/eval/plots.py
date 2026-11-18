"""Figures for the evaluation of one case."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_outlet(ref, pred, names, path):
    """Outlet mole fractions, temperature and velocity: reference (lines) vs PINN (dashed).

    ref = None: PINN only (cases without a solver reference)."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, nm in enumerate(names):
        if ref is not None:
            ax[0].plot(ref["t"], ref["y_out"][:, i], "-", color=colors[i], lw=1.8, label="reference %s" % nm)
        ax[0].plot(pred["t"], pred["y_out"][:, i], "--", color=colors[i], lw=1.5, label="PINN %s" % nm)
    if ref is not None:
        ax[1].plot(ref["t"], ref["T_out"], "-", color="k", lw=1.8, label="reference")
    ax[1].plot(pred["t"], pred["T_out"], "--", color="tab:red", lw=1.5, label="PINN")
    if ref is not None and ref.get("v_out") is not None:  # datasets may lack v(L, t)
        ax[2].plot(ref["t"], ref["v_out"], "-", color="k", lw=1.8, label="reference")
    ax[2].plot(pred["t"], pred["v_out"], "--", color="tab:red", lw=1.5, label="PINN")
    for a, lab in zip(ax, ["outlet mole fraction", "outlet T (K)", "outlet velocity (m/s)"]):
        a.set_xlabel("t (s)")
        a.set_ylabel(lab)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(csv_path, path):
    """Loss terms and loss weights along training."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    step = np.array([float(r["step"]) for r in rows])
    skip = {"step", "elapsed_s", "ms_per_step", "lr", "causal_eps", "causal_w_min", "pool_size",
            "rar_mean_resid", "rar_added_resid", "theta_resamples"}
    terms = [k for k in rows[0] if k not in skip and not k.startswith("lambda_")]
    lambdas = [k for k in rows[0] if k.startswith("lambda_")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for k in terms:
        ax[0].semilogy(step, [max(float(r[k]), 1e-16) for r in rows], label=k)
    for k in lambdas:
        ax[1].semilogy(step, [max(float(r[k]), 1e-16) for r in rows], label=k)
    ax[1].semilogy(step, [max(float(r["causal_w_min"]), 1e-16) for r in rows], "k:", label="causal w_min")
    for a, title in zip(ax, ["loss terms", "weights"]):
        a.set_xlabel("step")
        a.set_title(title)
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fields(z, t, ref, pred, path):
    """(z~, t~) maps of each dimensionless field: reference, PINN and absolute error."""
    keys = [k for k in ref]
    fig, ax = plt.subplots(len(keys), 3, figsize=(13, 2.6 * len(keys)), squeeze=False)
    ext = [z.min(), z.max(), t.min(), t.max()]
    for r, k in enumerate(keys):
        lo, hi = min(ref[k].min(), pred[k].min()), max(ref[k].max(), pred[k].max())
        for c, (data, title) in enumerate([(ref[k], "reference"), (pred[k], "PINN"), (np.abs(pred[k] - ref[k]), "|error|")]):
            kw = dict(vmin=lo, vmax=hi) if c < 2 else {}
            im = ax[r, c].imshow(data, origin="lower", aspect="auto", extent=ext, cmap="viridis", **kw)
            ax[r, c].set_title("%s  %s" % (k, title), fontsize=9)
            ax[r, c].set_xlabel("z~")
            ax[r, c].set_ylabel("t~")
            fig.colorbar(im, ax=ax[r, c])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_collocation(pool, path):
    """Interior collocation pool; RAR points (red) gather near the fronts."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(pool.z[~pool.from_rar], pool.t[~pool.from_rar], s=0.3, c="0.6", label="initial (Sobol)")
    ax.scatter(pool.z[pool.from_rar], pool.t[pool.from_rar], s=0.6, c="tab:red", label="added by RAR")
    ax.set_xlabel("z~")
    ax.set_ylabel("t~")
    ax.legend(markerscale=10, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
