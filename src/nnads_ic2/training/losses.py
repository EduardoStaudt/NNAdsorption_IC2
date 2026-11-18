"""Loss terms of the hybrid PINN.

    L = lambda_data L_data
      + lambda_pde (w_mass L_mass + w_ldf L_ldf + w_energy L_energy + w_global L_global)
      + lambda_bc (L_inlet + L_outlet)
      + lambda_ic L_ic

Each term is a mean of squared dimensionless residuals (physics/residuals.py) or, for
L_data, of squared differences to solver values. L_global holds Eq. (5) and the closure
sum(c~) = P/(RT). Derivatives in z~ and t~ come from nested GradientTapes.
"""

import tensorflow as tf

from ..physics.residuals import gather_cases, inlet_residuals, outlet_residuals, pde_residuals

PDE_TERMS = ("mass", "ldf", "energy", "global")


def model_inputs(z, t, g):
    return dict(z=z, t=t, theta=g["theta"], c_ic=g["c_init"], q_ic=g["q_init"])


def compute_fields(model, z, t, g):
    """Network outputs and the derivatives the residuals need, by automatic differentiation.

    inner tape: first derivatives (d/dz~, d/dt~); outer tape: d2/dz~2 of c~ and T~.
    Gradients are taken column by column, so each one is the derivative of one output.
    """
    with tf.GradientTape(persistent=True) as outer:
        outer.watch(z)
        with tf.GradientTape(persistent=True) as inner:
            inner.watch([z, t])
            out = model(model_inputs(z, t, g))
            c_cols = tf.unstack(out["c"], axis=1)
            q_cols = tf.unstack(out["q"], axis=1)
            T, v = out["T"], out["v"]
        c_z_cols = [inner.gradient(ci, z) for ci in c_cols]
        T_z = inner.gradient(T, z)
    fields = dict(
        c=out["c"],
        c_t=tf.concat([inner.gradient(ci, t) for ci in c_cols], axis=1),
        c_z=tf.concat(c_z_cols, axis=1),
        c_zz=tf.concat([outer.gradient(ci_z, z) for ci_z in c_z_cols], axis=1),
        q=out["q"],
        q_t=tf.concat([inner.gradient(qi, t) for qi in q_cols], axis=1),
        T=T,
        T_t=inner.gradient(T, t),
        T_z=T_z,
        T_zz=outer.gradient(T_z, z),
        v=v,
        v_z=inner.gradient(v, z),
    )
    del inner, outer
    return fields


def compute_boundary_fields(model, z, t, g):
    """Outputs and first z-derivatives of c~ and T~ (all the boundary conditions need)."""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        out = model(model_inputs(z, t, g))
        c_cols = tf.unstack(out["c"], axis=1)
        T = out["T"]
    fields = dict(c=out["c"], c_z=tf.concat([tape.gradient(ci, z) for ci in c_cols], axis=1), T=T, T_z=tape.gradient(T, z))
    del tape
    return fields


def causal_weights(t, t_final, r2, n_bins, eps):
    """Causal training (Wang, Sankaran & Perdikaris, 2022).

    Time is cut into n_bins slices. Slice j gets weight w_j = exp(-eps * sum_{k<j} L_k), where
    L_k is the mean residual of slice k: later times only count once the earlier ones are solved,
    as in a time-marching solver. Weights carry no gradient.
    """
    bins = tf.cast(tf.floor(t[:, 0] / t_final[:, 0] * n_bins), tf.int32)
    bins = tf.clip_by_value(bins, 0, n_bins - 1)
    L = tf.math.unsorted_segment_mean(r2, bins, n_bins)
    w_bins = tf.stop_gradient(tf.exp(-eps * tf.cumsum(L, exclusive=True)))
    return tf.gather(w_bins, bins), w_bins


class LossBuilder:
    def __init__(self, model, table, loss_cfg, p_min):
        self.model = model
        self.table = table  # case constants, [n_cases, k] tensors (includes normalized theta)
        self.pde_w = {k: float(loss_cfg["pde_terms"][k]) for k in PDE_TERMS}
        self.causal_on = bool(loss_cfg["causal"]["enabled"])
        self.n_bins = int(loss_cfg["causal"]["n_bins"])
        self.p_min = float(p_min)
        # Cauchy (robust) loss on the PDE residual: rho(s) = c log(1 + s/c) of the per-point weighted
        # squared residual s. Equal to s for s << c; a point with s >> c keeps a bounded influence.
        # null or 0 = plain mean squares (Phases 0-1, stages a and b of Phase 2).
        self.robust_c = float(loss_cfg.get("robust_c") or 0.0)

    def interior(self, z, t, case, eps):
        """PDE terms at interior points. Also returns per-point squared residual (for RAR)."""
        g = gather_cases(self.table, case)
        f = compute_fields(self.model, z, t, g)
        res = pde_residuals(f, g, self.p_min)
        r2 = dict(
            mass=tf.reduce_mean(res["mass"] ** 2, axis=1),
            ldf=tf.reduce_mean(res["ldf"] ** 2, axis=1),
            energy=res["energy"][:, 0] ** 2,
            global_=res["global_"][:, 0] ** 2 + res["eos"][:, 0] ** 2,
        )
        r2["global"] = r2.pop("global_")
        total = tf.add_n([self.pde_w[k] * r2[k] for k in PDE_TERMS])
        if self.robust_c > 0:
            # d rho/ds = 1/(1 + s/c): each point's terms get this weight (no gradient through it), so
            # the gradient is that of mean(rho(s)); the causal slices see rho(s) instead of s
            c = self.robust_c
            r_w = tf.stop_gradient(1.0 / (1.0 + total / c))
            causal_input = c * tf.math.log1p(total / c)
        else:
            r_w, causal_input = tf.ones_like(total), total
        if self.causal_on:
            w, w_bins = causal_weights(t, g["t_final"], causal_input, self.n_bins, eps)
        else:
            w, w_bins = tf.ones_like(total), tf.ones((self.n_bins,), total.dtype)
        terms = {k: tf.reduce_mean(w * r_w * r2[k]) for k in PDE_TERMS}
        return terms, w_bins, total

    def boundary(self, z_in, t_in, case_in, z_out, t_out, case_out):
        g_in = gather_cases(self.table, case_in)
        rc, rT = inlet_residuals(compute_boundary_fields(self.model, z_in, t_in, g_in), g_in)
        g_out = gather_cases(self.table, case_out)
        oc, oT = outlet_residuals(compute_boundary_fields(self.model, z_out, t_out, g_out), g_out)
        return dict(
            inlet=tf.reduce_mean(tf.reduce_mean(rc**2, axis=1) + rT[:, 0] ** 2),
            outlet=tf.reduce_mean(tf.reduce_mean(oc**2, axis=1) + oT[:, 0] ** 2),
        )

    def initial(self, z, t0, case):
        """Soft initial condition (used only when the hard constraint is switched off)."""
        g = gather_cases(self.table, case)
        out = self.model(model_inputs(z, t0, g))
        return tf.reduce_mean(
            tf.reduce_mean((out["c"] - g["c_init"]) ** 2, axis=1)
            + tf.reduce_mean((out["q"] - g["q_init"]) ** 2, axis=1)
            + out["T"][:, 0] ** 2
            + (out["v"][:, 0] - 1.0) ** 2
        )

    def data_outlet(self, z, t, case, y_obs, T_obs):
        """Solver outlet data: mole fractions y_i = c_i / sum c_j and T~ at z~ = 1."""
        g = gather_cases(self.table, case)
        out = self.model(model_inputs(z, t, g))
        y = out["c"] / tf.reduce_sum(out["c"], axis=1, keepdims=True)
        return tf.reduce_mean(tf.reduce_mean((y - y_obs) ** 2, axis=1) + (out["T"][:, 0] - T_obs[:, 0]) ** 2)

    def data_profiles(self, z, t, case, c_obs, q_obs, T_obs, v_obs):
        """Solver internal profiles (when the dataset has them): all 2n+2 outputs."""
        g = gather_cases(self.table, case)
        out = self.model(model_inputs(z, t, g))
        return tf.reduce_mean(
            tf.reduce_mean((out["c"] - c_obs) ** 2, axis=1)
            + tf.reduce_mean((out["q"] - q_obs) ** 2, axis=1)
            + (out["T"][:, 0] - T_obs[:, 0]) ** 2
            + (out["v"][:, 0] - v_obs[:, 0]) ** 2
        )
