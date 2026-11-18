"""Collocation at theta drawn from the domain (Phase 2).

The residuals of Eqs. (1)-(6) need no solver, so they can be imposed at any theta, not only at
the few cases that have solver data. The case table used in training has

    rows [0, n_fixed)                   the cases with data (or the base case of a subspace)
    rows [n_fixed, n_fixed + n_theta)   theta drawn by Sobol from the domain (data/domain.py)

and the collocation pool spreads `points_per_theta` (z~, t~) points per row on average. Every
`resample_every_epochs` passes over the pool, the Sobol rows get new theta: the Sobol sequence
continues, so the domain is covered more and more densely during training. The pool then gets new
(z~, t~) points; RAR points of the fixed rows are kept, those of the replaced rows are dropped
(they were chosen for the old theta). The table rows are tf.Variables, so the compiled training
step reads the new values without being traced again. resample_every_epochs = 0: the Sobol theta
are drawn once and kept for the whole run (fixed collocation theta, as the 50 cases of Phase 1).

Front sampling (Phase 3, `front_frac` > 0): at each renewal, that fraction of the new points is
put near the concentration fronts of the current network instead of uniformly. For every row,
`front_candidates` uniform candidates per point are screened and kept with probability
proportional to sum_i |dc~_i/dt~| (LossBuilder.front_weight), without replacement. Every row gets
the same number of front points, so a few theta with very steep fronts cannot take them all.
The rest of the pool stays uniform (Sobol); RAR is unchanged.
"""

import numpy as np

from ..data.sampling import CollocationPool
from ..experiment import case_rows
from ..physics.residuals import stack_arrays, stack_cases


class ThetaCollocation:
    def __init__(self, cfg, fixed_cases, sampler, scaler, dtype, steps):
        tc = cfg["theta_colloc"]
        self.cfg = cfg
        self.sampler = sampler
        self.scaler = scaler
        self.n_fixed = len(fixed_cases)
        self.n_theta = int(tc["n_theta"])
        self.seed = int(tc["seed"])
        self.fixed_rows, self.fixed_scales, _ = case_rows(fixed_cases, cfg, scaler)
        self.theta_cases = sampler.draw(self.n_theta)
        rows, _, _ = case_rows(self.theta_cases, cfg, scaler)
        self.table = stack_cases(self.fixed_rows + rows, dtype, variable=True)
        n_rows = self.n_fixed + self.n_theta
        n_init = n_rows * int(tc["points_per_theta"])
        rar = cfg["sampling"]["rar"]
        n_rar = int(rar["n_add"]) * (int(steps) // int(rar["every"]) + 1) if rar["enabled"] else 0
        self.pool = CollocationPool(n_init, n_rows, float(cfg["scales"]["t_final_factor"]), n_init + n_rar, self.seed)
        epoch = n_init / int(cfg["sampling"]["n_interior"])  # steps per pass over the pool
        every = float(tc["resample_every_epochs"] or 0)
        self.every_steps = max(1, int(round(every * epoch))) if every > 0 else 0  # 0: never renewed
        self.n_resamples = 0
        self.front_frac = float(tc.get("front_frac") or 0.0)
        self.front_candidates = int(tc.get("front_candidates") or 4)
        self.front_per_row = int(self.front_frac * n_init) // n_rows
        self.n_front_draws = 0

    @property
    def n_theta_seen(self):
        return self.n_theta * (self.n_resamples + 1)

    def resample(self, weight_fn=None):
        """New Sobol theta for rows n_fixed.. and new collocation points (part of them near the
        fronts when front_frac > 0 and `weight_fn(z, t, case)` gives the front weight)."""
        self.theta_cases = self.sampler.draw(self.n_theta)
        rows, _, _ = case_rows(self.theta_cases, self.cfg, self.scaler)
        new = stack_arrays(self.fixed_rows + rows)
        for key, var in self.table.items():
            var.assign(new[key].astype(var.dtype.as_numpy_dtype))
        self.n_resamples += 1
        keep = self.pool.from_rar & (self.pool.case < self.n_fixed)
        self.pool.renew(self.seed + self.n_resamples, keep)
        if self.front_per_row > 0 and weight_fn is not None:
            self.front_points(weight_fn)

    def front_points(self, weight_fn):
        """Replace front_per_row of the uniform points of every row by points near the fronts."""
        n_rows, k = self.n_fixed + self.n_theta, self.front_per_row
        m = k * self.front_candidates
        rng = np.random.default_rng(self.seed + 7919 * (self.n_resamples + 1))
        z = rng.uniform(0.0, 1.0, (n_rows, m))
        t = rng.uniform(0.0, self.pool.t_final, (n_rows, m))
        case = np.repeat(np.arange(n_rows), m).reshape(n_rows, m)
        w = np.asarray(weight_fn(z.ravel(), t.ravel(), case.ravel()), dtype=float).reshape(n_rows, m)
        w = np.where(np.isfinite(w), w, 0.0)
        w += 1e-6 * w.max(axis=1, keepdims=True) + 1e-30  # every candidate keeps a small chance
        keys = np.log(w) + rng.gumbel(size=w.shape)  # k largest keys = k draws without replacement, P ~ w
        pick = (np.arange(n_rows)[:, None], np.argpartition(-keys, k - 1, axis=1)[:, :k])
        self.pool.replace_initial(z[pick].ravel(), t[pick].ravel(), case[pick].ravel())
        self.n_front_draws += 1

    def info(self):
        return dict(n_fixed=self.n_fixed, n_theta=self.n_theta, every_steps=self.every_steps,
                    n_resamples=self.n_resamples, n_theta_seen=self.n_theta_seen, pool_init=self.pool.n_init,
                    front_frac=self.front_frac, front_per_row=self.front_per_row, n_front_draws=self.n_front_draws,
                    sobol_drawn=self.sampler.n_drawn, sobol_rejected=self.sampler.n_rejected)
