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

    @property
    def n_theta_seen(self):
        return self.n_theta * (self.n_resamples + 1)

    def resample(self):
        """New Sobol theta for rows n_fixed.. and new collocation points."""
        self.theta_cases = self.sampler.draw(self.n_theta)
        rows, _, _ = case_rows(self.theta_cases, self.cfg, self.scaler)
        new = stack_arrays(self.fixed_rows + rows)
        for key, var in self.table.items():
            var.assign(new[key].astype(var.dtype.as_numpy_dtype))
        self.n_resamples += 1
        keep = self.pool.from_rar & (self.pool.case < self.n_fixed)
        self.pool.renew(self.seed + self.n_resamples, keep)

    def info(self):
        return dict(n_fixed=self.n_fixed, n_theta=self.n_theta, every_steps=self.every_steps,
                    n_resamples=self.n_resamples, n_theta_seen=self.n_theta_seen, pool_init=self.pool.n_init,
                    sobol_drawn=self.sampler.n_drawn, sobol_rejected=self.sampler.n_rejected)
