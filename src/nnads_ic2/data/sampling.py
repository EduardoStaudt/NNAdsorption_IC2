"""Collocation points (where the residuals are evaluated) and residual-based adaptive refinement.

Interior points live in (z~, t~) in (0, 1) x (0, t~_F) and carry the index of the case they
belong to. The pool starts quasi-random (Sobol: evenly spread, no clusters or holes).

RAR (Lu et al., 2021): every K steps the residual is evaluated on many random candidates and the
worst ones join the pool. They pile up near the moving fronts, where the solution is steep and
uniform points are too sparse.
"""

import numpy as np
from scipy.stats import qmc


def sobol_unit(n, dim, seed):
    """n scrambled Sobol points in [0, 1)^dim."""
    m = int(np.ceil(np.log2(max(n, 2))))
    return qmc.Sobol(d=dim, scramble=True, seed=seed).random_base2(m)[:n]


class CollocationPool:
    """Interior collocation points (z~, t~, case) with t~ in [t_start, t_final] (t_start > 0: one
    window of time marching)."""

    def __init__(self, n_init, n_cases, t_final, max_size, seed, t_start=0.0):
        self.rng = np.random.default_rng(seed)
        u = sobol_unit(n_init, 2, seed)
        self.n_init = int(n_init)
        self.n_cases = int(n_cases)
        self.t_start = float(t_start)
        self.t_final = float(t_final)
        self.max_size = int(max_size)
        self.z = u[:, 0]
        self.t = self._times(u[:, 1])
        self.case = self.rng.integers(0, self.n_cases, n_init)
        self.from_rar = np.zeros(n_init, dtype=bool)

    @property
    def size(self):
        return self.z.size

    def _times(self, u):
        """[0, 1) -> [t_start, t_final)."""
        return self.t_start + u * (self.t_final - self.t_start)

    def sample(self, n):
        """Random mini-batch of interior points (with replacement)."""
        idx = self.rng.integers(0, self.size, n)
        return self.z[idx], self.t[idx], self.case[idx]

    def candidates(self, n):
        """Fresh uniform random points to be screened by RAR."""
        return (
            self.rng.uniform(0.0, 1.0, n),
            self.rng.uniform(self.t_start, self.t_final, n),
            self.rng.integers(0, self.n_cases, n),
        )

    def add(self, z, t, case):
        """Add RAR points; above max_size, drop random initial (non-RAR) points first."""
        self.z = np.concatenate([self.z, z])
        self.t = np.concatenate([self.t, t])
        self.case = np.concatenate([self.case, case])
        self.from_rar = np.concatenate([self.from_rar, np.ones(z.size, dtype=bool)])
        overflow = self.size - self.max_size
        if overflow > 0:
            pick_from = np.flatnonzero(~self.from_rar)
            if pick_from.size < overflow:
                pick_from = np.arange(self.size)
            drop = self.rng.choice(pick_from, overflow, replace=False)
            keep = np.ones(self.size, dtype=bool)
            keep[drop] = False
            self.z, self.t, self.case, self.from_rar = self.z[keep], self.t[keep], self.case[keep], self.from_rar[keep]


    def replace_initial(self, z, t, case):
        """Put the given points in place of the first z.size initial (non-RAR) points."""
        idx = np.flatnonzero(~self.from_rar)[: z.size]
        self.z[idx], self.t[idx], self.case[idx] = z[: idx.size], t[: idx.size], case[: idx.size]

    def renew(self, seed, keep):
        """New quasi-random initial points (new Sobol scramble, random case for each), plus the
        current points where `keep` is True (e.g. RAR points of cases whose theta did not change)."""
        u = sobol_unit(self.n_init, 2, seed)
        self.z = np.concatenate([u[:, 0], self.z[keep]])
        self.t = np.concatenate([self._times(u[:, 1]), self.t[keep]])
        self.case = np.concatenate([self.rng.integers(0, self.n_cases, self.n_init), self.case[keep]])
        self.from_rar = np.concatenate([np.zeros(self.n_init, dtype=bool), self.from_rar[keep]])


def select_largest(values, n):
    """Indices of the n largest values (RAR selection)."""
    n = min(int(n), values.size)
    return np.argpartition(-values, n - 1)[:n]


def boundary_points(rng, n, z_value, t_final, n_cases, t_start=0.0):
    """n points on the line z~ = z_value (0: inlet, 1: outlet), uniform in time over [t_start, t_final]."""
    return np.full(n, float(z_value)), rng.uniform(t_start, t_final, n), rng.integers(0, n_cases, n)


def initial_points(rng, n, n_cases):
    """n points at t~ = 0 (only used when the initial state is a soft loss term)."""
    return rng.uniform(0.0, 1.0, n), np.zeros(n), rng.integers(0, n_cases, n)
