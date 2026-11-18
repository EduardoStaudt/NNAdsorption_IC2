"""Loss weights: fixed, or adapted by gradient norms; and the causal-training schedule."""

import numpy as np
import tensorflow as tf

GROUPS = ("data", "pde", "bc", "ic")


class LossWeights:
    """Group weights lambda_data, lambda_pde, lambda_bc, lambda_ic.

    method 'fixed': the config values are used as they are.
    method 'grad_norm': learning-rate annealing (Wang, Teng & Perdikaris, 2021). Every `every`
    steps, for each group k other than the reference (PDE):
        lambda_hat_k = max |grad(lambda_pde L_pde)| / mean |grad L_k|
        lambda_k    <- (1 - alpha) lambda_k + alpha lambda_hat_k
    A group whose gradients are small (e.g. boundary terms) gets a larger weight, so no term is
    ignored by the optimizer. Groups with initial weight 0 stay off.

    Every weight is capped at `max`; the data weight at `max_data` (lower: with the cap of 100 the
    Phase 1 hybrid fitted its training outlets and generalized worse). Groups listed in `fixed`
    keep their initial weight (e.g. fixed: [data] -> lambda_data constant, no annealing).
    """

    def __init__(self, loss_cfg, dtype):
        init = loss_cfg["weights"]
        ad = loss_cfg["adaptive"]
        self.method = ad["method"]
        if self.method not in ("fixed", "grad_norm"):
            raise ValueError("adaptive.method must be 'fixed' or 'grad_norm'")
        self.every = int(ad["every"])
        self.alpha = float(ad["alpha"])
        self.lam_min, self.lam_max = float(ad["min"]), float(ad["max"])
        self.lam_max_group = {"data": float(ad.get("max_data", self.lam_max))}
        self.fixed = tuple(ad.get("fixed") or ())
        unknown = set(self.fixed) - set(GROUPS)
        if unknown:
            raise ValueError("adaptive.fixed: unknown groups %s" % sorted(unknown))
        self.ref = "pde"
        self.lam = {k: tf.Variable(float(init[k]), trainable=False, dtype=dtype, name="lambda_" + k) for k in GROUPS}
        self.active = tuple(k for k in GROUPS if float(init[k]) > 0)

    @property
    def adaptive(self):
        return self.method == "grad_norm"

    def update(self, stats):
        """stats[group] = (max |grad|, mean |grad|) of the unweighted group loss."""
        if not all(np.all(np.isfinite(v)) for v in stats.values()):
            return  # a non-finite batch: keep the current weights
        ref_max = float(stats[self.ref][0]) * float(self.lam[self.ref].numpy())
        for k in self.active:
            if k == self.ref or k in self.fixed or k not in stats:
                continue
            lam_hat = ref_max / max(float(stats[k][1]), 1e-30)
            new = (1.0 - self.alpha) * float(self.lam[k].numpy()) + self.alpha * lam_hat
            self.lam[k].assign(min(max(new, self.lam_min), self.lam_max_group.get(k, self.lam_max)))

    def as_dict(self):
        return {k: float(v.numpy()) for k, v in self.lam.items()}


class CausalSchedule:
    """eps of the causal weights (losses.causal_weights).

    Small eps: all times count almost equally. Large eps: strict time marching. When even the
    last time slice has weight above `w_min_threshold`, eps is multiplied by `growth` (up to
    `eps_max`), as in Wang, Sankaran & Perdikaris (2022).
    """

    def __init__(self, causal_cfg, dtype):
        self.enabled = bool(causal_cfg["enabled"])
        self.eps = tf.Variable(float(causal_cfg["eps"]), trainable=False, dtype=dtype, name="causal_eps")
        self.growth = float(causal_cfg["growth"])
        self.eps_max = float(causal_cfg["eps_max"])
        self.w_min_threshold = float(causal_cfg["w_min_threshold"])

    def update(self, w_min):
        eps = float(self.eps.numpy())
        if self.enabled and w_min > self.w_min_threshold and eps < self.eps_max:
            self.eps.assign(min(eps * self.growth, self.eps_max))
