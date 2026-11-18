"""Training loop.

Adam with cosine learning-rate decay, then optional L-BFGS (tensorflow-probability) on a fixed
batch. Each step draws interior, inlet, outlet (and, if used, initial/data) points; every
`adaptive.every` steps the loss weights are re-balanced; every `rar.every` steps RAR adds
collocation points; with a `resampler` (training/theta_colloc.py) the collocation theta are
replaced every `resampler.every_steps` steps. Every term of the loss, the weights and the time
per step go to a CSV file (and TensorBoard if enabled); checkpoints every `ckpt_every` steps.
An optional `evaluator(step)` runs every `eval_every` steps (validation curve); its time is kept
out of the logged ms/step and reported apart (timing["periodic_eval_s"]).
"""

import csv
import os
import time

import numpy as np
import tensorflow as tf

from ..data.sampling import boundary_points, initial_points, select_largest
from .losses import PDE_TERMS, LossBuilder
from .weighting import CausalSchedule, LossWeights


class Trainer:
    def __init__(self, cfg, model, table, pool, run_dir, data=None, resampler=None, evaluator=None, eval_every=0):
        self.cfg = cfg
        self.model = model
        self.pool = pool
        self.resampler = resampler
        self.evaluator = evaluator
        self.eval_every = int(eval_every or 0)
        self.run_dir = run_dir
        self.dtype = tf.keras.backend.floatx()
        self.rng = np.random.default_rng(int(cfg["runtime"]["seed"]) + 1)
        tcfg, scfg = cfg["train"], cfg["sampling"]
        self.steps = int(tcfg["steps"])
        self.sizes = dict(
            interior=int(scfg["n_interior"]),
            boundary=int(scfg["n_boundary"]),
            initial=int(scfg["n_initial"]),
            data=int(scfg["n_data"]),
        )
        self.weights = LossWeights(cfg["loss"], self.dtype)
        self.causal = CausalSchedule(cfg["loss"]["causal"], self.dtype)
        # a pool covering only part of [0, t~_F] is one window of time marching: causal slices over it
        partial = pool.t_start > 0 or pool.t_final < float(cfg["scales"]["t_final_factor"])
        window = (pool.t_start, pool.t_final) if partial else None
        self.losses = LossBuilder(model, table, cfg["loss"], cfg["physics"]["isotherm_p_min_atm"], t_window=window)
        self.data = {k: v for k, v in (data or {}).items() if v is not None}
        self.use_data = bool(self.data) and float(cfg["loss"]["weights"]["data"]) > 0
        self.use_ic = (not cfg["model"]["hard_ic"]) and float(cfg["loss"]["weights"]["ic"]) > 0
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            float(tcfg["lr"]), self.steps, alpha=float(tcfg["lr_final_frac"])
        )
        self.opt = tf.keras.optimizers.Adam(schedule)
        self._train_fn = tf.function(self._train_step, jit_compile=bool(tcfg["jit_compile"]))
        self._stats_fn = tf.function(self._grad_stats)
        self._resid_fn = tf.function(self._pointwise_residual)
        self._front_fn = tf.function(self.losses.front_weight)
        self.step_var = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.skipped = tf.Variable(0, dtype=tf.int64, trainable=False)  # steps with a non-finite loss/gradient
        ckpt = tf.train.Checkpoint(model=model, optimizer=self.opt, step=self.step_var)
        self.ckpt_mgr = tf.train.CheckpointManager(
            ckpt, os.path.join(run_dir, "ckpt"), max_to_keep=int(cfg["logging"]["keep_checkpoints"])
        )
        self.timing = {}

    # ---- batches ------------------------------------------------------------------
    def _col(self, a):
        return tf.constant(np.asarray(a).reshape(-1, 1), self.dtype)

    def _idx(self, a):
        return tf.constant(np.asarray(a), tf.int32)

    def make_batch(self, n_interior=None):
        n_int = n_interior or self.sizes["interior"]
        z, t, case = self.pool.sample(n_int)
        b = dict(z=self._col(z), t=self._col(t), case=self._idx(case))
        nb, t0, t1, nc = self.sizes["boundary"], self.pool.t_start, self.pool.t_final, self.pool.n_cases
        zi, ti, ci = boundary_points(self.rng, nb, 0.0, t1, nc, t_start=t0)
        zo, to, co = boundary_points(self.rng, nb, 1.0, t1, nc, t_start=t0)
        b.update(z_in=self._col(zi), t_in=self._col(ti), case_in=self._idx(ci))
        b.update(z_out=self._col(zo), t_out=self._col(to), case_out=self._idx(co))
        if self.use_ic:
            z0, t0, c0 = initial_points(self.rng, self.sizes["initial"], nc)
            b.update(z_ic=self._col(z0), t_ic=self._col(t0), case_ic=self._idx(c0))
        if self.use_data:
            for kind, d in self.data.items():
                k = self.rng.integers(0, d["z"].size, self.sizes["data"])
                for name, arr in d.items():
                    key = "%s_%s" % (kind, name)
                    b[key] = self._idx(arr[k]) if name == "case" else tf.constant(arr[k].reshape(k.size, -1), self.dtype)
        return b

    # ---- loss -------------------------------------------------------------------
    def _all_terms(self, b):
        terms, w_bins, _ = self.losses.interior(b["z"], b["t"], b["case"], self.causal.eps)
        terms.update(self.losses.boundary(b["z_in"], b["t_in"], b["case_in"], b["z_out"], b["t_out"], b["case_out"]))
        groups = dict(
            pde=tf.add_n([self.losses.pde_w[k] * terms[k] for k in PDE_TERMS]),
            bc=terms["inlet"] + terms["outlet"],
        )
        if self.use_ic:
            terms["ic"] = self.losses.initial(b["z_ic"], b["t_ic"], b["case_ic"])
            groups["ic"] = terms["ic"]
        if self.use_data:
            parts = []
            if "outlet" in self.data:
                terms["data_outlet"] = self.losses.data_outlet(
                    b["outlet_z"], b["outlet_t"], b["outlet_case"], b["outlet_y"], b["outlet_T"]
                )
                parts.append(terms["data_outlet"])
            if "profiles" in self.data:
                terms["data_profiles"] = self.losses.data_profiles(
                    b["profiles_z"], b["profiles_t"], b["profiles_case"],
                    b["profiles_c"], b["profiles_q"], b["profiles_T"], b["profiles_v"],
                )
                parts.append(terms["data_profiles"])
            groups["data"] = tf.add_n(parts)
        return terms, groups, w_bins

    def _total(self, groups):
        return tf.add_n([self.weights.lam[k] * groups[k] for k in groups])

    def _train_step(self, b):
        with tf.GradientTape() as tape:
            terms, groups, w_bins = self._all_terms(b)
            total = self._total(groups)
        grads = tape.gradient(total, self.model.trainable_variables)
        # one non-finite point is enough to turn every weight into NaN: such a step gets a zero
        # gradient (Adam then only coasts on its momentum) and is counted
        finite = tf.math.is_finite(total)
        for g in grads:
            finite = tf.logical_and(finite, tf.reduce_all(tf.math.is_finite(g)))
        grads = [tf.where(finite, g, tf.zeros_like(g)) for g in grads]
        self.skipped.assign_add(tf.cast(tf.logical_not(finite), tf.int64))
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return total, terms, tf.reduce_min(w_bins)

    def _grad_stats(self, b):
        """max and mean |gradient| of each (unweighted) group loss, for LossWeights.update."""
        with tf.GradientTape(persistent=True) as tape:
            _, groups, _ = self._all_terms(b)
        out = {}
        for k, L in groups.items():
            grads = [g for g in tape.gradient(L, self.model.trainable_variables) if g is not None]
            flat = tf.concat([tf.reshape(tf.abs(g), [-1]) for g in grads], axis=0)
            out[k] = tf.stack([tf.reduce_max(flat), tf.reduce_mean(flat)])
        del tape
        return out

    def _pointwise_residual(self, z, t, case):
        return self.losses.interior(z, t, case, self.causal.eps)[2]

    # ---- RAR, front sampling -------------------------------------------------------
    def _chunked(self, fn, z, t, case, chunk):
        """fn at many points, in chunks of a fixed size (the last one padded: no retracing)."""
        out = np.empty(z.size)
        for i in range(0, z.size, chunk):
            sl = slice(i, min(i + chunk, z.size))
            m = sl.stop - sl.start
            pad = lambda a: np.concatenate([a[sl], np.repeat(a[sl][-1:], chunk - m)])  # noqa: E731
            out[sl] = fn(self._col(pad(z)), self._col(pad(t)), self._idx(pad(case))).numpy()[:m]
        return out

    def front_weights(self, z, t, case):
        """sum_i |dc~_i/dt~| of the current network (theta_colloc front sampling)."""
        return self._chunked(self._front_fn, z, t, case, 4 * self.sizes["interior"])

    def rar(self):
        rc = self.cfg["sampling"]["rar"]
        n_cand, n_add = int(rc["n_candidates"]), int(rc["n_add"])
        z, t, case = self.pool.candidates(n_cand)
        vals = self._chunked(self._resid_fn, z, t, case, self.sizes["interior"])
        idx = select_largest(vals, n_add)
        self.pool.add(z[idx], t[idx], case[idx])
        return float(vals.mean()), float(vals[idx].mean())

    # ---- main loop ------------------------------------------------------------------
    def run(self):
        os.makedirs(self.run_dir, exist_ok=True)
        lc, rc = self.cfg["logging"], self.cfg["sampling"]["rar"]
        tb = None
        if lc["tensorboard"]:
            try:
                import tensorboard  # noqa: F401  (tf.summary needs the package; it is optional)

                tb = tf.summary.create_file_writer(os.path.join(self.run_dir, "tb"))
            except ImportError:
                print("tensorboard not installed: logging to CSV only (pip install tensorboard)", flush=True)
        log_file = open(os.path.join(self.run_dir, "train_log.csv"), "w", newline="", encoding="utf-8")
        writer = None
        t_start = time.time()
        t_last, steps_since = t_start, 0
        rar_info = (float("nan"), float("nan"))
        for step in range(1, self.steps + 1):
            total, terms, w_min = self._train_fn(self.make_batch())
            steps_since += 1
            if step == 1:
                self.timing["first_step_s"] = time.time() - t_start  # includes graph tracing
                t_last, steps_since = time.time(), 0
            if self.weights.adaptive and step % self.weights.every == 0:
                stats = {k: v.numpy() for k, v in self._stats_fn(self.make_batch()).items()}
                self.weights.update(stats)
            if rc["enabled"] and step % int(rc["every"]) == 0 and step < self.steps:
                rar_info = self.rar()
            if self.resampler is not None and self.resampler.every_steps > 0 and step % self.resampler.every_steps == 0                     and step < self.steps:
                self.resampler.resample(weight_fn=self.front_weights)
            if step % int(lc["every"]) == 0 or step == self.steps:
                w_min_v = float(w_min)
                self.causal.update(w_min_v)
                now = time.time()
                row = dict(
                    step=step,
                    elapsed_s=round(now - t_start, 3),
                    ms_per_step=round(1000.0 * (now - t_last) / max(steps_since, 1), 3),
                    lr=float(self.opt.learning_rate),
                    total=float(total),
                )
                row.update({k: float(v) for k, v in terms.items()})
                row.update({"lambda_" + k: v for k, v in self.weights.as_dict().items()})
                row.update(causal_eps=float(self.causal.eps.numpy()), causal_w_min=w_min_v, pool_size=self.pool.size,
                           rar_mean_resid=rar_info[0], rar_added_resid=rar_info[1],
                           skipped_steps=int(self.skipped.numpy()))
                if self.resampler is not None:
                    row["theta_resamples"] = self.resampler.n_resamples
                if writer is None:
                    writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                log_file.flush()
                if tb is not None:
                    with tb.as_default(step=step):
                        for k, v in row.items():
                            if k != "step":
                                tf.summary.scalar(k, v)
                print(
                    "step %6d | %.1f ms/step | total %.3e | %s | lam_bc %.2g | eps %.2g w_min %.3f | pool %d"
                    % (step, row["ms_per_step"], row["total"],
                       " ".join("%s %.2e" % (k, row[k]) for k in terms), row["lambda_bc"],
                       row["causal_eps"], w_min_v, self.pool.size),
                    flush=True,
                )
                t_last, steps_since = now, 0
            if step % int(lc["ckpt_every"]) == 0:
                self.step_var.assign(step)
                self.ckpt_mgr.save()
            if self.evaluator is not None and self.eval_every > 0 and step % self.eval_every == 0:
                t0 = time.time()
                self.evaluator(step)
                spent = time.time() - t0
                self.timing["periodic_eval_s"] = self.timing.get("periodic_eval_s", 0.0) + spent
                t_last += spent  # not part of the next logged ms/step
        self.timing["adam_s"] = time.time() - t_start
        log_file.close()
        if self.cfg["train"]["lbfgs"]["enabled"]:
            t0 = time.time()
            self.timing["lbfgs"] = self.run_lbfgs()
            self.timing["lbfgs_s"] = time.time() - t0
        self.timing["train_s"] = time.time() - t_start
        self.step_var.assign(self.steps)
        self.ckpt_mgr.save()
        self.model.save_weights(os.path.join(self.run_dir, "model.weights.h5"))
        return self.timing

    # ---- L-BFGS -------------------------------------------------------------------
    def run_lbfgs(self):
        """Quasi-Newton refinement on one fixed (large) batch, with the final loss weights."""
        import tensorflow_probability as tfp

        lcfg = self.cfg["train"]["lbfgs"]
        batch = self.make_batch(n_interior=int(lcfg["n_interior"]))
        # Causal weights depend on the current residuals; L-BFGS line searches need a fixed
        # objective, so all times get weight 1 here (eps = 0).
        self.causal.eps.assign(0.0)
        variables = self.model.trainable_variables
        shapes = [v.shape for v in variables]
        sizes = [int(np.prod(s)) for s in shapes]

        def assign(flat):
            for v, part, shape in zip(variables, tf.split(flat, sizes), shapes):
                v.assign(tf.reshape(part, shape))

        @tf.function
        def value_and_gradients(flat):
            assign(flat)
            with tf.GradientTape() as tape:
                _, groups, _ = self._all_terms(batch)
                total = self._total(groups)
            grads = tape.gradient(total, variables)
            return total, tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

        start = tf.concat([tf.reshape(v, [-1]) for v in variables], axis=0)
        f0 = float(value_and_gradients(start)[0])
        res = tfp.optimizer.lbfgs_minimize(
            value_and_gradients, initial_position=start, max_iterations=int(lcfg["max_iter"]),
            tolerance=float(lcfg["tolerance"]),
        )
        assign(res.position)
        info = dict(loss_before=f0, loss_after=float(res.objective_value), iterations=int(res.num_iterations),
                    converged=bool(res.converged))
        print("L-BFGS: loss %.3e -> %.3e in %d iterations" % (f0, info["loss_after"], info["iterations"]), flush=True)
        return info
