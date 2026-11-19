# Phase 4 results (determinism; stage (c) continued from the long run; provisional MOL data)

## Scope

- This is the last round before the v23b dataset: physics-only PINN, no MLP, no new MOL data.
  The validation numbers use the 20 stage (c) MOL validation seeds of Phases 2 and 3
  (provisional MOL data, not v23b).
- The advisor summary in Portuguese is [summary_pinn_ic2.md](summary_pinn_ic2.md).

## 1. Determinism

### Question

Phase 3 found that two runs of the same configuration and seeds drift apart: y R² 0.927 vs
0.903. Could `tf.config.experimental.enable_op_determinism()` fix this at an acceptable cost?
The rule: if the slowdown is at most 30 %, determinism is on by default; if not, it is an
option and comparisons keep two seeds.

### Seeds

Every random draw of the package already had a seed:

- the θ draws and the (z~, t~) collocation pool, which also draws the interior points of each
  batch and the RAR candidates: `theta_colloc.seed`;
- the Fourier features: `model.fourier.seed`;
- the inlet and outlet points of each step: `runtime.seed` + 1;
- the Keras initializers, through `tf.keras.utils.set_random_seed(runtime.seed)`, which also
  seeds Python and NumPy.

The new option `runtime.deterministic` adds `enable_op_determinism()` in
`runtime.setup_tf`.

### Test

- **Runs:** stage (c) physics (`configs/phase2_stage_c.yaml`), 5000 steps.
- **Each mode twice:** determinism off, then on; off, then on.
- **Comparison:** every logged value of `train_log.csv` (all loss terms and weights every 100
  steps) and the final validation y R².

| Mode | ms/step (median) | Two runs identical? |
|---|---|---|
| off | 9.00 / 9.01 | no: they differ from the first logged step (100); final total loss 0.19010 vs 0.19495 |
| on, first version | 146.5 | (stopped after one run) |
| on, after the fix below | 9.15 / 9.13 | **yes**: all 50 logged rows bit-identical, and the same validation y R² (0.301895374) |

### The fix

- **Cause of the slowdown.** The first deterministic run was 16 times slower. The causal weights
  computed the mean residual of each time slice with `tf.math.unsorted_segment_mean`. Under op
  determinism, XLA runs that scatter as a serial loop.
- **The change.** The slice means are now one-hot products (`training/losses.py`). The result is
  the same; a new test compares it with the segment means.
- **Cost now.** The deterministic run costs +1.5 % per step (9.14 vs 9.00 ms).

### Decision

- **Determinism on by default:** `runtime.deterministic: true` in `configs/pinn_base.yaml`.
  The cost is 1.5 %, against a limit of 30 %.
- **The same configuration and seed now gives the same result.** A run is reproducible exactly,
  and a difference between two configurations with the same seed is no longer floating-point
  noise.
- **Two seeds are still needed.** Determinism does not remove the spread between seeds: another
  `runtime.seed` gives another initial network and other boundary points. That spread reached
  0.015 within one corner of the Phase 3 ablation, so comparisons keep two seeds (item 2 below).
- **Older runs.** Runs made before this change are not bit-reproducible. Every configuration in
  `configs/` inherits the new default.

## 2. Stage (c) continued from the long run (two seeds each)

### Setup

- **Start.** Both runs start from the final weights of `runs/phase3c_long` (step 200 000, y R²
  0.932). `phase2.init_from` takes its network, dT_ratio and θ scaler; the Adam state is fresh.
- **Common to all.** Physics only, 50 000 steps, validation every 10 000 steps, deterministic
  kernels.
- **Seeds.** Seeds 0 and 1 (`runtime.seed`) change only the inlet and outlet points drawn at each
  step. The network starts from the same weights, and the θ, the (z~, t~) pool and the interior
  batches come from `theta_colloc.seed`, the same in all four runs.
- **(a) Second learning-rate cycle:** `configs/phase4_stage_c_lr_cycle.yaml`. The cosine decay
  restarts at lr 3e-4; the long run had ended at 1e-5.
- **(b) Twice the points per θ:** `configs/phase4_stage_c_points2x.yaml`. It is (a) with 1024
  (z~, t~) points per θ instead of 512, so the pool has 2.15 M points.
  - The θ are renewed every 5 passes over the pool. That is the same 2622 steps as before, so the
    same 40 960 θ are seen.
  - The learning rate of (b) was not specified. It uses the same restart as (a), so (a) and (b)
    differ only in the points per θ.

### Results (20 stage (c) validation seeds)

| Run | y R² | T_out RMSE (K) | v_out R² | tb | t50 | t90 | ts | c~_1 field | T field (K) | Moment gas 1, PINN vs ref | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| start: long run, 200 000 steps | 0.932 | 15.34 | 0.58 | 20.9 % | 10.5 % | 14.1 % | 25.3 % | 0.041 | 6.68 | 4.7 % | – |
| (a) lr cycle, seed 0 | 0.931 | 15.13 | 0.57 | 19.5 % | 10.1 % | 13.8 % | 20.4 % | 0.041 | 6.65 | 5.8 % | 0.132 |
| (a) lr cycle, seed 1 | 0.931 | 15.07 | 0.58 | 19.4 % | 10.0 % | 14.0 % | 21.9 % | 0.041 | 6.64 | 5.8 % | 0.132 |
| (b) 2× points, seed 0 | 0.933 | 15.06 | 0.58 | 19.3 % | 9.8 % | 13.6 % | 20.6 % | 0.040 | 6.62 | 5.6 % | 0.132 |
| (b) 2× points, seed 1 | 0.931 | 15.18 | 0.58 | 19.8 % | 10.8 % | 13.3 % | 24.7 % | 0.041 | 6.66 | 5.4 % | 0.132 |

*Times: mean |relative error|.*

**Validation y R² during the runs**

| Step | (a) seed 0 | (a) seed 1 | (b) seed 0 | (b) seed 1 |
|---|---|---|---|---|
| 0 (the long run) | 0.932 | 0.932 | 0.932 | 0.932 |
| 10 000 | 0.921 | 0.930 | 0.924 | 0.922 |
| 20 000 | 0.926 | 0.926 | 0.927 | 0.929 |
| 30 000 | 0.925 | 0.927 | 0.929 | 0.931 |
| 40 000 | 0.931 | 0.931 | 0.931 | 0.932 |
| 50 000 | 0.931 | 0.931 | 0.933 | 0.931 |

### Observations

- **Neither change moves y R².**

| Run | seed 0 | seed 1 | mean | start (long run) |
|---|---|---|---|---|
| (a) lr cycle | 0.9315 | 0.9313 | 0.931 | 0.9316 |
| (b) 2× points | 0.9333 | 0.9310 | 0.932 | 0.9316 |

  Both means are within 0.001 of the start.
- **The restart costs accuracy first.** At 10 000 steps, while the learning rate is high, y R²
  drops to 0.921–0.930. The runs then come back to where they started.
- **Small gains elsewhere, the same in all four runs:**
  - T_out RMSE 15.34 → 15.06–15.18 K;
  - tb error 20.9 → 19.3–19.8 %;
  - ts error 25.3 → 20.4–24.7 %.

  The moment of gas 1 gets worse: 4.7 → 5.4–5.8 %.
- **Small seed spread from a common start:** 0.0002 for (a) and 0.0022 for (b). It is much
  smaller than from scratch (up to 0.015 in Phase 3), because these runs start from the same
  network and differ only in the boundary points.
- **Conclusion.** The long run had converged for this network and loss. Another learning-rate
  cycle or twice the points per θ does not move it, and with this setup stage (c) stays at y R²
  ≈ 0.93. A further gain has to come from elsewhere: the v23b data and the hybrid fine-tuning,
  or a change of model or loss (not explored).

## 3. Cost

| Runs | GPU h |
|---|---|
| determinism tests: 7 short runs of 600–5000 steps (0.21 of it for the slow first deterministic run) | 0.30 |
| item 2: 4 runs of 50 000 steps | 0.53 |
| **total** | **0.82** |

There was no MOL computation.

## 4. Incidents

The first determinism queue was stopped after its deterministic run (146 ms/step). The code
then changed (one-hot slice means), so the run that had just started was killed: its directory
`runs/phase4_det_off_b` is incomplete and was kept. The test was repeated in full with the new
code (`phase4_det2_*`).
