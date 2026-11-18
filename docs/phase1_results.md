# Phase 1 results (parametric PINN, provisional MOL data)

> **Provisional data.** Every number here comes from the small MOL solver of this repository
> (`physics/mol_solver.py`), not from the v23b solver. The dataset is tagged `provisório_MOL`
> (file name without the accent, attribute `dataset_tag`); keep these results apart from v23b runs.

## Dataset

File `data/provisorio_MOL_n2_60cases.h5` (79 MB, not in git), written by
`scripts/make_mol_dataset.py --n-cases 60 --workers 16` at commit 6531ffa, with
`provisorio_MOL_n2_60cases.cost.csv` (cost per seed) and `.log` beside it.

- Cases: IC1 sampler (`sample_cfg_22` draw order, IC1 `X_RANGES`, y_carrier in [0.20, 0.75]),
  seeds 0 to 70. The prefilter (slowest t_st ≤ 1800 s, IC1 severity < 0.9) rejected 8 seeds
  (7, 27, 32, 46, 51, 53, 55, 60). The MOL solver failed on 3: seed 21 (step size below
  round-off) and seeds 38 and 62 (more than 1 h, stopped). 60 cases remain.
- Each case: outlet c_i, y_i, T and v at 800 times (recorder convention t_k = (k+1) t_F/800,
  t_F = 2 t_ref), final profiles, and internal profiles c, q, T, v at 201 times × 281 cells.
  MOL settings: nx = 281, van Leer, BDF, rtol 1e-6.
- Spread: t_ref from 5.7 s to 1286 s (median 121 s); largest |T − T_in| at the outlet from
  0.02 K to 121 K (median 14 K); some beds mainly cool (seed 33: −13 K at t_F).
- Split (random, `split_seed` 0): 50 training cases, 10 validation cases (seeds 70, 5, 58,
  15, 33, 47, 67, 37, 16, 35). The PINN never sees the validation cases: no data, no
  collocation points.

## Cost of the data (MOL solver, CPU)

| Quantity | Value |
|---|---|
| seeds sampled / rejected by the prefilter / solver failures / kept | 71 / 8 / 3 / 60 |
| wall time per kept seed, one core each, 16 solves in parallel | median 175 s, mean 604 s |
| quartiles, 90th percentile, range | 86 s – 735 s, 1931 s, 11 s – 3287 s |
| total CPU time (failures included) | 12.1 core-hours |
| elapsed, 16 processes on a Ryzen 9 7900X (12 cores, 24 threads) | 80 min |
| one uncontended core (estimate from 4 re-solved seeds) | median ≈ 110 s, mean ≈ 370 s per seed |

The 16 parallel solves share 12 physical cores, so each one runs slower than alone. The fastest
seed and the quartile seeds (63, 2, 16, 68) were solved again with only 4 solves at a time (one
physical core each): 7, 53, 106 and 488 s, that is 0.62 (0.58 – 0.67) of their times in the
dataset run (`data/provisorio_MOL_n2_60cases.cost_alone.json`).

The time of a seed follows the number of Jacobian evaluations almost exactly (log-log
correlation 0.99). BDF builds the Jacobian by finite differences, 1407 right-hand-side calls
per Jacobian (about 0.7 s); cases with sharp fronts or strong thermal coupling need thousands
of Jacobian updates. A banded or analytic Jacobian would make the MOL solver much cheaper, but it
is only a stand-in for v23b, so this was not pursued.

## Setup

- Network as in Phase 0 (6 × 128 tanh, Fourier features, 136 710 parameters). θ normalized
  (log + z-score) with the 50 training cases only.
- 50 000 Adam steps (cosine 1e-3 → 1e-5), 4096 interior + 2 × 512 boundary points per step.
  Collocation pool: 262 144 Sobol points spread over the 50 training cases, plus RAR (1024
  points every 1000 steps). Grad-norm weights with cap 100, energy weight 10, causal weights
  (ε stayed at 0.1).
- ΔT_ref: r = median over the training cases of (outlet max|T − T_in|) / (feed estimate) =
  0.424; ΔT_ref = max(1 K, 0.424 × feed estimate) for every case, validation included (1 K to
  112 K). Per-case values would have used the validation references, so the median keeps
  ΔT_ref a function of θ. The ratio varies a lot between cases (quartiles 0.23 – 0.70, range
  0.09 – 3.0 over the 60 cases). This scalar is the only use of solver data in the physics run.
- **physics**: λ_data = 0, residuals and boundary conditions only.
- **hybrid**: plus the outlet y_i and T of the 50 training cases (200 random times per case,
  10 000 points); λ_data adapted by the grad-norm rule and ended at the cap of 100.

Run names in `runs/`: `phase1_physics`, `phase1_hybrid` (evaluation tables in `eval_val/` and
`eval_train/`).

## Results on the 10 validation seeds

Outlet, pooled over the 10 cases (8000 points). y is the mole fraction of the fed strong gas
(for two gases y_0 and y_1 have the same errors).

| Run | y RMSE | y R² | T_out RMSE (K) | T_out R² | v_out/v_s RMSE | v_out R² |
|---|---|---|---|---|---|---|
| physics | 0.174 | 0.654 | 12.9 | 0.38 | 0.212 | 0.12 |
| hybrid | 0.218 | 0.460 | 12.9 | 0.38 | 0.209 | 0.14 |

Characteristic times of the front: mean |relative error| over the cases where both the PINN
and the reference reach the level within t_F (in brackets: cases counted / cases where the
reference reaches it).

| Run | tb (5 %) | t10 | t50 | t90 | ts (95 %) |
|---|---|---|---|---|---|
| physics | 41 % (10/10) | 42 % (10/10) | 50 % (10/10) | 52 % (3/10) | 16 % (1/8) |
| hybrid | 73 % (10/10) | 72 % (10/10) | 62 % (10/10) | 24 % (4/10) | 20 % (1/8) |

Internal fields (mean over the cases of the RMSE on a 101 × 201 grid; never in the data):

| Run | c~_0 | q~_0 | c~_1 | q~_1 | T~ | T (K) | v~ |
|---|---|---|---|---|---|---|---|
| physics | 0.125 | 0.487 | 0.116 | 0.273 | 1.05 | 12.1 | 0.141 |
| hybrid | 0.161 | 0.532 | 0.160 | 0.280 | 0.96 | 11.6 | 0.124 |

Mole balance: mean |PINN − reference| of the outlet moment, in % of t_ref: physics 21.7 (gas 1)
and 80.4 (gas 0, the initial gas); hybrid 35.3 and 113.1. On this finite horizon the reference
itself misses the long-time identity by 16.9 % and 4.1 % on average, so only PINN against
reference is meaningful here.

Cost:

| Run | Training | GPU hours | Inference (800-point outlet curve) |
|---|---|---|---|
| physics | 28.5 min | 0.475 | 1.5 ms |
| hybrid | 30.0 min | 0.501 | 1.3 ms |

(About 1500 steps of the physics run were slowed by a CPU job, 46 instead of 34 ms per step:
about +0.3 min.)

## Diagnosis: training seeds against validation seeds

| Run | Seeds | y R² | T_out RMSE (K) | v_out R² | t50 error | c~_1 field | T field (K) |
|---|---|---|---|---|---|---|---|
| physics | 50 training | 0.975 | 5.7 | 0.89 | 5.2 % | 0.023 | 3.9 |
| physics | 10 validation | 0.654 | 12.9 | 0.12 | 50 % | 0.116 | 12.1 |
| hybrid | 50 training | 0.999* | 0.83* | 0.88 | 0.4 %* | 0.020 | 4.0 |
| hybrid | 10 validation | 0.460 | 12.9 | 0.14 | 62 % | 0.160 | 11.6 |

\* fitted to these very curves (training data).

## Findings

1. **Neither version generalizes to new cases with 50 training cases.** The physics run does
   solve the equations at the training θ (y R² 0.975, field errors about 0.02 on the training
   seeds), but validation errors are 2 to 5 times larger. Residuals are imposed only at 50
   points of a 31-dimensional θ space, and nothing ties the network between them.
2. **Outlet data made generalization worse.** The hybrid fits the 50 training outlets almost
   exactly (λ_data at the cap), but its internal fields on the training seeds are no better than
   the physics run's (unlike Phase 0, where outlet data halved the interior error), and on new
   cases its outlet is worse (y R² 0.46 against 0.65).
3. The largest errors come from cases unlike the training set: seed 5 heats up by ~120 K
   (T_out RMSE 35 K); seed 33 has a sharp front at 6 s and a bed that cools
   (`figures/phase1/outlet_*_val_seed33.png`).
4. ts (c_i/c_i,F ≥ 0.95) is reached within t_F by the reference in only 34 of the 60 cases:
   a hot outlet keeps c = yP/(RT) below 0.95 c_F. The level times on y (t10, t50, t90) are
   defined in more cases.
5. Each run costs about 0.5 GPU-hour; the 60 MOL cases cost 12 CPU-hours.

## Possible next steps (not done)

- Collocation points at new θ drawn from the parameter domain: residuals need no solver run,
  so the physics can cover the whole domain instead of 50 points. This is the PINN's main
  advantage over a data-only MLP, and it addresses finding 1.
- A lower cap for λ_data in the hybrid, so the data cannot outweigh the residuals.
- More training cases and steps; then the v23b dataset when it exists.

## Figures

- `figures/phase1/outlet_physics_val_seed15.png`: a good validation case (physics)
- `figures/phase1/outlet_physics_val_seed33.png`, `outlet_hybrid_val_seed33.png`: a bad one
- `figures/phase1/outlet_hybrid_train_seed17.png`: hybrid on a training case (fitted)
- `figures/phase1/history_physics.png`, `history_hybrid.png`: loss terms and weights
