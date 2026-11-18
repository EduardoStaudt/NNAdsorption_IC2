# Architecture of the IC2 hybrid PINN

The model and its equations are in `docs/modelo_solver_v23b_preview.pdf`. The derivation of the
dimensionless equations is in `docs/nondim.md`. This note maps the pieces to the code.

## Data flow

```
case (CaseParams) ──► scales + dimensionless groups ──► case table (one row per case)
      │                                                     │  normalized theta, groups,
      │                                                     │  isotherm coefficients, IC
      ▼                                                     ▼
reference solution                  PINN(z~, t~, theta^, c~_0, q~_0) ──► c~ (n), q~ (n), T~, v~
(MOL solver now,                                │
 v23b dataset later)                            ▼
      │                         nested GradientTapes: d/dt~, d/dz~, d2/dz~2
      │                                         │
      ▼                                         ▼
data loss (outlet / profiles)        residuals of Eqs. (1)-(6) ──► loss terms
      └───────────────► weighted sum (adaptive λ, causal weights) ──► Adam (+ L-BFGS)
```

## Modules (`src/nnads_ic2/`)

| Module | What it does |
|---|---|
| `runtime.py` | TensorFlow setup: XLA uses the pip `ptxas` (RTX 50xx in WSL2), TF32 off, float32/float64 (floatx **and** Keras dtype policy), seed |
| `config.py` | YAML configs with `inherit:` (deep merge) |
| `physics/params.py` | `CaseParams` (one case, n_comp gases) and the theta vector in the IC1 order (31 columns for n = 2, Dt at 29) |
| `physics/isotherm.py` | LRC isotherm, numpy and TF, identical to IC1 `qstar_slope` (pressures in atm) |
| `physics/scales.py` | scales, `stoich_times` (Eq. 7), `dimensionless_groups`, round trip dimensional ↔ dimensionless |
| `physics/residuals.py` | residuals of Eqs. (1)–(6), the case table (`case_arrays`, `stack_cases`, `gather_cases`) |
| `physics/mol_solver.py` | small reference solver (method of lines, van Leer FV, scipy BDF), **tests and Phase 0 only** |
| `data/cases.py` | `sample_case_ic1` (same draws as IC1 `recorder.sample_cfg_22`), IC1 severity, prefilter |
| `data/loader.py` | h5 datasets in the IC1 recorder format (+ optional `v_out` and internal profiles) |
| `data/sampling.py` | Sobol collocation pool, RAR selection, boundary/initial points |
| `models/pinn.py` | `PINN`: Fourier features, MLP trunk, separate heads, hard constraints, `Domain` list (XPINN hook) |
| `training/losses.py` | nested-tape derivatives, loss terms, causal weights |
| `training/weighting.py` | fixed or gradient-norm (annealing) loss weights; causal ε schedule |
| `training/trainer.py` | training loop, RAR, CSV/TensorBoard logs, checkpoints, L-BFGS |
| `eval/metrics.py` | RMSE, MAE, MBE, R², Willmott d, AARD (IC1 definitions); level-crossing times |
| `eval/mass_balance.py` | outlet first moment vs t_st,i − t_init,i |
| `eval/predict.py`, `eval/report.py`, `eval/plots.py` | predictions in SI units, evaluation reports, figures |
| `experiment.py` | shared steps of the scripts (case table, model, reference, data points) |

Scripts: `scripts/train_single_case.py` (Phase 0), `scripts/make_mol_dataset.py` (provisional
Phase 1 data from the MOL solver, tagged `provisório_MOL`), `scripts/train_parametric.py`
(Phase 1), `scripts/evaluate.py` (re-evaluate a run, e.g. on the 10k holdout).
Configs: `configs/pinn_base.yaml` (every number), `pinn_single_case.yaml`, `pinn_parametric.yaml`.

## Network

- Input per point: z~ and t~/t~_F (both in [0, 1]), their Fourier features
  [sin(2π x B), cos(2π x B)] (fixed random B, one row per coordinate, scale σ_z, σ_t), and the
  normalized theta (log on B_ref, kL, dp, Dm, then z-score with the training cases).
- Trunk: `depth` dense layers of `width` units, tanh or SiLU.
- Heads (separate small MLPs): N_c (n), N_q (n), N_T (1), N_v (1). Output 2n + 2. No y1 = 1 − y0.
- Hard constraints with s = 1 − exp(−t~/δ): c~ = (1−s) c~_0 + s softplus(N_c), same for q~,
  T~ = s N_T, v~ = 1 + z~ s N_v. Exact initial state, c~, q~ ≥ 0, v(0, t) = v_s.
- Soft constraints: Danckwerts inlet (c and T) and zero-gradient outlet.

## Loss

L = λ_data L_data + λ_pde (L_mass + L_ldf + L_energy + L_global) + λ_bc (L_inlet + L_outlet) + λ_ic L_ic

- L_global = Eq. (5) with C~_T = 1/(1 + ε_T T~) plus the closure Σ c~_i = C~_T.
- Residuals are divided by their largest coefficient (all O(1)).
- λ: fixed, or learning-rate annealing (Wang, Teng & Perdikaris 2021) every K steps, with the
  PDE group as reference.
- Causal weights (Wang, Sankaran & Perdikaris 2022): time slices of t~, weight
  w_j = exp(−ε Σ_{k<j} L_k); ε grows ×10 when the last slice reaches weight 0.99.
- RAR (Lu et al. 2021): every K steps the worst of many random candidates join the pool.
- L_data: outlet y_i and T~ (recorder datasets) or all fields when internal profiles exist
  (`data.use_internal_profiles`).

## Hooks for later (not implemented)

- **N up to 5 gases**: every array has the component on the last axis and no index is fixed
  ("carrier" = whatever fills the bed at t = 0, from the config). Missing: the feed composition
  sampler and the severity index for n > 2 (Section 2.3 to be defined).
- **Two adsorbent layers (XPINN)**: `PINN(domains=[Domain(0, a), Domain(a, 1)])` raises
  `NotImplementedError` for now. Plan: one `SubNet` per domain, points routed by z~, and
  `interface_loss` with continuity of c_i, T and of the convective + diffusive fluxes at z~ = a;
  theta gets per-layer blocks (`theta_names(n_layer=2)`).

## Environment

TensorFlow GPU only runs in WSL2 on Windows. Reference: Ubuntu 22.04, Python 3.11,
`~/ic2env`, `pip install -r requirements.txt`. Tests: `~/ic2env/bin/python -m pytest -q` from the
repository root (in `/mnt/c/...`).
