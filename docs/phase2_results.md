# Phase 2 results (collocation at new θ, dimension curriculum; provisional MOL data)

> **Provisional data.** Every reference here comes from the small MOL solver of this repository
> (`physics/mol_solver.py`), not from v23b. All files are tagged `provisório_MOL` (attribute
> `dataset_tag`; file names without the accent). Keep these results apart from v23b runs.

STATUS: in progress (sections marked PENDING are filled as the runs finish).

## 1. Quick adjustments

| Change | Where | Why |
|---|---|---|
| ts = first time with y_i/y_i,F ≥ 0.95, like t10, t50 and t90; the former ts (c_i/c_i,F ≥ 0.95, IC1 recorder) is kept as the extra column ts_c | `eval.times` in `configs/pinn_base.yaml` | a hot outlet keeps c = yP/(RT) below 0.95 c_F for a long time (Phase 1: ts_c undefined in 26 of 60 cases) |
| Seeds 21, 38 and 62 in the list `cases.hard_seeds`; `make_mol_dataset.py` never draws them again, and `--seeds` solves a given list | `configs/pinn_base.yaml`, `scripts/make_mol_dataset.py` | reported apart (Section 6) |
| λ_data has its own cap, `loss.adaptive.max_data: 10` (the other weights keep 100); `loss.adaptive.fixed: [data]` keeps λ_data at its initial value (no annealing) | `training/weighting.py` | at the cap of 100 the Phase 1 hybrid fitted its training outlets and generalized worse |

All Phase 2 hybrid runs use the adaptive λ_data with the cap of 10.

## 2. Collocation at θ drawn from the domain

The residuals need no solver, so they can be imposed at any θ. In Phase 2 the case table used in
training (`training/theta_colloc.py`) holds the cases with solver data (fixed rows) plus
**2048 θ drawn by Sobol from the domain** (rows that change):

- each θ gets 512 collocation points (z~, t~) on average: a pool of about 1.05 million points;
- every **10 epochs** (passes over the pool, `theta_colloc.resample_every_epochs`), i.e. every
  2561 steps, the Sobol rows get new θ (the Sobol sequence continues, so the domain is filled
  more and more densely) and the pool gets new points: 19 renewals in 50 000 steps, 40 960
  different θ per run;
- RAR is kept (1024 points every 1000 steps); RAR points of the renewed rows are dropped at
  each renewal (they were chosen for the old θ), those of the fixed rows are kept;
- collocation θ pass the same prefilter as the solver cases (slowest t_st ≤ 1800 s, IC1
  severity < 0.9, carrier = least adsorbed gas);
- θ is normalized (log + z-score) with 8192 Sobol θ of the domain plus the training cases, fixed
  for the run (Phase 1: with the 50 training cases).

The table rows are `tf.Variable`s, so new θ do not retrace the compiled training step. The
training step is compiled with **XLA** (`train.jit_compile: true`): 9.4 ms per step instead of
34 ms on the Phase 1 setup (600-step probe, same losses at step 100). Phase 2 runs took 10 to
11 ms per step while MOL validation solves ran on the CPU at low priority (nice 19).

## 3. Dimension curriculum

`domain.mode: subspace` fixes every θ column at the IC1 example_usage case
(`configs/cases/ic1_example_h2_co2.json`, the Phase 0 case) except the columns listed in
`domain.vary`, which vary over their IC1 ranges (log-uniform for kL, B_ref, dp, Dm).
`domain.mode: full` is the whole 31-dimensional domain (IC1 sampler).

| Stage | Varying columns |
|---|---|
| (a) 4 | `vs`, `y0` (carrier feed fraction; the strong gas gets 1 − y0, so y_F,strong ∈ [0.25, 0.80]), `Tin`, `L0_c1_kL` (kL of the strong gas) |
| (b) 10 | the 4 of (a) plus `L0_c1_B_ref`, `L0_c1_qm_ref`, `L0_c1_n_ref` (isotherm of the strong gas), `L0_c0_n_ref` (exponent of the carrier), `L0_c1_dH` (heat of adsorption of the strong gas), `L` |
| (c) 31 | all |

### Choice of the 6 extra columns of stage (b)

`scripts/rank_parameters.py` (no solver run): (1) each column swept alone over its range around
the base case (21 points; points rejected by the prefilter left out), recording the slowest
stoichiometric time t_st (Eq. 7, exact equilibrium value) and the heat of the net loading change
ΔT_net = Σ_i (−ΔH_i)(q*_i,F − q*_i,0)/C_p,s (adsorption of the fed gas minus desorption of the
carrier); (2) Spearman correlation of each column with the measured outlet rise max|T − T_in|
and with t_st over the 60 MOL cases of Phase 1 (every column varies there at once, so these are
only indicative). Full table: `figures/phase2/parameter_ranking.csv`.

| Stage | Column | Fraction of the range kept | t_st max/min | ΔT_net span (K) | ρ(rise) | ρ(t_st) |
|---|---|---|---|---|---|---|
| a | `vs` | 1.00 | 50.0 | 0 | +0.34 | −0.27 |
| a | `y0` | 1.00 | 1.02 | 16 | −0.25 | −0.03 |
| a | `Tin` | 1.00 | 1.50 | 59 | −0.26 | −0.17 |
| a | `L0_c1_kL` | 1.00 | 1.00 | 0 | +0.15 | −0.07 |
| **b** | `L0_c1_B_ref` | 0.52 | 5.71 | 238 | +0.72 | +0.72 |
| **b** | `L0_c1_qm_ref` | 1.00 | 2.41 | 71 | +0.32 | +0.08 |
| **b** | `L0_c1_n_ref` | 1.00 | 2.26 | 58 | +0.34 | +0.41 |
| **b** | `L0_c0_n_ref` | 1.00 | 1.70 | 81 | −0.38 | −0.16 |
| **b** | `L0_c1_dH` | 1.00 | 1.00 | 26 | +0.30 | +0.13 |
| **b** | `L` | 1.00 | 5.00 | 0 | +0.05 | +0.18 |
| not chosen | `L0_c0_k6` | 1.00 | 1.41 | 46 | +0.20 | +0.10 |
| not chosen | `L0_c1_k6` | 1.00 | 1.50 | 22 | −0.21 | −0.14 |
| not chosen | `L0_rho_b` | 1.00 | 1.68 | 0 | +0.17 | +0.11 |
| not chosen | `P` | 1.00 | 1.29 | 16 | +0.18 | −0.26 |
| not chosen | `L0_c0_B_ref` | 0.48 | 1.02 | 16 | +0.35 | +0.33 |
| not chosen | `L0_eb` | 1.00 | 1.26 | 0 | +0.29 | +0.16 |

Why these six:

- the isotherm of the strong gas (B_ref, qm_ref, n_ref) sets q*_F, hence both t_st and the heat
  released: it leads every column of the table (B_ref first in all of them; only the lower half
  of its log range passes the prefilter, above it t_st > 1800 s or the strong gas would be less
  adsorbed than the carrier);
- the exponent of the carrier, n_ref of component 0, sets the carrier's initial loading, i.e.
  the cooling by desorption: the largest ΔT_net span after B_ref and the strongest (negative)
  correlation with the measured rise after B_ref;
- ΔH of the strong gas is the heat source itself (26 K span, ρ = +0.30);
- L changes t_st five times (t_st ∝ L) and, in dimensionless form, the number of transfer units
  kL L/vs, the Péclet number and the wall losses.

Left out: the temperature coefficient k6 of the carrier (46 K span, but at T ≈ T_ref it mostly
repeats n_ref), ρ_b and ε_b (only t_st, as a holdup factor), P (smaller effects), B_ref of the
carrier (its correlation comes from the sorting of the components, which ties it to B_ref of the
strong gas).

### Data used by each stage

No new MOL training data were generated (as asked). Inside the subspaces of (a) and (b) the only
existing solution is the base case itself (Phase 0 MOL reference, cached), so:

- (a) and (b), physics: no solver data except the measured rise of the base case, which sets
  the temperature scale (ΔT_ref = max(1 K, 0.126 × feed estimate), Phase 1 rule);
- (a) and (b), hybrid: also the outlet y_i and T of the base case (801 times);
- (c): the 50 training cases of Phase 1 (data for the hybrid, fixed collocation θ for both runs,
  ΔT_ref ratio 0.424 as in Phase 1).

### MOL validation sets (20 seeds per stage, never seen in training)

| Stage | File | Seeds | Rejected by the prefilter | Solver failures | t_ref (s) | Outlet rise (K) | Cost: median / max per seed, CPU-h, elapsed |
|---|---|---|---|---|---|---|---|
| (a) | `provisorio_MOL_n2_stage_a_val20.h5` | 100000–100019 | 0 | 0 | 21–681 (median 54) | 1.3–14.8 (median 3.9) | 42 s / 87 s, 0.26 CPU-h, 2 min |
| (b) | `provisorio_MOL_n2_stage_b_val20.h5` | 200000–200035 | 16 | 0 | 17–1553 (median 78) | 2.4–209 (median 16) | 73 s / 1122 s, 0.79 CPU-h, 19 min |
| (c) | `provisorio_MOL_n2_stage_c_val20.h5` | 71– | PENDING | PENDING | PENDING | PENDING | PENDING |

16 workers (14 for (c)), one core each. Subspace seeds draw only the varying columns
(`Domain.sample`); full-domain seeds are IC1 recorder seeds, continuing after the 0–70 of the
Phase 1 dataset.

## 4. Results by stage (validation seeds)

Metrics as in Phases 0 and 1: outlet y (strong gas) and T pooled over all points of the 20
cases; characteristic times as mean |relative error| over the cases where both PINN and
reference reach the level within t_F (in brackets: counted / reached by the reference); internal
fields as mean RMSE over the cases (101 × 201 grid); mole balance as mean |PINN − reference| of
the outlet moment in % of t_ref.

### Stage (a): 4 parameters

| Run | y RMSE | y R² | T_out RMSE (K) | T_out R² | v_out/v_s RMSE | v_out R² |
|---|---|---|---|---|---|---|
| physics | 0.0374 | **0.967** | 3.07 | 0.95 | 0.0303 | 0.79 |
| hybrid | 0.0296 | 0.979 | 2.17 | 0.98 | 0.0288 | 0.81 |

| Run | tb (c, 5 %) | t10 | t50 | t90 | ts (y, 95 %) | ts_c (c, 95 %) |
|---|---|---|---|---|---|---|
| physics | 13.6 % (20/20) | 10.2 % (20/20) | 7.4 % (20/20) | 18.5 % (18/18) | 29.1 % (8/12) | 26.8 % (6/11) |
| hybrid | 9.0 % (20/20) | 8.2 % (20/20) | 6.9 % (20/20) | 10.8 % (17/18) | 15.5 % (11/12) | 18.8 % (10/11) |

| Run | c~_0 | q~_0 | c~_1 | q~_1 | T~ | T (K) | v~ |
|---|---|---|---|---|---|---|---|
| physics | 0.023 | 0.044 | 0.022 | 0.060 | 0.255 | 1.21 | 0.019 |
| hybrid | 0.021 | 0.039 | 0.020 | 0.053 | 0.210 | 1.03 | 0.018 |

| Run | Moment gas 1, PINN vs ref | Moment gas 0, PINN vs ref | Reference vs theory (1 / 0) | Steps | Training | GPU h | ms/step | Inference |
|---|---|---|---|---|---|---|---|---|
| physics | 4.6 % | 5.2 % | 6.6 % / 7.4 % | 50 000 | 9.7 min | 0.161 | 11.0 | 1.7 ms |
| hybrid | 3.3 % | 4.3 % | 6.6 % / 7.4 % | 50 000 | 10.0 min | 0.167 | 11.4 | 1.4 ms |

Physics passes y R² 0.95 on the validation seeds: next stage without variations.

PENDING: stages (b), (c), comparison, hard cases, conclusions.

## 8. Loader for the v23b dataset

`data/loader.py` now reads a recorder-format h5 file whatever its optional content:

- dataset names can be changed in the config (`data.keys`, e.g. `{theta: X_v3}`); defaults are
  the IC1 recorder names (`X_v2`, `y_out`, `c_out`, `T_out`, `Tz`, `qz`, `TF`, `NT`, `seeds`,
  `valid`) plus the IC2 extensions (`v_out`, `profiles/<row>/{t, c, q, T, v}`);
- the number of gases follows from the width of θ (10 n + 11 columns, IC1 layout: Dt at index
  9n + 11, y_0..y_{n−2} last), so n_comp up to 5 needs no code change; each gas keeps its own
  series (no y_1 = 1 − y_0);
- `c_out` optional: without it the loader uses the model's ideal-gas closure at constant
  pressure, c_i = y_i P/(R T_out), and records it (`SolverCase.derived`);
- `v_out` (v(L, t)) optional: the velocity metrics are skipped when it is missing;
- profiles optional, and inside them every field but t (e.g. no v): field errors are computed
  for the fields present;
- `dataset_info` reports the tag, the purpose, n_comp and which optional series exist.

Tested with a synthetic three-gas file with another θ name, no c_out, no v_out and profiles
without v (`tests/test_data.py`), and with an evaluation without velocity data
(`tests/test_report.py`).

## 9. Incidents

1. **Stage (a) validation inputs out of distribution (fixed).** The first evaluation of both
   stage (a) runs gave y R² −0.88 (physics) and −0.81 (hybrid): the PINN returned the feed state
   everywhere for every validation case, while its residuals at new Sobol θ (never trained) were
   as small as at the base case (one-off diagnostic, mean PDE residual per slice of t~, median
   over 20 cases: for t~ ≥ 0.01, 0.004 to 0.035 at new Sobol θ, 0.002 to 0.019 at the base
   case, 0.29 at the validation θ; for t~ < 0.01, 0.27, 0.11 and 4.8e3). Cause: the pressure is a fixed
   column in the subspace; the mean of 8193 identical pressures (2.5e6 Pa) has ~3e-7 Pa of
   round-off, above the absolute floor of the θ scaler (sd < 1e-8 → sd = 1), so the scaler
   divided by 3e-7, and the float32 copy of the same pressure in the validation file (IC1
   recorder format) became an input 2e5 standard deviations away. Fix (commit 51394b0):
   constant columns are detected relative to |μ| and get sd = max(1, |μ|); cases read from a file
   in subspace mode get the fixed columns of the base case back. The two trained models were
   re-evaluated with `scripts/evaluate.py` (same weights, same scaler; the first evaluation is
   kept in `eval_val_before_fix/` and in `summary.json` as `validation_before_fix`).
2. **Hard seed 21 failed again** with Radau (3 h limit): "Required step size is less than
   spacing between numbers" after 964 s, like BDF in Phase 1. Section 6.
