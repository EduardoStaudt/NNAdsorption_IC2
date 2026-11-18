# Phase 2 results (collocation at new θ, dimension curriculum; provisional MOL data)

> **Provisional data.** Every reference here comes from the small MOL solver of this repository
> (`physics/mol_solver.py`), not from v23b. All files are tagged `provisório_MOL` (attribute
> `dataset_tag`; file names without the accent). Keep these results apart from v23b runs.

## Summary

| Stage | Varying parameters | Physics y R² (20 new MOL seeds) | Hybrid y R² | Gate 0.95 | Variations (physics) | GPU-h, all runs of the stage |
|---|---|---|---|---|---|---|
| (a) | 4 | **0.967** | 0.979 | passed | – | 0.33 |
| (b) | 10 | **0.927** | 0.924 | not passed | wide net 0.912 (lr 3e-4; diverged at 1e-3), 8192 θ 0.917 | 1.07 |
| (c) | 31 | **0.901** | 0.707 (λ_data fixed at 1: 0.842) | not passed | wide net 0.875 (lr 3e-4), 8192 θ 0.832 | 1.20 |

- Collocation at Sobol θ, with no new solver data, is the main gain: on the 10 Phase 1
  validation seeds, with the same 50 training cases, physics y R² 0.654 (Phase 1) → **0.818**.
- Stage (c) could only be trained after two fixes: a temperature floor in the closures (NaN) and
  a Cauchy loss on the PDE residual (a few extreme θ blew the runs up).
- The hybrid helps only in (a). In (c) the outlet data are matched through a non-physical layer
  near the outlet and new cases get worse.
- The remaining errors come from large temperature rises and sharp fronts, which the PINN
  spreads out (tb early, t90 late) in every stage.
- Hard cases: seed 38 was solved (2.6 h of MOL) and the PINN gets y R² 0.910 there; 21 (flow
  reversal) and 62 (more than 3 h) have no reference.
- Total: 2.6 GPU-h of training (15 runs: 14 of 50 000 steps, plus the NaN run stopped near step
  600); MOL: 6.0 CPU-h for the three validation sets, 5.8 CPU-h for the hard seeds (21, 38, 62).

## 1. Quick adjustments

| Change | Where | Why |
|---|---|---|
| ts = first time with y_i/y_i,F ≥ 0.95, like t10, t50 and t90; the former ts (c_i/c_i,F ≥ 0.95, IC1 recorder) is kept as the extra column ts_c | `eval.times` in `configs/pinn_base.yaml` | a hot outlet keeps c = yP/(RT) below 0.95 c_F for a long time (Phase 1: ts_c undefined in 26 of 60 cases) |
| Seeds 21, 38 and 62 in the list `cases.hard_seeds`; `make_mol_dataset.py` never draws them again, and `--seeds` solves a given list | `configs/pinn_base.yaml`, `scripts/make_mol_dataset.py` | reported apart (Section 6) |
| λ_data has its own cap, `loss.adaptive.max_data: 10` (the other weights keep 100); `loss.adaptive.fixed: [data]` keeps λ_data at its initial value (no annealing) | `training/weighting.py` | at the cap of 100 the Phase 1 hybrid fitted its training outlets and generalized worse |

All Phase 2 hybrid runs use the adaptive λ_data with the cap of 10, except one stage (c) run with
λ_data fixed at 1 (Sections 4 and 5).

## 2. Collocation at θ drawn from the domain

The residuals need no solver, so they can be imposed at any θ. In Phase 2 the case table used in
training (`training/theta_colloc.py`) holds the cases with solver data (fixed rows) plus
**2048 θ drawn by Sobol from the domain** (rows that change):

- each θ gets 512 collocation points (z~, t~) on average: a pool of about 1.05 million points;
- every **10 epochs** (passes over the pool, `theta_colloc.resample_every_epochs`), i.e. every
  2561 steps (2622 in stage (c), whose table also holds the 50 training cases), the Sobol rows
  get new θ (the Sobol sequence continues, so the domain is filled
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
34 ms on the Phase 1 setup (600-step probe, same losses at step 100). Phase 2 runs of this net
took 9.6 to 11 ms per step (some while MOL validation solves ran on the CPU at low priority,
nice 19); the wide net of the variations about 21 ms.

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
| (c) | `provisorio_MOL_n2_stage_c_val20.h5` | 71–93 | 1 | 2 (timeout, 1 h) | 10–722 (median 147) | 0.02–167 (median 11.5) | 66 s / 3471 s, 4.9 CPU-h, 63 min |

16 workers (14 for (c)), one core each, on an AMD Ryzen 9 7900X (12 cores, 24 threads).
Subspace seeds draw only the varying columns (`Domain.sample`); full-domain seeds are IC1
recorder seeds, continuing after the 0–70 of the Phase 1 dataset.

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

On the base case itself (its outlet is the hybrid's training data) physics gives y R² 0.986 and
T_out RMSE 2.2 K, the hybrid 1.000 and 0.0 K. Typical validation case:
`figures/phase2/outlet_a_physics_seed100000.png`.

### Stage (b): 10 parameters

| Run | y RMSE | y R² | T_out RMSE (K) | T_out R² | v_out/v_s RMSE | v_out R² |
|---|---|---|---|---|---|---|
| physics | 0.0706 | 0.927 | 12.82 | 0.81 | 0.0713 | 0.87 |
| hybrid | 0.0718 | 0.924 | 12.99 | 0.80 | 0.0715 | 0.87 |
| physics, variation 1: wide net, lr 1e-3 | 0.2693 | −0.066 | 26.73 | 0.17 | 0.6371 | −9.66 |
| physics, variation 2: 8192 collocation θ | 0.0752 | 0.917 | 13.21 | 0.80 | 0.0738 | 0.86 |
| physics, variation 1b: wide net, lr 3e-4 | 0.0773 | 0.912 | 16.65 | 0.68 | 0.0787 | 0.84 |

| Run | tb (c, 5 %) | t10 | t50 | t90 | ts (y, 95 %) | ts_c (c, 95 %) |
|---|---|---|---|---|---|---|
| physics | 30.2 % (20/20) | 25.4 % (20/20) | 13.8 % (20/20) | 24.3 % (17/18) | 27.3 % (11/15) | 29.1 % (5/11) |
| hybrid | 27.2 % (20/20) | 24.4 % (20/20) | 14.2 % (20/20) | 25.0 % (15/18) | 28.9 % (11/15) | 28.3 % (4/11) |
| variation 1 | 99.4 % (20/20) | 99.4 % (20/20) | 98.5 % (20/20) | 50.8 % (18/18) | 53.8 % (15/15) | 98.8 % (5/11) |
| variation 2 | 29.8 % (20/20) | 26.3 % (20/20) | 15.3 % (20/20) | 30.7 % (16/18) | 31.1 % (10/15) | 42.7 % (3/11) |
| variation 1b | 35.7 % (20/20) | 31.2 % (20/20) | 18.4 % (20/20) | 27.8 % (14/18) | 41.4 % (8/15) | 66.8 % (2/11) |

| Run | c~_0 | q~_0 | c~_1 | q~_1 | T~ | T (K) | v~ |
|---|---|---|---|---|---|---|---|
| physics | 0.053 | 0.093 | 0.046 | 0.085 | 0.530 | 6.53 | 0.048 |
| hybrid | 0.054 | 0.097 | 0.048 | 0.086 | 0.542 | 6.69 | 0.049 |
| variation 1 | 0.402 | 0.577 | 0.244 | 0.774 | 1.621 | 18.25 | 0.359 |
| variation 2 | 0.056 | 0.098 | 0.049 | 0.088 | 0.557 | 6.93 | 0.051 |
| variation 1b | 0.058 | 0.102 | 0.054 | 0.103 | 0.631 | 7.56 | 0.053 |

| Run | Moment gas 1, PINN vs ref | Moment gas 0, PINN vs ref | Reference vs theory (1 / 0) | Parameters | Training | GPU h | ms/step | Inference |
|---|---|---|---|---|---|---|---|---|
| physics | 6.0 % | 9.7 % | 4.6 % / 8.9 % | 136 710 | 8.8 min | 0.147 | 9.8 | 1.5 ms |
| hybrid | 5.9 % | 8.7 % | 4.6 % / 8.9 % | 136 710 | 10.8 min | 0.180 | 12.3 (a) | 3.1 ms |
| variation 1 | 29.2 % | 82.1 % | 4.6 % / 8.9 % | 634 374 | 18.0 min | 0.300 | 21.1 | 1.4 ms |
| variation 2 | 5.6 % | 8.2 % | 4.6 % / 8.9 % | 136 710 | 8.9 min | 0.148 | 9.3 | 1.6 ms |
| variation 1b | 9.4 % | 11.4 % | 4.6 % / 8.9 % | 634 374 | 17.9 min | 0.299 | 21.2 | 2.2 ms |

(a) slowed by CPU jobs running at the same time (tests, a solver diagnostic that used 17 threads,
MOL solves; 10–90 % of its logged step times: 11.1–14.3 ms, against 10.6–12.0 ms for the stage
(c) hybrid). The wide net takes about 21 ms per step in both stages (10–90 %: 20.6–22.3 ms here,
21.3–22.9 ms in stage (c)), 2.2 times the base net for 4.6 times the parameters. All runs:
50 000 steps.

Physics stays below y R² 0.95 (0.927), so two variations were tried:

1. **Wide net** (width 256, depth 8, head 128: 634 374 parameters), same learning rate 1e-3: the
   training diverged. The total loss rose from 0.9 to 28 near step 3000 and then moved between 3
   and 150; the causal weight of the last time slice fell to 0 and stayed there, so only the first
   instants were trained and the PINN returns the feed state soon after t = 0 (times −99 %).
   Repeated with lr 3e-4 at the end of the queue (**variation 1b**): stable, but y R² 0.912, still
   below the base net. Over steps 40 000–50 000 its PDE terms are 1.2 to 1.4 times those of the
   base net: at this learning rate 50 000 steps are not enough for it (the same happened in stage
   (c)), so the capacity question stays open.
2. **8192 collocation θ** at a time (128 points each; 163 840 different θ in the run instead of
   40 960): y R² 0.917, no gain. How densely the 10-dimensional box is covered is not what limits
   this stage.

On the base case physics gives y R² 0.990 and T_out RMSE 0.9 K, the hybrid 1.000 and 0.0 K: one
outlet is fitted exactly but tells nothing about the other 9 directions, so on new cases the
hybrid is no better than physics (0.924 against 0.927).

Where the errors are: the y RMSE of a case grows with its measured outlet rise (Spearman 0.59
over the 20 cases), and the worst cases differ in kind from the base case. Seed 200019: short and
fast bed (L 0.35 m, v_s 0.046 m/s) with a strongly adsorbed gas (B_ref 0.99), a sharp front, t50
−21 %. Seed 200016: rise of 209 K, T_out RMSE 39 K. Seed 200009
(`figures/phase2/outlet_b_physics_seed200009.png`): the outlet first cools by 16 K, because the
carrier desorbs (n_ref,0 = 0.77: large initial loading) while the strong gas brings little heat
(q_m 1.2 mol/kg); the PINN predicts heating by 7 K instead and t50 +38 %. The signed time errors
of physics show the other systematic error: tb −1.8 %, t50 +2.8 %, t90 +22.8 %, ts +22.9 %, i.e.
the end of the fronts comes too late (fronts too spread out).

Advance to stage (c) as the rule says.

### Stage (c): all 31 parameters

Training cases: the 50 of Phase 1 (fixed collocation θ for both runs, outlet data for the hybrid),
plus the 2048 Sobol θ renewed every 10 epochs. The first three physics runs blew up (Section 9,
incidents 4 and 5): a few Sobol θ where q~* = q*/q_ref reaches 1e3 to 1e6 had squared residuals up
to 8e9 and took over the gradient. The runs below use the Cauchy loss on the PDE residual
(`loss.robust_c: 10`, `configs/phase2_stage_c.yaml`), lr 1e-3 as in stages (a) and (b).

| Run | y RMSE | y R² | T_out RMSE (K) | T_out R² | v_out/v_s RMSE | v_out R² |
|---|---|---|---|---|---|---|
| physics | 0.0851 | **0.901** | 16.13 | 0.54 | 0.1956 | 0.46 |
| hybrid | 0.1466 | 0.707 | 20.15 | 0.28 | 0.2491 | 0.12 |
| physics, plain mean squares, lr 1e-3 | 0.3304 | −0.487 | 22.60 | 0.09 | 0.2918 | −0.21 |
| physics, plain mean squares, lr 3e-4 | 0.3712 | −0.877 | 22.35 | 0.11 | 0.2718 | −0.05 |
| physics, variation 1: wide net, lr 3e-4 | 0.0958 | 0.875 | 17.18 | 0.47 | 0.1995 | 0.43 |
| physics, variation 2: 8192 collocation θ | 0.1110 | 0.832 | 17.70 | 0.44 | 0.2109 | 0.37 |
| hybrid, λ_data fixed at 1 | 0.1076 | 0.842 | 17.34 | 0.46 | 0.2143 | 0.35 |

| Run | tb (c, 5 %) | t10 | t50 | t90 | ts (y, 95 %) | ts_c (c, 95 %) |
|---|---|---|---|---|---|---|
| physics | 34.7 % (20/20) | 27.6 % (20/20) | 15.5 % (20/20) | 31.4 % (15/20) | 29.3 % (7/15) | 30.2 % (6/12) |
| hybrid | 68.2 % (20/20) | 55.9 % (20/20) | 40.0 % (18/20) | 35.9 % (2/20) | 13.1 % (1/15) | – (0/12) |
| variation 1 | 39.5 % (20/20) | 32.6 % (20/20) | 29.5 % (20/20) | 50.8 % (12/20) | 31.5 % (4/15) | 19.0 % (2/12) |
| variation 2 | 43.3 % (20/20) | 37.7 % (20/20) | 38.9 % (20/20) | 65.5 % (11/20) | 26.9 % (3/15) | 23.8 % (1/12) |
| hybrid, λ_data = 1 | 55.5 % (20/20) | 46.1 % (20/20) | 31.1 % (19/20) | 43.0 % (8/20) | 34.7 % (5/15) | 22.8 % (2/12) |

| Run | c~_0 | q~_0 | c~_1 | q~_1 | T~ | T (K) | v~ |
|---|---|---|---|---|---|---|---|
| physics | 0.066 | 0.171 | 0.057 | 0.157 | 0.395 | 7.59 | 0.082 |
| hybrid | 0.103 | 0.241 | 0.118 | 0.380 | 0.708 | 10.55 | 0.121 |
| variation 1 | 0.071 | 0.174 | 0.068 | 0.183 | 0.436 | 8.00 | 0.086 |
| variation 2 | 0.082 | 0.185 | 0.079 | 0.216 | 0.480 | 8.46 | 0.093 |
| hybrid, λ_data = 1 | 0.079 | 0.200 | 0.077 | 0.212 | 0.534 | 8.53 | 0.096 |

| Run | Moment gas 1, PINN vs ref | Moment gas 0, PINN vs ref | Reference vs theory (1 / 0) | Parameters | Training | GPU h | ms/step | Inference |
|---|---|---|---|---|---|---|---|---|
| physics | 11.5 % | 13.8 % | 8.6 % / 3.4 % | 136 710 | 8.4 min | 0.140 | 9.6 | 2.1 ms |
| hybrid | 34.4 % | 43.9 % | 8.6 % / 3.4 % | 136 710 | 9.5 min | 0.158 | 10.8 | 2.1 ms |
| variation 1 | 16.0 % | 16.9 % | 8.6 % / 3.4 % | 634 374 | 18.2 min | 0.303 | 21.5 | 1.7 ms |
| variation 2 | 20.0 % | 24.4 % | 8.6 % / 3.4 % | 136 710 | 9.0 min | 0.149 | 9.6 | 1.8 ms |
| hybrid, λ_data = 1 | 20.9 % | 23.9 % | 8.6 % / 3.4 % | 136 710 | 9.0 min | 0.149 | 10.4 | 1.3 ms |
| the three runs that blew up | | | | 136 710 | 1 + 8.4 + 8.7 min | 0.30 | | |

Physics reaches y R² 0.901 on the 20 new seeds, below 0.95, so two variations were tried (both
with the Cauchy loss):

1. **Wide net** (width 256, depth 8, head 128: 634 374 parameters) with lr 3e-4, since the wide
   net of stage (b) diverged at lr 1e-3: stable (no blow-up, causal w_min 0.84 at the end), but
   y R² 0.875. Over steps 40 000–50 000 its PDE terms are still 1.1 to 1.7 times those of the
   base net, i.e. with the lower learning rate 50 000 steps are not enough for it; the
   comparison mixes width and learning rate.
2. **8192 collocation θ** at a time (128 points each instead of 512; 163 840 different θ in the
   run): y R² 0.832, worse. With the same number of points per step, each θ gets a quarter of the
   points; how densely the 31-dimensional box is covered is not what limits.

Neither passes; stage (c) is the last stage of the curriculum. On the 10 Phase 1 validation
seeds: variation 1 y R² 0.786, variation 2 0.729 (base net 0.818).

As in stage (b), the errors follow the
temperature rise (Spearman 0.94 between the T_out RMSE and the measured rise of a case, 0.47 for
the y RMSE): seed 78 (rise 143 K) has T_out RMSE 40 K and y RMSE 0.23, seed 74 (rise 83 K) 21 K
and 0.12; the median case has y RMSE 0.067. Signed time errors: tb −20 %, t50 +3 %, t90 +30 %:
the fronts are too spread out at both ends. Seed 74 (`figures/phase2/outlet_c_physics_seed74.png`)
shows it: the reference front is sharp (outlet y of the strong gas from about 0.1 to 0.5 within
about 50 s, temperature peak of 387 K at 255 s), the PINN's is spread over some 300 s (peak
332 K at 220 s): tb −35 %, t50 −16 %, t90 +62 %.

The hybrid fits its 50 training outlets (y R² 0.999, T_out RMSE 0.6 K, t50 error 1.6 %; λ_data at
its cap of 10 from early on) but is worse than physics on new cases (0.707 against 0.901) and its
interior fields are worse even at the training θ (c~_1 0.085 against 0.050 for physics, T 9.5 K
against 6.7 K). Its outlet often never reaches 90 % of the feed within t_F (t90 defined in 2 of
20 cases). With λ_data fixed at 1 (no annealing) the hybrid gets closer, y R² 0.842, but stays
below physics (Section 5).

**Same seeds as Phase 1.** The run also evaluates the 10 Phase 1 validation seeds (`eval_phase1_val/`),
which Phase 1 trained on the same 50 cases without Sobol θ:

| Run | y R² | T_out RMSE (K) | v_out R² | t50 error | c~_1 field | T field (K) |
|---|---|---|---|---|---|---|
| Phase 1 physics | 0.654 | 12.9 | 0.12 | 50 % | 0.116 | 12.1 |
| Phase 1 hybrid | 0.460 | 12.9 | 0.14 | 62 % | 0.160 | 11.6 |
| Phase 2 (c) physics | **0.818** | 11.9 | 0.61 | 20 % | 0.067 | 9.0 |
| Phase 2 (c) hybrid | 0.660 | 12.8 | 0.12 | 53 % | 0.135 | 12.1 |
| Phase 2 (c) hybrid, λ_data = 1 | 0.747 | 11.5 | 0.63 | 30 % | 0.092 | 9.7 |

On the 50 training cases physics now fits less (y R² 0.907 against 0.975 in Phase 1, c~_1 field
0.050 against 0.023): the same network spreads over the whole domain instead of 50 points, and
generalizes better. Caveat: Phase 2 (c) also differs from Phase 1 by the Cauchy loss, the
temperature floor and XLA, so the gain is not due to the Sobol θ alone (no ablation was run).

## 5. Physics × hybrid

| Stage | Data of the hybrid | y R² physics / hybrid | T_out RMSE (K) | t50 error | c~_1 field | GPU h |
|---|---|---|---|---|---|---|
| (a) | outlet of the base case (1 case, 801 times) | 0.967 / **0.979** | 3.07 / 2.17 | 7.4 % / 6.9 % | 0.022 / 0.020 | 0.161 / 0.167 |
| (b) | outlet of the base case | **0.927** / 0.924 | 12.82 / 12.99 | 13.8 % / 14.2 % | 0.046 / 0.048 | 0.147 / 0.180 |
| (c) | outlets of the 50 Phase 1 training cases (200 times each) | **0.901** / 0.707 | 16.13 / 20.15 | 15.5 % / 40.0 % | 0.057 / 0.118 | 0.140 / 0.158 |
| (c), the 10 Phase 1 validation seeds | same | **0.818** / 0.660 | 11.94 / 12.78 | 20 % / 53 % | 0.067 / 0.135 | |
| (c), λ_data fixed at 1 | same | **0.901** / 0.842 | 16.13 / 17.34 | 15.5 % / 31.1 % | 0.057 / 0.077 | 0.140 / 0.149 |
| (c), λ_data fixed at 1, the 10 Phase 1 validation seeds | same | **0.818** / 0.747 | 11.94 / 11.51 | 20 % / 30 % | 0.067 / 0.092 | |

- (a): the one outlet helps a little, mostly at the end of the fronts (t90 error 18.5 → 10.8 %,
  ts 29 → 16 %): in 4 dimensions around the base case, its outlet says something about its
  neighbours.
- (b): no gain any more (one point in 10 dimensions).
- (c): the data hurt. At the end of training the hybrid's PDE terms are 3 to 7 times those of
  physics (equal in (a) and (b)). One-off diagnostic (raw residual, without the Cauchy and causal
  weights, 1024 points per θ, t~ ≥ 0.01): the hybrid's mean residual is 10 times that of physics
  at the 50 training θ (median over the cases; 5–95 %: 3 to 31 times) and 5 times at 256 new
  Sobol θ. It is not a few cases: the ratio does not follow the rise (Spearman +0.17) and is 18 to
  29 for cases with no rise at all. Along the bed the excess is about 6.5 times over the first
  three quarters and **42 times at z~ = 0.90–0.97**, just before the outlet where the data are:
  the network matches the 50 outlets (y RMSE ≈ 0.01) through a layer near z = L that does not
  satisfy the equations, instead of through the interior solution (its interior fields at the
  training θ are worse than those of physics, Section 4). With λ_data at its cap of 10 this
  shortcut costs less than correcting the interior.
- (c) with **λ_data fixed at 1** (`loss.adaptive.fixed: [data]`, run
  `phase2c_hybrid_robust_fixed1`): the conflict is smaller (PDE terms 1.6 to 2.5 times those of
  physics at the end of training instead of 3 to 7; causal w_min 0.78 against 0.53, physics
  0.88); the training outlets are still fitted (y R² 0.992, T_out RMSE 1.5 K) and the interior
  fields at the training θ are as good as those of physics (c~_1 0.046 against 0.050, T 6.1 K
  against 6.7 K). On new cases it is still below physics: y R² 0.842 against 0.901 (Phase 1
  validation seeds: 0.747 against 0.818). In stage (c) the 50 outlets did not improve new cases
  with either weight tried; the smaller weight hurt less.

In GPU time the hybrid costs 4 to 13 % more per step (the data term); the 0.180 GPU-h of the (b)
hybrid includes CPU contention.

## 6. Hard cases (seeds 21, 38 and 62)

Parameters (IC1 recorder seeds, whole domain):

| Seed | t_st (s) carrier / strong gas | Severity | Feed estimate of the rise | q* initial / feed (mol/kg) | Notes |
|---|---|---|---|---|---|
| 21 | 86 / 1150 | 0.62 | 250 K | carrier 4.72 → 0.62, strong gas 0 → 8.10 | −ΔH 43 and 36 kJ/mol, Da 1112 / 30, Pe 1111 |
| 38 | 11 / 1214 | 0.69 | 44 K | carrier 0.003, strong gas 4.26 at feed | Da 266 / 359 (fast transfer), carrier B_ref 1.4e-4 |
| 62 | 219 / 1248 | 0.48 | 93 K | carrier 3.60 → 0.60, strong gas 0 → 5.99 | slow transfer, kL 0.027 / 0.041 1/s |

**Solver.** Seed 21 failed with BDF in Phase 1 (284 s) and now with Radau (964 s): "Required
step size is less than spacing between numbers". A diagnostic run (BDF, the dataset settings)
shows why. The solution is smooth up to t ≈ 82 s (t_F = 2300 s): the carrier, loaded at
4.7 mol/kg with −ΔH = 43 kJ/mol, desorbs and cools part of the bed down to 213 K (T_in = 288 K),
while the strong gas heats another part up to 351 K. Between 81 and 85 s the lowest velocity in
the bed falls from 0.54 v_s to 0.04 v_s and then below zero (−1.07 v_s at 85 s): the model
predicts flow reversal, the uptake and the contraction of the cold gas taking more than the feed
brings. The solver's upwind fluxes and outlet condition assume v > 0, and the integration blows up
right after (v ≈ −3e7 v_s and c ≈ −4e3 C_F at 85.7 s). The fixed-point iteration between dT/dt
and v (Eq. 5) still converges at all these times (change of 3e-11 after 8 iterations), so it is
not the cause. Seed 21 needs a solver that admits v < 0 (to be checked with v23b).

Seeds 38 and 62 (more than 1 h in Phase 1): BDF again, 3 h limit, one worker each. **Seed 38
solved in 9236 s** (2.6 h, against a median of 66 s for the stage (c) validation seeds: 206 559
right-hand-side evaluations, 19 929 Jacobians, 63 476 LU factorizations); its reference closes
the mole balance (outlet moment against theory −0.5 % and 0.0 % of t_ref). **Seed 62 timed out
at 3 h.** The log has overflow warnings in the isotherm term exp(k4/T), i.e. trial states with T
of a few kelvin (which of the two seeds produced them is not recorded). Cost: 5.5 CPU-h for the
two (`data/provisorio_MOL_n2_hard_38_62.h5`, `.cost.csv`, `.log`).

**PINN on seed 38** (`scripts/evaluate.py --data data/provisorio_MOL_n2_hard_38_62.h5`, folder
`eval_hard/` of each run):

| Run | y RMSE | y R² | T_out RMSE (K) | v_out R² | c~_1 field | T field (K) | Moment gas 1, PINN vs ref |
|---|---|---|---|---|---|---|---|
| physics | 0.0614 | **0.910** | 7.18 | 0.88 | 0.051 | 6.12 | 10.3 % |
| hybrid (adaptive λ_data) | 0.1227 | 0.641 | 8.92 | 0.25 | 0.121 | 8.60 | 43.1 % |
| hybrid, λ_data = 1 | 0.0798 | 0.848 | 6.55 | 0.81 | 0.074 | 6.49 | 22.7 % |

Seed 38 is hard for the solver, not for the PINN: physics reaches the level of the validation
set (y R² 0.910 against 0.901), in about 1 ms. Its error is the usual one, made larger by an
almost discontinuous front (`figures/phase2/outlet_c_physics_hard38.png`): with fast transfer
(Da 266 and 359) the reference outlet jumps from feed-free gas to y_1 ≈ 0.2 at 960 s, with a
33 K temperature spike, while the PINN spreads the front over 500–1500 s (tb −36 %, t10 −30 %,
t50 −0.6 %, t90 +46 %, ts not reached within t_F; peak 307 K against 328 K). That sharp front is
probably also what makes the solver slow (many small steps across it). The reference-free check
of the next paragraph had predicted the moment error: +10.1 % against the theory for gas 1, true
PINN − reference +10.3 %.

**PINN without a reference** (`scripts/predict_cases.py`, stage (c) physics run with the Cauchy
loss). Only internal checks are possible: the outlet moment of each gas against its exact
long-time value (mole balance) and the mean squared PDE residual. Calibrated on 10 seeds of the
stage (c) validation set, where the true error is known: |moment − theory| (worse gas) follows the
y RMSE (Spearman 0.62 over the 10 seeds; values 1 to 46 %, median 11 %), the late residual
(t~ ≥ 0.01) hardly (0.21).

| Seed | PINN outlet: max |T − T_in| | v_out/v_s | Moment − theory, gas 0 / gas 1 (% t_ref) | Residual t~ < 0.01 / rest |
|---|---|---|---|---|
| 21 | 11.4 K | 0.80 – 1.00 | −0.3 / +4.2 | 14.8 / 0.012 |
| 38 | 12.4 K | 0.48 – 1.00 | −3.4 / +10.1 | 0.51 / 0.017 |
| 62 | 12.0 K | 0.76 – 1.00 | −16.7 / +8.8 | 2.5 / 0.021 |
| 72 (timeout in (c)) | 6.3 K | 0.34 – 1.00 | −14.8 / −6.1 | 2.6 / 0.066 |
| 81 (timeout in (c)) | 1.0 K | 0.96 – 1.01 | −11.6 / +10.0 | 1.9 / 0.048 |
| 10 validation seeds (range) | 0.0 – 40 K | | 1 – 46 (worse gas) | 0.26 – 53 / 0.015 – 0.63 |

The hard seeds fall inside the range of ordinary seeds, so these checks do not flag them. For
seed 21 the PINN shows neither flow reversal at the outlet nor a large outlet temperature change,
while the solver's state before breaking had a 75 K cold zone; whether the reversal is real can
only be settled with another solver.

## 7. What worked, what did not, next step

**Worked**

- **Collocation at Sobol θ** (no solver): the main gain of the phase. With the same 50 solver
  cases as Phase 1, physics y R² on the 10 Phase 1 validation seeds went from 0.654 to 0.818
  (T_out R² 0.47, v_out R² 0.61 against 0.12; interior c~_1 error 0.116 → 0.067). Caveat: the
  Cauchy loss, the temperature floor and XLA are new too (no ablation). Renewing the θ every
  10 epochs costs nothing measurable (tf.Variable rows, no retracing).
- **XLA**: 9.4 ms per step instead of 34 ms; a 50 000-step run takes 0.14–0.16 GPU-h.
- **Cauchy loss on the residual**: without it stage (c) could not be trained (three runs blew up).
- **The curriculum as a diagnostic**: with no solver data at all, physics generalizes in 4
  dimensions (y R² 0.967) and degrades as the subspace lets in more of the physics: 0.927 in 10,
  0.901 in 31. The worst cases are those with large temperature rises and sharp fronts in every
  stage, not those of a particular parameter.
- **Reference-free checks** (`scripts/predict_cases.py`) for cases the solver cannot solve: the
  outlet moment against its exact value follows the true error (Spearman 0.62 over 10 cases), the
  mean residual does not.

**Did not work**

- The gate (y R² ≥ 0.95) was reached only in stage (a). The variations (wide net, 4 times more
  collocation θ) never helped, in (b) or in (c).
- **Large rises and sharp fronts**: in (c) the T_out RMSE of a case follows its rise (Spearman
  0.94); in every stage the fronts come out too spread (tb early, t90 late).
- **Hybrid with many training outlets**: worse than physics in (c) (0.707 against 0.901), because
  it fits the outlets through a non-physical layer near z = L (Section 5). With λ_data fixed at 1 the gap shrinks (0.842) but does not close.
- **MOL solver on hard cases**: seed 21 is out of its reach (flow reversal), seed 62 did not
  finish in 3 h, seed 38 took 2.6 h;
  2 of 23 stage (c) seeds took more than 1 h.

**Next step suggested**

1. **Longer training of the stage (c) physics run** (150 000–200 000 steps, 0.4–0.6 GPU-h): the
   loss was still falling at the end (median 0.064 over steps 30 000–40 000, 0.048 over
   40 000–50 000, with the cosine decay of the learning rate); it is the cheapest test. Then the
   wide net with lr 3e-4 for as long, since at 50 000 steps it was undertrained.
2. **Sharp fronts**: more collocation points along the fronts (RAR adds 1024 points every 1000
   steps, about half a point per θ of the table, and the Sobol θ are renewed every 2622 steps),
   or a time-marching / time-decomposed PINN (the XPINN already planned for two adsorbent layers
   would use the same machinery).
3. **Hybrid**: a smaller fixed λ_data (0.1), or the outlet data only in a short fine-tuning
   after the physics training; with v23b profiles, data inside the bed as well, so the outlet
   cannot be matched by a layer near z = L alone.
4. **Ablation in stage (c)**: Sobol θ off, Cauchy loss on, to separate the two gains over
   Phase 1.
5. **v23b data**: the loader is ready (Section 8); with the v23b dataset, repeat stage (c) and
   evaluate on the 10k holdout for the PINN × MLP comparison; check seed 21 (flow reversal) with
   v23b.

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
   spacing between numbers" after 964 s, like BDF in Phase 1. Diagnosed in Section 6.
3. **Wide net of stage (b) diverged** with lr 1e-3 (Section 4): repeated with lr 3e-4.
4. **First stage (c) physics run: NaN at step ~550** (loss 116 at step 500, NaN from step 600, all
   weights NaN). Over the whole domain dT_ref/Tin reaches 0.8 at some Sobol θ, so a network output
   T~ = −1.3 already means T = 0 K, where C_T = P/(RT) and exp(k4/T) are infinite. Fix (commit
   bc07e96): the closures (C_T and its derivatives, the EOS residual, q*) see max(T, Tin/2), which
   only unphysical outputs reach; a step with a non-finite loss or gradient gets a zero gradient
   and is counted (`skipped_steps` in the training log); the adaptive weights ignore a non-finite
   batch. The NaN run is kept as `runs/phase2c_physics_try1_nan/`.
5. **Stage (c) blew up twice more without NaN.** With lr 1e-3 (`runs/phase2c_physics`): loss
   spikes up to 2e3 from step 600, causal weight of the last slice at 0 for the whole run,
   validation y R² −0.49. With lr 3e-4 (`runs/phase2c_physics_lr3e4`): healthy until step 5300
   (loss 0.3, causal w_min 0.2), then a jump to 600 right after the second renewal of the
   collocation θ (step 5244), and no recovery. With the step-5000 checkpoint, the mean squared
   residual per θ (t~ ≥ 0.01) has median 44 but a handful of θ at 1e4 to 8e9: the top 10 of 2048 θ
   carry practically all of the sum, in both the θ in use and the θ that came next. Most of the
   worst θ have a gas with q_ref at its floor (1e-6 mol/kg: it hardly adsorbs at Tin, β ~ 1e-7),
   the others an isotherm that changes by orders of magnitude with temperature; in both, q~* =
   q*/q_ref reaches 1e3 to 1e6 where the bed (or the untrained network) is a little colder or
   hotter than Tin. About 1 % of the Sobol θ are like that (8192 draws: q_ref at the floor in
   1.1 %; max q*/q_ref over ±40 K above 1e4 in 1 %). A relative floor on q_ref does not remove the
   tail (with q_ref ≥ 1 % of the larger loading, max q*/q_ref still reaches 1e5). Fix (commit
   a25eee3): optional Cauchy loss on the PDE residual, `loss.robust_c` = 10, on for stage (c)
   only: each point's squared residual s enters as 10 log(1 + s/10), equal to s for s ≪ 10, with
   a bounded influence for s ≫ 10.
6. **Stage (c) validation set: 2 seeds timed out** (72 and 81, 1 h limit) and 1 was rejected by the
   prefilter (88); 23 seeds sampled for 20 solved cases. 72 and 81 are candidates for the list of
   hard cases (not added: the list is the one given, 21, 38, 62).
7. **Provenance of the stage (c) file:** its `git_commit` attribute (4a55402) is the commit at the
   time the file was written; the cases were solved from 15:01 with commit 1877f5b. The solver and
   sampler code is the same in both (`git diff 1877f5b 4a55402` touches only `--set` parsing and
   the θ scaler).
8. **`--set` given twice** (all scripts): with `nargs="*"` alone a second `--set` replaced the
   first, so `--set loss.weights.data=0 --set model.width=256` would have trained a hybrid. Fixed
   in commit 574c19b (`action="extend"`); the `config.yaml` of every Phase 2 run shows the
   overrides it was meant to have (checked).
