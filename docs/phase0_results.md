# Phase 0 results (sanity check, one fixed case)

Case: IC1 `example_usage.py` (H2/CO2, `configs/cases/ic1_example_h2_co2.json`): P = 24.8 bar,
T_in = 307 K, v_s = 2.08 cm/s, L = 1.22 m. Scales: t_ref = 49.7 s, t_F = 99.4 s,
ΔT_ref = 33.6 K (feed formula; runs baseline to C) or 4.85 K (measured rise; run A′),
Pe = 623, Da = (25.2, 9.3).

Reference: MOL solver (`physics/mol_solver.py`, nx = 281, scipy BDF, 13 s). The v23b solver is
not available yet. Mole balance closes to 1e-11 and Σc = P/RT drifts < 1e-7. Grid check:
breakthrough times identical for nx = 141 and 281 on the 0.25 s output grid.

Network: 6 × 128 tanh, 64 Fourier features (σ_z = 2, σ_t = 1), heads with 1 hidden layer of
64 units, 136 710 parameters. Adam 20 000 steps, cosine decay 1e-3 → 1e-5, 4096 interior +
2 × 512 boundary points per step, RAR every 1000 steps, causal weights (32 slices). float32 on an
RTX 5070 Ti (WSL2), ~35 ms per step.

Metrics: IC1 definitions. tb/ts = 5 %/95 % of c_CO2/c_CO2,F at the outlet (IC1 recorder).
t10/t50/t90 = 10/50/90 % of y_CO2/y_CO2,F with interpolation (IC1 compute_level_times). t10 and
t90 were added after the first report and computed for the old runs with `scripts/evaluate.py`
(their other metrics were reproduced exactly). Field errors = RMSE of the dimensionless fields
over a 101 × 201 (z~, t~) grid; the temperature error is also given in kelvin, because T~
depends on ΔT_ref.

## Runs

| Run | What changes | Data in the loss |
|---|---|---|
| baseline | λ cap 1e4, energy weight 1 | none (forward problem) |
| A, forward | λ cap 100, energy weight 10 | none (forward problem) |
| B, hybrid | baseline + outlet data | outlet y_i and T (801 points) |
| C, hybrid | A + outlet data | outlet y_i and T (801 points) |
| A′, forward | A + ΔT_ref = measured rise (4.85 K) | none (forward problem) |

Run names in `runs/`: `phase0_adam20k`, `phase0_fwd_energy10`, `phase0_hybrid`, `phase0_hybrid_energy10`,
`phase0_fwd_dTmeasured`. The defaults in `configs/pinn_base.yaml` are the settings of A′
(λ cap 100, energy weight 10, ΔT_ref from the reference data).

> **Note on B and C: they measure fit, not generalization.** The hybrid runs were trained on the
> outlet curve (y_i and T) of this very case and are evaluated on the same curve, so their outlet
> numbers only say how well the network reproduces data it has seen. Only the quantities that
> were not in their data (outlet velocity, internal fields, mole-balance moment) say something
> about the physics they learned. Generalization to unseen cases is measured in Phase 1, on
> validation seeds that the PINN never sees.

## Outlet (PINN vs reference)

| Run | y RMSE | y R² | T_out RMSE (K) | T_out R² | v_out/v_s RMSE | v_out R² |
|---|---|---|---|---|---|---|
| baseline | 0.0191 | 0.989 | 2.53 | −0.99 | 0.0233 | −0.70 |
| A, forward | 0.0140 | 0.994 | 0.255 | 0.980 | 0.0073 | 0.83 |
| B, hybrid | 0.0003* | 1.000* | 0.002* | 1.000* | 0.0169 | 0.11 |
| C, hybrid | 0.0005* | 1.000* | 0.005* | 1.000* | 0.0058 | 0.89 |
| A′, forward | 0.0151 | 0.993 | 0.323 | 0.968 | 0.0070 | 0.85 |

\* B and C were trained on this same outlet curve: these numbers show the fit, not generalization
(see the note above). The outlet velocity and the internal fields were not in their data.

## Characteristic times of the CO2 front (relative error)

| Run | tb (5 %) | t10 | t50 | t90 | ts (95 %) |
|---|---|---|---|---|---|
| baseline | −20.7 % | −13.7 % | −5.2 % | −2.5 % | −4.9 % |
| A, forward | −11.0 % | −6.9 % | +1.3 % | +8.9 % | +13.9 % |
| B, hybrid | +0.3 %* | −0.0 %* | 0.0 %* | +0.0 %* | +0.6 %* |
| C, hybrid | −0.1 %* | −0.1 %* | 0.0 %* | +0.2 %* | +0.9 %* |
| A′, forward | −10.1 % | −7.2 % | −1.3 % | +10.5 % | +21.3 % |

Reference times: tb 26.4 s, t10 29.7 s, t50 46.4 s, t90 67.8 s, ts 76.8 s. t10 and t90 sit
where the curve is steeper than at 5 % and 95 %, so the same concentration error moves them less.

## Internal fields (dimensionless RMSE; never in the data)

| Run | c~_H2 | q~_H2 | c~_CO2 | q~_CO2 | T~ | T (K) | v~ |
|---|---|---|---|---|---|---|---|
| baseline | 0.026 | 0.039 | 0.018 | 0.132 | 0.090 | 3.04 | 0.016 |
| A, forward | 0.009 | 0.017 | 0.009 | 0.024 | 0.009 | 0.29 | 0.006 |
| B, hybrid | 0.015 | 0.040 | 0.011 | 0.074 | 0.045 | 1.50 | 0.012 |
| C, hybrid | 0.004 | 0.014 | 0.004 | 0.014 | 0.008 | 0.28 | 0.005 |
| A′, forward | 0.009 | 0.017 | 0.010 | 0.028 | 0.084 | 0.41 | 0.006 |

(T~ is relative to ΔT_ref: 33.6 K for baseline to C, 4.85 K for A′, so compare the kelvin
column. v~ = 0.005 is 0.5 % of v_s.)

## Mole balance identity (outlet first moment, error in % of t_ref)

| Run | CO2 vs theory | CO2 vs reference | H2 vs theory | H2 vs reference |
|---|---|---|---|---|
| reference (MOL) | −0.79 | — | −6.42 | — |
| baseline | −7.68 | −6.89 | −8.93 | −2.51 |
| A, forward | +2.13 | +2.92 | −8.15 | −1.73 |
| C, hybrid | −0.49 | +0.30 | −5.81 | +0.61 |
| A′, forward | +1.84 | +2.63 | −8.64 | −2.22 |

The identity is exact only when the bed has returned to the feed state. At t_F = 2 t_ref the
bed is still about 4 K warmer than T_in, so even the reference misses it by −0.8 % (CO2) and
−6.4 % (H2, which fills the bed at t = 0). The fair comparison is PINN against reference on the
same horizon.

## Cost

| Run | Training | GPU hours | Inference (801-point outlet curve) |
|---|---|---|---|
| baseline | 12.3 min | 0.205 | 1.2 ms |
| A | 11.5 min | 0.192 | 2.0 ms |
| B | 13.1 min | 0.219 | 1.5 ms |
| C | 13.0 min | 0.216 | 1.1 ms |
| A′ | 22.6 min† | 0.377† | 2.4 ms |

† A′ ran while the 16 MOL processes making the Phase 1 dataset used the CPU: 66 ms per step
instead of ~35 ms. Its GPU time is not comparable with the other runs (same steps, same network).

## Findings

1. **The forward PINN works for this case once the loss weights are sane** (run A). Temperature
   (RMSE 0.26 K), velocity and all internal fields agree with the reference. The weak points
   are the low-concentration foot of the CO2 front (tb −11 %) and the slow tail (ts +14 %): at
   5 % and 95 % the curve is flat, so small concentration errors become large time errors.
2. **Learning-rate annealing with a high cap is harmful here.** The boundary (and data) terms are
   easy, their gradients are tiny, and the Wang et al. (2021) ratio pushed λ_bc to the cap of
   1e4 within 500 steps. The training then oscillated over two orders of magnitude. Cap 100 fixed it.
3. **The energy equation needed weight 10.** With ΔT_ref from the formula of the plan
   (Σ y_i,F (−ΔH_i) q_ref,i / C_p,s = 33.6 K), the real rise is only 4.8 K: H2 desorbs
   (endothermic) while CO2 adsorbs, and the wall removes heat. So T~ ≤ 0.14 and the energy
   residual is small. Without the extra weight the baseline converged to T~ ≈ 0 and distorted q
   to cancel the heat terms.
4. **Outlet data plus physics gives the most accurate interior** (run C): field errors are about
   half of the forward run's. Outlet data alone with poorly balanced weights (B) gives a perfect
   outlet and a worse interior. B and C saw the outlet they are scored on, so this is a statement
   about the interior (not seen) and not about predicting new cases.
5. **Measuring ΔT_ref did not change the forward result** (A′ against A: fields, velocity and
   moments within the scatter of single runs; T_out 0.32 K against 0.26 K). This is expected:
   the energy residual is divided by its largest coefficient, here the heat-of-adsorption term
   Σ_i H_i ∝ 1/ΔT_ref (1.97 with 33.6 K, 13.6 with 4.85 K), so as a function of the physical
   fields it does not depend on ΔT_ref. What changes is the size of the network's temperature
   output (T~ up to ~1 instead of ~0.14) and, in hybrid runs, the weight of T in the data loss.
   The energy weight of 10 is still needed.

## Open points for the advisor

- ΔT_ref: since this report it is max(1 K, measured rise of the reference) when data exist,
  with the feed formula as fallback (warning in the log); see `docs/nondim.md`. A scale based on
  the net loading change, Σ(−ΔH_i)(q*_i,F − q*_i,0)/C_p,s (12.8 K here), would need no data.
- tb uses the IC1 recorder signal (c_i/c_i,F); the PINN's small offset before breakthrough
  (y_CO2 ≈ 0.005) moves it a lot. t10/t90 (added) are less sensitive.

## Figures

- `figures/phase0/outlet_baseline.png`, `outlet_forward.png`, `outlet_hybrid.png`: outlet curves
- `figures/phase0/fields_forward.png`: (z~, t~) maps, reference / PINN / error, run A
