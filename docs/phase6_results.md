# Phase 6 results: 400k steps and hot collocation on stage (c)

All runs use the t26 default (8x224 SiLU, Fourier 128, sigma_t 0.297, lr 6.2e-4 with cosine decay over
all steps, energy weight 2.23, Cauchy c 24.3, 1024 collocation theta), physics only (lambda_data 0),
2 seeds each, one job on the GPU at a time. Validation: the 20 stage (c) MOL cases
(`data/provisorio_MOL_n2_stage_c_val20.h5`), curve every 25k steps. No MOL data was generated.

Runs: `runs/phase6c_400k_seed{0,1}`, `runs/phase6c_hot_200k_seed{0,1}`, `runs/phase6c_hot_400k_seed{0,1}`.
Baseline: `runs/phase5c_long_seed{0,1}` (200k, uniform). Group numbers: `runs/phase6_groups.json`
(`scripts/compare_case_groups.py`). Queue: `scripts/phase6_queue.sh`, log `runs/phase6_queue.log`.

## 1. Case groups

- **hot (main metric)**: cases 6, 11, 2, 9, 16, the 5 largest measured outlet temperature rises of the
  MOL reference (143, 167, 83, 54, 43 K). The group was approved as "rise above 50 K", but case 16 has
  43 K, so the threshold that gives these 5 cases is effectively 40 K (the next case has 20 K). The 5
  approved cases were kept. The group is defined by the physics, not by the error.
- **worst3 (secondary)**: cases 6, 11, 2, the 3 worst cases of the Phase 5 long run (seed 1). They were
  chosen by looking at errors on this same validation set, so part of any gain there is selection bias
  (regression to the mean between seeds). Read it together with the hot group.
- **others**: the 15 cases outside the hot group.

Group values are means of the per-case y R2; "pooled" is the R2 of all 20 cases together (summary.json).

## 2. Hot collocation (method)

Feed estimate of the temperature rise of a theta (no solver):

    dT_est = sum_i y_feed,i * (-dH_i) * q_ref,i / Cps

with q_ref from `reference_loadings(p, q_ref_floor)`. On the 20 validation cases its Spearman correlation
with the measured outlet rise is 0.87. `theta_colloc.hot_frac` = 0.5: at each resample (every 2622 steps)
512 of the 1024 theta are Sobol uniform and 512 are picked from 4x512 Sobol candidates (separate Sobol
stream, seed + 500) with probability proportional to clip(dT_est, 1 K, cap), without replacement (Gumbel
top-k). The cap is the 99th percentile of dT_est over 4096 draws (307 K), so a few extreme theta cannot
take all picks. Mean dT_est of the picked theta: 109 K against 47 K for the uniform half.
Code: `src/nnads_ic2/training/theta_colloc.py`, tests in `tests/test_theta_colloc.py`. Default `hot_frac`
stays 0.0.

## 3. Results

Final validation (step 200k or 400k).

| run | seed | pooled R2 | hot5 | worst3 | others15 | T RMSE (K) |
|---|---|---|---|---|---|---|
| 200k uniform (Phase 5) | 0 | 0.9531 | 0.8415 | 0.7651 | 0.9792 | 14.53 |
| | 1 | 0.9616 | 0.8756 | 0.8019 | 0.9854 | 13.30 |
| 400k uniform | 0 | 0.9570 | 0.8569 | 0.7718 | 0.9835 | 13.73 |
| | 1 | 0.9602 | 0.8647 | 0.7822 | 0.9816 | 13.53 |
| 200k hot | 0 | 0.9630 | 0.8834 | 0.8160 | 0.9807 | 13.75 |
| | 1 | 0.9626 | 0.8963 | 0.8346 | 0.9771 | 13.09 |
| 400k hot | 0 | 0.9569 | 0.8722 | 0.8062 | 0.9785 | 13.73 |
| | 1 | 0.9597 | 0.8950 | 0.8358 | 0.9757 | 13.40 |

Means of the 2 seeds:

| config | pooled | hot5 | worst3 | others15 | GPU h per seed |
|---|---|---|---|---|---|
| 200k uniform | 0.9573 | 0.8586 | 0.7835 | 0.9823 | 1.11 |
| 400k uniform | 0.9586 | 0.8608 | 0.7770 | 0.9826 | 2.18 |
| **200k hot** | **0.9628** | **0.8899** | **0.8253** | 0.9789 | 1.15 |
| 400k hot | 0.9583 | 0.8836 | 0.8210 | 0.9771 | 2.27 |

Per case, hot group (seed 0 / seed 1):

| case | 200k uniform | 400k uniform | 200k hot | 400k hot |
|---|---|---|---|---|
| 6 | 0.689 / 0.704 | 0.692 / 0.707 | 0.736 / 0.749 | 0.734 / 0.748 |
| 11 | 0.707 / 0.790 | 0.731 / 0.721 | 0.779 / 0.791 | 0.760 / 0.818 |
| 2 | 0.899 / 0.912 | 0.893 / 0.919 | 0.933 / 0.964 | 0.925 / 0.941 |
| 9 | 0.968 / 0.988 | 0.984 / 0.988 | 0.990 / 0.987 | 0.970 / 0.983 |
| 16 | 0.945 / 0.985 | 0.985 / 0.989 | 0.979 / 0.991 | 0.973 / 0.985 |

### 3.1 400k vs 200k (item 1)

Doubling the steps gives nothing: pooled +0.001, hot5 +0.002, worst3 -0.007, all inside the seed spread
(about 0.01). The curves (`docs/figures/phase6/val_curve_400k_vs_200k.png`) show why: with cosine decay over
400k the lr stays high for longer, and both seeds have a large drop in the middle (R2 0.73 at 125k for seed 0,
0.76 at 150k for seed 1). The model then recovers in the last 150k steps, when the lr is small, and ends where
the 200k run ends. The plateau criterion (0.002 over 2 points) would have stopped these runs at 100k and 125k,
on a dip, so it is not safe to turn it on with this schedule.

### 3.2 Hot collocation at 200k (item 2)

hot5 +0.031, worst3 +0.042, pooled +0.006, T RMSE -0.5 K. The cost is on the cool cases: others15 -0.003.
The gain is larger than the seed spread for hot5 and worst3, and is seen in both seeds. Cases 6 and 2 gain
the most (+0.05 and +0.04). Case 11 gains less (+0.04 against seed 0 of Phase 5 but ~0 against seed 1).
Cases 9 and 16 were already near 0.98. Cost: same as the uniform run (the dT_est formula is cheap), ~1.15 GPU h.

### 3.3 Decision on item 3 and 400k hot

Rule (user): run item 3 only if mean worst3 rises by >= 0.03 over the 200k run and pooled falls by <= 0.005.
Result: worst3 +0.042, pooled +0.006, so item 3 ran.

400k hot is not better than 200k hot: pooled -0.0045, hot5 -0.006, worst3 -0.004. It has the same mid-run
collapse as the uniform 400k runs (seed 0 reaches R2 -0.12 at 125k, seed 1 0.55 at 175k,
`docs/figures/phase6/val_curve_hot_vs_uniform.png`) and recovers in the end. So the gain of the hot
collocation holds at 400k, but the extra 200k steps add nothing.

## 4. Conclusions

1. 200k steps is enough for t26 on stage (c); 400k with the same cosine schedule costs twice and only adds
   instability in the middle of the run.
2. Hot collocation (hot_frac 0.5) is the best config so far: pooled 0.963 (both seeds above 0.962), hot cases
   0.890. It is a candidate for the new default of stage (c); the default was not changed (needs approval).
3. Cases 6 and 11 (rises of 143 and 167 K) stay at 0.74 to 0.80: the hot sampling helps but does not close
   the gap of the strongest temperature fronts.

Best single run: `runs/phase6c_hot_200k_seed0` (pooled 0.9630).

## 5. Notes

- Seed 1 of hot 200k took 12.5 h of wall time for 1.15 h of training (00:37 to 13:06); the machine was
  probably suspended. Training time in summary.json is unaffected.
- The fine-tune config (`configs/phase3_finetune_c.yaml`) still points to `runs/phase5c_long_seed1`.
