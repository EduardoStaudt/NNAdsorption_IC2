# Optuna study on stage (b): results

Run on 2026-11-19 (branch `pinn-v23b`) as approved: 60 trials, 4 of them at a time on the GPU, then the
confirmation of the top 3 at 50 000 steps. Protocol, search space and cost estimate are in
`docs/optuna_stage_b.md`. Code: `scripts/optuna_stage_b.py`. Runs: `runs/optuna_b/` (trials `t0000`
to `t0059`, the study journal `study.log`, the confirmation in `confirm_b/`, `confirm_c/` and
`confirm.json`).

## Summary

- The study found a setup well above the current model. Trial 26 (8 x 224, SiLU, 128 Fourier
  features at scale 0.30, energy weight 2.2, Cauchy c 24.3, lr 6.2e-4, 1024 theta) reached a
  validation y R² of 0.952 at 15 000 steps. The current model reached 0.842 and ranked 25th of the
  29 complete trials.
- Confirmed with 2 seeds at 50 000 steps: 0.973 ± 0.001 against 0.928 ± 0.004 on stage (b), and
  0.940 ± 0.001 against 0.900 ± 0.002 on stage (c). On stage (c) this beats, in 50 000 steps, the
  0.932 that the current model reached in 200 000.
- Verdict: the gain (+0.045 and +0.040) is about three times the 0.015 that two seeds can differ
  by, so trial 26 is now the default of the stage configs. It costs 2.2 times more per step.
- An extra ablation shows that SiLU, the lower Fourier scale and the lower energy weight, at the
  size of the current network, give 0.939 at 15 000 steps: most of the gain at the old step time.
  SiLU matters most, not the energy weight that the importance analysis ranked first.
- 4 trials at a time on the GPU: 4.7 h for the study and the confirmation instead of about 5.9 h
  one at a time (1.36x for the study, 1.15x to 1.18x for the long runs). The results are the same, bit for
  bit, whether a run trains alone or next to others.

## Parallel workers

`run --workers 4` started 4 worker processes on one study. I read "each with its own run directory
and seed" this way: each trial has its own run directory, each worker has its own sampler seed (its
number, 0 to 3), and the training seeds stay those of the config (`runtime.seed` 0,
`theta_colloc.seed` 1, `model.fourier.seed` 0). So a trial's result depends only on its
hyperparameters, whichever worker runs it. The TPE sampler runs with "constant liar" (a running
trial counts as a bad result), so the 4 workers do not all sample around the same point.

The plan was to keep the study in SQLite. SQLite gave no lock error, but it failed in a worse way:
with 4 workers started together, all 4 took the same queued trial (trial 0). The Python `sqlite3`
driver opens the transaction only at the UPDATE, so the check "is the trial still queued?" and the
switch to "running" are not one atomic step across processes. The Optuna journal file
(`runs/optuna_b/study.log`) does not have this problem. In a stress test with 4 workers and 80 fake
trials, 10 of them queued, each trial ran exactly once. The study used the journal, and SQLite is
now refused for more than 1 worker.

Each training process gets a fixed 3072 MB of GPU memory. The largest network of the space peaks
at 2569 MB inside that limit. Before the study I measured the throughput with N copies of trial 0
(the current model, 15 000 steps) started together:

Table 1. Throughput with N trials at a time on the RTX 5070 Ti (16.3 GB, 1.3 GB of it used by Windows).

| Workers | Wall time | Trials per hour | Gain | ms/step per run | GPU use | GPU memory peak | Host memory left (min) |
|---|---|---|---|---|---|---|---|
| 1 | 171 s | 21.0 | 1.00x | 9.2 | 55 % | 4.6 GB | 10.6 GB |
| 2 | 289 s | 24.9 | 1.19x | 17.7 | 73 % | 8.0 GB | 9.0 GB |
| 3 | 395 s | 27.4 | 1.30x | 24.9 | 82 % | 11.3 GB | 7.4 GB |
| 4 | 503 s | 28.6 | 1.36x | 32.3 | 87 % | 14.6 GB | 5.7 GB |

N = 4 gives the most trials per hour within the memory, so the study used it. A fifth process
would not fit in the GPU memory with this limit. The runs share the GPU almost evenly: with N runs
at once each one trains about N times slower. The gain comes mostly from overlapping the parts of
a trial that hardly use the GPU (start-up, XLA compile, evaluations). Pure training gains only
1.14x (9.2 ms per step alone, 32.3 / 4 = 8.1 ms per step with 4 runs).

Determinism: the 10 runs of Table 1 and trial 0 of the study (which ran next to 3 other trials) all
gave y R² 0.8417393092931413, the value of the smoke trial that ran alone and without the memory
limit. Running trials in parallel does not change their results.

## Study

The 60 trials ran from 08:24 to 10:45 (2.34 h). 29 trials completed and 31 were pruned (27 of them
at the first check, 5 000 steps). None failed. The best value went up at trials 1 (0.910), 5
(0.930), 13 (0.946), 15 (0.947) and 26 (0.952). The last 33 trials did not beat trial 26, though
trial 59 came within 0.0003 of it.

Table 2. Top 10 complete trials and the current model: validation y R² of stage (b) at 15 000 steps.

| Rank | Trial | y R² | Mean of the cases | Worst case (case) | T_out RMSE (K) | Network | Fourier features (scale) | lr | Energy weight | Cauchy c | theta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 26 | 0.9521 | 0.953 | 0.839 (11) | 12.0 | 8 x 224 silu | 128 (0.30) | 6.2e-4 | 2.2 | 24.3 | 1024 |
| 2 | 59 | 0.9518 | 0.953 | 0.817 (11) | 12.9 | 7 x 192 silu | 128 (0.41) | 8.2e-4 | 1.1 | 46.2 | 1024 |
| 3 | 39 | 0.9516 | 0.948 | 0.841 (11) | 14.6 | 7 x 192 silu | 128 (0.55) | 1.6e-3 | 1.3 | 76.4 | 1024 |
| 4 | 55 | 0.9510 | 0.949 | 0.822 (11) | 15.1 | 8 x 192 silu | 128 (0.53) | 1.8e-3 | 1.0 | 80.2 | 1024 |
| 5 | 34 | 0.9480 | 0.948 | 0.850 (11) | 12.8 | 7 x 192 silu | 128 (0.31) | 7.9e-4 | 1.1 | 26.7 | 1024 |
| 6 | 29 | 0.9470 | 0.939 | 0.862 (11) | 15.2 | 8 x 192 silu | 128 (0.61) | 8.2e-4 | 2.0 | 40.3 | 1024 |
| 7 | 15 | 0.9470 | 0.951 | 0.794 (11) | 12.8 | 5 x 224 silu | 32 (0.97) | 2.0e-3 | 4.7 | 12.6 | 2048 |
| 8 | 13 | 0.9456 | 0.941 | 0.824 (11) | 14.1 | 8 x 160 silu | 128 (0.58) | 1.2e-3 | 5.2 | 39.9 | 1024 |
| 9 | 20 | 0.9445 | 0.951 | 0.763 (11) | 11.8 | 8 x 192 silu | 128 (0.26) | 1.1e-3 | 1.1 | 32.4 | 1024 |
| 10 | 25 | 0.9443 | 0.943 | 0.822 (11) | 13.1 | 7 x 160 silu | 128 (0.42) | 1.3e-3 | 4.0 | 64.0 | 1024 |
| 25 | 0 (current) | 0.8417 | 0.792 | 0.465 (3) | 22.3 | 6 x 128 tanh | 64 (1.00) | 1.0e-3 | 10.0 | 10.0 | 2048 |

The y R² is pooled over the 16 000 points of the 20 validation cases (the objective). The mean and
the worst of the 20 single-case R² are reported but were not optimized. The Fourier scale
multiplies both sigmas of the current model (sigma_z 2, sigma_t 1).

The best 10 trials have a lot in common:

- all use SiLU;
- 9 of 10 have 128 Fourier features with a low scale, 0.26 to 0.61, so sigma_z 0.5 to 1.2 and
  sigma_t 0.26 to 0.61, against 2 and 1 now;
- the energy weight is 1.0 to 5.2, against 10 now;
- 9 of 10 use 1024 theta, against 2048 now;
- the networks have 5 to 8 layers of 160 to 224 neurons, against 6 x 128 now. Measured alone
  after the study, one step of trial 26 takes 20.9 ms and one of trials 59 and 39 (7 x 192)
  17.2 ms, against 9.4 ms for the current model.

The temperature got better too, though it is not in the objective: T_out RMSE 11.8 to 15.2 K in
the top 10, against 22.3 K for the current model. Trial 26 against trial 0: T_out R² 0.83 against
0.42, v_out/vs R² 0.92 against 0.63, mean error of t50 9.7 % against 27.8 % and of t90 22.6 %
against 88.1 %. Trial 26 has a higher y R² than trial 0 in each of the 20 validation cases (its
lowest is 0.839, against 0.465).

The worst case of all top 10 trials is case 11 (R² 0.76 to 0.86). It has the sharpest and earliest
front of the validation set: the second shortest bed (L = 0.35 m), the fourth highest velocity,
and the highest kL and B_ref of the 20 cases, with the lowest heat of adsorption. Its outlet goes from the
breakthrough time tb (33.8 s) to t90 (35.8 s) in 2 s.

### Importance

I used fANOVA, the default evaluator of `optuna.importance`, over the 29 complete trials. It fits a
random forest to the values, so I repeated it with forest seeds 0 to 4. By default it leaves out
the Fourier scale, because 2 complete trials have no Fourier features.

Table 3. Importance of each parameter (fANOVA, complete trials).

| Parameter | Seed 0 | Mean of seeds 0 to 4 | Range |
|---|---|---|---|
| Energy weight | 0.759 | 0.773 | 0.671 to 0.891 |
| lr | 0.097 | 0.094 | 0.043 to 0.118 |
| Activation | 0.086 | 0.067 | 0.024 to 0.104 |
| Number of Fourier features | 0.044 | 0.043 | 0.022 to 0.076 |
| Number of theta | 0.012 | 0.013 | 0.002 to 0.020 |
| Cauchy c | 0.001 | 0.005 | 0.000 to 0.016 |
| Width | 0.000 | 0.004 | 0.000 to 0.011 |
| Depth | 0.000 | 0.001 | 0.000 to 0.002 |

Only the first place is a firm result. With the Fourier scale included (the 27 complete trials
with Fourier features), fANOVA puts the scale first (0.46, mean 0.39) and the energy weight second
(0.21, mean 0.31). PED-ANOVA (top 10 % against all) gives energy weight 0.32, Cauchy c 0.21, width
0.16 and depth 0.15. Below the energy weight the order changes with the method and with the forest
seed, which is expected from 29 trials. Also, the importance sees only the complete trials: the
pruned half is not in the fit.

The raw values say the same about the energy weight. The 23 complete trials with a weight up to 7
reached 0.910 to 0.952, and the 6 with a weight of 10 or more reached −0.942 to 0.905.

One thing the study cannot separate: the 6 tanh trials all came from the random start, and all had
an energy weight of 7.5 or more. TPE never went back to tanh with a low energy weight, so the
importance of the activation is mixed with that of the energy weight. The ablation below checks
this.

### Ablation (extra, 15 000 steps)

Trial 26 changes all 9 parameters at once. To see which changes carry the gain, I trained the
current model with one change of trial 26 at a time, and with the three changes that the top 10
share at the size of the current network. These are 4 extra runs with the protocol of the trials
(15 000 steps, seed 0, 4 at a time, in the throwaway study `runs/optuna_bench/ablation`).

Table 4. Ablation from the current model (validation y R² of stage b at 15 000 steps).

| Setup | y R² | Gain | Mean of the cases | Worst case | T_out RMSE (K) | Step alone |
|---|---|---|---|---|---|---|
| Current model (trial 0) | 0.842 | | 0.792 | 0.465 | 22.3 | 9.4 ms |
| Energy weight 10 to 2.2 | 0.866 | +0.024 | 0.828 | 0.595 | 22.2 | 9.4 ms |
| tanh to SiLU | 0.921 | +0.079 | 0.903 | 0.755 | 16.4 | about 9.8 ms |
| Fourier scale 1 to 0.30 | 0.879 | +0.037 | 0.872 | 0.721 | 15.0 | 9.4 ms |
| The three together | 0.939 | +0.097 | 0.946 | 0.752 | 11.8 | about 9.8 ms |
| Trial 26 (also 8 x 224, 128 Fourier features, lr, Cauchy c, 1024 theta) | 0.952 | +0.110 | 0.953 | 0.839 | 12.0 | 20.9 ms |

The step time of the SiLU rows is an estimate: the grid of the plan measured +0.4 ms for SiLU on
the 6 x 128 network.

SiLU alone gives the most (+0.079), and the energy weight alone the least (+0.024). So the fANOVA
ranking above overrated the energy weight (0.76). The 4 complete tanh trials all had a weight of
10 or more, and they are 4 of the 6 complete trials with such a weight, so part of the effect of
tanh went to the energy weight. The weight still matters: the other 2, with SiLU and a weight of
11.7 and 19.0, reached only 0.905 and 0.701. The three changes together give 0.939, 88 % of the gain
of trial 26 (+0.097 of +0.110), with a network of the current size, at less than half the step
time. Their gains do not add up (0.079 + 0.037 + 0.024 = 0.140): they partly fix the same thing.
The larger network of trial 26 matters most for the hardest case: its worst case is 0.839,
against 0.752 for the three changes alone. These are single runs at 15 000 steps. The ablation
does not say whether the small setup keeps up with trial 26 at 50 000 steps or on stage (c).

## Confirmation (50 000 steps, 2 seeds)

The top 3 trials other than the current model (26, 59 and 39) and the current model were trained
again for 50 000 steps (the cosine decay of the learning rate spans the 50 000 steps), with training
seeds 0 and 1, on the same stage (b) validation set. The 8 runs went 4 at a time.

Table 5. Confirmation on stage (b): validation y R² at 50 000 steps.

| Setup | y R², seed 0 / seed 1 | Mean ± seed difference | Mean of the cases | Worst case (case 11) | T_out RMSE (K) | Network |
|---|---|---|---|---|---|---|
| Current | 0.9263 / 0.9302 | 0.9283 ± 0.0040 | 0.921 ± 0.000 | 0.773 ± 0.041 | 13.1 ± 0.4 | 6 x 128 tanh, FF 64 (1.00) |
| Trial 26 | 0.9739 / 0.9726 | 0.9733 ± 0.0013 | 0.980 ± 0.001 | 0.842 ± 0.020 | 9.2 ± 0.2 | 8 x 224 silu, FF 128 (0.30) |
| Trial 59 | 0.9723 / 0.9734 | 0.9728 ± 0.0011 | 0.979 ± 0.001 | 0.843 ± 0.006 | 9.5 ± 0.5 | 7 x 192 silu, FF 128 (0.41) |
| Trial 39 | 0.9715 / 0.9730 | 0.9722 ± 0.0015 | 0.977 ± 0.004 | 0.850 ± 0.000 | 9.5 ± 0.3 | 7 x 192 silu, FF 128 (0.55) |

Here "±" is the difference between the 2 seeds, not a standard deviation. The other parameters of
each trial are in Table 2.

The ranking of the study held at 50 000 steps. The current model reached 0.928, which matches the
0.925 of Phase 3 within its seed noise, and the three tuned setups reached 0.972 to 0.973. Trial 26
has the best mean, 0.045 above the current model, three times the 0.015 that two seeds of one setup
can differ by. Its 2 seeds differ by only 0.001. The three tuned setups are within 0.001 of each
other, so the choice among them is close to a tie: trial 26 goes on because it has the best mean,
as planned. The gain shows in every metric. The mean of the single cases went from 0.921 to 0.980,
the worst case (case 11 for every setup) from 0.773 to 0.842, and the T_out RMSE from 13.1 to 9.2 K.

The gap shrank from 0.110 at 15 000 steps to 0.045 at 50 000 steps, because the current model
gains more from the longer training (0.842 to 0.928) than trial 26 (0.952 to 0.973). The tuned
setups learn faster, and the study, with its short trials, favored exactly that. They still end
clearly higher.

### Stage (c) test

Trial 26 beat the current model on stage (b), so both went to stage (c), where all 31 parameters
vary. The validation set is `data/provisorio_MOL_n2_stage_c_val20.h5`, 20 other cases than those of
stage (b). Again 50 000 steps, 2 seeds each, physics only.

Table 6. Stage (c) test: validation y R² at 50 000 steps.

| Setup | y R², seed 0 / seed 1 | Mean ± seed difference | Mean of the cases | Worst case (case) | T_out RMSE (K) |
|---|---|---|---|---|---|
| Current | 0.9011 / 0.8987 | 0.8999 ± 0.0024 | 0.859 ± 0.007 | 0.306 ± 0.004 (18, 18) | 16.2 ± 0.2 |
| Trial 26 | 0.9410 / 0.9397 | 0.9403 ± 0.0013 | 0.926 ± 0.002 | 0.617 ± 0.005 (11, 6) | 14.9 ± 0.2 |

The gain held on stage (c): +0.040. The current model gave 0.900, the same as the 0.901 of Phase 2
at this length. Trial 26 reached 0.940 in 50 000 steps, above the 0.932 of the long run of Phase 3
(`runs/phase3c_long`, 200 000 steps). That run was the plateau of the current model: the Phase 4
continuations did not move it. The worst case doubled (0.306 to 0.617) and the mean of the cases
went from 0.859 to 0.926. The temperature gained less than on stage (b): T_out RMSE 16.2 to 14.9 K.
Case numbers here are those of the stage (c) validation set.

## Time gain from the parallel workers

Table 7. Wall time with 4 workers against the same runs one at a time.

| Part | Runs | With 4 workers | One at a time | Gain |
|---|---|---|---|---|
| Benchmark | 4 x trial 0, 15 000 steps | 503 s | 684 s (measured, 4 x 171 s) | 1.36x |
| Study | 60 trials, 31 of them pruned | 2.34 h | 3.18 h (estimate) | 1.36x |
| Confirmation, stage (b) | 8 runs of 50 000 steps | 1.62 h | 1.86 h (estimate) | 1.15x |
| Confirmation, stage (c) | 4 runs of 50 000 steps | 0.76 h | 0.90 h (estimate) | 1.18x |
| Study and confirmation | | 4.72 h | 5.94 h | 1.26x |

The "one at a time" times of the study and the confirmation are estimates. For each run I took
its steps times its step time alone, plus 24 to 44 s of start-up, compile and evaluations (more
for larger networks and on stage c). After the study I measured the step time alone of the three
networks of the confirmation: 9.4 ms (current), 17.2 ms (trials 59 and 39) and 20.9 ms (trial
26). For the 60 trials I used the step-time grid of the plan. Its fit underestimated the two large
SiLU networks by 14 and 18 %, so I multiplied the SiLU trials by 1.16. Without that correction the
study gain would be 1.22x.

The 4 workers saved 1.2 h of about 5.9 h. They paid off most in the study: its trials are short,
so start-up, compile and evaluations are a large part of each one, and they overlap well with the
training of the others. In the 50 000 step runs that part is small. There the GPU is shared almost
evenly in time (the current model ran 5.4 to 5.6 times slower than alone, trial 26 only 2.8 times), so
training goes only a little faster than one run at a time.

## Verdict

Yes. Trial 26 beats the current model by much more than ~0.015 on both stages: +0.045 on stage
(b) and +0.040 on stage (c), 2 seeds each, and it is better in every other metric too (mean and
worst of the cases, T_out RMSE). So it is now the default (commit 23bc72a):

- `configs/phase2_base.yaml` has the 9 values of trial 26, and the Fourier sigmas and theta
  renewal that follow from them, so stages (a), (b) and (c) train it. Stage (a) was not tested.
- `configs/phase2_stage_c.yaml` sets the theta renewal for its 50 fixed theta (every 19.5 passes
  over the pool, about 2 600 steps as before) and takes the Cauchy loss from the base (c = 24.3
  instead of 10).
- The values used until now are in `configs/phase2_stage_b_before_optuna.yaml` and
  `configs/phase2_stage_c_before_optuna.yaml`. The configs of the Phase 3 and 4 experiments, of
  the fine-tune (it starts from `runs/phase3c_long`) and of the study inherit them. I checked that
  each of these configs resolves to exactly the same values as before the change.
- `tests/test_optuna_stage_b.py` checks that the stage configs are trial 26.

The price is the network: 475 750 weights instead of 136 710, and 20.9 ms per step alone instead
of 9.4 ms, so 50 000 steps cost 0.30 GPU-h instead of 0.14. The ablation suggests that a 6 x 128
network with SiLU, Fourier scale 0.30 and energy weight 2.2 could get most of the gain at the old
cost. Before the long stage (c) run with the new default, I would confirm that small setup with 2
seeds at 50 000 steps on stages (b) and (c), about 30 to 40 min with 4 workers. If it stays within
~0.015 of trial 26, it would be the better default. I did not run this, since it was not part of
the approved plan.

## Caveats

- **Selection on the validation set.** The 60 trials and the confirmation on stage (b) use the same
  20 validation cases. The 2 new seeds remove the luck of the initialization, but not the selection
  on these 20 cases. The stage (c) test runs on 20 other cases in the whole domain, so it is the
  check outside the selection, and the gain held there.
- **Short trials.** The study ranked setups by what they reach in 15 000 steps, which favors setups
  that learn fast: the gap was 0.110 at 15 000 steps and 0.045 at 50 000. A setup that learns
  slowly but ends higher would have been missed.
- **Only y is optimized.** The temperature and the velocity at the outlet improved too, but nothing
  in the study protected them.
- **No long run yet.** The current model went from 0.900 at 50 000 steps to 0.932 at 200 000 on
  stage (c). Whether trial 26 gains as much with a longer run is open.
- **Provisional references.** All numbers use the MOL data of this repository, not the v23b
  dataset.
