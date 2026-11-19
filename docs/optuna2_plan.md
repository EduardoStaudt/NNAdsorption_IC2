# Second Optuna study on stage (b): plan and cost

Prepared on 2026-11-19 (branch `pinn-v23b`). Not run yet. It waits for your approval of the estimate
and of three decisions: the number of workers (GPU memory), the w0 of the sine layers, and how a
diverged trial is recorded. The first study is explained in `docs/optuna_search_space.md` and its
results are in `docs/optuna_stage_b_results.md`. Code: `scripts/optuna_stage_b.py --space b2`,
tests `tests/test_optuna_stage_b.py`.

## Protocol

The same as in the first study (`docs/optuna_stage_b.md`, section "Protocol"):

- each trial is one `train_phase2.py` run of `configs/optuna_stage_b.yaml` with the sampled values
  as overrides, 15 000 steps (the cosine decay spans them), validation every 2 500 steps;
- the objective is the final validation y R² of stage (b), pooled over the 20 cases of
  `data/provisorio_MOL_n2_stage_b_val20.h5`;
- TPE sampler, multivariate and grouped, with "constant liar", seeded by the worker number, about
  10 random trials first;
- median pruner from 5 complete trials on, at 5 000, 7 500, 10 000 and 12 500 steps. It compares
  only trials of the second study;
- determinism on.

What changes:

- a new study, `stage_b2` in `runs/optuna_b2/study.log`. The first study stays as it is;
- trial 0 is trial 26 of the first study, queued by hand (a test checks that it comes out with the
  same values). With determinism on it should give y R² 0.9521175 again, the value of the first
  study;
- the search space (next section);
- the workers, the sine layers and the diverged trials, if you approve the decisions below.

## Search space

Table 1. Search space of the second study against the first.

| Parameter | First study | Second study | Distribution | Trial 26 | Why |
|---|---|---|---|---|---|
| Depth | 4 to 8 | 6 to 12 | uniform integer | 8 | trial 26 was at the top of the range |
| Width | 64 to 256, step 32 | 128 to 320, step 32 | uniform integer | 224 | kept around trial 26 |
| Activation | tanh, SiLU | tanh, SiLU, GELU, softplus, Mish, sin (SIREN) | categorical | SiLU | smooth activations only |
| Fourier features | 0, 32, 64, 128 | 64, 128, 256 | categorical | 128 | trial 26 had the largest choice |
| Fourier scale | 0.25 to 4 | 0.05 to 1 | log-uniform | 0.297 | trial 26 was 6 % from the bottom |
| Learning rate | 1e-4 to 3e-3 | 2e-4 to 2e-3 | log-uniform | 6.2e-4 | kept around trial 26 |
| Energy weight | 1 to 100 | 0.3 to 10 | log-uniform | 2.23 | half of the top 10 were at 1.0 to 1.3 |
| Cauchy c | 1 to 100 | 5 to 60 | log-uniform | 24.3 | kept around trial 26 |
| Theta per draw | 512, 1024, 2048, 4096 | 512, 1024, 2048 | categorical | 1024 | kept around trial 26 |

Source: `SPACE_B1` and `SPACE_B2` in `scripts/optuna_stage_b.py`.

Every trial now has Fourier features, so every trial has a Fourier scale. Trial 26 is inside every
range of the second study (a test checks this). ReLU and ELU are left out because the residuals
need a smooth second derivative in z, see `docs/optuna_search_space.md`. The same page describes the
new activations and the initialization of the sine layers.

## Cost benchmark

Short runs of 1 500 steps, one at a time on the GPU, in `runs/optuna_bench` (logs `b2_grid.log`,
`b2_extra.log` and `b2_siren_w0.log`). The other values are those of trial 26.

Table 2. Step time and peak GPU memory.

| Network | Activation | Fourier features | ms per step | Peak memory (MB) | Fits in 3072 MB? |
|---|---|---|---|---|---|
| 8 x 224 | SiLU (trial 26) | 128 | 20.2 | 2 253 | yes |
| 8 x 224 | tanh | 128 | 19.9 | 1 182 | yes |
| 8 x 224 | GELU | 128 | 21.3 | 2 225 | yes |
| 8 x 224 | softplus | 128 | 21.9 | 1 148 | yes |
| 8 x 224 | Mish | 128 | 20.5 | 2 980 | yes, 92 MB to spare |
| 8 x 224 | sin (w0 = 30) | 128 | 20.0 | 1 383 | yes |
| 6 x 128 | SiLU | 64 | 10.0 | 1 025 | yes |
| 6 x 128 | SiLU | 256 | 12.0 | 1 278 | yes |
| 12 x 128 | SiLU | 64 | 15.6 | 1 885 | yes |
| 8 x 224 | SiLU | 256 | 29.3 | 2 444 | yes |
| 6 x 320 | SiLU | 64 | 29.8 | 2 435 | yes |
| 10 x 256 | SiLU | 128 | 27.3 | 3 121 | no |
| 12 x 320 | SiLU | 64 | 44.5 | 4 565 | no |
| 12 x 320 | SiLU | 256 | 47.8 | 4 709 | no |
| 12 x 320 | Mish | 256 | 47.4 | 5 610 | no |

Source: `runs/optuna_bench/b2_grid.log` and `b2_extra.log`. I measured the last four rows with a
12 000 MB limit. At 3072 MB the 12 x 320 networks stopped with an out-of-memory error, with SiLU,
GELU, Mish and sin alike.

- **The activation hardly changes the step time** (−1 % to +8 % against SiLU at the size of trial
  26), **but it changes the memory a lot.** SiLU and GELU need about twice as much as tanh and
  softplus, Mish about 2.5 times.
- **Memory fit.** For SiLU the peak follows 114 + 1.13 d w + 1.06 nf MB (d depth, w width, nf Fourier
  features) within 3 % in all 9 runs. tanh needs 0.47 of the d w term of SiLU, softplus 0.45, sin
  0.57, GELU 0.99 and Mish 1.36 (factors from the runs at the size of trial 26; at 12 x 320 Mish
  measured 5 610 MB, less than the 6 319 MB this predicts).
- **Step-time fit.** t ≈ 7.2 + 1.7e-4 d w + 2.9e-5 d w² + 0.019 nf ms for SiLU. It is rough: the
  leave-one-out error is 25 % (rms). 256 Fourier features cost 9 ms more than 128 at 8 x 224, but
  only 2.0 and 3.3 ms more than 64 at 6 x 128 and at 12 x 320. The step time depends on more than
  these terms. For the estimate I use the measured times where I have them.
- **Start-up** and compile took 25 s on average (18 to 37 s to the first log line, at step 100).
  With the validations and the final evaluation I count 35 s per run.
- **The memory limit does not change the results.** Trial 26 at 7000 MB gave the same training log
  as in the first study at 3072 MB, bit for bit, for the 1 500 steps I compared
  (`b2_regress_t26_cap7000`). So the limit can be raised, trial 0 should still give 0.9521175, and
  the trial 26 runs can be reused in the confirmation.

## Decision 1: GPU memory and the number of workers

The first study ran 4 trials at a time with 3072 MB per process: 4 x (3072 + about 260 MB of CUDA
context) = 13.3 GB of the 15.0 GB that Windows leaves free on the RTX 5070 Ti. Table 2 shows that
the larger networks of the second study do not fit in 3072 MB. The peak is not the training step
but the gradient statistics of the adaptive loss weights (`grad_norm`, every 100 steps,
`Trainer._grad_stats`): one persistent tape and one backward pass per loss group.

A trial that runs out of memory is recorded as failed, and TPE learns only from complete and pruned
trials (`optuna/samplers/_tpe/sampler.py`, Optuna 5.1). So it would not learn to avoid the networks
that do not fit, and would keep proposing them.

Table 3. Options for the workers.

| Option | Memory per process | Sizes that fit, with 10 % margin | Trials per hour against one at a time | New code |
|---|---|---|---|---|
| **A. 2 workers (recommended)** | 7000 MB | all (largest measured peak: 5 610 MB) | about 1.19x | none |
| B. 3 workers | 4400 MB | tanh, softplus and sin all, SiLU 93 %, GELU 94 %, Mish 72 % | about 1.30x | none |
| C. 4 workers sharing 14 GB | what each trial needs, 3072 to 7000 MB | all | between 1.19x and 1.36x | a memory budget shared by the workers, with tests |

Source: memory fit above, over the 147 sizes of the space (7 depths x 7 widths x 3 Fourier
choices). Gains: 1.19x, 1.30x and 1.36x were measured in the first study with 2, 3 and 4 runs of
the small model at once (`docs/optuna_stage_b_results.md`, Table 1).

I recommend A: two values on the command line (`--workers 2 --gpu-mem-mb 7000`, 2 x 7.26 GB = 14.5
GB), and every network fits. Against C it loses at most about 13 % of the wall time (1.19x against
1.36x), likely less, because large networks already keep the GPU busy. B leaves the largest Mish,
SiLU and GELU networks out of memory, a part of the space TPE would keep proposing.

## Decision 2: w0 of the sine layers (SIREN)

The sine layers follow the SIREN paper (Sitzmann et al., 2020): y = sin(w0 (x W + b)), with the
initialization of the paper, which scales with w0 (`docs/optuna_search_space.md`). The paper uses
w0 = 30, and so does `configs/pinn_base.yaml`. Before the study I trained trial 26 with sin in place
of SiLU, with the trial protocol (15 000 step schedule), up to its first validation at 2 500 steps.

Table 4. Trial 26 with sin, at the first validation (2 500 steps).

| w0 | Learning rate | y R² at 2 500 steps | Total loss at steps 100 / 500 / 2 500 | What happened |
|---|---|---|---|---|
| 30 | 6.2e-4 (trial 26) | none | 1.4e8 / NaN / NaN | diverged: NaN by step 500, the run stops with an error at the validation |
| 30 | 2e-4 (bottom of the range) | −0.11 | 0.44 / 3.4e8 / 1.5e4 | the loss jumps by 9 orders of magnitude and comes back only in part |
| 30 | 6.2e-5 | −0.30 | 0.26 / 11 / 0.56 | a smaller jump, still far behind |
| 30 | 2e-5 | −1.18 | 0.26 / 0.57 / 0.29 | stable, but it learns too slowly |
| 5 | 6.2e-4 | −0.08 | 0.38 / 7.5 / 1.08 | a jump at step 200 (loss 109) |
| 5 | 2e-4 | −1.58 | 0.22 / 0.45 / 0.30 | stable, too slow |
| 1 | 6.2e-4 | 0.446 | 0.23 / 0.35 / 0.24 | stable, and it learns |

Source: `runs/optuna_bench/b2_extra.log` and `b2_siren_w0.log`, runs `b2_sin_*`. For reference, at
2 500 steps trial 26 (SiLU) had 0.852 and the tanh model before the first study had 0.33.

With w0 = 30 no learning rate gave a network that learns in 2 500 steps, and w0 = 5 did not either.
Only w0 = 1 did. One likely reason: the Fourier features already put high frequencies in front of
the trunk, and a large w0 makes every sine layer oscillate fast in z on top of that, so the z
derivatives in the residual and in the outlet condition start large. This is a guess. I did not
test it. With w0 = 1 the initialization is the same formula: hidden weights uniform in
(−√(6/n), √(6/n)), the He uniform limit (1.4 times the Glorot limit at 224 neurons), and a first
layer with weights in (−1/n, 1/n), almost linear at the start (the Fourier features in front of it
already give the high frequencies).

Table 5. Options for the sine layers.

| Option | Sine layers in the study | What to expect |
|---|---|---|
| A. w0 = 30, as in the paper (the current setting) | unchanged | most sin trials diverge (failed) or are pruned at 5 000 steps. Cheap, but sin gets no fair test |
| **B. w0 = 1, same initialization formula (recommended)** | `model.siren_w0=1` in every run of the second study (only sin uses it) | sin learns. Trial 26 and the other activations do not change |
| C. w0 sampled for sin trials only | a 10th parameter, for example log-uniform 1 to 30 | spends sin trials on values that did not work here (w0 of 5 and 30) |

## Decision 3: how a diverged trial is recorded

In the first study no trial diverged. In the code, a trial whose validation y R² turns NaN is
recorded as failed. In practice the validation never writes that NaN: the training process stops
with an error while it computes the metrics (the outlet metrics of a NaN network are empty), and
that is recorded as failed too. TPE ignores
failed trials, so it would not learn that a setup diverges, and could propose the same region
again (sin with w0 = 30 and a high learning rate, or a large network at the top learning rate).

- **A.** Keep "failed", as in the first study.
- **B. (recommended)** Record it as pruned, with a note "diverged at step N". TPE then counts it as a
  bad trial, like a pruned trial without a validation value (it ranks those last), and the median
  pruner is not affected (it uses complete trials only). The change is small: the objective also
  reads the training log, and a run whose loss is NaN there counts as diverged even when the process
  stops with an error. One new test. The report shows diverged trials in their own column.

## Estimate

Table 6. Cost of the second study: GPU hours with one run at a time, and the wall time with 2
workers (decision 1, option A).

| Part | TPE stays near trial 26 (20.2 ms) | Space average (25.2 ms) | TPE moves to the largest networks (47.8 ms) |
|---|---|---|---|
| 60 trials | 4.1 | 4.8 | 8.0 |
| Confirmation on stage (b), 6 runs x 50 000 steps | 1.7 | 2.2 | 4.0 |
| Stage (c), 2 runs x 50 000 steps | 0.6 | 0.7 | 1.3 |
| **Total, one run at a time** | **6.5** | **7.7** | **13.4** |
| Wall time with 2 workers | 5.5 to 5.8 | 6.6 to 6.9 | 11.4 to 12.1 |

Source: step times of Table 2, 35 s per run of start-up, compile and evaluations.

- Every column has 10 random trials at the space average first. The other 50 trials and the 8
  confirmation runs take the step time of the column.
- 31 of the 60 trials are pruned at about 5 600 steps, as in the first study.
- 2 workers make the trials 1.19x faster (measured in the first study) and the 50 000 step runs 1.0x
  to 1.15x faster (1.15x was measured with 4 workers).
- Upper bound, with no pruning and every run at the largest network: 17.9 GPU-h.
- For comparison, the first study took 4.7 h of wall time in all (5.9 GPU-h one run at a time).

I expect something between the first two columns: in the first study TPE stayed around the best
trials, and trial 26 starts the second study. If larger networks turn out better, the cost moves to
the third column.

## Confirmation and stage (c)

After the 60 trials, the `confirm` command:

1. trains the 3 best trials of the second study (trial 0 left out, it is trial 26) for 50 000 steps
   with seeds 0 and 1 on stage (b): 6 new runs. Trial 26 is not trained again: its two seeds at
   50 000 steps from the first study are reused (`runs/optuna_b/confirm_b/t0026_seed{0,1}`, y R²
   0.9739 and 0.9726);
2. takes the best of the 3 by the mean of its two seeds and trains it on stage (c), 50 000 steps,
   seeds 0 and 1: 2 new runs. It goes to stage (c) even if it does not beat trial 26 on stage (b).
   Trial 26 on stage (c) is reused as well (`runs/optuna_b/confirm_c/t0026_seed{0,1}`, 0.9410 and
   0.9397).

The results go to `runs/optuna_b2/confirm.json` and to `docs/optuna2_results.md`, with a table of
the best and median y R² per activation. I change no default without your approval.

## Caveats

- **Short trials.** The study ranks setups by what they reach in 15 000 steps. The confirmation at
  50 000 steps checks whether that ranking holds.
- **Seed noise.** Two seeds of one setup can differ by about 0.015 in y R² at 50 000 steps (the
  figure of the first study, whose own seed pairs differed by 0.001 to 0.004). A gain below that is
  within the noise.
- **Six activations in 60 trials.** The random start gives each activation fewer than 2 trials on
  average. TPE can drop an activation after one bad draw, so the table per activation will show how
  many trials each one had.
- **The estimate** rests on a rough step-time fit and on where TPE goes, which I cannot know before.

## Commands

Run these inside WSL, from the repository root (shown with decision 1, option A):

```
mkdir -p runs/optuna_b2
(setsid nohup ~/ic2env/bin/python scripts/optuna_stage_b.py run --space b2 --n-trials 60 --workers 2 --gpu-mem-mb 7000 > runs/optuna_b2/run.out 2>&1 < /dev/null &)
~/ic2env/bin/python scripts/optuna_stage_b.py report --space b2 --importance
~/ic2env/bin/python scripts/optuna_stage_b.py plot --space b2 --out docs/figures/optuna_b2
touch runs/optuna_b2/STOP     # pause: each worker finishes its running trial
~/ic2env/bin/python scripts/optuna_stage_b.py confirm --space b2 --workers 2 --gpu-mem-mb 7000
```

Each worker logs to `runs/optuna_b2/worker<i>.out`, and each trial to `runs/optuna_b2/t<number>/`.
