# Optuna study on stage (b): plan and cost

Prepared on 2026-11-19 (branch `pinn-v23b`) and approved the same day with 60 trials and several
trials at a time on the GPU. The parallel setup is in the section "Parallel workers". Results are
in `docs/optuna_stage_b_results.md`. Code: `scripts/optuna_stage_b.py`, base config
`configs/optuna_stage_b.yaml`, tests `tests/test_optuna_stage_b.py`.

## Protocol

Each trial is one `train_phase2.py` process with `configs/optuna_stage_b.yaml`. That config is the
best stage (b) setup of Phase 3 (Sobol theta, Cauchy loss, physics only), shortened to 15 000 steps
(the cosine decay spans those 15 000 steps), with the validation curve every 2 500 steps. The
sampled values go in as `--set` overrides. With no overrides the run is the current model (a test
checks this for stages b and c).

- **Objective:** the final validation y R² of stage (b). This is the R² of the outlet mole
  fraction of gas 1, pooled over the 16 000 points of the 20 cases in
  `data/provisorio_MOL_n2_stage_b_val20.h5`, the same metric as in all phase reports. The mean
  and the lowest R² of the single cases are reported too (report and confirm), but they are not
  optimized.
- **Sampler:** TPE, multivariate and grouped (the Fourier scale exists only with Fourier
  features). Its seed is the worker number. With "constant liar" on, a running trial counts as a
  bad result, so the workers do not all propose the same region. The first 10 trials are random.
  Trial 0 is the current model, put in the queue by hand.
- **Pruner:** median. A trial stops at 5 000, 7 500, 10 000 or 12 500 steps if its best y R² so
  far is below the median of the earlier trials at that step. Pruning starts after 5 complete
  trials.
- **Failures:** a trial whose y R² turns NaN, or whose process crashes, is marked failed, and the
  study goes on.
- **Storage:** an Optuna journal file, `runs/optuna_b/study.log`. SQLite was the plan, but with 4
  workers started together all of them took the same queued trial. The Python `sqlite3` driver
  opens the transaction only at the UPDATE, so the check "still queued?" and the switch to
  "running" are not atomic across processes. The journal records which worker owns each running
  trial, and SQLite is now refused for more than 1 worker.
  - To pause, create `runs/optuna_b/STOP`. Each worker finishes its running trial first.
  - To resume, delete the file and run the same command.
  - If the study is killed, its training processes get SIGTERM. The next `run` marks the trials
    it finds running as failed ("interrupted") and queues their parameters once more. Only one
    `run` may be active at a time.
- **Determinism:** on. Trial 0 must reproduce the smoke trial below (y R² 0.8417393) bit for bit,
  whether it runs alone or next to other trials (checked, see "Parallel workers").

## Parallel workers

`run --workers N` starts N worker processes on the same study. Each trial still has its own run
directory and the usual training seeds, so its result depends only on its hyperparameters. Each
training process gets a fixed 3072 MB of GPU memory (`runtime.gpu_memory_mb`). The largest network
of the space (8 x 256, SiLU, 128 Fourier features, 4096 theta) peaks at 2569 MB inside that limit,
and each process takes about 260 MB more outside it (CUDA context).

I measured the throughput by starting N copies of trial 0 together (15 000 steps each) for
N = 1 to 4. Every one of the 10 runs gave y R² 0.8417393092931413, the value of the smoke trial,
which ran alone and without the memory limit.

| Workers | Wall time | Trials per hour | Gain | ms/step per run | GPU use | GPU memory peak | Host memory left (min) |
|---|---|---|---|---|---|---|---|
| 1 | 171 s | 21.0 | 1.00x | 9.2 | 55 % | 4.6 GB | 10.6 GB |
| 2 | 289 s | 24.9 | 1.19x | 17.7 | 73 % | 8.0 GB | 9.0 GB |
| 3 | 395 s | 27.4 | 1.30x | 24.9 | 82 % | 11.3 GB | 7.4 GB |
| 4 | 503 s | 28.6 | 1.36x | 32.3 | 87 % | 14.6 GB | 5.7 GB |

The GPU has 16.3 GB, and the Windows desktop uses about 1.3 GB of it. WSL has 12.3 GB of host
memory. The trials share the GPU almost evenly: with N runs at once each one trains about N times
slower. So the gain comes from overlapping the parts of a trial that hardly use the GPU (start-up,
XLA compile, evaluations), not from faster training. I chose N = 4, the most trials per hour
within the memory, which leaves 1.7 GB of GPU memory free. The runs start together here, which
makes the measurement a lower bound: in the study the trials drift apart, and one trial's start-up
overlaps the others' training.

## Search space

| Parameter | Range | Current model |
|---|---|---|
| Trunk depth | 4 to 8 | 6 |
| Trunk width | 64 to 256, step 32 | 128 |
| Activation | tanh, SiLU | tanh |
| Fourier features | 0, 32, 64, 128 | 64 |
| Fourier scale (multiplies sigma_z = 2 and sigma_t = 1) | 0.25 to 4, log | 1 |
| Max learning rate (cosine down to 1 % of it) | 1e-4 to 3e-3, log | 1e-3 |
| Energy weight | 1 to 100, log | 10 |
| Cauchy scale c | 1 to 100, log | 10 |
| Sobol theta per draw | 512, 1024, 2048, 4096 | 2048 |

The points per theta stay at 512, so the pool has (1 + n_theta) x 512 points. The theta renewal
period stays at 2 561 steps, because `resample_every_epochs` is scaled with the pool. A trial
therefore sees about 6 draws of n_theta theta in 15 000 steps. The Fourier scale keeps the 2:1
ratio between sigma_z and sigma_t.

## Confirmation (`confirm` command)

1. **Stage (b):** the top 3 complete trials and the current model, 2 seeds each
   (`runtime.seed` 0 and 1), 50 000 steps. This is the length behind the Phase 2 and 3 stage (b)
   numbers, where the current model gave 0.932 and 0.917.
2. **Stage (c):** if the best of the 3 has a higher 2-seed mean than the current model, that setup
   and the current model also run on stage (c). Each gets 2 seeds of 50 000 steps with
   `phase2_stage_c.yaml`, physics only, validated on the 20 stage (c) cases.
3. **Results:** written to `runs/optuna_b/confirm.json`. Finished runs are reused, so the command
   can be resumed.

## Cost benchmark

All runs are in `runs/optuna_bench`.

**Smoke trial.** I ran the study script itself for 1 trial in a throwaway study
(`runs/optuna_bench/smoke`). The trial was the current model at 15 000 steps:

- 3.1 min wall time in total;
- training took 139 s (9.2 ms per step on average, 8.94 median);
- the XLA compile took 11 s and the start-up about 20 s (cold caches);
- the periodic validations took 2.3 s and the final evaluation about 5 s;
- the study process then waited up to 10 s to notice the end. It now reads the curve every 2 s.

Validation y R² along the run: 0.33, 0.64, 0.78, 0.82, 0.84 and 0.842 at 15 000 steps.

**Step time.** Short runs (1 500 steps, 3 000 for the theta pair) at the corners of the space:

| Network | ms/step (median) |
|---|---|
| 4 x 64 | 6.5 |
| 8 x 64 | 8.8 |
| 6 x 128 (current) | 8.9 |
| 4 x 256 | 11.6 |
| 8 x 256 | 20.0 |
| 6 x 128, SiLU | 9.3 |
| 6 x 128, 128 Fourier features | 9.7 |
| 6 x 128, no Fourier features | 8.8 |
| 6 x 128, 4096 / 512 theta | 9.0 / 8.9 |
| 8 x 256, SiLU, 128 Fourier features | 21.3 |

A fit of the tanh corners gives t ≈ 3.7 + 0.55 depth + 22.5 depth width² / 10⁶ ms, with residuals up
to 0.3 ms. Averaged over the search space this is 11.4 ms per step, which gives:

- 3.4 min for a full trial;
- 1.8 min for a trial pruned at about 7 000 steps;
- 5.9 min for a trial at the largest network.

## Estimate

GPU-h here equals wall hours, because runs go one at a time and the GPU is held during start-up.

| | 30 trials | 60 trials |
|---|---|---|
| Trials, with pruning (40 % pruned at about 7 000 steps) | 1.4 | 2.8 |
| Trials, no pruning | 1.7 | 3.4 |
| Trials, all at the largest network (upper bound) | 2.9 | 5.9 |
| Confirmation on stage (b), 8 runs x 50 000 steps | 1.3 (0.9 to 2.1) | 1.3 (0.9 to 2.1) |
| Stage (c) test, 4 runs x 50 000 steps, only if a setup wins | 0.6 (0.5 to 0.9) | 0.6 (0.5 to 0.9) |
| **Total with pruning** (without / with the stage c test) | **2.7 / 3.3** | **4.1 / 4.7** |
| Total, no pruning (without / with the stage c test) | 3.0 / 3.6 | 4.7 / 5.3 |

The ranges of the confirmation rows go from the smallest to the largest network winning.

With 4 workers (section "Parallel workers") the wall time is shorter, but by less than 4x. In the
benchmark 4 runs together gave 1.36x the trials per hour of 1 run. Most of that comes from
overlapping start-up, compile and evaluations: pure training went only from 9.2 ms per step alone
to 32.3 / 4 = 8.1 ms per step together, 1.14x. So the 15 000 step trials gain about 1.36x and the
50 000 step confirmation runs about 1.15x. For 60 trials with pruning and the stage (c) test this
gives about 2.1 + 1.1 + 0.5 = 3.7 h instead of 4.7 h. The GPU is shared, so GPU-h no longer equals
the sum of the run times: each run takes about N times longer than alone.

Actual (docs/optuna_stage_b_results.md): 2.34 + 1.62 + 0.76 = 4.72 h. The gains were as expected
(1.36x for the study, 1.15x and 1.18x for the confirmation), but the estimate above assumed the
average network of the space, and the study moved to large networks (17 to 21 ms per step alone).

## Caveats

- **Short trials.** At 15 000 steps the current model reaches 0.842. At 50 000 it reaches 0.925
  (mean of 2 seeds). The study ranks setups by what they reach in 15 000 steps, and the
  confirmation at 50 000 steps checks whether that ranking holds.
- **Seed noise.** The 2 seeds of the current model differ by 0.015 on stage (b) at 50 000 steps.
  The decision rule (a higher 2-seed mean than the current model) follows the request, but a gain
  below about 0.01 is within that noise.
- **Few trials for 9 parameters.** 30 trials means 10 random ones and then 20 guided ones, which is
  thin for 9 parameters. 60 trials give TPE more room.

## Commands

Run these inside WSL, from the repository root:

```
mkdir -p runs/optuna_b
(setsid nohup ~/ic2env/bin/python scripts/optuna_stage_b.py run --n-trials 60 --workers 4 > runs/optuna_b/run.out 2>&1 < /dev/null &)
~/ic2env/bin/python scripts/optuna_stage_b.py report --importance
touch runs/optuna_b/STOP     # pause: each worker finishes its running trial
~/ic2env/bin/python scripts/optuna_stage_b.py confirm --workers 4
```

Each worker logs to `runs/optuna_b/worker<i>.out`, and each trial to `runs/optuna_b/t<number>/`.
Running `run` again with a larger `--n-trials` continues the same study.
