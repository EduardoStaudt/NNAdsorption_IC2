"""Optuna study of the stage (b) hyperparameters of the physics-only parametric PINN.

Each trial is a short run of scripts/train_phase2.py with configs/optuna_stage_b.yaml (15 000 steps,
validation every 2 500 steps) and the sampled values as --set overrides. The validation y R2 of each
2 500 steps is reported to Optuna, and a median pruner stops a trial whose best value so far is below
the median of the earlier trials at the same step. The objective is the final validation y R2: outlet
mole fraction of gas 1, pooled over the 20 stage (b) validation cases, the metric of the phase reports.
The mean and the lowest R2 of the single cases are kept too (report and confirm), not optimized.

Several trials can train at the same time on the GPU: `run --workers N` starts N worker processes on
the same study, each with its own sampler seed, so that they do not propose the same trial. Every
trial has its own run directory and the usual training seeds, so its result depends only on its
hyperparameters, not on the worker or on N. Each training process gets a fixed share of the GPU
memory (--gpu-mem-mb, i.e. runtime.gpu_memory_mb). The default is the space's: 3072 MB for b1, 6800 MB
for b2, whose largest networks need up to 5.6 GB (run it with 2 workers). The limit does not change
the results (bit-identical at 3072 and 7000 MB, docs/optuna2_plan.md).

A run that turns NaN (its validation y R2, or its training loss up to the end when the evaluation
of the NaN network stops the process with an error) counts as pruned, with the user attribute
diverged_at, not as failed: TPE ignores failed trials, so it could propose the same region again,
while it ranks pruned trials with the bad ones. The median pruner uses complete trials only.

The study is stored in an Optuna journal file (runs/<name>/study.log), so it can be paused and
resumed. SQLite (--storage sqlite, runs/<name>/study.db) is kept for a single worker only: with the
Python sqlite3 driver the "still queued?" check and the switch to running are not one transaction,
and 4 workers started together all took the same queued trial (the journal records which worker
owns each running trial). The study processes do not import TensorFlow and hold no GPU memory: they
start the training processes and read their val_curve.csv and summary.json.

Two search spaces (--space): b1, the first study (runs/optuna_b, docs/optuna_stage_b.md), and b2,
the second one (runs/optuna_b2, docs/optuna2_plan.md): more activations and wider ranges where
trial 26 of the first study sat at the edge. Trial 0 of each study is its reference: the model
before the study (b1) or trial 26 of the first study (b2). In b2 every run sets model.siren_w0=1: the
"sin" activation is a sine layer with the SIREN initialization and w0 = 1, since w0 = 30 (the SIREN
paper) and w0 = 5 diverged or stalled in the benchmark (docs/optuna2_plan.md).

Commands, from the repository root inside WSL (~/ic2env/bin/python):
    python scripts/optuna_stage_b.py run --n-trials 60 --workers 3   start, or resume until 60 trials
    python scripts/optuna_stage_b.py report [--importance]            table of the trials
    python scripts/optuna_stage_b.py plot --out DIR                    y R2 against each parameter
    python scripts/optuna_stage_b.py confirm --workers 3              the top 3 trials and the
        reference, 2 seeds each, 50 000 steps; b1: if the best of the 3 beats the reference, both go
        to stage (c) with 2 seeds each, 50 000 steps; b2: the best of the 3 always goes to stage (c),
        and the runs of trial 26 from the first confirmation are reused (same config, deterministic)
Add --space b2 to each command for the second study.

Pause: create runs/<name>/STOP. Each worker finishes its running trial, then stops. Delete the file
and run the same command again to resume. If a study process is killed, its training processes get
SIGTERM, and its trial stays "running" in the storage. Only one `run` may be active at a time, so
`run` marks the trials it finds running as failed ("interrupted") and queues their parameters again
(once); such a trial counts against --n-trials.
"""
import argparse
import concurrent.futures
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import optuna
from optuna.trial import TrialState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train_phase2.py")
# Both base configs have the hyperparameters of the model before the study (CURRENT): the study made
# trial 26 the default of the stage configs (configs/phase2_base.yaml, docs/optuna_stage_b_results.md).
CONFIG_B = "configs/optuna_stage_b.yaml"
CONFIG_C = "configs/phase2_stage_c_before_optuna.yaml"
VAL_B = "data/provisorio_MOL_n2_stage_b_val20.h5"
VAL_C = "data/provisorio_MOL_n2_stage_c_val20.h5"
N_FIXED = {"b": 1, "c": 50}  # fixed collocation theta: the base case (b), the 50 training cases (c)
STUDY_NAME = "stage_b"
SEEDS = (0, 1)
TOP = 3
GPU_MEM_MB = 3072  # b1: largest network of its space (8 x 256, SiLU, 128 Fourier features, 4096 theta): peak 2.6 GB
# b2, 2 workers: largest measured peak 5.6 GB (12 x 320, Mish, 256 Fourier features). 2 x (6800 + about 260 MB
# of CUDA context) = 14.1 GB, about 0.5 GB below what Windows leaves free (it uses 1.4 to 1.7 GB of 16.3 GB).
B2_GPU_MEM_MB = 6800
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
BUSY = FINISHED + (TrialState.RUNNING,)  # counted against --n-trials, so that parallel workers do not overshoot
NAN = float("nan")
FAILED_RUN = dict(y_r2=NAN, y_r2_case_mean=NAN, y_r2_case_min=NAN, worst_case=None)

# The model before the study (as phase3c_long): the first trial of the study and the baseline of `confirm`.
CURRENT = dict(depth=6, width=128, activation="tanh", n_fourier=64, fourier_scale=1.0, lr=1.0e-3,
               energy_weight=10.0, cauchy_c=10.0, n_theta=2048)
BASE_SIGMA_Z, BASE_SIGMA_T = 2.0, 1.0  # Fourier sigmas of the current model (fourier_scale = 1)
BASE_N_THETA, BASE_EPOCHS = 2048, 10   # theta renewed every 10 passes over the pool
# Trial 26 of the first study, the default of the stage configs since then: trial 0 of the second study.
T26 = dict(depth=8, width=224, activation="silu", n_fourier=128, fourier_scale=0.2974353163373616,
           lr=0.0006211316335097482, energy_weight=2.227854425543525, cauchy_c=24.34852416892599, n_theta=1024)

# Search spaces, parameter -> ("int", low, high, step) | ("log", low, high) | ("cat", choices). The order
# is the order of the suggest calls; fourier_scale is only sampled with Fourier features.
SPACE_B1 = {
    "depth": ("int", 4, 8, 1),
    "width": ("int", 64, 256, 32),
    "activation": ("cat", ("tanh", "silu")),
    "n_fourier": ("cat", (0, 32, 64, 128)),
    "fourier_scale": ("log", 0.25, 4.0),
    "lr": ("log", 1.0e-4, 3.0e-3),
    "energy_weight": ("log", 1.0, 100.0),
    "cauchy_c": ("log", 1.0, 100.0),
    "n_theta": ("cat", (512, 1024, 2048, 4096)),
}
SPACE_B2 = {
    "depth": ("int", 6, 12, 1),
    "width": ("int", 128, 320, 32),
    "activation": ("cat", ("tanh", "silu", "gelu", "softplus", "mish", "sin")),
    "n_fourier": ("cat", (64, 128, 256)),
    "fourier_scale": ("log", 0.05, 1.0),
    "lr": ("log", 2.0e-4, 2.0e-3),
    "energy_weight": ("log", 0.3, 10.0),
    "cauchy_c": ("log", 5.0, 60.0),
    "n_theta": ("cat", (512, 1024, 2048)),
}


@dataclass(frozen=True)
class Space:
    study_name: str
    default_dir: str      # study directory under the runs directory
    params: dict
    reference: dict       # trial 0 of the study and the baseline of `confirm`
    reference_label: str
    reference_runs: str   # "": `confirm` trains the reference; else its runs ("<runs>/confirm_<stage>/..._seed<seed>")
    stage_c_always: bool  # False: the best goes to stage (c) only if it beats the reference on stage (b)
    fixed_sets: tuple = ()  # --set items of every run of the study (trials and confirmation)
    gpu_mem_mb: int = GPU_MEM_MB  # default GPU memory of each training process
    act_names: tuple = ()  # (activation, name in reports and figures) where the name is not the activation


SPACES = {
    "b1": Space("stage_b", "optuna_b", SPACE_B1, CURRENT, "current", "", False),
    "b2": Space("stage_b2", "optuna_b2", SPACE_B2, T26, "study1_t0026", "optuna_b/confirm_%s/t0026_seed%d", True,
                fixed_sets=("model.siren_w0=1",), gpu_mem_mb=B2_GPU_MEM_MB,
                act_names=(("sin", "sin (SIREN init, w0 = 1)"),)),
}


def act_name(space, activation):
    return dict(space.act_names).get(activation, activation)


def space_of(args):
    return SPACES[getattr(args, "space", "b1")]


if sys.platform.startswith("linux"):
    import ctypes

    _PRCTL = ctypes.CDLL("libc.so.6", use_errno=True).prctl  # looked up here, not in the forked child
else:
    _PRCTL = None


def suggest(trial, space=SPACE_B1):
    """Sample one point of the search space. fourier_scale multiplies both Fourier sigmas (only with
    Fourier features)."""
    p = {}
    for name, (kind, *a) in space.items():
        if name == "fourier_scale" and p["n_fourier"] == 0:
            continue
        if kind == "int":
            p[name] = trial.suggest_int(name, a[0], a[1], step=a[2])
        elif kind == "log":
            p[name] = trial.suggest_float(name, a[0], a[1], log=True)
        else:
            p[name] = trial.suggest_categorical(name, list(a[0]))
    return p


def full_params(params):
    """Trial parameters with the value of the conditional one when it was not sampled."""
    p = dict(params)
    p.setdefault("fourier_scale", 1.0)
    return p


def same_as(params, reference):
    p = full_params(params)
    return all(math.isclose(p[k], v) if isinstance(v, float) else p[k] == v for k, v in reference.items())


def same_as_current(params):
    return same_as(params, CURRENT)


def yaml_number(x):
    """A float written so that the YAML parser of --set reads it back as a float (PyYAML reads
    '1e-05' as a string: it needs a dot)."""
    s = ("%.12f" % float(x)).rstrip("0")
    return s + "0" if s.endswith(".") else s


def renewal_epochs(n_theta, n_fixed):
    """Passes over the pool between theta renewals that keep the renewal period (in steps) of the
    current setup: the pool has (n_fixed + n_theta) x points_per_theta points, so a larger pool
    needs fewer passes."""
    return BASE_EPOCHS * (n_fixed + BASE_N_THETA) / (n_fixed + n_theta)


def overrides(p, n_fixed):
    """--set KEY=VALUE items of train_phase2.py for the parameters p."""
    sets = [
        "model.depth=%d" % p["depth"],
        "model.width=%d" % p["width"],
        "model.activation=%s" % p["activation"],
        "model.fourier.n_features=%d" % p["n_fourier"],
        "train.lr=%s" % yaml_number(p["lr"]),
        "loss.pde_terms.energy=%s" % yaml_number(p["energy_weight"]),
        "loss.robust_c=%s" % yaml_number(p["cauchy_c"]),
        "theta_colloc.n_theta=%d" % p["n_theta"],
        "theta_colloc.resample_every_epochs=%s" % yaml_number(renewal_epochs(p["n_theta"], n_fixed)),
    ]
    if p["n_fourier"] > 0:
        s = p.get("fourier_scale", 1.0)
        sets += ["model.fourier.sigma_z=%s" % yaml_number(BASE_SIGMA_Z * s),
                 "model.fourier.sigma_t=%s" % yaml_number(BASE_SIGMA_T * s)]
    return sets


def common_sets(args):
    """Overrides of every run of the study: where it writes, its GPU memory, and the fixed settings of
    its search space."""
    sets = ["logging.runs_dir=%s" % args.runs_dir]
    if args.gpu_mem_mb > 0:
        sets.append("runtime.gpu_memory_mb=%d" % args.gpu_mem_mb)
    return sets + list(space_of(args).fixed_sets)


def train_command(args, config, val, run_name, sets):
    return [sys.executable, args.train_script, "--config", config, "--val", val, "--run-name", run_name, "--set", *sets]


def read_rows(path):
    """Complete rows of a CSV file that another process is still appending to (a row still being
    written is left for the next read)."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            lines = f.read().splitlines(keepends=True)
    except FileNotFoundError:
        return []
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
    return list(csv.DictReader(lines))


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _die_with_parent():
    """Runs in the child before exec: it gets SIGTERM when its parent dies (no orphan GPU run)."""
    _PRCTL(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG


def start(cmd, log):
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                            preexec_fn=_die_with_parent if _PRCTL else None)


def stop_process(proc, wait_s=60.0):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def fresh_dir(path):
    """An empty run directory. A leftover one with the same name (a run that did not finish, or a
    study started again from scratch) is renamed aside, not deleted."""
    if os.path.isdir(path) and os.listdir(path):
        os.rename(path, "%s_old_%d" % (path, int(time.time())))
    os.makedirs(path, exist_ok=True)


def run_and_watch(cmd, log_path, curve_path, on_eval, poll_s):
    """Run one training process; call on_eval(row) for each new row of its validation curve.
    on_eval returns True to stop the run. Returns 'stopped', 'done' or 'failed'."""
    seen = 0
    with open(log_path, "w", encoding="utf-8") as log:
        proc = start(cmd, log)
        try:
            while True:
                finished = proc.poll() is not None
                for row in read_rows(curve_path)[seen:]:
                    seen += 1
                    if on_eval(row):
                        stop_process(proc)
                        return "stopped"
                if finished:
                    return "done" if proc.returncode == 0 else "failed"
                time.sleep(poll_s)
        except BaseException:
            stop_process(proc)
            raise


def run_result(run_dir):
    """Metrics of a finished run from its summary.json."""
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    v, t = s["validation"], s.get("training", {})
    return dict(y_r2=float(v["outlet"]["y_c1"]["r2"]),
                y_r2_case_mean=as_float(v.get("per_case_mean", {}).get("r2_y_c1")),
                y_r2_case_min=as_float(v.get("per_case_min", {}).get("r2_y_c1")),
                worst_case=v.get("worst_case", {}).get("y_c1", {}).get("case"),
                T_out_rmse=as_float(v["outlet"].get("T_out", {}).get("rmse")),
                median_ms_per_step=t.get("median_ms_per_step"),
                gpu_peak_mb=t.get("gpu_peak_mb"),
                host_peak_rss_mb=t.get("host_peak_rss_mb"))


def train_seconds(run_dir):
    """Training time from the last row of train_log.csv (also for a stopped run)."""
    rows = read_rows(os.path.join(run_dir, "train_log.csv"))
    return float(rows[-1]["elapsed_s"]) if rows else 0.0


def failed(trial, msg):
    trial.set_user_attr("error", msg)
    return RuntimeError("trial %d: %s" % (trial.number, msg))


def diverged(trial, step):
    """A run that turned NaN counts as pruned, not failed: TPE ranks pruned trials with the bad ones
    (one without validation values last), while it ignores failed trials and could propose the same
    region again. The median pruner uses complete trials only, so it is not affected."""
    trial.set_user_attr("diverged_at", int(step))
    return optuna.TrialPruned("diverged (NaN) at step %d" % step)


def nan_loss_step(run_dir):
    """First logged step from which the total training loss stays non-finite up to the last row of
    train_log.csv; None if the last logged loss is finite or not logged."""
    rows = [r for r in read_rows(os.path.join(run_dir, "train_log.csv")) if r.get("total") not in (None, "")]
    if not rows or math.isfinite(as_float(rows[-1]["total"])):
        return None
    step = int(rows[-1]["step"])
    for r in reversed(rows):
        if math.isfinite(as_float(r["total"])):
            break
        step = int(r["step"])
    return step


def make_objective(args):
    space = space_of(args).params

    def objective(trial):
        p = suggest(trial, space)
        run_name = "%s/t%04d" % (args.name, trial.number)
        run_dir = os.path.join(args.runs_dir, run_name)
        fresh_dir(run_dir)
        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("worker", args.worker_id)
        sets = overrides(p, N_FIXED["b"]) + ["train.steps=%d" % args.steps, "phase2.eval_every=%d" % args.eval_every]
        last, nan_at = {}, []

        def on_eval(row):
            step, r2 = int(row["step"]), as_float(row["y_c1_r2"])
            last.update(row)
            if not math.isfinite(r2):
                nan_at.append(step)
                return True
            trial.report(r2, step)
            return step < args.steps and trial.should_prune()  # the last value is the result, not a check

        t0 = time.time()
        try:
            status = run_and_watch(train_command(args, CONFIG_B, VAL_B, run_name, sets + common_sets(args)),
                                   os.path.join(run_dir, "stdout.txt"), os.path.join(run_dir, "val_curve.csv"),
                                   on_eval, args.poll_s)
        finally:
            trial.set_user_attr("wall_s", time.time() - t0)
            trial.set_user_attr("train_s", train_seconds(run_dir))
        if last:  # single-case metrics of the last validation (all a pruned trial has)
            trial.set_user_attr("y_r2_case_mean", as_float(last.get("y_c1_r2_case_mean")))
            trial.set_user_attr("y_r2_case_min", as_float(last.get("y_c1_r2_case_min")))
        if nan_at:
            raise diverged(trial, nan_loss_step(run_dir) or nan_at[0])
        if status == "stopped":
            raise optuna.TrialPruned()
        if status == "failed":
            step = nan_loss_step(run_dir)
            if step is not None:  # evaluating a NaN network stops the training process with an error
                raise diverged(trial, step)
            raise failed(trial, "the training process failed, see %s/stdout.txt" % run_name)
        try:
            res = run_result(run_dir)
        except (OSError, KeyError, ValueError, TypeError) as e:
            raise failed(trial, "no result in %s/summary.json (%r)" % (run_name, e))
        for k, v in res.items():
            if k != "y_r2":
                trial.set_user_attr(k, v)
        if not math.isfinite(res["y_r2"]):
            raise diverged(trial, nan_loss_step(run_dir) or args.steps)
        return res["y_r2"]

    return objective


def make_storage(args):
    study_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(study_dir, exist_ok=True)
    if args.storage == "journal":
        path = os.path.join(study_dir, "study.log")
        lock = optuna.storages.journal.JournalFileOpenLock(path)
        return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path, lock_obj=lock))
    return optuna.storages.RDBStorage(
        "sqlite:///" + os.path.join(study_dir, "study.db"),
        engine_kwargs=dict(connect_args=dict(timeout=60)),  # wait up to 60 s for a lock held by another worker
        heartbeat_interval=60, grace_period=180,
        heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=1),
    )


def open_study(args, seed=0):
    return optuna.create_study(
        study_name=space_of(args).study_name, storage=make_storage(args), direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2 * args.eval_every),
    )


def stop_file_callback(path):
    def callback(study, trial):
        if os.path.exists(path):
            print("%s found: this worker stops after trial %d" % (path, trial.number), flush=True)
            study.stop()

    return callback


def fmt_params(p):
    p = full_params(p)
    ff = "FF %d x %.2f" % (p["n_fourier"], p["fourier_scale"]) if p["n_fourier"] else "no FF"
    return "%d x %d %s, %s, lr %.2e, energy %.1f, Cauchy %.1f, %d theta" % (
        p["depth"], p["width"], p["activation"], ff, p["lr"], p["energy_weight"], p["cauchy_c"], p["n_theta"])


def _last_report(t):
    step = max(t.intermediate_values, default=0)
    return step, t.intermediate_values.get(step, NAN)


def elapsed_hours(trials):
    """Hours from the first start to the last end of the finished trials (pauses included)."""
    done = [t for t in trials if t.datetime_start and t.datetime_complete]
    if not done:
        return 0.0
    return (max(t.datetime_complete for t in done) - min(t.datetime_start for t in done)).total_seconds() / 3600


def activation_table(trials, activations, space=None):
    """Per activation: trials by state (the diverged ones, NaN, are counted as pruned and also on their
    own), best and median final y R2 of the complete trials, and the median y R2 at the first pruning
    check (5 000 steps) of the trials that reached it."""
    space = space or SPACES["b1"]
    lines = ["| activation | complete / pruned / failed | diverged (NaN) | best y R2 | median y R2 | median at 5000 steps |",
             "|---|---|---|---|---|---|"]
    for a in activations:
        ts = [t for t in trials if t.params.get("activation") == a]
        done = [t.value for t in ts if t.state == TrialState.COMPLETE]
        first = [t.intermediate_values[5000] for t in ts if 5000 in t.intermediate_values]
        n = [sum(t.state == s for t in ts) for s in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
        lines.append("| %s | %d / %d / %d | %d | %s | %s | %s |" % (
            act_name(space, a), *n, sum("diverged_at" in t.user_attrs for t in ts),
            "%.4f" % max(done) if done else "-", "%.4f" % statistics.median(done) if done else "-",
            "%.4f" % statistics.median(first) if first else "-"))
    return "\n".join(lines)


def print_report(study, top=None, importance=False, space=None):
    space = space or SPACES["b1"]
    trials = study.get_trials(deepcopy=False)
    counts = ", ".join("%s %d" % (s.name.lower(), n) for s in TrialState if (n := sum(t.state == s for t in trials)))
    print("study %s: %s | sum of trial times %.2f h, elapsed %.2f h" % (
        study.study_name, counts or "no trials", sum(t.user_attrs.get("wall_s", 0.0) for t in trials) / 3600,
        elapsed_hours(trials)))
    done = sorted((t for t in trials if t.state == TrialState.COMPLETE), key=lambda t: t.value, reverse=True)
    pruned = sorted((t for t in trials if t.state == TrialState.PRUNED), key=_last_report, reverse=True)
    print("| trial | state | y R2 | case mean | worst case (case) | at step | min | worker | parameters |\n"
          "|---|---|---|---|---|---|---|---|---|")
    for t in (done + pruned)[:top]:
        step, r2 = _last_report(t)
        ua = t.user_attrs
        print("| %d%s | %s | %.4f | %.4f | %.4f (%s) | %d | %.1f | %s | %s |" % (
            t.number, " (%s)" % space.reference_label if same_as(t.params, space.reference) else "",
            "diverged" if "diverged_at" in t.user_attrs else t.state.name.lower(),
            t.value if t.state == TrialState.COMPLETE else r2, ua.get("y_r2_case_mean", NAN),
            ua.get("y_r2_case_min", NAN), ua.get("worst_case", "-"), step, ua.get("wall_s", 0.0) / 60,
            ua.get("worker", "-"), fmt_params(t.params)))
    for t in trials:
        if t.state == TrialState.FAIL:
            print("failed trial %d: %s" % (t.number, t.user_attrs.get("error", "see the log")))
        if "diverged_at" in t.user_attrs:
            print("diverged trial %d (counted as pruned): NaN from step %d, %s" % (
                t.number, t.user_attrs["diverged_at"], fmt_params(t.params)))
    print(activation_table(trials, space.params["activation"][1], space))
    if importance:
        imp = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
        print("importance (fANOVA over the complete trials): " + ", ".join("%s %.3f" % kv for kv in imp.items()))


def worker_command(args, i):
    return [sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(i), "--name", args.name,
            "--space", args.space, "--runs-dir", args.runs_dir, "--storage", args.storage, "--eval-every", str(args.eval_every),
            "--poll-s", str(args.poll_s), "--gpu-mem-mb", str(args.gpu_mem_mb), "--train-script", args.train_script,
            "--n-trials", str(args.n_trials), "--steps", str(args.steps)]


def recover_interrupted(study):
    """Trials still running when `run` starts were cut off by a killed study (only one `run` is active
    at a time): mark them failed and queue their parameters again, once."""
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study._storage.set_trial_user_attr(t._trial_id, "error", "interrupted (the study was stopped while it ran)")
        study.tell(t.number, state=TrialState.FAIL)
        if t.params and "retry_of" not in t.user_attrs:
            study.enqueue_trial(t.params, user_attrs={"retry_of": t.number})
        print("trial %d was interrupted: failed%s" % (t.number, ", queued again" if t.params else ""), flush=True)


def cmd_run(args):
    """Start (or resume) the study with args.workers worker processes and wait for them."""
    if args.storage == "sqlite" and args.workers > 1:
        sys.exit("SQLite cannot hand queued trials to parallel workers safely: use --storage journal")
    study = open_study(args)
    study_dir = os.path.join(args.runs_dir, args.name)
    stop_path = os.path.join(study_dir, "STOP")
    if os.path.exists(stop_path):
        sys.exit("%s exists: delete it to run the study" % stop_path)
    recover_interrupted(study)
    if not study.trials:
        study.enqueue_trial(dict(space_of(args).reference))  # the reference first: a baseline for the pruner
    n = min(args.workers, args.n_trials - len(study.get_trials(deepcopy=False, states=BUSY)))
    procs = []
    for i in range(max(n, 0)):
        log = open(os.path.join(study_dir, "worker%d.out" % i), "a", encoding="utf-8")
        procs.append((start(worker_command(args, i), log), log))
    for proc, log in procs:
        proc.wait()
        log.close()
    print_report(study, top=10, space=space_of(args))


def cmd_worker(args):
    """One worker: trials until the study has args.n_trials (running ones included) or STOP exists."""
    study = open_study(args, seed=args.worker_id)
    stop_path = os.path.join(args.runs_dir, args.name, "STOP")
    if os.path.exists(stop_path) or len(study.get_trials(deepcopy=False, states=BUSY)) >= args.n_trials:
        return
    study.optimize(make_objective(args), catch=(RuntimeError,),
                   callbacks=[optuna.study.MaxTrialsCallback(args.n_trials, states=BUSY), stop_file_callback(stop_path)])


def confirm_run_name(args, stage, label, seed):
    """Run name of one confirmation run. The reference of the second study (trial 26) already has its
    confirmation runs from the first study, with the same config: training is deterministic, so they
    are reused while their summary.json exists (never trained again in the first study's directory)."""
    space = space_of(args)
    if label == space.reference_label and space.reference_runs:
        old = space.reference_runs % (stage, seed)
        if os.path.exists(os.path.join(args.runs_dir, old, "summary.json")):
            return old
    return "%s/confirm_%s/%s_seed%d" % (args.name, stage, label, seed)


def run_once(args, stage, label, p, seed):
    """One confirmation run (args.confirm_steps steps, physics only). A run whose summary.json exists
    is not trained again, so `confirm` can be resumed. Returns its metrics (NaN if it failed)."""
    config, val = (CONFIG_B, VAL_B) if stage == "b" else (CONFIG_C, VAL_C)
    run_name = confirm_run_name(args, stage, label, seed)
    run_dir = os.path.join(args.runs_dir, run_name)
    if not os.path.exists(os.path.join(run_dir, "summary.json")):
        fresh_dir(run_dir)
        sets = overrides(p, N_FIXED[stage]) + [
            "train.steps=%d" % args.confirm_steps, "phase2.eval_every=%d" % (args.confirm_steps // 5),
            "loss.weights.data=0.0", "runtime.seed=%d" % seed] + common_sets(args)
        print("%s stage (%s), %s, seed %d: %s" % (time.strftime("%H:%M:%S"), stage, label, seed, fmt_params(p)), flush=True)
        status = run_and_watch(train_command(args, config, val, run_name, sets), os.path.join(run_dir, "stdout.txt"),
                               os.path.join(run_dir, "val_curve.csv"), lambda row: False, args.poll_s)
        if status != "done":
            return dict(FAILED_RUN)
    try:
        return run_result(run_dir)
    except (OSError, KeyError, ValueError, TypeError):
        return dict(FAILED_RUN)


def run_many(args, stage, labelled):
    """run_once for every (label, params) and seed, args.workers runs at a time: {label: [result per seed]}."""
    jobs = [(label, p, seed) for label, p in labelled for seed in SEEDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        results = list(ex.map(lambda job: run_once(args, stage, *job), jobs))
    out = {}
    for (label, _, _), r in zip(jobs, results):
        out.setdefault(label, []).append(r)
    return out


def mean_or_minus_inf(values):
    m = statistics.fmean(values)
    return m if math.isfinite(m) else -math.inf


def decide(y_r2, reference="current"):
    """{label: validation y R2 per seed} of stage (b) -> (best label other than the reference, whether
    its mean beats the mean of the reference). A failed run (NaN) makes its mean -inf."""
    mean = {k: mean_or_minus_inf(v) for k, v in y_r2.items()}
    best = max((k for k in mean if k != reference), key=mean.get)
    return best, bool(mean[best] > mean[reference])


def print_confirm(res):
    def pm(values):
        return "%.4f +/- %.4f" % (statistics.fmean(values), abs(values[0] - values[-1]))

    print("| stage | setup | y R2 per seed | mean +/- seed difference | case mean | worst case | parameters |\n"
          "|---|---|---|---|---|---|---|")
    for stage in ("stage_b", "stage_c"):
        for label, v in res[stage].items():
            runs = v["runs"]
            print("| %s | %s | %s | %s | %s | %s | %s |" % (
                stage[-1], label, " / ".join("%.4f" % r["y_r2"] for r in runs), pm([r["y_r2"] for r in runs]),
                pm([r["y_r2_case_mean"] for r in runs]), pm([r["y_r2_case_min"] for r in runs]), fmt_params(v["params"])))
    d = res["decision"]
    print("best of the top %d: %s | stage (c) test: %s" % (TOP, d["best"], "yes" if d["stage_c"] else "no"))


def cmd_confirm(args):
    space = space_of(args)
    study = open_study(args)
    done = sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), key=lambda t: t.value, reverse=True)
    top = [t for t in done if not same_as(t.params, space.reference)][:TOP]
    if not top:
        sys.exit("no complete trial other than the reference")
    out_path = os.path.join(args.runs_dir, args.name, "confirm.json")
    res = dict(steps=args.confirm_steps, seeds=list(SEEDS), workers=args.workers, stage_b={}, stage_c={})

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    ref = space.reference_label
    labelled = [(ref, dict(space.reference))] + [("t%04d" % t.number, full_params(t.params)) for t in top]
    t0 = time.time()
    runs = run_many(args, "b", labelled)
    res["stage_b"] = {label: dict(params=p, y_r2=[r["y_r2"] for r in runs[label]], runs=runs[label],
                                  run_names=[confirm_run_name(args, "b", label, s) for s in SEEDS]) for label, p in labelled}
    res["stage_b_wall_s"] = time.time() - t0
    best, beats = decide({k: v["y_r2"] for k, v in res["stage_b"].items()}, reference=ref)
    go = beats or space.stage_c_always
    res["decision"] = dict(best=best, stage_c=go)
    save()
    if go:
        labelled = [(label, res["stage_b"][label]["params"]) for label in (ref, best)]
        t0 = time.time()
        runs = run_many(args, "c", labelled)
        res["stage_c"] = {label: dict(params=p, y_r2=[r["y_r2"] for r in runs[label]], runs=runs[label],
                                      run_names=[confirm_run_name(args, "c", label, s) for s in SEEDS])
                          for label, p in labelled}
        res["stage_c_wall_s"] = time.time() - t0
        save()
    print_confirm(res)


PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")  # one per activation, in space order
MARKERS = ("s", "o", "^", "D", "v", "P")
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#dcdcd6"
LABELS = dict(depth="Trunk depth (layers)", width="Trunk width (neurons per layer)", activation="Activation",
              n_fourier="Fourier features (frequencies)", fourier_scale="Fourier scale (multiplies sigma_z = 2, sigma_t = 1)",
              lr="Maximum learning rate", energy_weight="Energy weight", cauchy_c="Cauchy scale c",
              n_theta="Sobol theta per draw")


def describe(kind, a):
    if kind == "int":
        return "uniform integer, %d to %d%s" % (a[0], a[1], ", step %d" % a[2] if a[2] > 1 else "")
    if kind == "log":
        return "log-uniform, %g to %g" % (a[0], a[1])
    return "categorical, %d choices" % len(a[0])


def log_ticks(low, high):
    """Ticks of a log axis: the two ends of the range and the 1-2-5 values between them that are at
    least a quarter of a decade away from the ends."""
    inside = [m * 10.0 ** k for k in range(-6, 4) for m in (1, 2, 5)]
    far = [v for v in inside if low < v < high and min(abs(math.log10(v / low)), abs(math.log10(high / v))) >= 0.25]
    return [low] + far + [high]


def cmd_plot(args):
    """One scatter per parameter: final validation y R2 of every complete trial against its value,
    marked by activation; the best trial is circled. Trials below --ymin sit on the bottom edge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import NullFormatter

    halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    space = space_of(args)
    done = list(open_study(args).get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if not done:
        sys.exit("no complete trial")
    best = max(done, key=lambda t: t.value)
    acts = list(space.params["activation"][1])
    top = max(t.value for t in done) + 0.012
    os.makedirs(args.out, exist_ok=True)
    for name, (kind, *a) in space.params.items():
        ts = [t for t in done if name in t.params]
        fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=150)

        def x_of(t):
            jitter = np.random.default_rng(t.number).uniform(-0.18, 0.18)
            if kind == "cat":
                return list(a[0]).index(t.params[name]) + jitter
            if kind == "int":
                return t.params[name] + jitter * a[2]
            return t.params[name]

        def frac(t):
            """Position of the point of t across the axes, 0 to 1."""
            v = x_of(t)
            return (v / max(len(a[0]) - 1, 1) if kind == "cat" else
                    math.log(v / a[0]) / math.log(a[1] / a[0]) if kind == "log" else (v - a[0]) / (a[1] - a[0]))

        def label(text, t, y, dx, dy, flip=False, **kw):
            """Text beside the point of t, on the side that stays inside the axes (the other side if flip)."""
            right = (frac(t) <= 0.75) != flip
            ax.annotate(text, (x_of(t), y), xytext=(dx if right else -dx, dy), textcoords="offset points",
                        ha="left" if right else "right", path_effects=halo, zorder=5, **kw)

        for i, act in enumerate(acts):
            group = [t for t in ts if t.params["activation"] == act]
            if group:
                ax.scatter([x_of(t) for t in group], [max(t.value, args.ymin + 0.004) for t in group], s=34,
                           c=PALETTE[i], marker=MARKERS[i], edgecolors="white", linewidths=0.6,
                           label=act_name(space, act), zorder=3)
        level, last = 0, None
        for t in sorted((t for t in ts if t.value < args.ymin), key=frac):  # off the scale: on the bottom edge
            level = level + 1 if last is not None and frac(t) - last < 0.15 else 0  # close labels stack up
            last = frac(t)
            label("t%d: %s" % (t.number, ("%.2f" % t.value).replace("-", "−")), t, args.ymin + 0.004, 7,
                  3 + 9 * level, fontsize=7, color=MUTED)
        for t in ts:  # the best trial (circled) and the reference, labelled on opposite sides of their points
            if t is best or same_as(t.params, space.reference):
                y = max(t.value, args.ymin + 0.004)
                if t is best:
                    ax.scatter([x_of(t)], [y], s=150, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=4)
                label("t%d" % t.number, t, y, 10 if t is best else 6, 0, flip=t is not best, va="center",
                      fontsize=7.5, color=INK)
        if kind == "cat":
            ax.set_xticks(range(len(a[0])), [act_name(space, c).replace(" (", "\n(").replace(", w0", ",\nw0")
                                             if name == "activation" else str(c) for c in a[0]])
            ax.set_xlim(-0.6, len(a[0]) - 0.4)
        else:
            if kind == "log":
                ax.set_xscale("log")
                ax.set_xlim(a[0] / 1.25, a[1] * 1.25)
                ticks = log_ticks(a[0], a[1])
                ax.set_xticks(ticks, ["%g" % v for v in ticks])
                ax.xaxis.set_minor_formatter(NullFormatter())
                for edge in a[:2]:  # the limits of the sampled range (integers: the first and last tick)
                    ax.axvline(edge, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
            else:
                ax.set_xticks(range(a[0], a[1] + 1, a[2]))
                ax.set_xlim(a[0] - 0.6 * a[2], a[1] + 0.6 * a[2])
        ax.set_ylim(args.ymin, top)
        ax.grid(axis="y", color=RULE, linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("%s: %s" % (LABELS[name], describe(kind, a)), fontsize=8.5, color=INK)
        ax.set_ylabel("validation y R² at %s steps" % format(args.steps, ",").replace(",", " "), fontsize=8.5, color=INK)
        if name != "activation" and len({t.params["activation"] for t in ts}) > 1:
            ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(acts),
                      handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, name + ".png"), facecolor="white")
        plt.close(fig)
    print("%d figures in %s (%d complete trials)" % (len(space.params), args.out, len(done)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Optuna study of the stage (b) hyperparameters (see the module docstring).")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "worker", "report", "plot", "confirm"):
        sp = sub.add_parser(name)
        sp.add_argument("--space", choices=sorted(SPACES), default="b1", help="search space: b1 (first study) or b2")
        sp.add_argument("--name", default=None, help="study directory under the runs directory (default: of the space)")
        sp.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
        sp.add_argument("--storage", choices=("journal", "sqlite"), default="journal",
                        help="journal file (default) or SQLite (a single worker only)")
        sp.add_argument("--eval-every", type=int, default=2500, help="steps between validations (pruner)")
        sp.add_argument("--poll-s", type=float, default=2.0, help="seconds between reads of the validation curve")
        sp.add_argument("--gpu-mem-mb", type=int, default=None,
                        help="GPU memory of each training process (0: as needed; default: of the space)")
        sp.add_argument("--train-script", default=TRAIN_SCRIPT, help=argparse.SUPPRESS)  # tests: a stand-in trainer
        if name in ("run", "worker"):
            sp.add_argument("--n-trials", type=int, required=True, help="finished trials to reach")
            sp.add_argument("--steps", type=int, default=15000, help="training steps per trial")
        if name in ("run", "confirm"):
            sp.add_argument("--workers", type=int, default=1, help="training processes at the same time on the GPU")
        if name == "worker":
            sp.add_argument("--worker-id", type=int, required=True, help="also the seed of its sampler")
        if name == "confirm":
            sp.add_argument("--confirm-steps", type=int, default=50000)
        if name == "report":
            sp.add_argument("--importance", action="store_true", help="parameter importance (fANOVA)")
        if name == "plot":
            sp.add_argument("--out", required=True, help="directory of the figures (one PNG per parameter)")
            sp.add_argument("--ymin", type=float, default=0.65, help="bottom of the y axis; lower trials sit on it")
            sp.add_argument("--steps", type=int, default=15000, help="training steps per trial (axis label)")
    args = ap.parse_args(argv)
    args.name = args.name or SPACES[args.space].default_dir
    if args.gpu_mem_mb is None:
        args.gpu_mem_mb = SPACES[args.space].gpu_mem_mb
    args.runs_dir = os.path.abspath(args.runs_dir)
    args.train_script = os.path.abspath(args.train_script)
    commands = dict(run=cmd_run, worker=cmd_worker, confirm=cmd_confirm, plot=cmd_plot,
                    report=lambda a: print_report(open_study(a), importance=a.importance, space=space_of(a)))
    commands[args.command](args)


if __name__ == "__main__":
    main()
