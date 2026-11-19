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
memory (--gpu-mem-mb, i.e. runtime.gpu_memory_mb; 3072 MB fits the largest network of the space).

The study is stored in SQLite (runs/<name>/study.db) or, with --storage journal, in an Optuna journal
file (runs/<name>/study.log), so it can be paused and resumed. The study processes do not import
TensorFlow and hold no GPU memory: they start the training processes and read their val_curve.csv
and summary.json.

Commands, from the repository root inside WSL (~/ic2env/bin/python):
    python scripts/optuna_stage_b.py run --n-trials 60 --workers 3   start, or resume until 60 trials
    python scripts/optuna_stage_b.py report [--importance]            table of the trials
    python scripts/optuna_stage_b.py confirm --workers 3              the top 3 trials and the current
        model, 2 seeds each, 50 000 steps; if the best of the 3 beats the current model, both go to
        stage (c) with 2 seeds each, 50 000 steps

Pause: create runs/<name>/STOP. Each worker finishes its running trial, then stops. Delete the file
and run the same command again to resume. If a study process is killed, its training processes get
SIGTERM. With SQLite such a trial stays "running" until its heartbeat is 3 minutes old, then it is
marked failed and queued again (once). The journal storage has no heartbeat.
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

import optuna
from optuna.trial import TrialState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train_phase2.py")
CONFIG_B = "configs/optuna_stage_b.yaml"
CONFIG_C = "configs/phase2_stage_c.yaml"
VAL_B = "data/provisorio_MOL_n2_stage_b_val20.h5"
VAL_C = "data/provisorio_MOL_n2_stage_c_val20.h5"
N_FIXED = {"b": 1, "c": 50}  # fixed collocation theta: the base case (b), the 50 training cases (c)
STUDY_NAME = "stage_b"
SEEDS = (0, 1)
TOP = 3
GPU_MEM_MB = 3072  # largest network of the space (8 x 256, SiLU, 128 Fourier features, 4096 theta): peak 2.6 GB
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
BUSY = FINISHED + (TrialState.RUNNING,)  # counted against --n-trials, so that parallel workers do not overshoot
NAN = float("nan")
FAILED_RUN = dict(y_r2=NAN, y_r2_case_mean=NAN, y_r2_case_min=NAN, worst_case=None)

# The current model (as phase3c_long): the first trial of the study and the baseline of `confirm`.
CURRENT = dict(depth=6, width=128, activation="tanh", n_fourier=64, fourier_scale=1.0, lr=1.0e-3,
               energy_weight=10.0, cauchy_c=10.0, n_theta=2048)
BASE_SIGMA_Z, BASE_SIGMA_T = 2.0, 1.0  # Fourier sigmas of the current model (fourier_scale = 1)
BASE_N_THETA, BASE_EPOCHS = 2048, 10   # theta renewed every 10 passes over the pool

if sys.platform.startswith("linux"):
    import ctypes

    _PRCTL = ctypes.CDLL("libc.so.6", use_errno=True).prctl  # looked up here, not in the forked child
else:
    _PRCTL = None


def suggest(trial):
    """The search space. fourier_scale multiplies both Fourier sigmas (only with Fourier features)."""
    p = dict(
        depth=trial.suggest_int("depth", 4, 8),
        width=trial.suggest_int("width", 64, 256, step=32),
        activation=trial.suggest_categorical("activation", ["tanh", "silu"]),
        n_fourier=trial.suggest_categorical("n_fourier", [0, 32, 64, 128]),
    )
    if p["n_fourier"] > 0:
        p["fourier_scale"] = trial.suggest_float("fourier_scale", 0.25, 4.0, log=True)
    p["lr"] = trial.suggest_float("lr", 1.0e-4, 3.0e-3, log=True)
    p["energy_weight"] = trial.suggest_float("energy_weight", 1.0, 100.0, log=True)
    p["cauchy_c"] = trial.suggest_float("cauchy_c", 1.0, 100.0, log=True)
    p["n_theta"] = trial.suggest_categorical("n_theta", [512, 1024, 2048, 4096])
    return p


def full_params(params):
    """Trial parameters with the value of the conditional one when it was not sampled."""
    p = dict(params)
    p.setdefault("fourier_scale", 1.0)
    return p


def same_as_current(params):
    p = full_params(params)
    return all(math.isclose(p[k], v) if isinstance(v, float) else p[k] == v for k, v in CURRENT.items())


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
    """Overrides of every run of the study: where it writes, and its GPU memory."""
    sets = ["logging.runs_dir=%s" % args.runs_dir]
    if args.gpu_mem_mb > 0:
        sets.append("runtime.gpu_memory_mb=%d" % args.gpu_mem_mb)
    return sets


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


def make_objective(args):
    def objective(trial):
        p = suggest(trial)
        run_name = "%s/t%04d" % (args.name, trial.number)
        run_dir = os.path.join(args.runs_dir, run_name)
        fresh_dir(run_dir)
        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("worker", args.worker_id)
        sets = overrides(p, N_FIXED["b"]) + ["train.steps=%d" % args.steps, "phase2.eval_every=%d" % args.eval_every]
        last, diverged = {}, []

        def on_eval(row):
            step, r2 = int(row["step"]), as_float(row["y_c1_r2"])
            last.update(row)
            if not math.isfinite(r2):
                diverged.append(step)
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
        if diverged:
            raise failed(trial, "non-finite validation y R2 at step %d" % diverged[0])
        if status == "stopped":
            raise optuna.TrialPruned()
        if status == "failed":
            raise failed(trial, "the training process failed, see %s/stdout.txt" % run_name)
        try:
            res = run_result(run_dir)
        except (OSError, KeyError, ValueError, TypeError) as e:
            raise failed(trial, "no result in %s/summary.json (%r)" % (run_name, e))
        for k, v in res.items():
            if k != "y_r2":
                trial.set_user_attr(k, v)
        if not math.isfinite(res["y_r2"]):
            raise failed(trial, "non-finite validation y R2")
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
        study_name=STUDY_NAME, storage=make_storage(args), direction="maximize", load_if_exists=True,
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


def print_report(study, top=None, importance=False):
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
            t.number, " (current)" if same_as_current(t.params) else "", t.state.name.lower(),
            t.value if t.state == TrialState.COMPLETE else r2, ua.get("y_r2_case_mean", NAN),
            ua.get("y_r2_case_min", NAN), ua.get("worst_case", "-"), step, ua.get("wall_s", 0.0) / 60,
            ua.get("worker", "-"), fmt_params(t.params)))
    for t in trials:
        if t.state == TrialState.FAIL:
            print("failed trial %d: %s" % (t.number, t.user_attrs.get("error", "see the log")))
    if importance:
        imp = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
        print("importance (fANOVA over the complete trials): " + ", ".join("%s %.3f" % kv for kv in imp.items()))


def worker_command(args, i):
    return [sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(i), "--name", args.name,
            "--runs-dir", args.runs_dir, "--storage", args.storage, "--eval-every", str(args.eval_every),
            "--poll-s", str(args.poll_s), "--gpu-mem-mb", str(args.gpu_mem_mb), "--train-script", args.train_script,
            "--n-trials", str(args.n_trials), "--steps", str(args.steps)]


def cmd_run(args):
    """Start (or resume) the study with args.workers worker processes and wait for them."""
    study = open_study(args)
    study_dir = os.path.join(args.runs_dir, args.name)
    stop_path = os.path.join(study_dir, "STOP")
    if os.path.exists(stop_path):
        sys.exit("%s exists: delete it to run the study" % stop_path)
    optuna.storages.fail_stale_trials(study)  # trials left running by killed workers (SQLite only)
    if not study.trials:
        study.enqueue_trial(dict(CURRENT))  # the current model first: a reference for the pruner
    n = min(args.workers, args.n_trials - len(study.get_trials(deepcopy=False, states=BUSY)))
    procs = []
    for i in range(max(n, 0)):
        log = open(os.path.join(study_dir, "worker%d.out" % i), "a", encoding="utf-8")
        procs.append((start(worker_command(args, i), log), log))
    for proc, log in procs:
        proc.wait()
        log.close()
    print_report(study, top=10)


def cmd_worker(args):
    """One worker: trials until the study has args.n_trials (running ones included) or STOP exists."""
    study = open_study(args, seed=args.worker_id)
    stop_path = os.path.join(args.runs_dir, args.name, "STOP")
    if os.path.exists(stop_path) or len(study.get_trials(deepcopy=False, states=BUSY)) >= args.n_trials:
        return
    study.optimize(make_objective(args), catch=(RuntimeError,),
                   callbacks=[optuna.study.MaxTrialsCallback(args.n_trials, states=BUSY), stop_file_callback(stop_path)])


def run_once(args, stage, label, p, seed):
    """One confirmation run (args.confirm_steps steps, physics only). A run whose summary.json exists
    is not trained again, so `confirm` can be resumed. Returns its metrics (NaN if it failed)."""
    config, val = (CONFIG_B, VAL_B) if stage == "b" else (CONFIG_C, VAL_C)
    run_name = "%s/confirm_%s/%s_seed%d" % (args.name, stage, label, seed)
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


def decide(y_r2):
    """{label: validation y R2 per seed} of stage (b) -> (best label other than 'current', whether its
    mean beats the mean of the current model). A failed run (NaN) makes its mean -inf."""
    mean = {k: mean_or_minus_inf(v) for k, v in y_r2.items()}
    best = max((k for k in mean if k != "current"), key=mean.get)
    return best, bool(mean[best] > mean["current"])


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
    study = open_study(args)
    done = sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), key=lambda t: t.value, reverse=True)
    top = [t for t in done if not same_as_current(t.params)][:TOP]
    if not top:
        sys.exit("no complete trial other than the current model")
    out_path = os.path.join(args.runs_dir, args.name, "confirm.json")
    res = dict(steps=args.confirm_steps, seeds=list(SEEDS), workers=args.workers, stage_b={}, stage_c={})

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    labelled = [("current", dict(CURRENT))] + [("t%04d" % t.number, full_params(t.params)) for t in top]
    t0 = time.time()
    runs = run_many(args, "b", labelled)
    res["stage_b"] = {label: dict(params=p, y_r2=[r["y_r2"] for r in runs[label]], runs=runs[label]) for label, p in labelled}
    res["stage_b_wall_s"] = time.time() - t0
    best, go = decide({k: v["y_r2"] for k, v in res["stage_b"].items()})
    res["decision"] = dict(best=best, stage_c=go)
    save()
    if go:
        labelled = [(label, res["stage_b"][label]["params"]) for label in ("current", best)]
        t0 = time.time()
        runs = run_many(args, "c", labelled)
        res["stage_c"] = {label: dict(params=p, y_r2=[r["y_r2"] for r in runs[label]], runs=runs[label]) for label, p in labelled}
        res["stage_c_wall_s"] = time.time() - t0
        save()
    print_confirm(res)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Optuna study of the stage (b) hyperparameters (see the module docstring).")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "worker", "report", "confirm"):
        sp = sub.add_parser(name)
        sp.add_argument("--name", default="optuna_b", help="study directory under the runs directory")
        sp.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
        sp.add_argument("--storage", choices=("sqlite", "journal"), default="sqlite")
        sp.add_argument("--eval-every", type=int, default=2500, help="steps between validations (pruner)")
        sp.add_argument("--poll-s", type=float, default=2.0, help="seconds between reads of the validation curve")
        sp.add_argument("--gpu-mem-mb", type=int, default=GPU_MEM_MB, help="GPU memory of each training process (0: as needed)")
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
    args = ap.parse_args(argv)
    args.runs_dir = os.path.abspath(args.runs_dir)
    args.train_script = os.path.abspath(args.train_script)
    commands = dict(run=cmd_run, worker=cmd_worker, confirm=cmd_confirm,
                    report=lambda a: print_report(open_study(a), importance=a.importance))
    commands[args.command](args)


if __name__ == "__main__":
    main()
