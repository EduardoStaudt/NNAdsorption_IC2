"""Optuna study of the stage (b) hyperparameters of the physics-only parametric PINN.

Each trial is a short run of scripts/train_phase2.py with configs/optuna_stage_b.yaml (15 000 steps,
validation every 2 500 steps) and the sampled values as --set overrides. The validation y R2 of each
2 500 steps is reported to Optuna, and a median pruner stops a trial whose best value so far is below
the median of the earlier trials at the same step. The objective is the final validation y R2: outlet
mole fraction of gas 1, pooled over the 20 stage (b) validation cases, the metric of the phase reports.

The study is stored in SQLite (runs/<name>/study.db), so it can be paused and resumed. This process
does not import TensorFlow and holds no GPU memory: it starts the training processes one at a time
and reads their val_curve.csv and summary.json.

Commands, from the repository root inside WSL (~/ic2env/bin/python):
    python scripts/optuna_stage_b.py run --n-trials 30   start, or resume until 30 trials are finished
    python scripts/optuna_stage_b.py report              table of the trials
    python scripts/optuna_stage_b.py confirm             the top 3 trials and the current model, 2 seeds
                                                         each, 50 000 steps; if the best of the 3 beats
                                                         the current model, both go to stage (c) with
                                                         2 seeds each, 50 000 steps

Pause: create runs/<name>/STOP. The running trial finishes, then the study stops. Delete the file and
run the same command again to resume. If this process is killed, its training process gets SIGTERM.
The trial stays "running" in the database until its heartbeat is 3 minutes old, then it is marked
failed and queued again (once).
"""
import argparse
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

CONFIG_B = "configs/optuna_stage_b.yaml"
CONFIG_C = "configs/phase2_stage_c.yaml"
VAL_B = "data/provisorio_MOL_n2_stage_b_val20.h5"
VAL_C = "data/provisorio_MOL_n2_stage_c_val20.h5"
N_FIXED = {"b": 1, "c": 50}  # fixed collocation theta: the base case (b), the 50 training cases (c)
STUDY_NAME = "stage_b"
SEEDS = (0, 1)
TOP = 3
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)

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


def train_command(config, val, run_name, sets):
    return [sys.executable, os.path.join(ROOT, "scripts", "train_phase2.py"), "--config", config,
            "--val", val, "--run-name", run_name, "--set", *sets]


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


def _die_with_parent():
    """Runs in the child before exec: it gets SIGTERM when this process dies (no orphan GPU run)."""
    _PRCTL(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG


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
    """Run one training process; call on_eval(step, y_r2) for each new row of its validation curve.
    on_eval returns True to stop the run. Returns 'stopped', 'done' or 'failed'."""
    seen = 0
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                preexec_fn=_die_with_parent if _PRCTL else None)
        try:
            while True:
                finished = proc.poll() is not None
                for row in read_rows(curve_path)[seen:]:
                    seen += 1
                    if on_eval(int(row["step"]), float(row["y_c1_r2"])):
                        stop_process(proc)
                        return "stopped"
                if finished:
                    return "done" if proc.returncode == 0 else "failed"
                time.sleep(poll_s)
        except BaseException:
            stop_process(proc)
            raise


def run_result(run_dir):
    """Validation y R2 of a finished run, and a few other numbers of its summary.json."""
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    out = s["validation"]["outlet"]
    return float(out["y_c1"]["r2"]), dict(T_out_rmse=out["T_out"]["rmse"],
                                          median_ms_per_step=s["training"].get("median_ms_per_step"))


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
        sets = overrides(p, N_FIXED["b"]) + ["train.steps=%d" % args.steps, "phase2.eval_every=%d" % args.eval_every,
                                             "logging.runs_dir=%s" % args.runs_dir]
        diverged = []

        def on_eval(step, r2):
            if not math.isfinite(r2):
                diverged.append(step)
                return True
            trial.report(r2, step)
            return step < args.steps and trial.should_prune()  # the last value is the result, not a check

        t0 = time.time()
        try:
            status = run_and_watch(train_command(CONFIG_B, VAL_B, run_name, sets), os.path.join(run_dir, "stdout.txt"),
                                   os.path.join(run_dir, "val_curve.csv"), on_eval, args.poll_s)
        finally:
            trial.set_user_attr("wall_s", time.time() - t0)
            trial.set_user_attr("train_s", train_seconds(run_dir))
        if diverged:
            raise failed(trial, "non-finite validation y R2 at step %d" % diverged[0])
        if status == "stopped":
            raise optuna.TrialPruned()
        if status == "failed":
            raise failed(trial, "the training process failed, see %s/stdout.txt" % run_name)
        try:
            value, extra = run_result(run_dir)
        except (OSError, KeyError, ValueError) as e:
            raise failed(trial, "no result in %s/summary.json (%r)" % (run_name, e))
        for k, v in extra.items():
            trial.set_user_attr(k, v)
        if not math.isfinite(value):
            raise failed(trial, "non-finite validation y R2")
        return value

    return objective


def open_study(args):
    study_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(study_dir, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        "sqlite:///" + os.path.join(study_dir, "study.db"),
        heartbeat_interval=60, grace_period=180,
        heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=1),
    )
    return optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2 * args.eval_every),
    )


def stop_file_callback(path):
    def callback(study, trial):
        if os.path.exists(path):
            print("%s found: the study stops after trial %d" % (path, trial.number), flush=True)
            study.stop()

    return callback


def fmt_params(p):
    p = full_params(p)
    ff = "FF %d x %.2f" % (p["n_fourier"], p["fourier_scale"]) if p["n_fourier"] else "no FF"
    return "%d x %d %s, %s, lr %.2e, energy %.1f, Cauchy %.1f, %d theta" % (
        p["depth"], p["width"], p["activation"], ff, p["lr"], p["energy_weight"], p["cauchy_c"], p["n_theta"])


def _last_report(t):
    step = max(t.intermediate_values, default=0)
    return step, t.intermediate_values.get(step, float("nan"))


def print_report(study, top=None):
    trials = study.get_trials(deepcopy=False)
    counts = ", ".join("%s %d" % (s.name.lower(), n) for s in TrialState if (n := sum(t.state == s for t in trials)))
    print("study %s: %s | wall time %.2f h" % (study.study_name, counts or "no trials",
                                               sum(t.user_attrs.get("wall_s", 0.0) for t in trials) / 3600))
    done = sorted((t for t in trials if t.state == TrialState.COMPLETE), key=lambda t: t.value, reverse=True)
    pruned = sorted((t for t in trials if t.state == TrialState.PRUNED), key=_last_report, reverse=True)
    print("| trial | state | y R2 | at step | min | parameters |\n|---|---|---|---|---|---|")
    for t in (done + pruned)[:top]:
        step, r2 = _last_report(t)
        print("| %d%s | %s | %.4f | %d | %.1f | %s |" % (
            t.number, " (current)" if same_as_current(t.params) else "", t.state.name.lower(),
            t.value if t.state == TrialState.COMPLETE else r2, step, t.user_attrs.get("wall_s", 0.0) / 60,
            fmt_params(t.params)))
    for t in trials:
        if t.state == TrialState.FAIL:
            print("failed trial %d: %s" % (t.number, t.user_attrs.get("error", "see the log")))


def cmd_run(args):
    study = open_study(args)
    stop_path = os.path.join(args.runs_dir, args.name, "STOP")
    if os.path.exists(stop_path):
        sys.exit("%s exists: delete it to run the study" % stop_path)
    if not study.trials:
        study.enqueue_trial(dict(CURRENT))  # the current model first: a reference for the pruner
    if len(study.get_trials(deepcopy=False, states=FINISHED)) < args.n_trials:
        study.optimize(make_objective(args), catch=(RuntimeError,),
                       callbacks=[optuna.study.MaxTrialsCallback(args.n_trials, states=FINISHED),
                                  stop_file_callback(stop_path)])
    print_report(study, top=10)


def run_once(args, stage, label, p, seed):
    """One confirmation run (args.confirm_steps steps, physics only). A run whose summary.json exists
    is not trained again, so `confirm` can be resumed. Returns its validation y R2 (NaN if it failed)."""
    config, val = (CONFIG_B, VAL_B) if stage == "b" else (CONFIG_C, VAL_C)
    run_name = "%s/confirm_%s/%s_seed%d" % (args.name, stage, label, seed)
    run_dir = os.path.join(args.runs_dir, run_name)
    if not os.path.exists(os.path.join(run_dir, "summary.json")):
        fresh_dir(run_dir)
        sets = overrides(p, N_FIXED[stage]) + [
            "train.steps=%d" % args.confirm_steps, "phase2.eval_every=%d" % (args.confirm_steps // 5),
            "loss.weights.data=0.0", "runtime.seed=%d" % seed, "logging.runs_dir=%s" % args.runs_dir]
        print("%s stage (%s), %s, seed %d: %s" % (time.strftime("%H:%M:%S"), stage, label, seed, fmt_params(p)), flush=True)
        status = run_and_watch(train_command(config, val, run_name, sets), os.path.join(run_dir, "stdout.txt"),
                               os.path.join(run_dir, "val_curve.csv"), lambda step, r2: False, args.poll_s)
        if status != "done":
            return float("nan")
    try:
        return run_result(run_dir)[0]
    except (OSError, KeyError, ValueError):
        return float("nan")


def mean_or_minus_inf(values):
    m = statistics.fmean(values)
    return m if math.isfinite(m) else -math.inf


def decide(y_r2):
    """{label: validation y R2 per seed} of stage (b) -> (best label other than 'current', whether its
    mean beats the mean of the current model). A failed run (NaN) makes its mean -inf."""
    mean = {k: mean_or_minus_inf(v) for k, v in y_r2.items()}
    best = max((k for k in mean if k != "current"), key=mean.get)
    return best, bool(mean[best] > mean["current"])


def cmd_confirm(args):
    study = open_study(args)
    done = sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), key=lambda t: t.value, reverse=True)
    top = [t for t in done if not same_as_current(t.params)][:TOP]
    if not top:
        sys.exit("no complete trial other than the current model")
    out_path = os.path.join(args.runs_dir, args.name, "confirm.json")
    res = dict(steps=args.confirm_steps, seeds=list(SEEDS), stage_b={}, stage_c={})

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    for label, p in [("current", dict(CURRENT))] + [("t%04d" % t.number, full_params(t.params)) for t in top]:
        res["stage_b"][label] = dict(params=p, y_r2=[run_once(args, "b", label, p, s) for s in SEEDS])
        save()
    best, go = decide({k: v["y_r2"] for k, v in res["stage_b"].items()})
    res["decision"] = dict(best=best, stage_c=go)
    save()
    if go:
        for label in ("current", best):
            p = res["stage_b"][label]["params"]
            res["stage_c"][label] = dict(params=p, y_r2=[run_once(args, "c", label, p, s) for s in SEEDS])
            save()
    for stage in ("stage_b", "stage_c"):
        for label, v in res[stage].items():
            print("%s %-8s y R2 %s, mean %.4f | %s" % (stage, label, " ".join("%.4f" % x for x in v["y_r2"]),
                                                       statistics.fmean(v["y_r2"]), fmt_params(v["params"])))
    print("best of the top %d: %s | stage (c) test: %s" % (TOP, best, "yes" if go else "no"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Optuna study of the stage (b) hyperparameters (see the module docstring).")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "report", "confirm"):
        sp = sub.add_parser(name)
        sp.add_argument("--name", default="optuna_b", help="study directory under the runs directory")
        sp.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
        sp.add_argument("--eval-every", type=int, default=2500, help="steps between validations (pruner)")
        sp.add_argument("--poll-s", type=float, default=2.0, help="seconds between reads of the validation curve")
        if name == "run":
            sp.add_argument("--n-trials", type=int, required=True, help="finished trials to reach")
            sp.add_argument("--steps", type=int, default=15000, help="training steps per trial")
        if name == "confirm":
            sp.add_argument("--confirm-steps", type=int, default=50000)
    args = ap.parse_args(argv)
    args.runs_dir = os.path.abspath(args.runs_dir)
    dict(run=cmd_run, report=lambda a: print_report(open_study(a)), confirm=cmd_confirm)[args.command](args)


if __name__ == "__main__":
    main()
