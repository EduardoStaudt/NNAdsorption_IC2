import argparse
import importlib.util
import json
import os

import optuna
import pytest

from conftest import REPO_ROOT
from nnads_ic2.config import apply_overrides, load_config

spec = importlib.util.spec_from_file_location("optuna_stage_b", os.path.join(REPO_ROOT, "scripts", "optuna_stage_b.py"))
osb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osb)

pytestmark = pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")

CONFIGS = {"b": osb.CONFIG_B, "c": osb.CONFIG_C}
S = optuna.trial.TrialState


@pytest.mark.parametrize("stage", ["b", "c"])
def test_current_parameters_change_nothing(stage):
    """The base config of the trials (b) and of the stage (c) test is the current model."""
    path = os.path.join(REPO_ROOT, CONFIGS[stage])
    base, cfg = load_config(path), load_config(path)
    apply_overrides(cfg, osb.overrides(osb.CURRENT, osb.N_FIXED[stage]))
    assert cfg == base
    assert osb.same_as_current(osb.CURRENT)


# Trial 26 of the study (docs/optuna_stage_b_results.md), confirmed on stages b and c.
TRIAL_26 = dict(depth=8, width=224, activation="silu", n_fourier=128, fourier_scale=0.2974353163373616,
                lr=0.0006211316335097482, energy_weight=2.227854425543525, cauchy_c=24.34852416892599, n_theta=1024)


@pytest.mark.parametrize("config, stage", [("phase2_stage_a.yaml", "b"), ("phase2_stage_b.yaml", "b"),
                                           ("phase2_stage_c.yaml", "c")])
def test_stage_configs_default_to_trial_26(config, stage):
    """The stage configs train trial 26, with the values of its confirmation runs (stage a: 1 fixed theta, as b)."""
    path = os.path.join(REPO_ROOT, "configs", config)
    base, cfg = load_config(path), load_config(path)
    apply_overrides(cfg, osb.overrides(TRIAL_26, osb.N_FIXED[stage]))
    assert cfg == base


@pytest.mark.parametrize("stage", ["b", "c"])
def test_theta_renewal_period_does_not_depend_on_n_theta(stage):
    """Renewal period in steps as in ThetaCollocation: resample_every_epochs x pool / n_interior.
    Within 1 step: in stage (c) the exact period is 2622.5 steps, rounded to 2622 or 2623."""
    periods = set()
    for n_theta in (512, 1024, 2048, 4096):
        cfg = load_config(os.path.join(REPO_ROOT, CONFIGS[stage]))
        apply_overrides(cfg, osb.overrides(dict(osb.CURRENT, n_theta=n_theta), osb.N_FIXED[stage]))
        tc = cfg["theta_colloc"]
        pool = (osb.N_FIXED[stage] + tc["n_theta"]) * tc["points_per_theta"]
        periods.add(max(1, int(round(tc["resample_every_epochs"] * pool / cfg["sampling"]["n_interior"]))))
    assert max(periods) - min(periods) <= 1


@pytest.mark.parametrize("x", [1.0e-4, 3.0e-3, 1.2345678e-4, 10.0, 0.25, 4.0, 5.001220405174])
def test_numbers_are_read_back_as_floats(x):
    cfg = {}
    apply_overrides(cfg, ["a.b=%s" % osb.yaml_number(x)])
    assert isinstance(cfg["a"]["b"], float) and cfg["a"]["b"] == pytest.approx(x, rel=1e-7)


def test_search_space():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(200):
        p = osb.suggest(study.ask())
        assert 4 <= p["depth"] <= 8 and p["width"] in range(64, 257, 32)
        assert ("fourier_scale" in p) == (p["n_fourier"] > 0)
        sets = osb.overrides(p, osb.N_FIXED["b"])
        assert any(s.startswith("model.fourier.sigma_z=") for s in sets) == (p["n_fourier"] > 0)


def test_the_enqueued_trial_is_the_current_model():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True))
    study.enqueue_trial(dict(osb.CURRENT))
    assert osb.suggest(study.ask()) == osb.CURRENT


def test_space_b1_keeps_the_distributions_of_the_first_study():
    D = optuna.distributions
    study = optuna.create_study()
    study.enqueue_trial(dict(osb.CURRENT))  # with Fourier features: all 9 parameters
    trial = study.ask()
    osb.suggest(trial)
    assert trial.distributions == dict(
        depth=D.IntDistribution(4, 8), width=D.IntDistribution(64, 256, step=32),
        activation=D.CategoricalDistribution(["tanh", "silu"]), n_fourier=D.CategoricalDistribution([0, 32, 64, 128]),
        fourier_scale=D.FloatDistribution(0.25, 4.0, log=True), lr=D.FloatDistribution(1.0e-4, 3.0e-3, log=True),
        energy_weight=D.FloatDistribution(1.0, 100.0, log=True), cauchy_c=D.FloatDistribution(1.0, 100.0, log=True),
        n_theta=D.CategoricalDistribution([512, 1024, 2048, 4096]))


def test_search_space_b2():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    acts = set()
    for _ in range(300):
        p = osb.suggest(study.ask(), osb.SPACE_B2)
        assert 6 <= p["depth"] <= 12 and p["width"] in range(128, 321, 32)
        assert p["n_fourier"] in (64, 128, 256) and 0.05 <= p["fourier_scale"] <= 1.0
        assert 2.0e-4 <= p["lr"] <= 2.0e-3 and 0.3 <= p["energy_weight"] <= 10.0 and 5.0 <= p["cauchy_c"] <= 60.0
        assert p["n_theta"] in (512, 1024, 2048)
        acts.add(p["activation"])
    assert acts == {"tanh", "silu", "gelu", "softplus", "mish", "sin"}


def test_trial_26_is_inside_space_b2_and_its_first_trial():
    assert osb.T26 == TRIAL_26
    for name, (kind, *a) in osb.SPACE_B2.items():
        v = osb.T26[name]
        assert (v in a[0]) if kind == "cat" else (a[0] <= v <= a[1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True))
    study.enqueue_trial(dict(osb.T26))
    assert osb.suggest(study.ask(), osb.SPACE_B2) == osb.T26


def test_read_rows_leaves_the_row_being_written(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text("step,y_c1_r2\r\n2500,0.5\r\n5000,0.6")
    assert [r["step"] for r in osb.read_rows(str(path))] == ["2500"]
    assert osb.read_rows(str(tmp_path / "missing.csv")) == []


def test_decision_of_the_confirmation():
    nan = float("nan")
    assert osb.decide({"current": [0.92, 0.93], "t1": [0.94, 0.93], "t2": [nan, 0.99]}) == ("t1", True)
    assert osb.decide({"current": [0.95, 0.95], "t1": [0.94, 0.95]}) == ("t1", False)


# Stand-in for train_phase2.py (--train-script): writes what the study reads (val_curve.csv,
# train_log.csv, summary.json) with the behaviour that plan.json gives to its run name.
FAKE_TRAINER = r'''
import csv, json, os, sys, time

args = sys.argv[1:]
run_name = args[args.index("--run-name") + 1]
sets = dict(s.split("=", 1) for s in args[args.index("--set") + 1:])
run_dir = os.path.join(sets["logging.runs_dir"], run_name)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")) as f:
    plan = json.load(f)
b = plan.get(run_name.split("/", 1)[1], plan.get("default"))
steps, every = int(sets["train.steps"]), int(sets["phase2.eval_every"])
with open(os.path.join(run_dir, "sets.json"), "w") as f:
    json.dump(sets, f)
with open(os.path.join(run_dir, "train_log.csv"), "w") as f:
    f.write("step,elapsed_s\n%d,1.5\n" % steps)
for k in range(every, steps + 1, every):
    time.sleep(b.get("sleep", 0.0))
    r2 = float("nan") if b.get("nan_at") == k else b["final"] * k / steps
    new = not os.path.exists(os.path.join(run_dir, "val_curve.csv"))
    with open(os.path.join(run_dir, "val_curve.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "y_c1_r2", "y_c1_r2_case_mean", "y_c1_r2_case_min"])
        if new:
            w.writeheader()
        w.writerow(dict(step=k, y_c1_r2=r2, y_c1_r2_case_mean=r2 - 0.05, y_c1_r2_case_min=r2 - 0.3))
    if b.get("stop_file_at") == k:
        open(os.path.join(run_dir, "..", "STOP"), "w").close()
if b.get("exit"):
    sys.exit(b["exit"])
y = b["final"]
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump(dict(validation=dict(outlet=dict(y_c1=dict(r2=y), T_out=dict(rmse=1.0)),
                                   per_case_mean=dict(r2_y_c1=y - 0.05), per_case_min=dict(r2_y_c1=y - 0.3),
                                   worst_case=dict(y_c1=dict(case=3, r2=y - 0.3))),
                   training=dict(median_ms_per_step=1.0, gpu_peak_mb=100.0, host_peak_rss_mb=900.0)), f)
'''


def fake_setup(tmp_path, plan):
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "fake_train.py").write_text(FAKE_TRAINER)
    return ["--runs-dir", str(tmp_path / "runs"), "--poll-s", "0.02", "--train-script", str(tmp_path / "fake_train.py")]


def load(study_dir, storage):
    if storage == "journal":
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(study_dir / "study.log")))
    else:
        storage = "sqlite:///%s" % (study_dir / "study.db")
    return optuna.load_study(study_name=osb.STUDY_NAME, storage=storage)


def test_study_prunes_fails_pauses_resumes_and_confirms(tmp_path, capsys):
    plan = {"t%04d" % i: {"final": 0.80 + 0.01 * i} for i in range(5)}  # pruner start-up trials
    plan["t0002"]["stop_file_at"] = 15000                 # STOP file: pause after trial 2
    plan["t0005"] = {"final": 0.30}                        # below the median at 5000 steps: pruned
    plan["t0006"] = {"final": 0.90, "nan_at": 7500}        # diverges: failed
    plan["t0007"] = {"final": 0.90, "exit": 1}             # training process crashes: failed
    plan["t0008"] = {"final": 0.95}
    finals = {"confirm_b": {"current": (0.90, 0.91), "t0008": (0.93, 0.94), "t0004": (0.85, 0.86), "t0003": (0.84, 0.85)},
              "confirm_c": {"current": (0.90, 0.90), "t0008": (0.92, 0.91)}}
    for stage, labels in finals.items():
        for label, values in labels.items():
            for seed, final in enumerate(values):
                plan["%s/%s_seed%d" % (stage, label, seed)] = {"final": final}
    common = fake_setup(tmp_path, plan)
    study_dir = tmp_path / "runs" / "optuna_b"

    osb.main(["run", "--n-trials", "9", *common])
    assert (study_dir / "STOP").exists()
    with pytest.raises(SystemExit):
        osb.main(["run", "--n-trials", "9", *common])
    (study_dir / "STOP").unlink()
    osb.main(["run", "--n-trials", "9", *common])  # resumes from the journal file

    study = load(study_dir, "journal")
    assert [t.state for t in study.trials] == [S.COMPLETE] * 5 + [S.PRUNED, S.FAIL, S.FAIL, S.COMPLETE]
    assert osb.same_as_current(study.trials[0].params)
    assert max(study.trials[5].intermediate_values) == 5000  # first check after the warm-up
    assert study.trials[5].user_attrs["y_r2_case_mean"] == pytest.approx(0.30 / 3 - 0.05)  # from the last validation
    assert "step 7500" in study.trials[6].user_attrs["error"]
    assert study.best_trial.number == 8 and study.best_value == pytest.approx(0.95)
    ua = study.best_trial.user_attrs
    assert ua["y_r2_case_mean"] == pytest.approx(0.90) and ua["y_r2_case_min"] == pytest.approx(0.65) and ua["worst_case"] == 3
    sets = json.loads((study_dir / "t0000" / "sets.json").read_text())
    assert sets["model.depth"] == "6" and sets["train.steps"] == "15000" and sets["phase2.eval_every"] == "2500"
    assert sets["runtime.gpu_memory_mb"] == str(osb.GPU_MEM_MB) and "model.siren_w0" not in sets

    osb.main(["report", "--importance", *common])
    assert "importance (fANOVA" in capsys.readouterr().out

    osb.main(["confirm", "--confirm-steps", "1000", "--workers", "3", *common])
    osb.main(["confirm", "--confirm-steps", "1000", "--workers", "3", *common])  # second call: every run is reused
    res = json.loads((study_dir / "confirm.json").read_text())
    assert list(res["stage_b"]) == ["current", "t0008", "t0004", "t0003"]
    assert res["decision"] == {"best": "t0008", "stage_c": True}
    assert res["stage_c"]["t0008"]["y_r2"] == pytest.approx([0.92, 0.91])
    assert res["stage_b"]["t0008"]["runs"][1]["y_r2_case_min"] == pytest.approx(0.94 - 0.3)
    sets = json.loads((study_dir / "confirm_c" / "t0008_seed1" / "sets.json").read_text())
    assert sets["runtime.seed"] == "1" and sets["loss.weights.data"] == "0.0"
    assert not [d for d in os.listdir(study_dir / "confirm_b") if "_old_" in d]


def test_second_study_starts_from_trial_26_and_reuses_its_confirmation(tmp_path, capsys):
    plan = {"default": {"final": 0.90}}
    plan.update({"t%04d" % i: {"final": 0.90 + 0.01 * i} for i in range(1, 6)})
    finals = {"confirm_b": {"t0005": (0.95, 0.96), "t0004": (0.94, 0.95), "t0003": (0.93, 0.94)},
              "confirm_c": {"t0005": (0.91, 0.92)}}
    for stage, labels in finals.items():
        for label, values in labels.items():
            for seed, final in enumerate(values):
                plan["%s/%s_seed%d" % (stage, label, seed)] = {"final": final}
    common = fake_setup(tmp_path, plan)
    runs = tmp_path / "runs"
    for stage, values in (("b", (0.973, 0.972)), ("c", (0.941, 0.940))):  # trial 26 confirmed in the first study
        for seed, y in enumerate(values):
            d = runs / "optuna_b" / ("confirm_%s" % stage) / ("t0026_seed%d" % seed)
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps(dict(validation=dict(outlet=dict(y_c1=dict(r2=y))), training={})))

    osb.main(["run", "--space", "b2", "--n-trials", "6", *common])
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(runs / "optuna_b2" / "study.log")))
    study = optuna.load_study(study_name="stage_b2", storage=storage)
    assert len(study.trials) == 6 and osb.same_as(study.trials[0].params, osb.T26)
    sets = json.loads((runs / "optuna_b2" / "t0000" / "sets.json").read_text())
    assert sets["model.activation"] == "silu" and sets["model.depth"] == "8" and sets["model.fourier.n_features"] == "128"
    assert sets["model.siren_w0"] == "1" and sets["runtime.gpu_memory_mb"] == "7000"  # every run of b2

    osb.main(["report", "--space", "b2", *common])
    out = capsys.readouterr().out
    assert "0 (study1_t0026)" in out and "| activation | complete / pruned / failed |" in out
    assert "| sin (SIREN init, w0 = 1) | " in out

    osb.main(["plot", "--space", "b2", "--out", str(tmp_path / "figs"), *common])
    assert sorted(os.listdir(tmp_path / "figs")) == sorted(name + ".png" for name in osb.SPACE_B2)

    osb.main(["confirm", "--space", "b2", "--confirm-steps", "1000", "--workers", "2", *common])
    res = json.loads((runs / "optuna_b2" / "confirm.json").read_text())
    assert list(res["stage_b"]) == ["study1_t0026", "t0005", "t0004", "t0003"]
    assert res["stage_b"]["study1_t0026"]["y_r2"] == pytest.approx([0.973, 0.972])  # reused, not trained again
    assert res["stage_b"]["study1_t0026"]["run_names"] == ["optuna_b/confirm_b/t0026_seed0", "optuna_b/confirm_b/t0026_seed1"]
    assert res["decision"] == {"best": "t0005", "stage_c": True}  # below trial 26 on (b), still tested on (c)
    assert res["stage_c"]["study1_t0026"]["y_r2"] == pytest.approx([0.941, 0.940])
    assert res["stage_c"]["t0005"]["y_r2"] == pytest.approx([0.91, 0.92])
    assert sorted(os.listdir(runs / "optuna_b2" / "confirm_b")) == ["t%04d_seed%d" % (t, s) for t in (3, 4, 5) for s in (0, 1)]
    assert sorted(os.listdir(runs / "optuna_b" / "confirm_b")) == ["t0026_seed0", "t0026_seed1"]  # nothing new there
    sets = json.loads((runs / "optuna_b2" / "confirm_c" / "t0005_seed1" / "sets.json").read_text())
    assert sets["model.siren_w0"] == "1" and sets["runtime.gpu_memory_mb"] == "7000" and sets["runtime.seed"] == "1"


def study_args(tmp_path):
    return argparse.Namespace(runs_dir=str(tmp_path / "runs"), name="optuna_b", storage="journal", eval_every=2500)


def test_parallel_workers_share_one_study(tmp_path):
    common = fake_setup(tmp_path, {"default": {"final": 0.9, "sleep": 0.25}})
    study = osb.open_study(study_args(tmp_path))
    for _ in range(6):  # queued trials: the workers must not take the same one (SQLite did)
        study.enqueue_trial(dict(osb.CURRENT))
    osb.main(["run", "--n-trials", "9", "--workers", "3", *common])
    trials = load(tmp_path / "runs" / "optuna_b", "journal").trials
    assert len(trials) == 9 and all(t.state == S.COMPLETE for t in trials)  # no extra trials at the end
    assert {t.user_attrs["worker"] for t in trials} == {0, 1, 2}
    assert all(osb.same_as_current(t.params) for t in trials[:6])
    assert len({json.dumps(t.params, sort_keys=True) for t in trials[6:]}) == 3  # each worker samples its own trials
    assert sorted(os.listdir(tmp_path / "runs" / "optuna_b" / "t0000")) == ["sets.json", "stdout.txt", "summary.json",
                                                                              "train_log.csv", "val_curve.csv"]
    assert sorted(d for d in os.listdir(tmp_path / "runs" / "optuna_b") if d.startswith("t")) == ["t%04d" % i for i in range(9)]
    spans = sorted((t.datetime_start, t.datetime_complete) for t in trials)
    assert any(b0 < a1 for (a0, a1), (b0, b1) in zip(spans, spans[1:]))  # some trials ran at the same time


def test_sqlite_is_for_one_worker_only(tmp_path):
    common = fake_setup(tmp_path, {"default": {"final": 0.9}})
    with pytest.raises(SystemExit, match="journal"):
        osb.main(["run", "--n-trials", "4", "--workers", "2", "--storage", "sqlite", *common])


def test_run_recovers_trials_left_running(tmp_path):
    common = fake_setup(tmp_path, {"default": {"final": 0.9}})
    study = osb.open_study(study_args(tmp_path))
    osb.suggest(study.ask())  # a trial cut off by a killed study: running, its parameters set
    osb.main(["run", "--n-trials", "3", "--workers", "2", *common])
    trials = load(tmp_path / "runs" / "optuna_b", "journal").trials
    assert [t.state for t in trials] == [S.FAIL, S.COMPLETE, S.COMPLETE]
    assert "interrupted" in trials[0].user_attrs["error"]
    assert trials[1].params == trials[0].params and trials[1].user_attrs["retry_of"] == 0
