import importlib.util
import json
import os
import sys

import optuna
import pytest

from conftest import REPO_ROOT
from nnads_ic2.config import apply_overrides, load_config

spec = importlib.util.spec_from_file_location("optuna_stage_b", os.path.join(REPO_ROOT, "scripts", "optuna_stage_b.py"))
osb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osb)

pytestmark = pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")

CONFIGS = {"b": osb.CONFIG_B, "c": osb.CONFIG_C}


@pytest.mark.parametrize("stage", ["b", "c"])
def test_current_parameters_change_nothing(stage):
    """The base config of the trials (b) and of the stage (c) test is the current model."""
    path = os.path.join(REPO_ROOT, CONFIGS[stage])
    base, cfg = load_config(path), load_config(path)
    apply_overrides(cfg, osb.overrides(osb.CURRENT, osb.N_FIXED[stage]))
    assert cfg == base
    assert osb.same_as_current(osb.CURRENT)


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


def test_read_rows_leaves_the_row_being_written(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text("step,y_c1_r2\r\n2500,0.5\r\n5000,0.6")
    assert [r["step"] for r in osb.read_rows(str(path))] == ["2500"]
    assert osb.read_rows(str(tmp_path / "missing.csv")) == []


def test_decision_of_the_confirmation():
    nan = float("nan")
    assert osb.decide({"current": [0.92, 0.93], "t1": [0.94, 0.93], "t2": [nan, 0.99]}) == ("t1", True)
    assert osb.decide({"current": [0.95, 0.95], "t1": [0.94, 0.95]}) == ("t1", False)


# Stand-in for train_phase2.py: writes what the study reads (val_curve.csv, train_log.csv,
# summary.json) with the behaviour that plan.json gives to its run name.
FAKE_TRAINER = r'''
import csv, json, os, sys

args = sys.argv[1:]
run_name = args[args.index("--run-name") + 1]
sets = dict(s.split("=", 1) for s in args[args.index("--set") + 1:])
run_dir = os.path.join(sets["logging.runs_dir"], run_name)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")) as f:
    b = json.load(f)[run_name.split("/", 1)[1]]
steps, every = int(sets["train.steps"]), int(sets["phase2.eval_every"])
with open(os.path.join(run_dir, "sets.json"), "w") as f:
    json.dump(sets, f)
with open(os.path.join(run_dir, "train_log.csv"), "w") as f:
    f.write("step,elapsed_s\n%d,1.5\n" % steps)
for k in range(every, steps + 1, every):
    new = not os.path.exists(os.path.join(run_dir, "val_curve.csv"))
    with open(os.path.join(run_dir, "val_curve.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "y_c1_r2"])
        if new:
            w.writeheader()
        w.writerow(dict(step=k, y_c1_r2=float("nan") if b.get("nan_at") == k else b["final"] * k / steps))
    if b.get("stop_file_at") == k:
        open(os.path.join(run_dir, "..", "STOP"), "w").close()
if b.get("exit"):
    sys.exit(b["exit"])
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump(dict(validation=dict(outlet=dict(y_c1=dict(r2=b["final"]), T_out=dict(rmse=1.0))),
                   training=dict(median_ms_per_step=1.0)), f)
'''


def test_study_prunes_fails_pauses_resumes_and_confirms(tmp_path, monkeypatch):
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
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    fake = tmp_path / "fake_train.py"
    fake.write_text(FAKE_TRAINER)
    monkeypatch.setattr(osb, "train_command", lambda config, val, run_name, sets: [
        sys.executable, str(fake), "--config", config, "--val", val, "--run-name", run_name, "--set", *sets])
    common = ["--runs-dir", str(tmp_path / "runs"), "--poll-s", "0.02"]
    study_dir = tmp_path / "runs" / "optuna_b"

    osb.main(["run", "--n-trials", "9", *common])
    assert (study_dir / "STOP").exists()
    with pytest.raises(SystemExit):
        osb.main(["run", "--n-trials", "9", *common])
    (study_dir / "STOP").unlink()
    osb.main(["run", "--n-trials", "9", *common])  # resumes from the SQLite file

    study = optuna.load_study(study_name=osb.STUDY_NAME, storage="sqlite:///%s" % (study_dir / "study.db"))
    S = optuna.trial.TrialState
    assert [t.state for t in study.trials] == [S.COMPLETE] * 5 + [S.PRUNED, S.FAIL, S.FAIL, S.COMPLETE]
    assert osb.same_as_current(study.trials[0].params)
    assert max(study.trials[5].intermediate_values) == 5000  # first check after the warm-up
    assert "step 7500" in study.trials[6].user_attrs["error"]
    assert study.best_trial.number == 8 and study.best_value == pytest.approx(0.95)
    sets = json.loads((study_dir / "t0000" / "sets.json").read_text())
    assert sets["model.depth"] == "6" and sets["train.steps"] == "15000" and sets["phase2.eval_every"] == "2500"

    osb.main(["confirm", "--confirm-steps", "1000", *common])
    osb.main(["confirm", "--confirm-steps", "1000", *common])  # second call: every run is reused
    res = json.loads((study_dir / "confirm.json").read_text())
    assert list(res["stage_b"]) == ["current", "t0008", "t0004", "t0003"]
    assert res["decision"] == {"best": "t0008", "stage_c": True}
    assert res["stage_c"]["t0008"]["y_r2"] == pytest.approx([0.92, 0.91])
    sets = json.loads((study_dir / "confirm_c" / "t0008_seed1" / "sets.json").read_text())
    assert sets["runtime.seed"] == "1" and sets["loss.weights.data"] == "0.0"
    assert not [d for d in os.listdir(study_dir / "confirm_b") if "_old_" in d]
