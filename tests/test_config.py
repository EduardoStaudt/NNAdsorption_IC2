import os

from conftest import REPO_ROOT
from nnads_ic2.config import apply_overrides, load_config


def test_child_config_inherits_and_overrides():
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "pinn_single_case.yaml"))
    assert cfg["loss"]["weights"]["data"] == 0.0  # child value
    assert cfg["loss"]["weights"]["pde"] == 1.0  # inherited from pinn_base.yaml
    assert cfg["case"]["path"].endswith("ic1_example_h2_co2.json")
    par = load_config(os.path.join(REPO_ROOT, "configs", "pinn_parametric.yaml"))
    assert par["loss"]["weights"]["data"] == 1.0 and par["phase1"]["n_train_cases"] == 50


def test_command_line_overrides_are_parsed_as_yaml():
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "pinn_base.yaml"))
    apply_overrides(cfg, ["loss.pde_terms.energy=10", "train.lr=5.0e-4", "train.lbfgs.enabled=true", "new.key=[1, 2]"])
    assert cfg["loss"]["pde_terms"]["energy"] == 10
    assert cfg["train"]["lr"] == 5e-4
    assert cfg["train"]["lbfgs"]["enabled"] is True
    assert cfg["new"]["key"] == [1, 2]
