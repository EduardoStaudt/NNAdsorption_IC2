import pytest

from nnads_ic2.training.plateau import plateau_step

STEPS = [25000, 50000, 75000, 100000, 125000, 150000]


def test_flat_for_two_points_in_a_row():
    # gains: 0.05, 0.001, 0.0015 -> flat at the 2nd small gain
    assert plateau_step(STEPS[:4], [0.80, 0.85, 0.851, 0.8525]) == 100000


def test_one_small_gain_is_not_enough():
    # a small gain followed by a large one resets the count
    assert plateau_step(STEPS[:5], [0.80, 0.801, 0.83, 0.831, 0.86]) is None


def test_a_drop_counts_as_flat():
    assert plateau_step(STEPS[:3], [0.90, 0.88, 0.881]) == 75000


def test_short_curve_and_one_point():
    assert plateau_step([25000], [0.9]) is None
    assert plateau_step(STEPS[:2], [0.9, 0.9], n_points=1) == 50000
    with pytest.raises(ValueError):
        plateau_step(STEPS, [0.0] * 6, n_points=0)


def test_base_config_is_off(tmp_path):
    from nnads_ic2.config import load_config
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "phase2_base.yaml"))
    ps = cfg["phase2"]["plateau_stop"]
    assert ps == {"enabled": False, "min_gain": 0.002, "n_points": 2}
