import copy
import os

import numpy as np
import pytest

from conftest import EXAMPLE_CASE, REPO_ROOT
from nnads_ic2.config import load_config
from nnads_ic2.data.cases import sample_case_ic1
from nnads_ic2.data.domain import Domain, ThetaSampler, sort_components_by_affinity
from nnads_ic2.physics.params import CaseParams, theta_names

CFG = load_config(os.path.join(REPO_ROOT, "configs", "pinn_base.yaml"))
STAGE_A = ["vs", "y0", "Tin", "L0_c1_kL"]


def subspace_cfg(vary):
    cfg = copy.deepcopy(CFG)
    cfg["domain"].update(mode="subspace", vary=list(vary))
    return cfg


def test_subspace_cases_keep_the_base_values_outside_vary():
    dom = Domain(subspace_cfg(STAGE_A), REPO_ROOT)
    base = CaseParams.from_ic1_json(EXAMPLE_CASE).to_theta()
    names = theta_names(2)
    fixed = [j for j, n in enumerate(names) if n not in STAGE_A]
    for seed in range(20):
        p, ok, info = dom.sample(seed)
        th = p.to_theta()
        np.testing.assert_allclose(th[fixed], base[fixed], rtol=1e-12)
        for name, (lo, hi, _) in zip(dom.vary, dom.ranges):
            assert lo <= th[names.index(name)] <= hi
        assert info["tst_max_s"] > 0
    assert dom.ranges[STAGE_A.index("y0")] == (0.20, 0.75, False)
    assert dom.ranges[STAGE_A.index("L0_c1_kL")][2]  # kL varies log-uniformly


def test_full_domain_seed_is_the_ic1_recorder_case():
    dom = Domain(CFG, REPO_ROOT)
    cc = CFG["cases"]
    p, _, _ = dom.sample(5)
    np.testing.assert_array_equal(p.to_theta(), sample_case_ic1(5, cc["x_ranges"], cc["y0_range"], cc["log_params"]).to_theta())
    assert dom.dim == 31


def test_unit_cube_maps_to_the_range_ends():
    dom = Domain(subspace_cfg(["vs", "L0_c1_kL"]), REPO_ROOT)
    np.testing.assert_allclose(dom.values_from_unit([0.0, 0.0]), [0.001, 0.01])
    np.testing.assert_allclose(dom.values_from_unit([1.0, 1.0]), [0.05, 1.0])
    np.testing.assert_allclose(dom.values_from_unit([0.5, 0.5])[1], 0.1)  # log-uniform: geometric midpoint


def test_sorting_puts_the_least_adsorbed_component_first():
    p = CaseParams.from_ic1_json(EXAMPLE_CASE)
    th = p.to_theta()
    swapped = th.copy()
    swapped[:9], swapped[9:18] = th[9:18], th[:9]
    np.testing.assert_array_equal(sort_components_by_affinity(swapped, 2), th)


def test_sobol_theta_pass_the_prefilter_and_fill_the_box():
    dom = Domain(subspace_cfg(STAGE_A), REPO_ROOT)
    sampler = ThetaSampler(dom, seed=0, chunk=256)
    first = sampler.draw(300)
    second = sampler.draw(300)
    assert len(first) == len(second) == 300
    assert all(dom.accept(p)[0] for p in first + second)
    vs = np.array([p.vs for p in first + second])
    # evenly spread: every tenth of the vs range gets points (prefilter removes only slow beds)
    counts, _ = np.histogram(vs, bins=10, range=(0.001, 0.05))
    assert counts.min() > 20
    assert not {p.vs for p in first} & {p.vs for p in second}  # the sequence continues


def test_full_domain_sobol_keeps_ic1_component_order():
    dom = Domain(CFG, REPO_ROOT)
    for p in ThetaSampler(dom, seed=1, chunk=128).draw(50):
        assert p.B_ref[0] <= p.B_ref[1]


def test_unknown_column_is_rejected():
    with pytest.raises(ValueError):
        Domain(subspace_cfg(["vs", "kL"]), REPO_ROOT)


def test_snap_restores_the_fixed_columns_of_a_float32_copy():
    dom = Domain(subspace_cfg(STAGE_A), REPO_ROOT)
    p, _, _ = dom.sample(3)
    stored = CaseParams.from_theta(p.to_theta().astype(np.float32).astype(np.float64), 2)  # as read from an h5 file
    snapped = dom.snap(stored)
    fixed = [j for j in range(31) if j not in dom.cols]
    np.testing.assert_array_equal(snapped.to_theta()[fixed], dom.base.to_theta()[fixed])
    np.testing.assert_allclose(snapped.to_theta()[dom.cols], p.to_theta()[dom.cols], rtol=1e-7)


def test_scaler_treats_round_off_of_large_constant_columns_as_constant():
    from nnads_ic2.experiment import fit_theta_scaler

    dom = Domain(subspace_cfg(STAGE_A), REPO_ROOT)
    cases = ThetaSampler(dom, seed=0).draw(4096)
    sc = fit_theta_scaler(cases, CFG)
    P = theta_names(2).index("P")
    assert sc.sd[P] == pytest.approx(abs(sc.mu[P]))  # constant column: sd = |mu|, not its round-off
    stored = dom.base.to_theta().astype(np.float32).astype(np.float64)
    assert np.abs(sc.transform(stored[None, :])[0, P]) < 1e-6  # a float32 copy stays in distribution
