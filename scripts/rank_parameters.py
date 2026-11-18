"""Which theta columns change the stoichiometric time and the temperature rise the most?

Used to choose the 10 parameters of Phase 2 stage (b) (the 4 of stage (a) plus 6). No solver run:
  1. one at a time around the base case (IC1 example_usage): each column swept over its range
     (21 points, the others fixed; points that fail the prefilter are left out), with
       t_st      the slowest stoichiometric time, Eq. (7) (exact equilibrium value)
       dT_net    sum_i (-dH_i)(q*_i,F - q*_i,0) / Cps: heat of the net loading change per heat
                 capacity of the solid (adsorption of the fed gas minus desorption of the carrier)
     effect = max/min ratio over the sweep, as a factor (t_st) or in kelvin (dT_net);
  2. over the 60 MOL cases of the provisional dataset (every column varies at once): Spearman
     rank correlation of each column with the measured outlet rise max|T - Tin| and with t_st.

    ~/ic2env/bin/python scripts/rank_parameters.py --data data/provisorio_MOL_n2_60cases.h5
"""

import argparse
import copy
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2.config import load_config  # noqa: E402
from nnads_ic2.data.domain import Domain  # noqa: E402
from nnads_ic2.data.loader import load_recorder_h5  # noqa: E402
from nnads_ic2.physics.params import theta_names  # noqa: E402
from nnads_ic2.physics.scales import feed_and_initial_state, measured_temperature_rise, stoich_times  # noqa: E402


def net_rise(p):
    st = feed_and_initial_state(p)
    return float(np.sum(p.dH * (st["q_feed"] - st["q_init"])) / p.Cps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "phase2_stage_a.yaml"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "provisorio_MOL_n2_60cases.h5"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "phase2", "parameter_ranking.csv"))
    ap.add_argument("--n-sweep", type=int, default=21)
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = theta_names(int(cfg["physics"]["n_comp"]))
    base_cfg = copy.deepcopy(cfg)
    base_cfg["domain"]["vary"] = ["vs"]
    base = Domain(base_cfg, ROOT).base
    tst0, rise0 = float(np.nanmax(stoich_times(base))), net_rise(base)
    print("base case: t_st = %.1f s, dT_net = %.2f K" % (tst0, rise0))

    cases = load_recorder_h5(args.data, 2, use_internal_profiles=False)
    Theta = np.stack([c.theta for c in cases])
    rise_meas = np.array([measured_temperature_rise(c.T_out, c.params.Tin) for c in cases])
    tst_data = np.array([np.nanmax(stoich_times(c.params)) for c in cases])

    rows = []
    for j, name in enumerate(names):
        one = copy.deepcopy(cfg)
        one["domain"]["vary"] = [name]
        dom = Domain(one, ROOT)
        tst, rise = [], []
        for u in np.linspace(0.0, 1.0, args.n_sweep):
            p = dom.case_from_unit([u])
            if dom.accept(p)[0]:
                tst.append(float(np.nanmax(stoich_times(p))))
                rise.append(net_rise(p))
        lo, hi, log = dom.ranges[0]
        rho_rise = spearmanr(Theta[:, j], rise_meas)[0]
        rho_tst = spearmanr(Theta[:, j], tst_data)[0]
        rows.append(dict(
            column=name, low=lo, high=hi, log=log, frac_accepted=len(tst) / args.n_sweep,
            tst_factor=max(tst) / min(tst) if tst else float("nan"),
            dT_net_span_K=max(rise) - min(rise) if rise else float("nan"),
            spearman_rise=rho_rise, spearman_tst=rho_tst,
        ))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%-14s %8s %10s %12s %10s %10s" % ("column", "accepted", "t_st x", "dT_net span", "rho rise", "rho t_st"))
    for r in sorted(rows, key=lambda r: -np.nan_to_num(np.log(r["tst_factor"])) - np.nan_to_num(r["dT_net_span_K"]) / 10):
        print("%-14s %8.2f %10.2f %10.2f K %10.2f %10.2f" % (r["column"], r["frac_accepted"], r["tst_factor"],
                                                            r["dT_net_span_K"], r["spearman_rise"], r["spearman_tst"]))
    print("written", args.out)


if __name__ == "__main__":
    main()
