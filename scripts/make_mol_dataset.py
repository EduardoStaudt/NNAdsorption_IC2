"""Provisional datasets from the MOL reference solver (NOT the v23b solver).

Run inside WSL2, from the repository root:
    ~/ic2env/bin/python scripts/make_mol_dataset.py --n-cases 60 --workers 16
    ~/ic2env/bin/python scripts/make_mol_dataset.py --config configs/phase2_stage_a.yaml \\
        --n-cases 20 --seed-start 100000 --purpose validation --out data/<name>.h5
    ~/ic2env/bin/python scripts/make_mol_dataset.py --seeds 21 38 62 --max-solve-s 10800 ...

Seeds seed_start, seed_start + 1, ... are drawn from the `domain:` of the config (data/domain.py:
IC1 sampler for the full domain, or a subspace around the base case); the cases that pass the
prefilter (`cases:` section) are solved with physics/mol_solver.py (`solver:` section), each in
its own process on one CPU core, until n_cases are solved. A solver failure or a solve longer
than --max-solve-s is skipped and the next seed takes its place. The seeds in cases.hard_seeds
are skipped. With --seeds, exactly those seeds are solved (failures are not replaced).

Output: the IC1 recorder h5 format plus v_out and internal profiles. The tag "provisório_MOL" is
in the file name (without the accent) and in the attribute `dataset_tag`, so these data are never
mixed with the v23b dataset. The cost of each seed (wall and CPU time of its solve, solver
statistics) goes to solve_info/ in the file and to <file>.cost.csv.
"""

import os

# One BLAS thread per process (set before numpy loads): each solve runs on one core, so the time
# per seed is a single-core cost and the workers do not compete for cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nnads_ic2.config import apply_overrides, load_config  # noqa: E402
from nnads_ic2.data.domain import Domain  # noqa: E402
from nnads_ic2.data.loader import write_recorder_h5  # noqa: E402
from nnads_ic2.physics.mol_solver import solve_case  # noqa: E402
from nnads_ic2.physics.scales import compute_scales  # noqa: E402

DATASET_TAG = "provisório_MOL"
FILE_TAG = "provisorio_MOL"  # ASCII version for file names
SOURCE = "MOL reference solver (src/nnads_ic2/physics/mol_solver.py), NOT the v23b solver: provisional Phase 1 data"


class SolveTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise SolveTimeout()


def solve_one(seed, p, scales_cfg, solver_cfg, max_solve_s):
    """Solve one case in a worker process. Returns (seed, sol or None, error, wall_s, cpu_s)."""
    t_final = compute_scales(p, **scales_cfg).t_final  # t_F = factor * t_ref (does not depend on dT_ref)
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(max_solve_s))
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        sol, err = solve_case(p, t_final, solver_cfg), None
    except SolveTimeout:
        sol, err = None, "timeout after %d s" % max_solve_s
    except Exception as e:  # solver failure: recorded, the seed is replaced
        sol, err = None, repr(e)
    finally:
        signal.alarm(0)
    return seed, sol, err, time.perf_counter() - wall0, time.process_time() - cpu0


def accepted_cases(domain, seeds, rejected, skip=()):
    """Yield (seed, CaseParams, prefilter info) of the seeds that pass the prefilter."""
    for seed in seeds:
        if seed in skip:
            continue
        p, ok, info = domain.sample(seed)
        if ok:
            yield seed, p, info
        else:
            rejected.append(seed)


def git_version():
    """Short commit hash, with -dirty if tracked files changed. autocrlf/fileMode: a Windows checkout
    read from WSL would otherwise look modified (CRLF line ends, 777 file modes)."""
    git = ["git", "-c", "core.autocrlf=true", "-c", "core.fileMode=false"]
    try:
        rev = subprocess.run(git + ["rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "pinn_base.yaml"))
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="solve exactly these seeds (no replacement)")
    ap.add_argument("--purpose", default="training", help="stored as an attribute (training, validation, hard cases)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-seeds", type=int, default=100000, help="give up after this many sampled seeds")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--profile-stride", type=int, default=4, help="keep every k-th output time in the internal profiles")
    ap.add_argument("--max-solve-s", type=int, default=3600, help="skip a seed whose solve takes longer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="config overrides, e.g. solver.nx=141")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    n_comp = int(cfg["physics"]["n_comp"])
    domain = Domain(cfg, ROOT)
    if args.seeds:
        args.n_cases = len(args.seeds)
    out = args.out or os.path.join(ROOT, "data", "%s_n%d_%dcases.h5" % (FILE_TAG, n_comp, args.n_cases))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print("%s -> %s | %d workers | solver %s | domain %s" % (DATASET_TAG, out, args.workers, json.dumps(cfg["solver"]),
                                                           json.dumps(domain.describe())), flush=True)

    rejected, failed, done, info = [], [], [], {}
    if args.seeds:
        candidates = accepted_cases(domain, args.seeds, rejected)
    else:
        seeds = range(args.seed_start, args.seed_start + args.max_seeds)
        candidates = accepted_cases(domain, seeds, rejected, skip=set(cfg["cases"].get("hard_seeds") or ()))
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        running = {}

        def submit_next():
            """Start the next accepted seed; False when there are no seeds left."""
            nxt = next(candidates, None)
            if nxt is None:
                return False
            seed, p, pre = nxt
            info[seed] = (p, pre)
            running[ex.submit(solve_one, seed, p, cfg["scales"], cfg["solver"], args.max_solve_s)] = seed
            return True

        def fill():
            n_target = args.n_cases - len(failed) if args.seeds else args.n_cases  # --seeds: no replacement
            while len(done) + len(running) < n_target:
                if not submit_next():
                    if not args.seeds:
                        raise RuntimeError("no more seeds: raise --max-seeds")
                    return

        fill()
        while running:
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                running.pop(fut)
                seed, sol, err, wall, cpu = fut.result()
                if err is None:
                    done.append((seed, sol, wall, cpu))
                    print("seed %6d  solved in %6.1f s  (%d/%d)" % (seed, wall, len(done), args.n_cases), flush=True)
                else:
                    failed.append(dict(seed=seed, error=err, wall_s=wall))
                    print("seed %6d  FAILED after %.1f s: %s" % (seed, wall, err), flush=True)
            fill()
    elapsed = time.perf_counter() - t_start
    if not done:
        print(json.dumps(dict(failed=failed, rejected=rejected), indent=1), flush=True)
        sys.exit("no seed was solved")

    done.sort(key=lambda d: d[0])
    items = [(seed, info[seed][0], sol) for seed, sol, _, _ in done]
    rows = []
    for seed, sol, wall, cpu in done:
        pre = info[seed][1]
        rows.append(dict(seed=seed, tst_max_s=pre["tst_max_s"], severity=pre["severity"], t_final_s=float(sol["t"][-1]),
                         wall_s=wall, cpu_s=cpu, nfev=sol["info"]["nfev"], njev=sol["info"]["njev"], nlu=sol["info"]["nlu"]))
    walls = np.array([r["wall_s"] for r in rows])
    stats = dict(n_cases=len(rows), n_failed=len(failed), n_rejected_prefilter=len(rejected),
                 n_seeds_sampled=len(info) + len(rejected),
                 wall_per_seed_mean_s=float(walls.mean()), wall_per_seed_median_s=float(np.median(walls)),
                 wall_per_seed_min_s=float(walls.min()), wall_per_seed_max_s=float(walls.max()),
                 cpu_hours_total=float(sum(r["cpu_s"] for r in rows) + sum(f["wall_s"] for f in failed)) / 3600.0,
                 elapsed_s=elapsed, workers=args.workers)
    attrs = dict(
        dataset_tag=DATASET_TAG,
        source=SOURCE,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=git_version(),
        solver_cfg=json.dumps(cfg["solver"]),
        cases_cfg=json.dumps(cfg["cases"]),
        domain=json.dumps(domain.describe()),
        purpose=args.purpose,
        t_final_factor=float(cfg["scales"]["t_final_factor"]),
        seed_start=args.seed_start,
        rejected_seeds=json.dumps(rejected),
        failed_seeds=json.dumps(failed),
        profile_stride=args.profile_stride,
        cost=json.dumps(stats),
        machine=json.dumps(dict(cpu=cpu_model(), python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, threads_per_solve=1)),
    )
    row_data = {k: [r[k] for r in rows] for k in ("wall_s", "cpu_s", "nfev", "njev", "nlu", "t_final_s", "tst_max_s", "severity")}
    write_recorder_h5(out, items, write_profiles=True, profile_stride=args.profile_stride, attrs=attrs, row_data=row_data)
    with open(os.path.splitext(out)[0] + ".cost.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(stats, indent=1), flush=True)
    print("written %s (%.1f MB)" % (out, os.path.getsize(out) / 2**20), flush=True)


if __name__ == "__main__":
    main()
