"""Solver datasets in the h5 format of IC1 recorder.py, plus optional internal profiles.

Recorder keys (one row per seed):
    X_v2   [N, d]   theta vector (IC1 order)
    c_out  [N, n]   variable-length outlet series c_i(L, t) [mol/m3]
    y_out  [N, n]   variable-length outlet series y_i(L, t)
    T_out  [N]      variable-length outlet temperature [K]
    Tz [N, NX], qz [N, n, NX]   profiles at the final time t = TF
    TF, NT, seeds, valid [N]
The recorder stores the state after each time step, so sample k (0-based) of an outlet series is
at t_k = (k + 1) TF / NT. Axial grid of Tz/qz: file attribute `grid` ("nodes": z_j = j L/(NX-1),
the IC1 solver; "cells": z_j = (j + 1/2) L/NX, finite volumes). IC1 files have no attribute and
use nodes.

IC2 extensions (optional; the v23b format is not known yet):
    v_out  [N]      variable-length outlet velocity [m/s]
    profiles/<row>/{t, c, q, T, v}   internal fields at a few times (z grid in attribute `z`)
    solve_info/<name> [N]            per-row solver statistics (e.g. wall time per seed)
    file attributes, e.g. `dataset_tag` ("provisório_MOL": MOL data, not v23b) and `source`
`use_internal_profiles=True` reads the profiles when present; `dataset_info` reads the attributes.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from ..physics.params import CaseParams


@dataclass
class SolverCase:
    row: int
    seed: int
    params: CaseParams
    theta: np.ndarray
    TF: float
    t: np.ndarray  # outlet time grid [s]
    y_out: np.ndarray  # [nt, n]
    T_out: np.ndarray  # [nt]
    c_out: np.ndarray = None  # [nt, n]
    v_out: np.ndarray = None  # [nt]
    final: dict = None  # z [NX], T [NX], q [n, NX] at t = TF
    profiles: dict = None  # t [k], z [NX], c [k, n, NX], q [k, n, NX], T [k, NX], v [k, NX]


def _axial_grid(L, nx, grid):
    if grid == "cells":
        return (np.arange(nx) + 0.5) * L / nx
    return np.arange(nx) * L / (nx - 1)


def _text(v):
    return v.decode() if isinstance(v, bytes) else v


def dataset_info(path):
    """File attributes of a dataset (tag, source, solver settings...) plus its row counts."""
    with h5py.File(path, "r") as f:
        info = {k: _text(v) for k, v in f.attrs.items()}
        info = {k: (v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else v)
                for k, v in info.items()}
        n = f["X_v2"].shape[0]
        info["n_rows"] = int(n)
        info["n_valid"] = int(np.sum(f["valid"][:])) if "valid" in f else int(n)
    return info


def load_recorder_h5(path, n_comp, carrier=0, use_internal_profiles=False, valid_only=True, rows=None):
    """Read the cases of a recorder-format h5 file into a list of SolverCase."""
    out = []
    with h5py.File(path, "r") as f:
        N = f["X_v2"].shape[0]
        grid = _text(f.attrs.get("grid", "nodes"))
        rows = np.arange(N) if rows is None else np.asarray(rows)
        valid = f["valid"][:] if "valid" in f else np.ones(N, dtype=np.int8)
        for r in rows:
            r = int(r)
            if valid_only and not valid[r]:
                continue
            theta = np.asarray(f["X_v2"][r], dtype=np.float64)
            p = CaseParams.from_theta(theta, n_comp, carrier=carrier)
            TF, NT = float(f["TF"][r]), int(f["NT"][r])
            y_out = np.stack([np.asarray(f["y_out"][r, i], dtype=np.float64) for i in range(n_comp)], axis=1)
            nt = y_out.shape[0]
            case = SolverCase(
                row=r,
                seed=int(f["seeds"][r]) if "seeds" in f else r,
                params=p,
                theta=theta,
                TF=TF,
                t=(np.arange(nt) + 1) * TF / NT,
                y_out=y_out,
                T_out=np.asarray(f["T_out"][r], dtype=np.float64),
            )
            if "c_out" in f:
                case.c_out = np.stack([np.asarray(f["c_out"][r, i], dtype=np.float64) for i in range(n_comp)], axis=1)
            if "v_out" in f:
                case.v_out = np.asarray(f["v_out"][r], dtype=np.float64)
            if "Tz" in f and "qz" in f:
                Tz = np.asarray(f["Tz"][r], dtype=np.float64)
                case.final = dict(z=_axial_grid(p.L, Tz.size, grid), T=Tz, q=np.asarray(f["qz"][r], dtype=np.float64))
            if use_internal_profiles and "profiles" in f and str(r) in f["profiles"]:
                g = f["profiles"][str(r)]
                case.profiles = {k: np.asarray(g[k], dtype=np.float64) for k in ("t", "c", "q", "T", "v")}
                case.profiles["z"] = np.asarray(g.attrs["z"], dtype=np.float64)
            out.append(case)
    return out


def write_recorder_h5(path, items, write_profiles=False, profile_stride=1, attrs=None, row_data=None):
    """Write solutions in the recorder format (+ v_out and, optionally, internal profiles).

    items: list of (seed, CaseParams, sol) with `sol` as returned by physics/mol_solver.py.
    The MOL solver saves its initial state too, so the first sample is dropped and the series
    follow the recorder convention t_k = (k + 1) TF / NT.
    attrs: extra file attributes (dataset tag, solver settings...); row_data: {name: [N] values}
    written to solve_info/<name>.
    """
    N = len(items)
    n = items[0][1].n_comp
    nx = items[0][2]["z"].size
    vlf = h5py.special_dtype(vlen=np.float32)
    with h5py.File(path, "w") as f:
        f.attrs.update(dict(config="%dcomp_1layer" % n, nmax=n, NX=nx, grid="cells", source="MOL reference solver"))
        f.attrs.update(attrs or {})
        if row_data:
            g = f.create_group("solve_info")
            for name, values in row_data.items():
                g.create_dataset(name, data=np.asarray(values))
        X = f.create_dataset("X_v2", (N, items[0][1].to_theta().size), dtype=np.float32)
        dsc = f.create_dataset("c_out", (N, n), dtype=vlf)
        dsy = f.create_dataset("y_out", (N, n), dtype=vlf)
        dst = f.create_dataset("T_out", (N,), dtype=vlf)
        dsv = f.create_dataset("v_out", (N,), dtype=vlf)
        Tz = f.create_dataset("Tz", (N, nx), dtype=np.float32)
        qz = f.create_dataset("qz", (N, n, nx), dtype=np.float32)
        TF = f.create_dataset("TF", (N,), dtype=np.float32)
        NT = f.create_dataset("NT", (N,), dtype=np.int32)
        seeds = f.create_dataset("seeds", (N,), dtype=np.int64)
        valid = f.create_dataset("valid", (N,), dtype=np.int8)
        prof = f.create_group("profiles") if write_profiles else None
        for r, (seed, p, sol) in enumerate(items):
            X[r] = p.to_theta()
            for i in range(n):
                dsc[r, i] = sol["c_out"][1:, i]
                dsy[r, i] = sol["y_out"][1:, i]
            dst[r] = sol["T_out"][1:]
            dsv[r] = sol["v_out"][1:]
            Tz[r] = sol["T"][-1]
            qz[r] = sol["q"][-1]
            TF[r] = sol["t"][-1]
            NT[r] = sol["t"].size - 1
            seeds[r] = seed
            valid[r] = 1
            if prof is not None:
                g = prof.create_group(str(r))
                k = slice(None, None, profile_stride)
                g.create_dataset("t", data=sol["t"][k])
                g.create_dataset("c", data=sol["c"][k].astype(np.float32))
                g.create_dataset("q", data=sol["q"][k].astype(np.float32))
                g.create_dataset("T", data=sol["T"][k].astype(np.float32))
                g.create_dataset("v", data=sol["v_cell"][k].astype(np.float32))
                g.attrs["z"] = sol["z"]
