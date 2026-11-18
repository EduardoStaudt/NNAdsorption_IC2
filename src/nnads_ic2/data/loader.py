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

IC2 extensions (optional; the exact v23b layout is not known yet):
    v_out  [N]      variable-length outlet velocity v(L, t) [m/s]
    profiles/<row>/{t, c, q, T, v}   internal fields at a few times (z grid in attribute `z`);
                                     every field except t is optional (e.g. no v)
    solve_info/<name> [N]            per-row solver statistics (e.g. wall time per seed)
    file attributes, e.g. `dataset_tag` ("provisório_MOL": MOL data, not v23b) and `source`
`use_internal_profiles=True` reads the profiles when present; `dataset_info` reads the attributes.

Prepared for the v23b dataset: the key names can be changed with `keys` (config data.keys,
defaults = recorder names above), n_comp follows from the width of the theta vector (10 n + 11
columns, IC1 layout), and c_out, v_out and the profiles are optional. Without c_out, the outlet
concentrations come from the model's ideal-gas closure at constant pressure, c_i = y_i P/(R T)
(the case records it in `derived`). Every component keeps its own series (no y1 = 1 - y0).
"""

from dataclasses import dataclass

import h5py
import numpy as np

from ..physics.constants import R
from ..physics.params import CaseParams

DEFAULT_KEYS = dict(theta="X_v2", c_out="c_out", y_out="y_out", T_out="T_out", v_out="v_out", Tz="Tz", qz="qz",
                    TF="TF", NT="NT", seeds="seeds", valid="valid", profiles="profiles")
PROFILE_FIELDS = ("c", "q", "T", "v")


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
    profiles: dict = None  # t [k], z [NX], c [k, n, NX], q [k, n, NX], T [k, NX], v [k, NX] (fields optional)
    derived: tuple = ()  # series not in the file, computed by the loader (e.g. "c_out")


def _axial_grid(L, nx, grid):
    if grid == "cells":
        return (np.arange(nx) + 0.5) * L / nx
    return np.arange(nx) * L / (nx - 1)


def _text(v):
    return v.decode() if isinstance(v, bytes) else v


def dataset_info(path, keys=None):
    """File attributes of a dataset (tag, source, solver settings...) plus its row counts."""
    k = dict(DEFAULT_KEYS, **(keys or {}))
    with h5py.File(path, "r") as f:
        info = {name: _text(v) for name, v in f.attrs.items()}
        info = {name: (v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else v)
                for name, v in info.items()}
        n = f[k["theta"]].shape[0]
        info["n_rows"] = int(n)
        info["n_valid"] = int(np.sum(f[k["valid"]][:])) if k["valid"] in f else int(n)
        info["n_comp"] = infer_n_comp(f[k["theta"]].shape[1])
        info["has"] = {name: k[name] in f for name in ("c_out", "v_out", "Tz", "qz", "profiles")}
    return info


def infer_n_comp(width):
    """Number of gases from the width of the theta vector: 9 n + 3 + 9 + (n - 1) = 10 n + 11 columns."""
    n, rest = divmod(int(width) - 11, 10)
    if rest or n < 1:
        raise ValueError("theta width %d does not fit the IC1 layout (10 n + 11 columns)" % width)
    return n


def _series(f, key, r, n_comp):
    return np.stack([np.asarray(f[key][r, i], dtype=np.float64) for i in range(n_comp)], axis=1)


def load_recorder_h5(path, n_comp=None, carrier=0, use_internal_profiles=False, valid_only=True, rows=None, keys=None):
    """Read the cases of a recorder-format h5 file into a list of SolverCase.

    n_comp=None: taken from the theta width (and checked against it when given).
    keys: other dataset names, e.g. {"theta": "X_v3"} (defaults: DEFAULT_KEYS).
    """
    k = dict(DEFAULT_KEYS, **(keys or {}))
    out = []
    with h5py.File(path, "r") as f:
        N, width = f[k["theta"]].shape
        n_file = infer_n_comp(width)
        if n_comp is not None and int(n_comp) != n_file:
            raise ValueError("%s has theta for %d gases, n_comp=%d was requested" % (path, n_file, n_comp))
        n_comp = n_file
        grid = _text(f.attrs.get("grid", "nodes"))
        rows = np.arange(N) if rows is None else np.asarray(rows)
        valid = f[k["valid"]][:] if k["valid"] in f else np.ones(N, dtype=np.int8)
        for r in rows:
            r = int(r)
            if valid_only and not valid[r]:
                continue
            theta = np.asarray(f[k["theta"]][r], dtype=np.float64)
            p = CaseParams.from_theta(theta, n_comp, carrier=carrier)
            TF, NT = float(f[k["TF"]][r]), int(f[k["NT"]][r])
            y_out = _series(f, k["y_out"], r, n_comp)
            nt = y_out.shape[0]
            case = SolverCase(
                row=r,
                seed=int(f[k["seeds"]][r]) if k["seeds"] in f else r,
                params=p,
                theta=theta,
                TF=TF,
                t=(np.arange(nt) + 1) * TF / NT,
                y_out=y_out,
                T_out=np.asarray(f[k["T_out"]][r], dtype=np.float64),
            )
            if k["c_out"] in f:
                case.c_out = _series(f, k["c_out"], r, n_comp)
            else:  # ideal gas at the (uniform, constant) column pressure
                case.c_out = y_out * p.P / (R * case.T_out[:, None])
                case.derived = ("c_out",)
            if k["v_out"] in f:
                case.v_out = np.asarray(f[k["v_out"]][r], dtype=np.float64)
            if k["Tz"] in f and k["qz"] in f:
                Tz = np.asarray(f[k["Tz"]][r], dtype=np.float64)
                case.final = dict(z=_axial_grid(p.L, Tz.size, grid), T=Tz, q=np.asarray(f[k["qz"]][r], dtype=np.float64))
            if use_internal_profiles and k["profiles"] in f and str(r) in f[k["profiles"]]:
                g = f[k["profiles"]][str(r)]
                case.profiles = {name: np.asarray(g[name], dtype=np.float64) for name in ("t",) + PROFILE_FIELDS if name in g}
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
