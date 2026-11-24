"""Error metrics with the same definitions as the IC1 article (avaliacao/stratify_error.py),
and characteristic times of the breakthrough curves."""

import numpy as np


def regression_metrics(pred, ref, aard_min):
    """RMSE, MAE, MBE (bias), R2, Willmott d and AARD (%) of pred against ref.

    AARD is averaged only where |ref| > aard_min, as in IC1 (relative errors explode near 0).
    """
    p = np.asarray(pred, dtype=float).ravel()
    r = np.asarray(ref, dtype=float).ravel()
    ok = np.isfinite(p) & np.isfinite(r)
    p, r = p[ok], r[ok]
    if p.size == 0:
        return None
    res = p - r
    ss = float(np.sum(res**2))
    st = float(np.sum((r - r.mean()) ** 2))
    den = float(np.sum((np.abs(p - r.mean()) + np.abs(r - r.mean())) ** 2))
    big = np.abs(r) > aard_min
    return dict(
        n=int(p.size),
        rmse=float(np.sqrt(np.mean(res**2))),
        mae=float(np.mean(np.abs(res))),
        mbe=float(np.mean(res)),
        r2=1.0 - ss / st if st > 0 else float("nan"),
        d=1.0 - ss / den if den > 0 else float("nan"),
        aard=float(np.mean(np.abs(res[big] / r[big])) * 100) if big.any() else float("nan"),
    )


def level_time(t, signal, level, interpolate=True):
    """First time `signal` reaches `level` (NaN if it never does).

    interpolate=True: linear interpolation between the two grid points around the crossing
    (IC1 compute_level_times); False: first grid point at or above the level (IC1 recorder).
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    idx = np.flatnonzero(s >= level)
    if idx.size == 0:
        return float("nan")
    k = idx[0]
    if k == 0 or not interpolate:
        return float(t[k])
    frac = (level - s[k - 1]) / max(s[k] - s[k - 1], 1e-12)
    return float(t[k - 1] + frac * (t[k] - t[k - 1]))


def front_components(y_init, y_feed):
    """Components that form an advancing front: fed, but absent from the bed at t = 0."""
    return [i for i in range(len(y_feed)) if y_init[i] == 0 and y_feed[i] > 0]


def characteristic_times(t, c_out, y_out, c_feed, y_feed, components, times_cfg, interpolate):
    """{name: [time per component]} for each entry of eval.times (e.g. tb, t50, ts)."""
    out = {}
    for name, spec in times_cfg.items():
        vals = []
        for i in components:
            if spec["signal"] == "c_ratio":
                sig = c_out[:, i] / c_feed[i]
            elif spec["signal"] == "y_ratio":
                sig = y_out[:, i] / y_feed[i]
            else:
                raise ValueError("unknown signal %r" % spec["signal"])
            vals.append(level_time(t, sig, float(spec["level"]), interpolate))
        out[name] = vals
    return out
