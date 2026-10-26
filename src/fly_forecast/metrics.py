"""Error metrics, skill scores and a daily block bootstrap for paired model comparisons.

Skill score against a reference forecast (persistence in this project):
    SS = 1 - RMSE_model / RMSE_reference

Usage:
    table = score(ds, pred, TEST_START, TEST_END)                  # one row per target and horizon
    lo, hi = bootstrap_rmse_diff(err_a, err_b, days, rng)          # 95 % interval of RMSE_a - RMSE_b
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fly_forecast.data import HORIZONS, TARGETS, Dataset


def score(ds: Dataset, pred: np.ndarray, start: pd.Timestamp, end: pd.Timestamp,
          allow_missing: bool = False) -> pd.DataFrame:
    """MAE and RMSE (MW) per target and horizon over target times in [start, end); predictions clipped at 0.

    A non-finite prediction (a diverged model) counts as an infinite error unless ``allow_missing`` (forecasts that
    may simply be unavailable for some hours, such as the PSE plan), where those hours are left out.
    """
    out = []
    k = 0
    for s in TARGETS:
        for h in HORIZONS:
            rows = ds.eval_rows(h, start, end)
            y, p = ds.Y[rows, k], np.clip(pred[rows, k], 0, None)
            if not allow_missing:
                p = np.where(np.isfinite(p), p, np.inf)
            ok = np.isfinite(y) & ~np.isnan(p)
            e = p[ok] - y[ok]
            out.append({"target": s, "h": h, "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                        "n": int(ok.sum())})
            k += 1
    return pd.DataFrame(out)


def stack(columns: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    """{(target, h): vector over issue times} -> T x 6 matrix in the column order of data.column_names()."""
    return np.column_stack([columns[(s, h)] for s in TARGETS for h in HORIZONS])


def skill(rmse: pd.Series | np.ndarray | float, rmse_ref: pd.Series | np.ndarray | float):
    return 1 - np.asarray(rmse) / np.asarray(rmse_ref)


def bootstrap_rmse_diff(err_a: np.ndarray, err_b: np.ndarray, day: np.ndarray, rng: np.random.Generator,
                        reps: int = 2000) -> tuple[float, float]:
    """95 % interval of RMSE(a) - RMSE(b) resampling whole days (paired errors on the same target times)."""
    days, inv = np.unique(day, return_inverse=True)
    sa = np.bincount(inv, weights=err_a ** 2, minlength=len(days))
    sb = np.bincount(inv, weights=err_b ** 2, minlength=len(days))
    cnt = np.bincount(inv, minlength=len(days)).astype(float)
    pick = rng.integers(0, len(days), size=(reps, len(days)))
    diff = np.sqrt(sa[pick].sum(1) / cnt[pick].sum(1)) - np.sqrt(sb[pick].sum(1) / cnt[pick].sum(1))
    return float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))
