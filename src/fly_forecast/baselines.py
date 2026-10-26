"""Reference forecasts: persistence, daily persistence, ridge and gradient boosting on lags, PSE day-ahead plan.

Lag features at issue time t: wind and PV at t, t-1, ..., t-23, demand at t, and the calendar terms of data.calendar
(for the tree model the plain hour of day instead). All use the gap-filled inputs, i.e. only information available
at t.

Usage:
    L = lag_matrix(ds)
    pred = persistence(ds, "wind", 6)            # MW, aligned with issue times
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fly_forecast.data import Dataset, calendar

LAGS = 24


def lag_matrix(ds: Dataset, tree: bool = False) -> np.ndarray:
    f = ds.filled
    cols = {f"{s}_lag{k}": f[s].shift(k) for s in ("wind", "pv") for k in range(LAGS)}
    cols["demand_lag0"] = f["demand"]
    out = pd.DataFrame(cols, index=ds.index)
    if tree:
        # hour only: with about one year of training data a day-of-year split memorises that year's weather
        out["hour"] = ds.index.hour
    else:
        out = out.join(calendar(ds.index))
    return out.bfill().to_numpy(dtype=np.float64)


def persistence(ds: Dataset, series: str, h: int) -> np.ndarray:
    """y_hat[t + h] = y[t]."""
    return ds.filled[series].to_numpy()


def daily_persistence(ds: Dataset, series: str, h: int) -> np.ndarray:
    """y_hat[t + h] = y[t + h - 24] (same hour of the previous day)."""
    return ds.filled[series].shift(24 - h).to_numpy()


def pse_plan(ds: Dataset, series: str, h: int) -> np.ndarray:
    """PSE day-ahead plan for the target hour t + h (published on D-1, lead about 4 to 32 h)."""
    return ds.frame[f"{series}_plan"].shift(-h).to_numpy()


def gbm_fit_predict(X: np.ndarray, Y: np.ndarray, train: np.ndarray, seed: int = 0) -> np.ndarray:
    """One HistGradientBoosting model per target column, fitted on rows ``train``; predictions for all rows."""
    out = np.empty((len(X), Y.shape[1]))
    for k in range(Y.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=len(train) > 2000,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
        m.fit(X[train], Y[train, k])
        out[:, k] = m.predict(X)
    return out
