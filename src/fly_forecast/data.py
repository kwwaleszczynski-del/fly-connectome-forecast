"""Hourly national wind and PV series, inputs, targets and the train / validation / test split.

Timeline (hours in UTC, end exclusive):
    washout     the first WASHOUT_HOURS of the series (reservoir transient), never used for fitting
    train       series start .. VAL_START          (model selection)
    validation  VAL_START .. TEST_START             (hyperparameters, ridge penalty)
    test        TEST_START .. TEST_END              (12 months, touched once with the selected settings)
The final models are refitted on train + validation. A training row is an issue time t whose target t + h lies in
the same window, so no target leaks across a boundary.

Usage:
    ds = Dataset.from_hourly(hourly_frame)
    rows = ds.train_rows(ds.start, TEST_START)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS = ["wind", "pv"]
HORIZONS = [1, 6, 24]
VAL_START = pd.Timestamp("2025-04-01", tz="UTC")
TEST_START = pd.Timestamp("2025-10-01", tz="UTC")
TEST_END = pd.Timestamp("2026-10-01", tz="UTC")
WASHOUT_HOURS = 168
INPUT_COLS = ["wind", "pv", "demand"]
MAX_GAP_HOURS = 6


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour of day and day of year as sine/cosine pairs (UTC, so free of daylight-saving jumps)."""
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    doy = index.day_of_year.to_numpy() - 1 + hour / 24
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                         "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25)},
                        index=index)


def column_names() -> list[str]:
    return [f"{s}_h{h}" for s in TARGETS for h in HORIZONS]


@dataclass
class Dataset:
    frame: pd.DataFrame      # hourly MW: wind, pv, demand (observed, NaN in gaps), wind_plan, pv_plan
    filled: pd.DataFrame     # inputs with gaps up to MAX_GAP_HOURS interpolated
    U: np.ndarray            # T x d model input: 1, z-scored wind / pv / demand, calendar
    Y: np.ndarray            # T x 6 targets y[t + h] in MW, columns as column_names()
    mean: pd.Series
    std: pd.Series

    @classmethod
    def from_hourly(cls, frame: pd.DataFrame, stats_end: pd.Timestamp = TEST_START) -> "Dataset":
        frame = frame.sort_index()
        full = pd.date_range(frame.index[0], frame.index[-1], freq="1h", tz="UTC")
        frame = frame.reindex(full)
        filled = frame[INPUT_COLS].interpolate(limit=MAX_GAP_HOURS, limit_area="inside").ffill().bfill()
        fit = filled[filled.index < stats_end]
        mean, std = fit.mean(), fit.std()
        Y = np.column_stack([frame[s].shift(-h).to_numpy() for s in TARGETS for h in HORIZONS])
        return cls(frame, filled, cls._inputs(filled, mean, std), Y, mean, std)

    @staticmethod
    def _inputs(filled: pd.DataFrame, mean: pd.Series, std: pd.Series) -> np.ndarray:
        z = (filled - mean) / std
        return np.hstack([np.ones((len(filled), 1)), z.to_numpy(), calendar(filled.index).to_numpy()])

    def with_noise(self, sigma: float, rng: np.random.Generator) -> "Dataset":
        """Copy whose observed inputs carry Gaussian noise of ``sigma`` standard deviations; targets stay clean."""
        noise = rng.normal(0, sigma, size=self.filled.shape) * self.std.to_numpy()
        filled = self.filled + noise
        return Dataset(self.frame, filled, self._inputs(filled, self.mean, self.std), self.Y, self.mean, self.std)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    @property
    def start(self) -> pd.Timestamp:
        return self.index[0] + pd.Timedelta(hours=WASHOUT_HOURS)

    def train_rows(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """Issue times in [start, end) whose targets (all horizons) fall before ``end`` and are observed."""
        start = max(start, self.start)
        t = self.index
        ok = (t >= start) & (t + pd.Timedelta(hours=max(HORIZONS)) < end) & np.isfinite(self.Y).all(axis=1)
        return np.flatnonzero(ok)

    def eval_rows(self, h: int, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """Issue times t whose target time t + h lies in [start, end)."""
        tau = self.index + pd.Timedelta(hours=h)
        return np.flatnonzero((tau >= start) & (tau < end) & (self.index >= self.start))
