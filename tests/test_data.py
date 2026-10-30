from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pse_opendata
import pytest

from fly_forecast import baselines as bl
from fly_forecast import pse
from fly_forecast.data import HORIZONS, TARGETS, Dataset, column_names
from fly_forecast.metrics import bootstrap_rmse_diff, score, stack


@pytest.fixture
def ds():
    idx = pd.date_range("2025-01-01", periods=24 * 40, freq="1h", tz="UTC")
    k = np.arange(len(idx), dtype=float)
    frame = pd.DataFrame({"wind": 1000 + k, "pv": 10 * k, "demand": 15000 + np.sin(k),
                          "wind_plan": 2000 + k, "pv_plan": 5 * k}, index=idx)
    frame.iloc[100:103, frame.columns.get_loc("wind")] = np.nan
    return Dataset.from_hourly(frame, stats_end=idx[24 * 30])


def test_targets_are_shifted_by_horizon(ds):
    cols = column_names()
    t = 200
    for h in HORIZONS:
        assert ds.Y[t, cols.index(f"wind_h{h}")] == 1000 + t + h
        assert ds.Y[t, cols.index(f"pv_h{h}")] == 10 * (t + h)


def test_gaps_are_interpolated_for_inputs_only(ds):
    assert ds.filled["wind"].iloc[101] == pytest.approx(1101)
    assert np.isnan(ds.frame["wind"].iloc[101])
    assert np.isnan(ds.Y[101 - 1, column_names().index("wind_h1")])


def test_train_rows_do_not_reach_past_window_end(ds):
    start, end = ds.index[24 * 10], ds.index[24 * 20]
    rows = ds.train_rows(start, end)
    assert ds.index[rows].min() >= start
    assert (ds.index[rows] + pd.Timedelta(hours=max(HORIZONS))).max() < end
    assert np.isfinite(ds.Y[rows]).all()


def test_eval_rows_select_target_times(ds):
    start, end = ds.index[24 * 30], ds.index[24 * 35]
    for h in HORIZONS:
        rows = ds.eval_rows(h, start, end)
        tau = ds.index[rows] + pd.Timedelta(hours=h)
        assert tau.min() == start and tau.max() == end - pd.Timedelta(hours=1)


def test_inputs_are_standardised_on_the_fit_period(ds):
    z = ds.U[: 24 * 30, 1:4]
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
    assert ds.U[:, 0].min() == 1


def test_noise_changes_inputs_not_targets(ds):
    dn = ds.with_noise(0.5, np.random.default_rng(0))
    assert not np.allclose(dn.U[:, 1:4], ds.U[:, 1:4])
    assert np.array_equal(dn.Y, ds.Y, equal_nan=True)
    assert np.array_equal(dn.U[:, 4:], ds.U[:, 4:])


def test_baselines_alignment_and_perfect_score(ds):
    t, cols = 300, column_names()
    assert bl.persistence(ds, "wind", 6)[t] == 1000 + t
    assert bl.daily_persistence(ds, "wind", 6)[t] == 1000 + t + 6 - 24
    assert bl.pse_plan(ds, "pv", 24)[t] == 5 * (t + 24)
    perfect = stack({(s, h): ds.Y[:, cols.index(f"{s}_h{h}")] for s in TARGETS for h in HORIZONS})
    table = score(ds, perfect, ds.index[24 * 30], ds.index[24 * 35])
    assert (table["rmse"] == 0).all() and (table["n"] == 24 * 5).all()
    # persistence on a linear ramp misses by exactly h steps
    pers = stack({(s, h): bl.persistence(ds, s, h) for s in TARGETS for h in HORIZONS})
    t2 = score(ds, pers, ds.index[24 * 30], ds.index[24 * 35]).set_index(["target", "h"])
    assert t2.loc[("wind", 6), "mae"] == pytest.approx(6)
    assert t2.loc[("pv", 24), "rmse"] == pytest.approx(240)


def test_bootstrap_interval():
    rng = np.random.default_rng(0)
    e = rng.normal(size=24 * 50)
    day = np.repeat(np.arange(50), 24)
    assert bootstrap_rmse_diff(e, e, day, rng) == (0.0, 0.0)
    lo, hi = bootstrap_rmse_diff(2 * e, e, day, rng)
    assert 0 < lo < 1 < hi < 1.5


class FakeResponse:
    def __init__(self, payload):
        self.status_code, self.headers, self.reason, self.payload = 200, {}, "OK", payload

    def json(self):
        return self.payload


class FakeSession:
    """Answers every pse-opendata request with the same records."""

    def __init__(self, records: list[dict]):
        self.records = records

    def get(self, url, params=None, timeout=None, headers=None):
        return FakeResponse({"value": self.records})


def test_pse_records_are_indexed_by_interval_start(tmp_path):
    recs = [{"dtime_utc": f"2025-06-14 22:{m:02d}:00", "business_date": "2025-06-15", "demand": 1.0, "pv": 2.0,
             "wi": float(m)} for m in (15, 30, 45)]
    recs.append({"dtime_utc": "2025-06-14 23:00:00", "business_date": "2025-06-15", "demand": 1.0, "pv": 2.0,
                 "wi": 60.0})
    client = pse_opendata.Client(session=FakeSession(recs), cache_dir=tmp_path, sleep=lambda s: None, min_interval=0,
                                 retries=1, now=lambda: datetime(2025, 7, 1, tzinfo=UTC))
    df = pse.load_series("mix", date(2025, 6, 15), date(2025, 6, 15), tmp_path, client=client)
    assert df.index[0] == pd.Timestamp("2025-06-14 22:00", tz="UTC")
    h = pse.hourly(df)
    assert len(h) == 1 and h["wind"].iloc[0] == pytest.approx(37.5)
    assert pse.hourly(df.iloc[:3])["wind"].isna().all()


def test_diverged_forecast_scores_infinite_unless_missing_allowed(ds):
    cols = column_names()
    pred = stack({(s, h): ds.Y[:, cols.index(f"{s}_h{h}")] for s in TARGETS for h in HORIZONS})
    pred[24 * 31, 0] = np.nan
    start, end = ds.index[24 * 30], ds.index[24 * 35]
    assert np.isinf(score(ds, pred, start, end)["rmse"].iloc[0])
    allowed = score(ds, pred, start, end, allow_missing=True)
    assert allowed["rmse"].iloc[0] == 0 and allowed["n"].iloc[0] == 24 * 5 - 1
