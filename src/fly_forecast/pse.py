"""PSE series of the analysis, read with the pse-opendata client (https://pypi.org/project/pse-opendata/).

Series (15-minute intervals, business dates in Europe/Warsaw, available from 14 June 2024):
    mix   his-wlk-cal   realised national totals, MW: demand (KSE demand), pv (all photovoltaic sources),
                        wind (all wind sources)
    plan  pdgopkd       PSE day-ahead coordination plan (PKD), MW: wind_plan (gen_wi), pv_plan (gen_fv);
                        published on D-1, mostly between 15:30 and 20:30 local time

pse-opendata downloads the public PSE reports API (no key) and caches whole months as Parquet files in ``cache_dir``
(``api/<report>/<YYYY-MM>.parquet``); a month is downloaded again until the revision window of its report has
passed (60 days for his-wlk-cal, 3 days for pdgopkd). Frames returned here are indexed by interval START (UTC).

Usage:
    from fly_forecast.pse import load_series
    df = load_series("mix", date(2024, 6, 14), date(2026, 10, 25), cache_dir=Path("data/pse"))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pse_opendata as pse

HISTORY_START = date(2024, 6, 14)   # first business date of his-wlk-cal and pdgopkd in this API
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Series:
    endpoint: str
    fields: dict[str, str]   # API field -> column name


SERIES = {
    "mix": Series("his-wlk-cal", {"demand": "demand", "pv": "pv", "wi": "wind"}),
    "plan": Series("pdgopkd", {"gen_wi": "wind_plan", "gen_fv": "pv_plan"}),
}


def empty_frame(series: str) -> pd.DataFrame:
    cols = list(SERIES[series].fields.values())
    idx = pd.DatetimeIndex([], tz="UTC", name="ts_utc")
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in cols}, index=idx)
    out["business_date"] = pd.Series(dtype=object)
    return out


def to_frame(raw: pd.DataFrame, series: str) -> pd.DataFrame:
    """pse-opendata frame -> frame indexed by interval START (UTC) with the column names of ``SERIES``."""
    spec = SERIES[series]
    if raw.empty:
        return empty_frame(series)
    out = raw.set_index("start_utc")[list(spec.fields)].rename(columns=spec.fields)
    out.index.name = "ts_utc"
    out["business_date"] = raw["business_date"].to_numpy()
    return out


def load_series(series: str, start: date, end: date, cache_dir: Path, offline: bool = False,
                client: pse.Client | None = None) -> pd.DataFrame:
    """Load business dates start..end month by month.

    Whole months are requested, so that pse-opendata caches them and an offline run rebuilds the same result. When a
    download fails, the cached copy of the month is used; a month that cannot be loaded is skipped. Rows without any
    value (future intervals) are dropped. ``offline`` uses the cache only.
    """
    spec = SERIES[series]
    client = client or pse.Client(cache_dir=cache_dir, offline=offline, allow_stale=True)
    frames = []
    for month_start in pd.date_range(date(start.year, start.month, 1), end, freq="MS"):
        label = f"{month_start:%Y-%m}"
        m_end = (month_start + pd.offsets.MonthEnd(0)).date()
        try:
            raw = client.get(spec.endpoint, month_start.date(), m_end, fields=list(spec.fields), dropna=True)
        except pse.PseError as e:
            log.error("%s %s: not loaded (%s)", series, label, e)
            continue
        log.info("%s %s: %d intervals", series, label, len(raw))
        frames.append(to_frame(raw, series))
    frames = [f for f in frames if not f.empty]
    out = pd.concat(frames).sort_index() if frames else empty_frame(series)
    out = out[~out.index.duplicated(keep="last")]
    return out[(out["business_date"] >= start) & (out["business_date"] <= end)]


def hourly(df: pd.DataFrame) -> pd.DataFrame:
    """15-minute MW -> hourly mean MW on a complete UTC hourly index; hours with fewer than 4 intervals are NaN."""
    num = df.drop(columns="business_date")
    grouped = num.resample("1h")
    out = grouped.mean()
    out[grouped.count().min(axis=1) < 4] = float("nan")
    return out
