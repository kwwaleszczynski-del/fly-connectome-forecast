"""Client for the public PSE reports API (https://api.raporty.pse.pl) with a monthly Parquet cache.

Series (15-minute intervals, business dates in Europe/Warsaw, available from 14 June 2024):
    mix   his-wlk-cal   realised national totals, MW: demand (KSE demand), pv (all photovoltaic sources),
                        wind (all wind sources)
    plan  pdgopkd       PSE day-ahead coordination plan (PKD), MW: wind_plan (gen_wi), pv_plan (gen_fv);
                        published on D-1, mostly between 15:30 and 20:30 local time

The API needs no key. ``dtime_utc`` is the END of each interval; frames returned here are indexed by interval START
(UTC). Pagination: ``$filter`` on business_date, ``$first=20000`` and ``nextLink``.

Usage:
    from fly_forecast.pse import load_series
    df = load_series("mix", date(2024, 6, 14), date(2026, 10, 25), cache_dir=Path("data/pse"), today=date.today())
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.raporty.pse.pl/api"
INTERVAL = pd.Timedelta("15min")
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


def parse_records(records: list[dict], series: str) -> pd.DataFrame:
    """API records -> frame indexed by interval START (UTC). Rows without any value (future intervals) are dropped."""
    if not records:
        return empty_frame(series)
    spec = SERIES[series]
    df = pd.DataFrame.from_records(records)
    out = pd.DataFrame({col: pd.to_numeric(df[api], errors="coerce") if api in df else float("nan")
                        for api, col in spec.fields.items()})
    end = pd.to_datetime(df["dtime_utc"], format="ISO8601")
    out.index = pd.DatetimeIndex(end, name="ts_utc").tz_localize("UTC") - INTERVAL
    out["business_date"] = pd.to_datetime(df["business_date"]).dt.date.to_numpy()
    out = out.dropna(subset=list(spec.fields.values()), how="all")
    return out[~out.index.duplicated(keep="last")].sort_index()


def _get(session: requests.Session, url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            log.warning("PSE API request failed (%s), retrying", e)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_period(series: str, start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch one series for business dates start..end (inclusive), following pagination links."""
    spec = SERIES[series]
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "fly-connectome-forecast (open-source analysis)")
    params = {
        "$filter": f"business_date ge '{start:%Y-%m-%d}' and business_date le '{end:%Y-%m-%d}'",
        "$select": ",".join(["dtime_utc", "business_date", *spec.fields]),
        "$first": 20000,
    }
    payload = _get(session, f"{BASE_URL}/{spec.endpoint}", params)
    records = list(payload.get("value", []))
    while payload.get("nextLink"):
        payload = _get(session, payload["nextLink"])
        records += payload.get("value", [])
    return parse_records(records, series)


def load_series(series: str, start: date, end: date, cache_dir: Path, today: date, refresh_days: int = 10,
                offline: bool = False) -> pd.DataFrame:
    """Load business dates start..end month by month.

    Months that ended more than ``refresh_days`` before ``today`` are read from the cache once the cached copy reaches
    the last day of the month; other months are downloaded again. ``offline`` uses the cache only.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    frames = []
    for month_start in pd.date_range(date(start.year, start.month, 1), end, freq="MS"):
        label = f"{month_start:%Y-%m}"
        m_end = (month_start + pd.offsets.MonthEnd(0)).date()
        cache = cache_dir / f"{series}_{label}.parquet"
        cached = pd.read_parquet(cache) if cache.exists() else None
        final = m_end < today - timedelta(days=refresh_days)
        complete = cached is not None and len(cached) > 0 and cached["business_date"].max() >= m_end
        if (offline and cached is not None) or (final and complete):
            frames.append(cached)
            continue
        if offline:
            log.warning("%s %s: not in cache", series, label)
            continue
        try:
            df = fetch_period(series, month_start.date(), min(m_end, end), session)
        except (requests.RequestException, ValueError, KeyError) as e:
            log.error("%s %s: download failed (%s)%s", series, label, e, ", using cache" if cached is not None else "")
            if cached is not None:
                frames.append(cached)
            continue
        if not df.empty:
            df.to_parquet(cache)
        frames.append(df)
        log.info("%s %s: %d intervals", series, label, len(df))
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
