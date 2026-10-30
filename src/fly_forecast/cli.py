"""Command line interface.

Usage:
    python -m fly_forecast fetch                     # PSE wind / PV / demand and PSE day-ahead plan -> data/pse
    python -m fly_forecast connectome                # FlyWire v783 table (852 MB, once) -> subgraphs in connectome/
    python -m fly_forecast run --jobs 4              # experiments -> results/ (about 1 h on 4 cores)
    python -m fly_forecast run --steps curves noise  # only some steps (tuning results are reused)
    python -m fly_forecast report                    # README figures (docs/img), summary.json, docs/index.html
    python -m fly_forecast all --jobs 4

The PSE cache is ./data/pse (override with --data-dir or FLYFC_DATA_DIR); the FlyWire table goes to
<data-dir>/flywire. The committed subgraph files in connectome/ are enough to rerun ``run`` and ``report`` without the
FlyWire download.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from fly_forecast import connectome as cn
from fly_forecast.data import TEST_END, Dataset
from fly_forecast.pse import HISTORY_START, hourly, load_series

log = logging.getLogger("fly_forecast")

SIZES = [500, 1000, 2000, 5000]
REGION_SIZE = 1000
SUBGRAPHS = [f"top{n}" for n in SIZES] + list(cn.REGIONS)
STEPS = ("tune", "main", "matched", "curves", "noise")


def load_hourly(data_dir: Path, today: date, offline: bool = False) -> pd.DataFrame:
    """Hourly MW frame (wind, pv, demand, wind_plan, pv_plan) up to two days after the end of the test period."""
    end = min(today - timedelta(days=1), (TEST_END + pd.Timedelta(days=2)).date())
    mix = load_series("mix", HISTORY_START, end, data_dir / "pse", offline=offline)
    plan = load_series("plan", HISTORY_START, end, data_dir / "pse", offline=offline)
    frame = hourly(mix).join(hourly(plan), how="left")
    frame = frame[frame.index < TEST_END + pd.Timedelta(days=2)]
    log.info("hourly frame %s .. %s, %d hours, missing wind %d, pv %d, plan %d", frame.index[0], frame.index[-1],
             len(frame), frame["wind"].isna().sum(), frame["pv"].isna().sum(), frame["wind_plan"].isna().sum())
    return frame


def load_matrices(folder: Path, names: list[str]):
    return {name: cn.Subgraph.load(folder, name).matrix() for name in names}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fly-forecast", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "connectome", "run", "report", "all"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("FLYFC_DATA_DIR", "data")))
        p.add_argument("--connectome-dir", type=Path, default=Path("connectome"),
                       help="folder of the reservoir subgraph files")
        p.add_argument("--results", type=Path, default=Path("results"))
        p.add_argument("--docs", type=Path, default=Path("docs"))
        p.add_argument("--offline", action="store_true", help="PSE cache only, no API calls")
        p.add_argument("--jobs", type=int, default=1, help="worker processes for the experiments")
        p.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
        p.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    today = date.today()

    if args.cmd in ("fetch", "all"):
        load_hourly(args.data_dir, today)
    if args.cmd in ("connectome", "all"):
        path = cn.download(args.data_dir / "flywire")
        cn.build_subgraphs(path, SIZES, REGION_SIZE, args.connectome_dir)
    if args.cmd in ("run", "all"):
        from fly_forecast.experiment import run_all

        ds = Dataset.from_hourly(load_hourly(args.data_dir, today, offline=args.offline))
        mats = load_matrices(args.connectome_dir, SUBGRAPHS)
        run_all(ds, mats, args.results, args.data_dir / "runs", jobs=args.jobs, steps=tuple(args.steps))
    if args.cmd in ("report", "all"):
        from fly_forecast.report import build_report

        ds = Dataset.from_hourly(load_hourly(args.data_dir, today, offline=args.offline))
        build_report(ds, args.connectome_dir, args.results, args.data_dir / "runs", args.docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
