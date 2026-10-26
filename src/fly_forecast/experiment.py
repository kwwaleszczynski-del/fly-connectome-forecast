"""Experiments: hyperparameter selection, test scores by reservoir size and variant, learning curves, input noise.

Protocol (dates in data.py):
    1. tune      for every variant at the main size (seed 0): grid over spectral radius, leak rate and input
                 scaling, fitted on train, scored on validation; ridge penalty per target column from ALPHAS.
                 Score = mean over the 6 target columns of RMSE / RMSE(persistence).
    2. main      every size x variant x seed with its selected settings, refitted on train + validation, scored
                 on the 12-month test period. The input weights depend only on the seed, so variants with the same
                 seed are paired.
    3. matched   main size: every variant with recurrence at the setting selected for the connectome and at the
                 setting selected for the random ESN (as in Costi et al., only the matrix differs), each with its
                 own ridge penalties at that setting from the tuning table.
    4. curves    matched settings: training windows of the last 7 ... all days before the test period; ridge penalty
                 fixed (every alpha in ALPHAS) or chosen inside the window ("inner": fitted on its first 80 %,
                 scored on the last 20 %, then refitted); train and test error as the training set shrinks.
    5. noise     main size, each variant at its own setting: Gaussian noise of 0 ... 0.5 standard deviations on the observed inputs, clean targets.
The baselines (persistence, daily persistence, ridge and gradient boosting on lags, PSE day-ahead plan) go through
the same windows.

Usage:
    run_all(ds, subgraphs, out_dir=Path("results"), jobs=4)
"""
from __future__ import annotations

import itertools
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fly_forecast import baselines as bl
from fly_forecast.data import HORIZONS, TARGETS, TEST_END, TEST_START, VAL_START, Dataset, column_names
from fly_forecast.esn import RidgePath, features, input_weights, run_reservoir
from fly_forecast.metrics import score, stack
from fly_forecast.reservoir import VARIANTS, make_variant, scale_to_radius

log = logging.getLogger(__name__)

ALPHAS = np.logspace(-10, 2, 13)
GRID = {"rho": [0.5, 0.9, 0.99, 1.1], "leak": [0.1, 0.3, 0.6, 1.0], "input_scale": [0.02, 0.05, 0.1, 0.5, 1.0]}
INNER_VAL = 0.2           # learning curves: share of each training window used to choose the ridge penalty
MAIN_SIZE = "top1000"
SEEDS = 5
SEEDS_LARGE = 3            # for reservoirs above LARGE_N neurons
LARGE_N = 3000
TRAIN_DAYS = [7, 14, 30, 60, 120, 240, 0]   # 0 = everything from the end of the washout
NOISE = [0.0, 0.1, 0.25, 0.5]
MATCHED = ["connectome", "random"]   # variants whose selected settings are imposed on all others
COLS = column_names()


@dataclass(frozen=True)
class Params:
    rho: float
    leak: float
    input_scale: float


# ---------------------------------------------------------------- worker context (one copy per process)
_CTX: dict = {}


def _init(ds: Dataset, mats: dict[str, sp.csr_matrix], threads: int | None) -> None:
    if threads:
        from threadpoolctl import threadpool_limits
        threadpool_limits(threads)
    _CTX.clear()
    _CTX.update(ds=ds, mats=mats, unit={})


def _map(fn, jobs: list, ds: Dataset, mats: dict, n_jobs: int) -> list:
    if n_jobs <= 1 or len(jobs) <= 1:
        _init(ds, mats, None)
        return [fn(j) for j in jobs]
    threads = max(1, (os.cpu_count() or 2) // n_jobs)
    with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(ds, mats, threads)) as ex:
        return list(ex.map(fn, jobs))


def unit_variant(size: str, variant: str, seed: int) -> sp.csr_matrix:
    """Variant of the connectome matrix ``size`` scaled to spectral radius 1 (cached per process)."""
    key = (size, variant, seed)
    if key not in _CTX["unit"]:
        rng = np.random.default_rng([seed, VARIANTS.index(variant)])
        _CTX["unit"][key] = scale_to_radius(make_variant(variant, _CTX["mats"][size], rng), 1.0)
    return _CTX["unit"][key]


def esn_features(size: str, variant: str, seed: int, p: Params, U: np.ndarray) -> np.ndarray:
    W = unit_variant(size, variant, seed) * p.rho
    Win = input_weights(W.shape[0], U.shape[1], p.input_scale, np.random.default_rng([seed, 999]))
    return features(run_reservoir(W, Win, U, p.leak), U)


def _reference(ds: Dataset) -> np.ndarray:
    return stack({(s, h): bl.persistence(ds, s, h) for s in TARGETS for h in HORIZONS})


def _rmse_by_alpha(ds: Dataset, path: RidgePath, F: np.ndarray, start, end) -> np.ndarray:
    """len(ALPHAS) x 6 RMSE matrix over target times in [start, end)."""
    return np.array([score(ds, path.predict(F, a), start, end)["rmse"].to_numpy() for a in ALPHAS])


def _fit_rmse(ds: Dataset, path: RidgePath, F: np.ndarray, rows: np.ndarray, alpha) -> np.ndarray:
    """RMSE on rows with observed targets (training rows); a non-finite prediction counts as an infinite error."""
    with np.errstate(all="ignore"):
        pred = np.clip(path.predict(F[rows], alpha), 0, None)
    pred = np.where(np.isfinite(pred), pred, np.inf)
    return np.sqrt(np.mean((pred - ds.Y[rows]) ** 2, axis=0))


def participation_ratio(X: np.ndarray) -> float:
    """Effective number of dimensions of the states X (rows = time): (tr C)^2 / tr(C^2) of their covariance C."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    C = Xc.T @ Xc
    return float(np.trace(C) ** 2 / (C ** 2).sum()) if C.any() else 0.0


def inner_alpha(ds: Dataset, F: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Penalty per target chosen inside a training window: fit on its first part, score on the last INNER_VAL."""
    val = fit[int((1 - INNER_VAL) * len(fit)):]
    first = ds.index[val[0]]
    inner = fit[ds.index[fit] + pd.Timedelta(hours=max(HORIZONS)) < first]
    path = RidgePath.fit(F[inner], ds.Y[inner])
    rm = np.array([_fit_rmse(ds, path, F, val, a) for a in ALPHAS])
    return ALPHAS[rm.argmin(axis=0)]


# ---------------------------------------------------------------- 1. tuning
def _tune_job(job: tuple) -> dict:
    size, variant, p = job
    ds = _CTX["ds"]
    F = esn_features(size, variant, 0, p, ds.U)
    path = RidgePath.fit(F[ds.train_rows(ds.start, VAL_START)], ds.Y[ds.train_rows(ds.start, VAL_START)])
    rm = _rmse_by_alpha(ds, path, F, VAL_START, TEST_START)
    ref = score(ds, _reference(ds), VAL_START, TEST_START)["rmse"].to_numpy()
    best = rm.argmin(axis=0)
    ratio = rm[best, range(len(COLS))] / ref
    return {"variant": variant, **asdict(p), "score": float(ratio.mean()),
            **{f"ratio_{c}": float(r) for c, r in zip(COLS, ratio)},
            **{f"alpha_{c}": float(ALPHAS[b]) for c, b in zip(COLS, best)}}


def tune(ds: Dataset, mats: dict, jobs: int, variants=VARIANTS) -> pd.DataFrame:
    grid = []
    for v in variants:
        rhos = [GRID["rho"][-1]] if v == "no_recurrence" else GRID["rho"]
        for rho, leak, isc in itertools.product(rhos, GRID["leak"], GRID["input_scale"]):
            grid.append((MAIN_SIZE, v, Params(rho, leak, isc)))
    log.info("tuning: %d settings", len(grid))
    return pd.DataFrame(_map(_tune_job, grid, ds, mats, jobs))


def selected(tuning: pd.DataFrame) -> dict[str, dict]:
    """Best setting per variant: Params fields plus one alpha per target column."""
    out = {}
    for v, g in tuning.groupby("variant", sort=False):
        r = g.loc[g["score"].idxmin()]
        out[v] = {"params": {k: float(r[k]) for k in ("rho", "leak", "input_scale")},
                  "alpha": [float(r[f"alpha_{c}"]) for c in COLS], "val_score": float(r["score"])}
    return out


# ---------------------------------------------------------------- 2. main comparison
def _main_job(job: tuple) -> tuple[pd.DataFrame, np.ndarray | None]:
    size, variant, seed, sel, keep = job
    ds = _CTX["ds"]
    p, alpha = Params(**sel["params"]), np.array(sel["alpha"])
    F = esn_features(size, variant, seed, p, ds.U)
    fit = ds.train_rows(ds.start, TEST_START)
    path = RidgePath.fit(F[fit], ds.Y[fit])
    pred = path.predict(F, alpha)
    out = score(ds, pred, TEST_START, TEST_END)
    out["train_rmse"] = _fit_rmse(ds, path, F, fit, alpha)
    out["state_pr"] = participation_ratio(F[fit, :-ds.U.shape[1]])
    out.insert(0, "seed", seed)
    out.insert(0, "variant", variant)
    out.insert(0, "size", size)
    return out, (pred if keep else None)


def main_runs(ds: Dataset, mats: dict, sel: dict, jobs: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    todo = []
    for size, W in mats.items():
        seeds = SEEDS_LARGE if W.shape[0] > LARGE_N else SEEDS
        for v in sel:
            for seed in range(seeds):
                todo.append((size, v, seed, sel[v], size == MAIN_SIZE and seed == 0))
    # largest reservoirs first so that the pool is not left waiting for them at the end
    todo.sort(key=lambda j: -mats[j[0]].shape[0])
    log.info("main runs: %d", len(todo))
    res = _map(_main_job, todo, ds, mats, jobs)
    preds = {f"esn_{j[1]}": r[1] for j, r in zip(todo, res) if r[1] is not None}
    return pd.concat([r[0] for r in res], ignore_index=True), preds


def matched_settings(tuning: pd.DataFrame, sel: dict) -> dict[str, dict[str, dict]]:
    """{setting: {variant: {"params", "alpha"}}}: variants with recurrence at the hyperparameters selected for
    ``setting``, each with its own best ridge penalties at that point of the grid."""
    out = {}
    for setting in MATCHED:
        p = sel[setting]["params"]
        at = tuning[np.isclose(tuning["rho"], p["rho"]) & np.isclose(tuning["leak"], p["leak"])
                    & np.isclose(tuning["input_scale"], p["input_scale"])]
        out[setting] = {r["variant"]: {"params": p, "alpha": [float(r[f"alpha_{c}"]) for c in COLS]}
                        for r in at.to_dict("records") if r["variant"] != "no_recurrence"}
    return out


def matched_runs(ds: Dataset, mats: dict, matched: dict, jobs: int) -> pd.DataFrame:
    todo, labels = [], []
    for setting, by_variant in matched.items():
        for v, s in by_variant.items():
            for seed in range(SEEDS):
                todo.append((MAIN_SIZE, v, seed, s, False))
                labels.append(setting)
    log.info("matched runs: %d", len(todo))
    res = _map(_main_job, todo, ds, mats, jobs)
    return pd.concat([r[0].assign(setting=lab) for r, lab in zip(res, labels)], ignore_index=True)


def baseline_runs(ds: Dataset, alpha_ridge: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    fit = ds.train_rows(ds.start, TEST_START)
    L = bl.lag_matrix(ds)
    preds = {
        "persistence": _reference(ds),
        "daily_persistence": stack({(s, h): bl.daily_persistence(ds, s, h) for s in TARGETS for h in HORIZONS}),
        "ridge_lags": RidgePath.fit(L[fit], ds.Y[fit]).predict(L, alpha_ridge),
        "gbm_lags": bl.gbm_fit_predict(bl.lag_matrix(ds, tree=True), ds.Y, fit),
        "pse_plan": stack({(s, h): bl.pse_plan(ds, s, h) for s in TARGETS for h in HORIZONS}),
    }
    rows = []
    for name, pred in preds.items():
        t = score(ds, pred, TEST_START, TEST_END, allow_missing=name == "pse_plan")
        t.insert(0, "model", name)
        rows.append(t)
    return pd.concat(rows, ignore_index=True), preds


def tune_ridge_lags(ds: Dataset) -> np.ndarray:
    L = bl.lag_matrix(ds)
    fit = ds.train_rows(ds.start, VAL_START)
    rm = _rmse_by_alpha(ds, RidgePath.fit(L[fit], ds.Y[fit]), L, VAL_START, TEST_START)
    return ALPHAS[rm.argmin(axis=0)]


# ---------------------------------------------------------------- 3. learning curves
def _window(ds: Dataset, days: int) -> np.ndarray:
    start = ds.start if days == 0 else TEST_START - pd.Timedelta(days=days)
    return ds.train_rows(start, TEST_START)


def _curve_rows(ds: Dataset, F: np.ndarray, label: dict, alpha_sel: np.ndarray) -> list[dict]:
    out = []
    for days in TRAIN_DAYS:
        fit = _window(ds, days)
        path = RidgePath.fit(F[fit], ds.Y[fit])
        a_inner = inner_alpha(ds, F, fit)
        for a in [*ALPHAS, "selected", "inner"]:
            alpha = alpha_sel if a == "selected" else a_inner if a == "inner" else a
            test = score(ds, path.predict(F, alpha), TEST_START, TEST_END)["rmse"].to_numpy()
            train = _fit_rmse(ds, path, F, fit, alpha)
            for c, te, tr in zip(COLS, test, train):
                out.append({**label, "days": days, "rows": len(fit), "alpha": str(a), "column": c,
                            "test_rmse": float(te), "train_rmse": float(tr)})
    return out


def _curve_job(job: tuple) -> list[dict]:
    variant, seed, sel, setting = job
    ds = _CTX["ds"]
    F = esn_features(MAIN_SIZE, variant, seed, Params(**sel["params"]), ds.U)
    return _curve_rows(ds, F, {"model": f"esn_{variant}", "setting": setting, "seed": seed}, np.array(sel["alpha"]))


def learning_curves(ds: Dataset, mats: dict, matched: dict, alpha_ridge: np.ndarray, jobs: int) -> pd.DataFrame:
    todo = [(v, seed, s, setting) for setting, d in matched.items() for v, s in d.items() for seed in range(SEEDS)]
    log.info("learning curves: %d reservoirs", len(todo))
    rows = list(itertools.chain.from_iterable(_map(_curve_job, todo, ds, mats, jobs)))
    rows += _curve_rows(ds, bl.lag_matrix(ds), {"model": "ridge_lags", "setting": "", "seed": 0}, alpha_ridge)
    T = bl.lag_matrix(ds, tree=True)
    for days in TRAIN_DAYS:
        fit = _window(ds, days)
        pred = bl.gbm_fit_predict(T, ds.Y, fit)
        test = score(ds, pred, TEST_START, TEST_END)["rmse"].to_numpy()
        train = np.sqrt(np.nanmean((np.clip(pred[fit], 0, None) - ds.Y[fit]) ** 2, axis=0))
        for c, te, tr in zip(COLS, test, train):
            rows.append({"model": "gbm_lags", "setting": "", "seed": 0, "days": days, "rows": len(fit), "alpha": "inner",
                         "column": c, "test_rmse": float(te), "train_rmse": float(tr)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. input noise
def _noise_job(job: tuple) -> pd.DataFrame:
    variant, seed, sigma, sel = job
    ds = _CTX["ds"].with_noise(sigma, np.random.default_rng([seed, 4242, int(sigma * 1000)]))
    F = esn_features(MAIN_SIZE, variant, seed, Params(**sel["params"]), ds.U)
    fit = ds.train_rows(ds.start, TEST_START)
    out = score(ds, RidgePath.fit(F[fit], ds.Y[fit]).predict(F, np.array(sel["alpha"])), TEST_START, TEST_END)
    out.insert(0, "sigma", sigma)
    out.insert(0, "seed", seed)
    out.insert(0, "model", f"esn_{variant}")
    return out


def noise_runs(ds: Dataset, mats: dict, sel: dict, alpha_ridge: np.ndarray, jobs: int) -> pd.DataFrame:
    todo = [(v, seed, s, sel[v]) for v in sel for seed in range(SEEDS) for s in NOISE]
    rows = _map(_noise_job, todo, ds, mats, jobs)
    for seed in range(SEEDS):
        for s in NOISE:
            dn = ds.with_noise(s, np.random.default_rng([seed, 4242, int(s * 1000)]))
            fit = dn.train_rows(dn.start, TEST_START)
            L, T = bl.lag_matrix(dn), bl.lag_matrix(dn, tree=True)
            for name, pred in (("ridge_lags", RidgePath.fit(L[fit], dn.Y[fit]).predict(L, alpha_ridge)),
                               ("gbm_lags", bl.gbm_fit_predict(T, dn.Y, fit, seed=seed))):
                t = score(dn, pred, TEST_START, TEST_END)
                t.insert(0, "sigma", s)
                t.insert(0, "seed", seed)
                t.insert(0, "model", name)
                rows.append(t)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- everything
def run_all(ds: Dataset, mats: dict[str, sp.csr_matrix], out_dir: Path, run_dir: Path, jobs: int,
            steps: tuple[str, ...] = ("tune", "main", "matched", "curves", "noise")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    sel_path = out_dir / "selected_params.json"
    if "tune" in steps:
        tuning = tune(ds, {MAIN_SIZE: mats[MAIN_SIZE]}, jobs)
        tuning.to_csv(out_dir / "tuning.csv", index=False, float_format="%.6g")
        alpha_ridge = tune_ridge_lags(ds)
        sel = {"esn": selected(tuning), "ridge_lags_alpha": [float(a) for a in alpha_ridge]}
        sel_path.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
        log.info("selected: %s", json.dumps(sel["esn"]))
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    esn_sel, alpha_ridge = sel["esn"], np.array(sel["ridge_lags_alpha"])
    if "main" in steps:
        base, base_preds = baseline_runs(ds, alpha_ridge)
        base.round(3).to_csv(out_dir / "baselines.csv", index=False)
        main, preds = main_runs(ds, mats, esn_sel, jobs)
        main.round(3).to_csv(out_dir / "main.csv", index=False)
        frame = pd.DataFrame(index=ds.index)
        for name, pred in {**base_preds, **preds}.items():
            for k, c in enumerate(COLS):
                frame[f"{name}|{c}"] = pred[:, k].astype(np.float32)
        frame.to_parquet(run_dir / "predictions.parquet")
    matched = matched_settings(pd.read_csv(out_dir / "tuning.csv"), esn_sel)
    if "matched" in steps:
        m = matched_runs(ds, {MAIN_SIZE: mats[MAIN_SIZE]}, matched, jobs)
        m.round(3).to_csv(out_dir / "matched.csv", index=False)
    if "curves" in steps:
        curves = learning_curves(ds, {MAIN_SIZE: mats[MAIN_SIZE]}, matched, alpha_ridge, jobs)
        curves.round(3).to_csv(out_dir / "learning_curves.csv.gz", index=False)
    if "noise" in steps:
        noise = noise_runs(ds, {MAIN_SIZE: mats[MAIN_SIZE]}, esn_sel, alpha_ridge, jobs)
        noise.round(3).to_csv(out_dir / "noise.csv", index=False)
