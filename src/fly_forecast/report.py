"""Summary tables, README figures (matplotlib PNG) and the static dashboard (Plotly.js) from the experiment results.

Usage:
    build_report(ds, Path("connectome"), Path("results"), Path("data/runs"), Path("docs"))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fly_forecast import connectome as cn
from fly_forecast.data import HORIZONS, TARGETS, TEST_END, TEST_START, Dataset, column_names
from fly_forecast.experiment import MAIN_SIZE
from fly_forecast.metrics import bootstrap_rmse_diff
from fly_forecast.reservoir import VARIANTS, make_variant, scale_to_radius

log = logging.getLogger(__name__)

COLS = column_names()
LABEL = {
    "esn_connectome": "ESN, fly connectome", "esn_random": "ESN, random (classic)",
    "esn_shuffled": "ESN, connectome, shuffled weights", "esn_random_topology": "ESN, random wiring, fly weights",
    "esn_topology_random_w": "ESN, fly wiring, random weights", "esn_degree_preserving": "ESN, degree-preserving rewired",
    "esn_no_recurrence": "ESN, no recurrence (W = 0)", "persistence": "Persistence",
    "daily_persistence": "Persistence, same hour yesterday", "ridge_lags": "Ridge on 24 h of lags",
    "gbm_lags": "Gradient boosting on lags", "pse_plan": "PSE day-ahead plan (uses weather forecasts)",
}
COLOUR = {
    "esn_connectome": "#d9480f", "esn_random": "#495057", "esn_shuffled": "#f59f00", "esn_random_topology": "#7048e8",
    "esn_topology_random_w": "#e64980", "esn_degree_preserving": "#1098ad", "esn_no_recurrence": "#adb5bd",
    "persistence": "#ced4da", "daily_persistence": "#dee2e6", "ridge_lags": "#1c7ed6", "gbm_lags": "#2b8a3e",
    "pse_plan": "#212529",
}
NT_COLOUR = {"ACH": "#d9480f", "GABA": "#1c7ed6", "GLUT": "#7048e8",
             "DA": "#868e96", "SER": "#adb5bd", "OCT": "#ced4da"}   # modulatory: inputs only, no outgoing weights
TARGET_NAME = {"wind": "Wind", "pv": "PV"}


def col_label(c: str) -> str:
    s, h = c.split("_h")
    return f"{TARGET_NAME[s]} +{h} h"


# ---------------------------------------------------------------- tables
def main_table(main: pd.DataFrame, base: pd.DataFrame, size: str = MAIN_SIZE) -> pd.DataFrame:
    """Test RMSE / MAE per model and column at one reservoir size: mean and std over seeds, skill vs persistence."""
    m = main[main["size"] == size].assign(model=lambda d: "esn_" + d["variant"])
    g = m.groupby(["model", "target", "h"])
    esn = pd.DataFrame({"rmse": g["rmse"].mean(), "rmse_std": g["rmse"].std(), "mae": g["mae"].mean(),
                        "mae_std": g["mae"].std(), "train_rmse": g["train_rmse"].mean(), "seeds": g.size()})
    b = base.set_index(["model", "target", "h"])[["rmse", "mae"]].assign(rmse_std=0.0, mae_std=0.0, seeds=1)
    out = pd.concat([esn, b]).reset_index()
    ref = b.loc["persistence"]
    key = list(zip(out["target"], out["h"]))
    out["skill_rmse"] = 1 - out["rmse"].to_numpy() / ref.loc[key, "rmse"].to_numpy()
    out["skill_mae"] = 1 - out["mae"].to_numpy() / ref.loc[key, "mae"].to_numpy()
    return out


def paired_vs_random(main: pd.DataFrame) -> pd.DataFrame:
    """Per size, variant and column: RMSE ratio to the random ESN with the same seed (same input weights)."""
    key = ["size", "seed", "target", "h"]
    rnd = main[main["variant"] == "random"].set_index(key)["rmse"]
    m = main.set_index(key)
    m = m.assign(ratio=m["rmse"] / rnd.reindex(m.index)).reset_index()
    g = m.groupby(["size", "variant", "target", "h"])["ratio"]
    return pd.DataFrame({"ratio_mean": g.mean(), "ratio_std": g.std(), "ratio_min": g.min(), "ratio_max": g.max(),
                         "better_seeds": g.apply(lambda r: int((r < 1).sum())), "seeds": g.size()}).reset_index()


def bootstrap_table(ds: Dataset, preds: pd.DataFrame, a: str, b: str, rng_seed: int = 0) -> pd.DataFrame:
    """95 % daily-block bootstrap interval of RMSE(a) - RMSE(b) on the test period (seed 0 predictions)."""
    rows = []
    rng = np.random.default_rng(rng_seed)
    for k, c in enumerate(COLS):
        h = HORIZONS[k % len(HORIZONS)]
        r = ds.eval_rows(h, TEST_START, TEST_END)
        y = ds.Y[r, k]
        pa = np.clip(preds[f"{a}|{c}"].to_numpy()[r], 0, None)
        pb = np.clip(preds[f"{b}|{c}"].to_numpy()[r], 0, None)
        day = ((ds.index[r] + pd.Timedelta(hours=h)) - TEST_START) // pd.Timedelta(days=1)
        ea, eb = pa - y, pb - y
        lo, hi = bootstrap_rmse_diff(ea, eb, np.asarray(day), rng)
        rows.append({"column": c, "a": a, "b": b, "rmse_a": float(np.sqrt(np.mean(ea ** 2))),
                     "rmse_b": float(np.sqrt(np.mean(eb ** 2))), "diff_lo": lo, "diff_hi": hi})
    return pd.DataFrame(rows)


def curve_table(curves: pd.DataFrame) -> pd.DataFrame:
    g = curves.groupby(["model", "setting", "days", "alpha", "column"])
    return pd.DataFrame({"rows": g["rows"].first(), "test_rmse": g["test_rmse"].mean(),
                         "test_median": g["test_rmse"].median(), "test_std": g["test_rmse"].std(),
                         "train_rmse": g["train_rmse"].mean()}).reset_index()


def curve_pairs(curves: pd.DataFrame, a: str = "esn_connectome", b: str = "esn_random") -> pd.DataFrame:
    """Learning curves paired by seed: median over seeds of test RMSE(a) / RMSE(b), and seeds where a is better."""
    esn = curves[curves["model"].isin([a, b])]
    q = esn.pivot_table(index=["setting", "seed", "days", "alpha", "column"], columns="model", values="test_rmse")
    r = (q[a] / q[b]).rename("ratio").reset_index()
    g = r.groupby(["setting", "days", "alpha", "column"])["ratio"]
    return pd.DataFrame({"ratio_median": g.median(), "better_seeds": g.apply(lambda x: int((x < 1).sum())),
                         "seeds": g.size()}).reset_index()


def spectrum_stats(eig: dict[str, np.ndarray]) -> dict:
    """Shape of each spectrum after scaling to radius 1: share of |lambda| > 0.5 and > 1/1.1, second / first."""
    out = {}
    for v, e in eig.items():
        a = np.sort(np.abs(e))[::-1]
        out[v] = {"share_above_0.5": round(float((a > 0.5).mean()), 4),
                  "count_above_1/1.1": int((a > 1 / 1.1).sum()), "second_over_first": round(float(a[1] / a[0]), 4)}
    return out


def noise_table(noise: pd.DataFrame) -> pd.DataFrame:
    noise = noise.assign(column=noise["target"] + "_h" + noise["h"].astype(str))
    g = noise.groupby(["model", "sigma", "column"])["rmse"]
    out = pd.DataFrame({"rmse": g.mean(), "rmse_std": g.std()}).reset_index()
    clean = out[out["sigma"] == 0].set_index(["model", "column"])["rmse"]
    out["relative"] = out["rmse"] / clean.reindex(list(zip(out["model"], out["column"]))).to_numpy()
    return out


def summarise(ds: Dataset, results: Path, run_dir: Path, connectome_dir: Path, views: dict) -> dict:
    main = pd.read_csv(results / "main.csv")
    matched = pd.read_csv(results / "matched.csv")
    base = pd.read_csv(results / "baselines.csv")
    sel = json.loads((results / "selected_params.json").read_text(encoding="utf-8"))
    tab = main_table(main, base)
    pairs = paired_vs_random(main)
    mpairs = paired_vs_random(matched.drop(columns="size").rename(columns={"setting": "size"}))
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    boot = pd.concat([bootstrap_table(ds, preds, "esn_connectome", "esn_random"),
                      bootstrap_table(ds, preds, "esn_connectome", "ridge_lags")], ignore_index=True)
    raw = pd.read_csv(results / "learning_curves.csv.gz").fillna({"setting": ""})
    curves, cpairs = curve_table(raw), curve_pairs(raw)
    noise = noise_table(pd.read_csv(results / "noise.csv"))
    tab.round(4).to_csv(results / "summary_main.csv", index=False)
    pairs.round(4).to_csv(results / "summary_vs_random.csv", index=False)
    boot.round(2).to_csv(results / "bootstrap.csv", index=False)
    mpairs.round(4).to_csv(results / "summary_matched_vs_random.csv", index=False)
    curves.round(3).to_csv(results / "summary_curves.csv", index=False)
    cpairs.round(4).to_csv(results / "summary_curves_paired.csv", index=False)
    noise.round(4).to_csv(results / "summary_noise.csv", index=False)

    def cell(model, c, field="rmse"):
        s, h = c.split("_h")
        r = tab[(tab["model"] == model) & (tab["target"] == s) & (tab["h"] == int(h))]
        return round(float(r[field].iloc[0]), 4)

    p_main = pairs[pairs["size"] == MAIN_SIZE]
    pr = main.groupby(["size", "variant"])["state_pr"].mean().round(2)
    tuning = pd.read_csv(results / "tuning.csv")
    same = {}
    for setting in ("connectome", "random"):
        p = sel["esn"][setting]["params"]
        at = tuning[np.isclose(tuning["rho"], p["rho"]) & np.isclose(tuning["leak"], p["leak"])
                    & np.isclose(tuning["input_scale"], p["input_scale"])]
        same[setting] = {**p, **at.set_index("variant")["score"].round(4).to_dict()}

    def ratios(table, key):
        out = {}
        for k, g in table.groupby(key):
            g = g[g["variant"] == "connectome"]
            out[k] = {c: [round(float(r["ratio_mean"]), 4), int(r["better_seeds"]), int(r["seeds"])]
                      for c, r in zip(g["target"] + "_h" + g["h"].astype(str), g.to_dict("records"))}
        return out

    lc, lc_all = {}, {}
    for (setting, alpha), g in cpairs[cpairs["alpha"].isin(["inner", "1e-08"])].groupby(["setting", "alpha"]):
        key = f"{setting} setting, alpha {alpha}"
        lc[key] = {int(d): {c: [round(float(r["ratio_median"]), 3), int(r["better_seeds"])]
                            for c, r in zip(gd["column"], gd.to_dict("records"))} for d, gd in g.groupby("days")}
        part = g[g["days"] > 0]
        lc_all[key] = {"cells": len(part), "median_ratio_below_1": int((part["ratio_median"] < 1).sum()),
                       "all_seeds_better": int((part["better_seeds"] == part["seeds"]).sum()),
                       "all_seeds_worse": int((part["better_seeds"] == 0).sum()),
                       "geometric_mean_ratio": round(float(np.exp(np.log(part["ratio_median"]).mean())), 3)}
    summary = {
        "state_participation_ratio": {s: g.droplevel(0).to_dict() for s, g in pr.groupby(level=0)},
        "validation_score_at_setting_of": same,
        "connectome_vs_random_matched_test": ratios(mpairs, "size"),
        "learning_curves_overall": lc_all,
        "learning_curves_connectome_over_random_median": lc,
        "spectrum": spectrum_stats(views["eig"]),
        "test_period": f"{TEST_START:%Y-%m-%d} .. {TEST_END:%Y-%m-%d} (UTC, end exclusive)",
        "selected": sel,
        "rmse_mw": {m: {c: cell(m, c) for c in COLS} for m in tab["model"].unique()},
        "skill_vs_persistence": {m: {c: cell(m, c, "skill_rmse") for c in COLS} for m in tab["model"].unique()},
        "connectome_vs_random_ratio": {
            size: {c: round(float(g[(g["variant"] == "connectome") & (g["target"] + "_h" + g["h"].astype(str) == c)]
                                    ["ratio_mean"].iloc[0]), 4) for c in COLS}
            for size, g in pairs.groupby("size")},
        "variants_vs_random_main": {
            v: {c: [round(float(r["ratio_mean"]), 4), round(float(r["ratio_std"]), 4), int(r["better_seeds"])]
                for c, r in zip(g["target"] + "_h" + g["h"].astype(str), g.to_dict("records"))}
            for v, g in p_main.groupby("variant")},
        "bootstrap": boot.round(2).to_dict("records"),
        "graph_stats": json.loads((connectome_dir / "graph_stats.json").read_text(encoding="utf-8")),
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"tab": tab, "pairs": pairs, "mpairs": mpairs, "boot": boot, "curves": curves, "cpairs": cpairs,
            "noise": noise, "preds": preds, "main": main, "base": base, "summary": summary}


# ---------------------------------------------------------------- reservoir structure
def reservoir_views(connectome_dir: Path, name: str = MAIN_SIZE, seed: int = 0) -> dict:
    """Matrices (scaled to radius 1) of the main variants and their eigenvalues, for figures and the dashboard."""
    sub = cn.Subgraph.load(connectome_dir, name)
    W = sub.matrix()
    order = np.lexsort((-sub.nodes["syn_total"].to_numpy(), sub.nodes["top_neuropil"].astype(str).to_numpy()))
    out = {"nodes": sub.nodes, "edges": sub.edges, "order": order, "mats": {}, "eig": {}}
    for v in VARIANTS[:-1]:
        M = scale_to_radius(make_variant(v, W, np.random.default_rng([seed, VARIANTS.index(v)])), 1.0)
        out["mats"][v] = M
        out["eig"][v] = np.linalg.eigvals(M.toarray())
    return out


def block_image(M, order: np.ndarray, blocks: int = 200) -> np.ndarray:
    """|W| summed into blocks x blocks cells after reordering, log10(1 + count of synapse weight)."""
    A = abs(M.tocsr()[order][:, order]).tocoo()
    n = M.shape[0]
    bi, bj = A.row * blocks // n, A.col * blocks // n
    img = np.zeros((blocks, blocks))
    np.add.at(img, (bi, bj), A.data)
    return np.log10(1 + img / img[img > 0].min()) if (img > 0).any() else img


def view_limits(pos: np.ndarray, q: float = 1.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits that keep the central 100 - 2q % of nodes (isolated neurons are pushed far out by the layout)."""
    lo, hi = np.percentile(pos, [q, 100 - q], axis=0)
    pad = 0.05 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def layout(nodes: pd.DataFrame, edges: pd.DataFrame, seed: int = 1) -> np.ndarray:
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for a, b, w in zip(edges["pre"], edges["post"], np.log1p(np.abs(edges["weight"]))):
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a, b, weight=w)
    pos = nx.spring_layout(G, weight="weight", seed=seed, iterations=120)
    return np.array([pos[i] for i in range(len(nodes))])


# ---------------------------------------------------------------- report
def build_report(ds: Dataset, connectome_dir: Path, results: Path, run_dir: Path, docs: Path) -> None:
    from fly_forecast import dashboard, figures

    views = reservoir_views(connectome_dir)
    res = summarise(ds, results, run_dir, connectome_dir, views)
    views["pos"] = layout(views["nodes"], views["edges"])
    img = docs / "img"
    img.mkdir(parents=True, exist_ok=True)
    for path in figures.save_all(ds, res, views, img):
        log.info("figure %s", path)
    dashboard.write(ds, res, views, docs / "index.html")
    log.info("dashboard %s", docs / "index.html")
