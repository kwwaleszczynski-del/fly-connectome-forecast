"""README figures (matplotlib, PNG).

Usage:
    paths = save_all(ds, res, views, Path("docs/img"))   # res from report.summarise, views from report.reservoir_views
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fly_forecast.data import TEST_START, Dataset, column_names  # noqa: E402
from fly_forecast.experiment import MAIN_SIZE, TRAIN_DAYS  # noqa: E402
from fly_forecast.report import COLOUR, LABEL, NT_COLOUR, block_image, col_label, view_limits  # noqa: E402

COLS = column_names()
TASK_COLOUR = {"wind_h1": "#1864ab", "wind_h6": "#4dabf7", "wind_h24": "#a5d8ff",
               "pv_h1": "#c92a2a", "pv_h6": "#ff8787", "pv_h24": "#ffc9c9"}


ESN_ORDER = ["esn_connectome", "esn_random", "esn_shuffled", "esn_random_topology", "esn_topology_random_w",
             "esn_degree_preserving", "esn_no_recurrence"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9.5, "legend.frameon": False})


def _save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_reservoir(views: dict, path: Path) -> Path:
    fig = plt.figure(figsize=(12.5, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1.05])
    ax = fig.add_subplot(gs[0])
    pos, nodes, edges = views["pos"], views["nodes"], views["edges"]
    strong = edges.reindex(edges["weight"].abs().sort_values(ascending=False).index[:2500])
    for a, b in zip(strong["pre"], strong["post"]):
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#adb5bd", lw=0.3, alpha=0.4, zorder=1)
    size = 4 + 40 * nodes["syn_total"] / nodes["syn_total"].max()
    for nt, colour in NT_COLOUR.items():
        m = (nodes["nt"] == nt).to_numpy()
        if not m.any():
            continue
        ax.scatter(pos[m, 0], pos[m, 1], s=size[m], c=colour, label=f"{nt} ({m.sum()})", zorder=2, lw=0)
    ax.set_title(f"{len(nodes)} most connected FlyWire neurons", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, markerscale=1.5)
    (x0, x1), (y0, y1) = view_limits(pos)
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for k, (v, title) in enumerate([("connectome", "connectome |W|"), ("degree_preserving", "degree-preserving"),
                                    ("random", "random (same density)")]):
        ax = fig.add_subplot(gs[k + 1])
        ax.imshow(block_image(views["mats"][v], views["order"]), cmap="magma_r", interpolation="nearest")
        ax.set_title(title + "\n(rows: post, columns: pre)", fontsize=9.5)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    fig.suptitle("Reservoir matrices, neurons ordered by main neuropil", y=1.02)
    return _save(fig, path)


def fig_spectra(views: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True, sharey=True)
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, v in zip(axes, ["connectome", "degree_preserving", "shuffled", "random"]):
        e = views["eig"][v]
        ax.scatter(e.real, e.imag, s=3, color=COLOUR[f"esn_{v}"], lw=0)
        ax.plot(np.cos(t), np.sin(t), color="#868e96", lw=0.6)
        ax.set_aspect("equal")
        ax.set_title(LABEL[f"esn_{v}"].replace("ESN, ", ""), fontsize=9)
    axes[0].set_ylabel("imaginary")
    for ax in axes:
        ax.set_xlabel("real")
    fig.suptitle("Eigenvalues of the reservoir matrices scaled to spectral radius 1", y=1.04)
    return _save(fig, path)


def fig_skill(res: dict, path: Path) -> Path:
    tab = res["tab"].assign(column=lambda d: d["target"] + "_h" + d["h"].astype(str))
    models = ["esn_connectome", "esn_random", "esn_degree_preserving", "ridge_lags", "gbm_lags", "pse_plan"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.8 / len(models)
    x = np.arange(len(COLS))
    for k, m in enumerate(models):
        t = tab[tab["model"] == m].set_index("column").reindex(COLS)
        if m == "pse_plan":   # a day-ahead forecast: only comparable with the +24 h horizon
            t.loc[[c for c in COLS if not c.endswith("_h24")], "skill_rmse"] = np.nan
        err = (t["rmse_std"] / res["tab"].query("model == 'persistence'")["rmse"].to_numpy()).fillna(0)
        ax.bar(x + (k - len(models) / 2 + 0.5) * w, t["skill_rmse"], w, yerr=err, color=COLOUR[m], label=LABEL[m],
               error_kw={"lw": 0.8})
    ax.axhline(0, color="#495057", lw=0.8)
    ax.set_xticks(x, [col_label(c) for c in COLS])
    ax.set_ylabel("RMSE skill vs persistence")
    ax.set_title(f"Test period from {TEST_START:%d %b %Y}, 12 months: higher is better "
                 "(error bars: std over seeds)", fontsize=10)
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1))
    return _save(fig, path)


def _ratio_bars(ax, p: pd.DataFrame, variants: list[str], title: str) -> None:
    w = 0.8 / len(variants)
    x = np.arange(len(COLS))
    for k, m in enumerate(variants):
        t = p[p["variant"] == m[4:]].assign(column=lambda d: d["target"] + "_h" + d["h"].astype(str))
        t = t.set_index("column").reindex(COLS)
        xs = x + (k - len(variants) / 2 + 0.5) * w
        ax.bar(xs, t["ratio_mean"] - 1, w, bottom=1, color=COLOUR[m], label=LABEL[m].replace("ESN, ", ""))
        ax.errorbar(xs, t["ratio_mean"], yerr=[t["ratio_mean"] - t["ratio_min"], t["ratio_max"] - t["ratio_mean"]],
                    fmt="none", ecolor="#212529", lw=0.7)
    ax.axhline(1, color="#212529", lw=0.9)
    ax.set_xticks(x, [col_label(c) for c in COLS])
    ax.set_ylabel("RMSE / random ESN")
    ax.set_title(title, fontsize=10, loc="left")


def fig_controls(res: dict, path: Path) -> Path:
    variants = [v for v in ESN_ORDER if v != "esn_random"]
    sel = res["summary"]["selected"]["esn"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.6), sharex=True)
    p = res["pairs"]
    _ratio_bars(axes[0], p[(p["size"] == MAIN_SIZE) & (p["variant"] != "random")], variants,
                "(a) each reservoir with its own tuned setting")
    for ax, setting, tag in zip(axes[1:], ["random", "connectome"], ["b", "c"]):
        q = sel[setting]["params"]
        m = res["mpairs"]
        _ratio_bars(ax, m[(m["size"] == setting) & (m["variant"] != "random")], variants[:-1],
                    f"({tag}) all at the setting tuned for the {setting} ESN "
                    f"(spectral radius {q['rho']}, leak {q['leak']}, input scaling {q['input_scale']})")
    axes[0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.suptitle("Test RMSE relative to the random ESN with the same seed (same input weights); below 1 = better; "
                 "whiskers: min-max over 5 seeds", y=1.04, fontsize=10.5)
    return _save(fig, path)


def fig_sizes(res: dict, path: Path) -> Path:
    p = res["pairs"].assign(column=lambda d: d["target"] + "_h" + d["h"].astype(str))
    sizes = sorted({s for s in p["size"] if s.startswith("top")}, key=lambda s: int(s[3:]))
    n = [int(s[3:]) for s in sizes]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, v in zip(axes, ["connectome", "degree_preserving"]):
        for col in COLS:
            t = p[(p["variant"] == v) & (p["column"] == col)].set_index("size")
            ax.errorbar(n, t["ratio_mean"].reindex(sizes), yerr=t["ratio_std"].reindex(sizes), marker="o", ms=4,
                        color=TASK_COLOUR[col], lw=1.3, capsize=2, label=col_label(col))
            mb = t["ratio_mean"].get("mb_right")
            if mb is not None and v == "connectome":
                ax.plot([1000], [mb], marker="*", ms=11, color=TASK_COLOUR[col], mec="#212529", mew=0.5, ls="none")
        ax.axhline(1, color="#212529", lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks(n, [f"{k:,}" for k in n])
        ax.minorticks_off()
        ax.set_xlabel("reservoir neurons")
        ax.set_title(LABEL[f"esn_{v}"].replace("ESN, ", "") + (" (stars: right mushroom body, 1,000 neurons)"
                                                               if v == "connectome" else ""), fontsize=9.5)
    axes[0].set_ylabel("test RMSE / random ESN\n(mean of seeds, bars: std)")
    axes[0].legend(fontsize=8, ncol=6, loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.suptitle("Reservoir size: each reservoir at its own tuned setting, paired with the random ESN", y=1.02,
                 fontsize=10.5)
    return _save(fig, path)


def TARGET(t: str) -> str:
    return {"wind": "Wind", "pv": "PV"}[t]


def _x_days(c: pd.DataFrame) -> dict:
    full = c[c["days"] == 0]["rows"].max() / 24
    return {d: d for d in TRAIN_DAYS if d > 0} | {0: round(full)}


def fig_curves(res: dict, path: Path) -> Path:
    c = res["curves"]
    c = c[c["alpha"] == "inner"]
    xs_map = _x_days(c)
    lines = [("esn_connectome", "random", "-"), ("esn_random", "random", "-"), ("esn_connectome", "connectome", "--"),
             ("esn_random", "connectome", "--"), ("ridge_lags", "", "-"), ("gbm_lags", "", "-")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for ax, col in zip(axes.ravel(), COLS):
        for m, setting, ls in lines:
            t = c[(c["model"] == m) & (c["column"] == col) & (c["setting"] == setting)]
            t = t.assign(x=t["days"].map(xs_map)).sort_values("x")
            lab = LABEL[m] + (f", {setting}-ESN setting" if setting else "")
            ax.plot(t["x"], t["test_median"], ls, marker="o", ms=3, color=COLOUR[m], label=lab, lw=1.3)
        ref = res["base"].query("model == 'persistence'")
        s, h = col.split("_h")
        r0 = ref[(ref["target"] == s) & (ref["h"] == int(h))]["rmse"].iloc[0]
        ax.axhline(r0, color="#adb5bd", lw=1)
        ax.set_xscale("log")
        ax.set_title(col_label(col), fontsize=10)
        ax.set_ylim(0, 2.2 * r0)
    for ax in axes[1]:
        ax.set_xlabel("training days before the test period")
    for ax in axes[:, 0]:
        ax.set_ylabel("test RMSE, MW (median of seeds)")
    axes[1, 2].legend(fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(-2.3, -0.2))
    fig.suptitle("Learning curves, ridge penalty chosen inside each training window (grey: persistence)", y=1.0)
    return _save(fig, path)


def fig_overfit(res: dict, path: Path) -> Path:
    p = res["cpairs"]
    xs_map = _x_days(res["curves"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, alpha, title in zip(axes, ["1e-08", "inner"], ["weak ridge (penalty 1e-8)",
                                                           "ridge penalty chosen inside each window"]):
        for col in COLS:
            for setting, ls in (("random", "-"), ("connectome", "--")):
                t = p[(p["alpha"] == alpha) & (p["column"] == col) & (p["setting"] == setting)]
                t = t.assign(x=t["days"].map(xs_map)).sort_values("x")
                ax.plot(t["x"], t["ratio_median"], ls, marker="o", ms=3, color=TASK_COLOUR[col], lw=1.3,
                        label=f"{col_label(col)}, {setting}-ESN setting")
        ax.axhline(1, color="#212529", lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training days before the test period")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("test RMSE connectome / random, median of 5 paired seeds", fontsize=8.5)
    axes[0].legend(fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.suptitle("Overfitting test: below 1 = the fly reservoir generalises better than a random one of the same "
                 "size, density and setting", y=1.02, fontsize=10.5)
    return _save(fig, path)


def fig_noise(res: dict, path: Path) -> Path:
    n = res["noise"]
    models = ["esn_connectome", "esn_random", "esn_degree_preserving", "ridge_lags", "gbm_lags"]
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.9), sharey=True)
    for ax, col in zip(axes, COLS):
        for m in models:
            t = n[(n["model"] == m) & (n["column"] == col)].sort_values("sigma")
            ax.plot(t["sigma"], t["relative"], "o-", ms=3, color=COLOUR[m], label=LABEL[m])
        ax.set_title(col_label(col), fontsize=9.5)
        ax.set_xlabel("input noise, std")
    axes[0].set_ylabel("test RMSE / RMSE without noise")
    axes[0].legend(fontsize=7.5, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.28))
    return _save(fig, path)


def fig_week(ds: Dataset, res: dict, path: Path, start: str = "2026-04-20") -> Path:
    p = res["preds"]
    t0 = pd.Timestamp(start, tz="UTC")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True)
    for ax, s in zip(axes, ["wind", "pv"]):
        col = f"{s}_h24"
        idx = ds.index + pd.Timedelta(hours=24)
        win = (idx >= t0) & (idx < t0 + pd.Timedelta(days=7))
        ax.plot(idx[win], ds.frame[s].reindex(idx[win]), color="#212529", lw=2, label="actual")
        for m in ["esn_connectome", "esn_random", "ridge_lags", "pse_plan"]:
            ax.plot(idx[win], np.clip(p[f"{m}|{col}"].to_numpy()[win], 0, None), color=COLOUR[m], lw=1.2,
                    ls="--" if m == "pse_plan" else "-", label=LABEL[m])
        ax.set_ylabel("MW")
        ax.set_title(f"{TARGET(s)}, forecasts issued 24 h ahead (UTC)", fontsize=10)
    axes[1].legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.15))
    return _save(fig, path)


def save_all(ds: Dataset, res: dict, views: dict, img: Path) -> list[Path]:
    return [
        fig_reservoir(views, img / "reservoir.png"),
        fig_spectra(views, img / "spectra.png"),
        fig_skill(res, img / "skill.png"),
        fig_controls(res, img / "controls.png"),
        fig_sizes(res, img / "sizes.png"),
        fig_curves(res, img / "learning_curves.png"),
        fig_overfit(res, img / "overfitting.png"),
        fig_noise(res, img / "noise.png"),
        fig_week(ds, res, img / "week.png"),
    ]
