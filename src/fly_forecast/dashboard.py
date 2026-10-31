"""Static HTML dashboard (Plotly.js from the official CDN) for GitHub Pages.

Figures are plain Plotly JSON with colour tokens ("@ink", "@grid", ...) that the page resolves for the light or dark
theme, so one file follows the reader's colour scheme.

Usage:
    write(ds, res, views, Path("docs/index.html"))
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fly_forecast.data import HORIZONS, TARGETS, TEST_END, TEST_START, Dataset, column_names
from fly_forecast.experiment import MAIN_SIZE, TRAIN_DAYS
from fly_forecast.report import COLOUR, LABEL, NT_COLOUR, block_image, col_label, view_limits

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"
# the two near-black series of the figures would vanish on the dark theme: use theme tokens instead
COLOUR = {**COLOUR, "esn_random": "@ink2", "pse_plan": "@muted"}
REPO_URL = "https://github.com/kwwaleszczynski-del/fly-connectome-forecast"
COLS = column_names()
ESN_MODELS = ["esn_connectome", "esn_random", "esn_shuffled", "esn_random_topology", "esn_topology_random_w",
              "esn_degree_preserving", "esn_no_recurrence"]
BASE_MODELS = ["persistence", "daily_persistence", "ridge_lags", "gbm_lags", "pse_plan"]


def _num(values, nd: int = 3) -> list:
    out = []
    for v in values:
        v = float(v)
        out.append(None if math.isnan(v) else round(v, nd))
    return out


def network_fig(views: dict) -> dict:
    pos, nodes, edges = views["pos"], views["nodes"], views["edges"]
    strong = edges.reindex(edges["weight"].abs().sort_values(ascending=False).index[:3000])
    ex, ey = [], []
    for a, b in zip(strong["pre"], strong["post"]):
        ex += [round(pos[a, 0], 4), round(pos[b, 0], 4), None]
        ey += [round(pos[a, 1], 4), round(pos[b, 1], 4), None]
    data = [{"type": "scattergl", "mode": "lines", "x": ex, "y": ey, "hoverinfo": "skip", "showlegend": False,
             "line": {"color": "@grid", "width": 0.5}}]
    smax = nodes["syn_total"].max()
    for nt, colour in NT_COLOUR.items():
        m = (nodes["nt"] == nt).to_numpy()
        if not m.any():
            continue
        sub = nodes[m]
        text = [f"root id {r}<br>{nt}, main neuropil {npl}<br>{s:,} synapses in graph"
                for r, npl, s in zip(sub["root_id"], sub["top_neuropil"], sub["syn_total"])]
        data.append({"type": "scattergl", "mode": "markers", "x": _num(pos[m, 0], 4), "y": _num(pos[m, 1], 4),
                     "name": f"{nt} ({int(m.sum())})", "text": text, "hovertemplate": "%{text}<extra></extra>",
                     "marker": {"color": colour, "size": _num(4 + 14 * np.sqrt(sub["syn_total"] / smax), 1),
                                "line": {"width": 0}}})
    (x0, x1), (y0, y1) = view_limits(pos)
    return {"data": data, "layout": {"xaxis": {"visible": False, "range": [x0, x1]},
                                     "yaxis": {"visible": False, "range": [y0, y1]}, "height": 560,
                                     "legend": {"orientation": "h", "y": -0.02}, "hovermode": "closest"}}


def matrix_fig(views: dict) -> dict:
    data, layout = [], {"height": 360, "margin": {"t": 40}}
    names = [("connectome", "connectome"), ("degree_preserving", "degree-preserving"), ("random", "random ESN")]
    for k, (v, title) in enumerate(names):
        img = block_image(views["mats"][v], views["order"], blocks=150)
        ax = "" if k == 0 else str(k + 1)
        data.append({"type": "heatmap", "z": [_num(r, 2) for r in img], "colorscale": "Magma", "reversescale": True,
                     "showscale": False, "xaxis": f"x{ax}", "yaxis": f"y{ax}",
                     "hovertemplate": "post block %{y}, pre block %{x}<br>log10 weight %{z}<extra></extra>"})
        layout[f"xaxis{ax}"] = {"domain": [k / 3 + 0.01, (k + 1) / 3 - 0.01], "showticklabels": False,
                                "title": {"text": title}}
        layout[f"yaxis{ax}"] = {"autorange": "reversed", "showticklabels": False, "anchor": f"x{ax}",
                                "scaleanchor": f"x{ax}"}
    return {"data": data, "layout": layout}


def spectra_fig(views: dict) -> dict:
    t = np.linspace(0, 2 * np.pi, 120)
    data = [{"type": "scattergl", "mode": "lines", "x": _num(np.cos(t)), "y": _num(np.sin(t)), "name": "|z| = 1",
             "line": {"color": "@muted", "width": 1}, "hoverinfo": "skip"}]
    for v in ("random", "shuffled", "degree_preserving", "connectome"):
        e = views["eig"][v]
        data.append({"type": "scattergl", "mode": "markers", "x": _num(e.real), "y": _num(e.imag),
                     "name": LABEL[f"esn_{v}"].replace("ESN, ", ""), "visible": True if v in ("connectome", "random")
                     else "legendonly", "marker": {"size": 4, "color": COLOUR[f"esn_{v}"]},
                     "hovertemplate": "%{x:.3f} %{y:+.3f}i<extra></extra>"})
    return {"data": data, "layout": {"height": 460, "xaxis": {"title": {"text": "real"}, "zeroline": False},
                                     "yaxis": {"title": {"text": "imaginary"}, "scaleanchor": "x", "zeroline": False}}}


def skill_fig(res: dict) -> dict:
    tab = res["tab"].assign(column=lambda d: d["target"] + "_h" + d["h"].astype(str))
    x = [col_label(c) for c in COLS]
    data = []
    for m in ESN_MODELS + ["ridge_lags", "gbm_lags", "daily_persistence", "pse_plan"]:
        t = tab[tab["model"] == m].set_index("column").reindex(COLS)
        if m == "pse_plan":   # a day-ahead forecast: only comparable with the +24 h horizon
            t.loc[[c for c in COLS if not c.endswith("_h24")], "skill_rmse"] = np.nan
        data.append({"type": "bar", "x": x, "y": _num(t["skill_rmse"]), "name": LABEL[m],
                     "marker": {"color": COLOUR[m]}, "customdata": _num(t["rmse"], 0),
                     "visible": True if m in ("esn_connectome", "esn_random", "ridge_lags", "gbm_lags", "pse_plan")
                     else "legendonly",
                     "hovertemplate": "%{y:.3f} (RMSE %{customdata} MW)<extra>" + LABEL[m] + "</extra>"})
    return {"data": data, "layout": {"barmode": "group", "height": 440, "yaxis": {"title": {"text": "RMSE skill vs persistence"}},
                                     "legend": {"orientation": "h", "y": -0.18}}}


def controls_fig(p: pd.DataFrame, size: str, models: list[str]) -> dict:
    p = p[(p["size"] == size) & (p["variant"] != "random")]
    x = [col_label(c) for c in COLS]
    data = []
    for m in models:
        t = p[p["variant"] == m[4:]].assign(column=lambda d: d["target"] + "_h" + d["h"].astype(str))
        t = t.set_index("column").reindex(COLS)
        data.append({"type": "bar", "x": x, "y": _num(t["ratio_mean"] - 1, 4), "base": 1, "name": LABEL[m],
                     "marker": {"color": COLOUR[m]},
                     "customdata": [[lo, hi, n] for lo, hi, n in zip(_num(t["ratio_min"], 3), _num(t["ratio_max"], 3),
                                                                      t["better_seeds"].astype(int).tolist())],
                     "hovertemplate": "ratio %{y:.3f} + 1<br>min %{customdata[0]}, max %{customdata[1]}<br>"
                                      "better than random in %{customdata[2]} of 5 seeds<extra>" + LABEL[m] + "</extra>"})
    return {"data": data, "layout": {"barmode": "group", "height": 400,
                                     "yaxis": {"title": {"text": "RMSE / RMSE of random ESN"}},
                                     "legend": {"orientation": "h", "y": -0.18}}}


def _x_days(c: pd.DataFrame) -> dict:
    full = c[c["days"] == 0]["rows"].max() / 24
    return {d: d for d in TRAIN_DAYS if d > 0} | {0: round(full)}


def curves_fig(res: dict, col: str) -> dict:
    c = res["curves"]
    xs = _x_days(c)
    data = []
    lines = [(m, st) for m in ESN_MODELS[:-1] for st in ("random", "connectome")] + [("ridge_lags", ""), ("gbm_lags", "")]
    for m, setting in lines:
        for alpha in ("inner", "1e-08"):
            t = c[(c["model"] == m) & (c["column"] == col) & (c["alpha"] == alpha) & (c["setting"] == setting)]
            if t.empty:
                continue
            t = t.assign(x=t["days"].map(xs)).sort_values("x")
            name = (LABEL[m] + (f", {setting}-ESN setting" if setting else "")
                    + ("" if alpha == "inner" else ", weak ridge"))
            show = alpha == "inner" and m in ("esn_connectome", "esn_random", "ridge_lags", "gbm_lags")
            dash = "dot" if alpha != "inner" else "dash" if setting == "connectome" else "solid"
            data.append({"type": "scatter", "mode": "lines+markers", "x": _num(t["x"], 1),
                         "y": _num(t["test_median"], 0), "name": name, "line": {"color": COLOUR[m], "dash": dash},
                         "marker": {"size": 5}, "customdata": _num(t["train_rmse"], 0),
                         "visible": True if show else "legendonly",
                         "hovertemplate": "%{x:.0f} days: test %{y} MW (median of seeds), train %{customdata} MW"
                                          "<extra>" + name + "</extra>"})
    s, h = col.split("_h")
    b = res["base"]
    ref = b[(b["model"] == "persistence") & (b["target"] == s) & (b["h"] == int(h))]["rmse"].iloc[0]
    return {"data": data, "layout": {"height": 460, "xaxis": {"type": "log", "title": {"text": "training days"}},
                                     "yaxis": {"title": {"text": "test RMSE, MW"}, "range": [0, 2.2 * ref]},
                                     "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": ref,
                                                 "y1": ref, "line": {"color": "@muted", "dash": "dash", "width": 1}}],
                                     "legend": {"orientation": "h", "y": -0.2}}}


TASK_COLOUR = {"wind_h1": "#1864ab", "wind_h6": "#4dabf7", "wind_h24": "#74c0fc",
               "pv_h1": "#c92a2a", "pv_h6": "#ff8787", "pv_h24": "#ffa8a8"}


def overfit_fig(res: dict, alpha: str) -> dict:
    p = res["cpairs"]
    xs = _x_days(res["curves"])
    data = []
    for col in COLS:
        for setting, dash in (("random", "solid"), ("connectome", "dash")):
            t = p[(p["alpha"] == alpha) & (p["column"] == col) & (p["setting"] == setting)]
            t = t.assign(x=t["days"].map(xs)).sort_values("x")
            name = f"{col_label(col)}, {setting}-ESN setting"
            data.append({"type": "scatter", "mode": "lines+markers", "x": _num(t["x"], 1),
                         "y": _num(t["ratio_median"], 3), "name": name,
                         "line": {"color": TASK_COLOUR[col], "dash": dash}, "marker": {"size": 5},
                         "customdata": t["better_seeds"].astype(int).tolist(),
                         "hovertemplate": "%{x:.0f} days: ratio %{y} (connectome better in %{customdata} of 5 seeds)"
                                          "<extra>" + name + "</extra>"})
    return {"data": data, "layout": {"height": 440, "xaxis": {"type": "log", "title": {"text": "training days"}},
                                     "yaxis": {"type": "log", "title": {"text": "RMSE connectome / random (median)"}},
                                     "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 1, "y1": 1,
                                                 "line": {"color": "@ink", "width": 1}}],
                                     "legend": {"orientation": "h", "y": -0.2}}}


def noise_fig(res: dict, col: str) -> dict:
    n = res["noise"]
    data = []
    for m in ESN_MODELS + ["ridge_lags", "gbm_lags"]:
        t = n[(n["model"] == m) & (n["column"] == col)].sort_values("sigma")
        data.append({"type": "scatter", "mode": "lines+markers", "x": _num(t["sigma"]), "y": _num(t["rmse"], 0),
                     "name": LABEL[m], "line": {"color": COLOUR[m]}, "customdata": _num(t["relative"]),
                     "visible": True if m in ("esn_connectome", "esn_random", "ridge_lags", "gbm_lags") else "legendonly",
                     "hovertemplate": "noise %{x}: %{y} MW (x%{customdata})<extra>" + LABEL[m] + "</extra>"})
    return {"data": data, "layout": {"height": 400, "xaxis": {"title": {"text": "input noise, standard deviations"}},
                                     "yaxis": {"title": {"text": "test RMSE, MW"}}, "legend": {"orientation": "h", "y": -0.2}}}


def forecast_data(ds: Dataset, res: dict) -> dict:
    """Test-period series per column (hourly, MW) for the forecast explorer: actual and every model."""
    p = res["preds"]
    out = {"t": None, "cols": {}}
    for k, col in enumerate(COLS):
        h = HORIZONS[k % len(HORIZONS)]
        r = ds.eval_rows(h, TEST_START, TEST_END)
        tau = ds.index[r] + pd.Timedelta(hours=h)
        if out["t"] is None:
            out["t"] = [f"{t:%Y-%m-%dT%H}" for t in tau]
        series = {"actual": _num(ds.Y[r, k], 0)}
        for m in ["esn_connectome", "esn_random", "esn_degree_preserving", "ridge_lags", "gbm_lags", "pse_plan"]:
            series[m] = _num(np.clip(p[f"{m}|{col}"].to_numpy()[r], 0, None), 0)
        out["cols"][col] = series
    return out


def table_rows(res: dict) -> str:
    tab = res["tab"].assign(column=lambda d: d["target"] + "_h" + d["h"].astype(str))
    rows = []
    for m in ESN_MODELS + BASE_MODELS:
        t = tab[tab["model"] == m].set_index("column").reindex(COLS)
        cells = "".join(
            f"<td>{r.rmse:,.0f}<span class='sd'>{'' if r.seeds <= 1 else f' ±{r.rmse_std:,.0f}'}</span>"
            f"<br><span class='sk'>{r.skill_rmse:+.2f}</span></td>" for r in t.itertuples())
        rows.append(f"<tr><th>{html.escape(LABEL[m])}</th>{cells}</tr>")
    return "\n".join(rows)


def _setting(summary: dict, name: str) -> str:
    p = summary["selected"]["esn"][name]["params"]
    return f"spectral radius {p['rho']}, leak rate {p['leak']}, input scaling {p['input_scale']}"


def kpis(res: dict) -> str:
    p = res["pairs"]
    p = p[(p["size"] == MAIN_SIZE) & (p["variant"] == "connectome")]
    m = res["mpairs"]
    m = m[(m["size"] == "random") & (m["variant"] == "connectome")]
    tab = res["tab"].set_index(["model", "target", "h"])
    spec = res["summary"]["spectrum"]
    skill = {k: tab.loc[(k, "wind", 24), "skill_rmse"] for k in ("esn_connectome", "ridge_lags", "pse_plan")}
    items = [
        (f"{p['ratio_mean'].mean():.3f}", "connectome / random ESN test RMSE, mean of 6 tasks, each tuned (1,000 neurons)"),
        (f"{int((p['better_seeds'] == p['seeds']).sum())} / {int((m['better_seeds'] == m['seeds']).sum())} of 6",
         "tasks where the connectome beats the random ESN in all 5 seeds: own settings / classic ESN setting"),
        (f"{spec['connectome']['share_above_0.5']:.1%} vs {spec['random']['share_above_0.5']:.0%}",
         "eigenvalues above half the spectral radius: connectome vs random matrix"),
        (f"{skill['esn_connectome']:+.2f} / {skill['ridge_lags']:+.2f}", "wind +24 h skill: connectome ESN / ridge on lags"),
        (f"{skill['pse_plan']:+.2f}", "wind +24 h skill of the PSE day-ahead plan (uses weather forecasts)"),
    ]
    return "".join(f"<div class='kpi'><b>{v}</b><span>{html.escape(t)}</span></div>" for v, t in items)


def write(ds: Dataset, res: dict, views: dict, path: Path) -> None:
    s = res["summary"]
    figs = {"network": network_fig(views), "matrices": matrix_fig(views), "spectra": spectra_fig(views),
            "skill": skill_fig(res),
            "controls": controls_fig(res["pairs"], MAIN_SIZE, ESN_MODELS[2:] + ["esn_connectome"]),
            "controls_random": controls_fig(res["mpairs"], "random", ESN_MODELS[2:-1] + ["esn_connectome"]),
            "controls_connectome": controls_fig(res["mpairs"], "connectome", ESN_MODELS[2:-1] + ["esn_connectome"]),
            "controls_mb": controls_fig(res["pairs"], "mb_right", ESN_MODELS[2:] + ["esn_connectome"]),
            "overfit_weak": overfit_fig(res, "1e-08"), "overfit_inner": overfit_fig(res, "inner")}
    for col in COLS:
        figs[f"curves_{col}"] = curves_fig(res, col)
        figs[f"noise_{col}"] = noise_fig(res, col)
    payload = {"figs": figs, "forecast": forecast_data(ds, res), "labels": {m: LABEL[m] for m in LABEL},
               "colours": COLOUR, "cols": COLS, "colLabels": [col_label(c) for c in COLS]}
    gs = s["graph_stats"]
    page = TEMPLATE
    for key, value in {
        "PLOTLY_JS": PLOTLY_JS, "REPO_URL": REPO_URL, "DATA": json.dumps(payload, separators=(",", ":")),
        "TABLE_HEAD": "".join(f"<th>{col_label(c)}</th>" for c in COLS), "TABLE_ROWS": table_rows(res),
        "TEST": f"{TEST_START:%d %b %Y} to {TEST_END - pd.Timedelta(days=1):%d %b %Y}",
        "NEURONS": f"{gs['neurons']:,}", "PAIRS": f"{gs['pairs']:,}", "SYNAPSES": f"{gs['synapses']:,}",
        "N_MAIN": f"{len(views['nodes']):,}", "E_MAIN": f"{len(views['edges']):,}",
        "COL_OPTIONS": "".join(f"<option value='{c}'>{col_label(c)}</option>" for c in COLS),
        "KPIS": kpis(res),
        "SET_RANDOM": _setting(s, "random"), "SET_CONN": _setting(s, "connectome"),
    }.items():
        page = page.replace("{{" + key + "}}", value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fly Connectome Forecast</title>
<meta name="description" content="The FlyWire fruit-fly connectome as an echo state network reservoir for Polish wind and PV forecasts, against random and rewired reservoirs and standard baselines.">
<script src="{{PLOTLY_JS}}"></script>
<style>
:root { --bg:#ffffff; --panel:#f8f9fa; --ink:#212529; --ink2:#495057; --muted:#868e96; --grid:#dee2e6; --accent:#d9480f; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181b; --panel:#1f2226; --ink:#e9ecef; --ink2:#ced4da; --muted:#868e96; --grid:#343a40; --accent:#ff8a4c; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.7rem; margin:0 0 6px; } h2 { font-size:1.2rem; margin:34px 0 6px; } p { margin:6px 0 10px; color:var(--ink2); max-width:900px; }
.lead { font-size:1.02rem; color:var(--ink); }
.card { background:var(--panel); border-radius:10px; padding:10px 12px; margin:10px 0; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:14px 0; }
.kpi { background:var(--panel); border-radius:10px; padding:10px 12px; } .kpi b { display:block; font-size:1.35rem; color:var(--accent); }
.kpi span { font-size:.85rem; color:var(--ink2); }
table { border-collapse:collapse; width:100%; font-size:.86rem; } th, td { padding:5px 7px; border-bottom:1px solid var(--grid); text-align:right; }
th:first-child { text-align:left; font-weight:500; } thead th { font-weight:600; }
.sd { color:var(--muted); font-size:.78rem; } .sk { color:var(--ink2); font-size:.8rem; }
.scroll { overflow-x:auto; } select { background:var(--panel); color:var(--ink); border:1px solid var(--grid); border-radius:6px; padding:4px 8px; font:inherit; }
a { color:var(--accent); } footer { margin-top:40px; font-size:.82rem; color:var(--muted); }
</style>
</head>
<body><main>
<h1>Does a fruit-fly brain forecast Polish wind and solar power?</h1>
<p class="lead">The wiring of {{N_MAIN}} neurons from the FlyWire whole-brain connectome of <i>Drosophila melanogaster</i>
is used as the fixed recurrent layer (reservoir) of an echo state network that forecasts national wind and PV
generation in Poland 1, 6 and 24 hours ahead. It is compared with random reservoirs of the same size and density,
with rewired and reshuffled versions of the same connectome, and with standard forecasting baselines.
Test period: {{TEST}} (hourly, PSE data). Code and method: <a href="{{REPO_URL}}">{{REPO_URL}}</a>.</p>
<div class="kpis">{{KPIS}}</div>

<h2>The reservoir</h2>
<p>The {{N_MAIN}} neurons with the most synapses in FlyWire v783 ({{NEURONS}} neurons, {{PAIRS}} connections with at least
5 synapses, {{SYNAPSES}} synapses after removing modulatory neurons), with {{E_MAIN}} connections between them. Colour:
predicted transmitter (ACh excitatory, GABA and glutamate inhibitory; grey: dopamine and serotonin neurons, which
only receive input in the reservoir); size: synapse count. Layout: force-directed,
strongest 3,000 connections drawn. Hover a neuron for its FlyWire root id.</p>
<div class="card"><div id="network"></div></div>
<p>Connection matrices ordered by each neuron's main neuropil. The real wiring is clustered into brain regions; the
degree-preserving control keeps every neuron's number of inputs and outputs but scatters them, the random ESN has no
structure at all.</p>
<div class="card"><div id="matrices"></div></div>
<p>Eigenvalues after scaling to spectral radius 1. A random matrix fills a disc (circular law); the connectome has a
few dominant eigenvalues and most of the spectrum close to zero.</p>
<div class="card"><div id="spectra"></div></div>

<h2>Forecast skill on the test year</h2>
<p>RMSE skill against persistence (1 = perfect, 0 = as good as "the next hours look like now", negative = worse).
Click legend entries to show the other controls.</p>
<div class="card"><div id="skill"></div></div>
<div class="card scroll"><table><thead><tr><th>Model: test RMSE in MW (± std over seeds) / skill</th>{{TABLE_HEAD}}</tr></thead>
<tbody>{{TABLE_ROWS}}</tbody></table></div>

<h2>Does the wiring matter? Controls paired by seed</h2>
<p>Each control uses the same input weights as the random ESN with the same seed, so the ratio isolates the
recurrent matrix. Below 1 = better than a classic random reservoir. First with every reservoir at its own tuned
setting (the fair forecasting comparison):</p>
<div class="card"><div id="controls"></div></div>
<p>Then all reservoirs at one shared setting, as in Costi et al., so that only the matrix differs: at the setting
tuned for the random ESN ({{SET_RANDOM}}),</p>
<div class="card"><div id="controls_random"></div></div>
<p>and at the setting tuned for the connectome ({{SET_CONN}}), where the reference is the random ESN at that
same setting:</p>
<div class="card"><div id="controls_connectome"></div></div>
<p>A regional reservoir: 1,000 neurons of the right mushroom body (the fly's learning centre), only synapses inside
its neuropils, each variant at its own tuned setting.</p>
<div class="card"><div id="controls_mb"></div></div>

<h2>Overfitting: learning curves and noisy inputs</h2>
<p>Training windows from 7 days to all data before the test year, every reservoir at the two shared settings (solid:
random-ESN setting; dashed: connectome setting). The ratio is the median over 5 seeds of the connectome's test RMSE
divided by the random reservoir's with the same seed. With an almost unregularised readout (ridge penalty 1e-8),
where overfitting shows first:</p>
<div class="card"><div id="overfit_weak"></div></div>
<p>With the ridge penalty chosen inside each training window (fitted on its first 80%, scored on the last 20%):</p>
<div class="card"><div id="overfit_inner"></div></div>
<p>Absolute learning curves with the baselines. <select id="curve_col">{{COL_OPTIONS}}</select> Hover for the
training error; dashed grey: persistence.</p>
<div class="card"><div id="curves"></div></div>
<p>Gaussian noise added to the observed wind, PV and demand inputs (targets stay clean), each reservoir at its own
setting. <select id="noise_col">{{COL_OPTIONS}}</select></p>
<div class="card"><div id="noise"></div></div>

<h2>Forecast explorer</h2>
<p><select id="fc_col">{{COL_OPTIONS}}</select> Drag on the chart to zoom; double click to reset.</p>
<div class="card"><div id="forecast"></div></div>

<footer>
<p>Data: PSE S.A. (www.pse.pl), reports API, national wind and PV generation, demand and the day-ahead coordination
plan, obtained in October 2026 and processed (hourly means, forecasts by the author); PSE is not responsible for
these results. Connectome: FlyWire Consortium, FlyWire Whole-brain Connectome Connectivity Data v783,
doi:10.5281/zenodo.10676866, CC BY 4.0 (Dorkenwald et al. 2024; Schlegel et al. 2024; transmitter predictions
Eckstein, Bates et al. 2024); subgraphs derived and re-weighted by the author. Method after Costi et al. 2025,
Biomimetics 10(5):341. Code: MIT licence.</p>
</footer>
</main>
<script>
const D = {{DATA}};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function theme(o) {
  if (Array.isArray(o)) return o.map(theme);
  if (o && typeof o === "object") { const r = {}; for (const k in o) r[k] = theme(o[k]); return r; }
  if (typeof o === "string" && o.startsWith("@")) return css("--" + o.slice(1));
  return o;
}
function baseLayout(extra) {
  const ax = { gridcolor: css("--grid"), zerolinecolor: css("--grid"), color: css("--ink2") };
  const l = Object.assign({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: css("--ink"), size: 12 },
    margin: { l: 60, r: 16, t: 20, b: 50 }, xaxis: Object.assign({}, ax), yaxis: Object.assign({}, ax) }, theme(extra));
  for (const k in l) if (/^[xy]axis/.test(k)) l[k] = Object.assign({}, ax, l[k]);
  return l;
}
function draw(id, fig) {
  let lay = fig.layout, data = fig.data;
  if (id === "curves") {  // 27 legend entries: a column on the right on a wide screen, below a taller chart on a phone
    const narrow = innerWidth < 760;
    lay = Object.assign({}, lay, narrow
      ? { height: 1700, margin: { l: 60, r: 16, t: 20, b: 1160 },
          legend: { orientation: "v", x: 0, y: -0.14, yanchor: "top", font: { size: 11 } } }
      : { height: 600, legend: { orientation: "v", x: 1.02, xanchor: "left", y: 1, yanchor: "top" } });
    // on a phone the long names break before the setting and the weak-ridge note so they fit the width
    if (narrow) data = data.map(t => Object.assign({}, t, { name: t.name.replace(/, ([a-z]+-ESN setting|weak ridge)/g, "<br>$1") }));
  }
  if (innerWidth < 760) lay = phone(lay, data, id !== "curves");
  Plotly.react(id, theme(data), baseLayout(lay), { responsive: true, displaylogo: false });
}
// narrow screens: legend above the plot, smaller ticks with automargin, slanted category labels,
// panel titles broken over two lines
function phone(lay, data, moveLegend) {
  const l = Object.assign({}, lay);
  const x0 = data.length && data[0].x ? data[0].x[0] : null;
  const cat = typeof x0 === "string" && !/^[0-9]{4}-/.test(x0);
  if (cat && !("xaxis" in l)) l.xaxis = {};
  for (const k of Object.keys(l)) {
    if (!/^[xy]axis[0-9]*$/.test(k)) continue;
    const a = Object.assign({ automargin: true }, l[k]);
    a.tickfont = Object.assign({ size: 10 }, a.tickfont);
    if (a.title && a.title.text) {
      const split = k.startsWith("x") && "xaxis2" in l;
      a.title = Object.assign({}, a.title, { font: Object.assign({ size: 11 }, a.title.font),
        text: split ? a.title.text.replace(/[- ]/, s => s === "-" ? "-<br>" : "<br>") : a.title.text });
    }
    if (k === "xaxis" && cat) a.tickangle = -45;
    l[k] = a;
  }
  if (moveLegend) l.legend = Object.assign({}, l.legend, { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { size: 10 } });
  return l;
}
function drawAll() {
  for (const id of ["network", "matrices", "spectra", "skill", "controls", "controls_random", "controls_connectome",
                    "controls_mb", "overfit_weak", "overfit_inner"]) draw(id, D.figs[id]);
  draw("curves", D.figs["curves_" + document.getElementById("curve_col").value]);
  draw("noise", D.figs["noise_" + document.getElementById("noise_col").value]);
  drawForecast();
}
function drawForecast() {
  const col = document.getElementById("fc_col").value, S = D.forecast.cols[col], t = D.forecast.t;
  const order = ["actual", "pse_plan", "ridge_lags", "gbm_lags", "esn_random", "esn_degree_preserving", "esn_connectome"];
  const data = order.map(m => ({ type: "scattergl", mode: "lines", x: t, y: S[m], name: m === "actual" ? "Actual" : D.labels[m],
    line: { color: m === "actual" ? css("--ink") : theme(D.colours[m]), width: m === "actual" ? 2 : 1.2, dash: m === "pse_plan" ? "dash" : "solid" },
    visible: ["actual", "esn_connectome", "esn_random", "ridge_lags", "pse_plan"].includes(m) ? true : "legendonly" }));
  const end = t.length > 24 * 14 ? t[24 * 14] : t[t.length - 1];
  const lay = { height: 430, margin: { l: 60, r: 32, t: 20, b: 50 },  // room for the last date tick
    xaxis: { range: [t[0], end], rangeslider: { visible: true } },
    yaxis: { title: { text: "MW" } }, legend: { orientation: "h", y: -0.45 } };
  Plotly.react("forecast", data, baseLayout(innerWidth < 760 ? phone(lay, data, true) : lay), { responsive: true, displaylogo: false });
}
for (const id of ["curve_col", "noise_col", "fc_col"]) document.getElementById(id).addEventListener("change", drawAll);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
drawAll();
</script>
</body>
</html>
"""
