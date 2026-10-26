"""FlyWire whole-brain connectome (v783) -> signed, weighted subgraphs used as ESN reservoirs.

Source: FlyWire Consortium, "FlyWire Whole-brain Connectome Connectivity Data", Zenodo record 10676866, version 783,
licence CC BY 4.0 (doi:10.5281/zenodo.10676866). Only ``proofread_connections_783.feather`` is used: one row per
(presynaptic neuron, postsynaptic neuron, neuropil) with the synapse count and the mean neurotransmitter
probabilities of those synapses (Eckstein, Bates et al. 2024).

Construction:
    1. Rows are summed over neuropils per neuron pair; pairs with fewer than ``MIN_SYNAPSES`` synapses are dropped
       (the threshold used for most analyses in Dorkenwald et al. 2024) and so are autapses.
    2. Each presynaptic neuron gets one transmitter (Dale's principle): the argmax of its synapse-weighted mean
       transmitter probabilities over all its outgoing connections.
    3. Sign: acetylcholine excitatory (+1); GABA and glutamate inhibitory (-1, glutamate acts mostly through the
       inhibitory GluCl receptor in the fly central brain, as in Shiu et al. 2024); edges of modulatory neurons
       (dopamine, serotonin, octopamine) are left out, as in Costi et al. 2025.
    4. Weight of edge pre -> post: sign(pre) x synapse count, stored as W[post, pre].

Subgraphs: the ``n`` neurons with the most synapses (in + out) inside the thresholded graph ("top"), or the same
restricted to the synapses of one region ("mb_right": the right mushroom body, neuropils MB_CA_R, MB_PED_R,
MB_VL_R, MB_ML_R).

Usage:
    path = download(Path("data/flywire"))
    graph = build_graph(path)
    sub = top_subgraph(graph, 1000)
    W = sub.matrix()            # scipy.sparse CSR, shape (1000, 1000)
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

log = logging.getLogger(__name__)

ZENODO_RECORD = "10676866"
FILE_NAME = "proofread_connections_783.feather"
FILE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{FILE_NAME}?download=1"
FILE_MD5 = "f48f972d262323a102aed49af1396b8a"
FILE_SIZE = 852_022_274
MIN_SYNAPSES = 5
NT_COLUMNS = ["ach_avg", "gaba_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
NT_NAMES = ["ACH", "GABA", "GLUT", "OCT", "SER", "DA"]
NT_SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "OCT": 0, "SER": 0, "DA": 0}
REGIONS = {"mb_right": ("MB_CA_R", "MB_PED_R", "MB_VL_R", "MB_ML_R")}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest_dir: Path, attempts: int = 5) -> Path:
    """Download the connection table once (852 MB), resuming a partial file, and check its MD5 against Zenodo."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / FILE_NAME
    if path.exists() and path.stat().st_size == FILE_SIZE:
        return path
    tmp = path.with_suffix(".part")
    log.info("downloading %s (%.0f MB) from Zenodo record %s", FILE_NAME, FILE_SIZE / 1e6, ZENODO_RECORD)
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have >= FILE_SIZE:
            break
        headers = {"User-Agent": "fly-connectome-forecast (open-source analysis)"}
        if have:
            headers["Range"] = f"bytes={have}-"
            log.info("resuming at %.0f MB", have / 1e6)
        try:
            with requests.get(FILE_URL, stream=True, timeout=120, headers=headers) as r:
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 22):
                        f.write(chunk)
        except requests.RequestException as e:
            log.warning("download interrupted (%s), attempt %d of %d", e, attempt + 1, attempts)
    md5 = _md5(tmp)
    if md5 != FILE_MD5:
        tmp.unlink()
        raise RuntimeError(f"MD5 mismatch for {FILE_NAME}: {md5} != {FILE_MD5}")
    tmp.replace(path)
    return path


@dataclass
class Graph:
    """Thresholded neuron-pair graph of the whole brain (or of one region)."""
    pairs: pd.DataFrame    # pre, post (root ids), syn (synapse count), sign of the presynaptic neuron
    neurons: pd.DataFrame  # index root_id: nt, sign, syn_total, top_neuropil


@dataclass
class Subgraph:
    """A reservoir-sized subgraph: nodes 0..n-1 and signed edges."""
    name: str
    nodes: pd.DataFrame    # node, root_id, nt, sign, syn_total, top_neuropil
    edges: pd.DataFrame    # pre, post (node indices), weight (signed synapse count)

    @property
    def n(self) -> int:
        return len(self.nodes)

    def matrix(self) -> sp.csr_matrix:
        """W[post, pre] = signed synapse count."""
        e = self.edges
        return sp.csr_matrix((e["weight"].to_numpy(float), (e["post"].to_numpy(), e["pre"].to_numpy())),
                             shape=(self.n, self.n))

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.nodes.to_csv(folder / f"{self.name}_nodes.csv", index=False)
        self.edges.to_csv(folder / f"{self.name}_edges.csv.gz", index=False)

    @classmethod
    def load(cls, folder: Path, name: str) -> "Subgraph":
        nodes = pd.read_csv(folder / f"{name}_nodes.csv")
        edges = pd.read_csv(folder / f"{name}_edges.csv.gz")
        return cls(name, nodes, edges)


def neuron_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-neuron transmitter (synapse-weighted argmax over outgoing rows), sign, synapse total and main neuropil."""
    w = rows["syn_count"].to_numpy(float)
    nt = pd.DataFrame(rows[NT_COLUMNS].to_numpy(float) * w[:, None], columns=NT_NAMES)
    nt["root_id"] = rows["pre_pt_root_id"].to_numpy()
    nt = nt.groupby("root_id").sum()
    out = pd.DataFrame({"nt": nt.idxmax(axis=1)})
    out["sign"] = out["nt"].map(NT_SIGN).astype(int)
    by_np = pd.concat([
        rows[["pre_pt_root_id", "neuropil", "syn_count"]].rename(columns={"pre_pt_root_id": "root_id"}),
        rows[["post_pt_root_id", "neuropil", "syn_count"]].rename(columns={"post_pt_root_id": "root_id"}),
    ]).groupby(["root_id", "neuropil"])["syn_count"].sum()
    top = by_np.reset_index().sort_values("syn_count", ascending=False).drop_duplicates("root_id")
    out["top_neuropil"] = top.set_index("root_id")["neuropil"].reindex(out.index)
    return out


def build_graph(rows: pd.DataFrame, neuropils: tuple[str, ...] | None = None,
                min_synapses: int = MIN_SYNAPSES) -> Graph:
    """Connection rows (one per pair and neuropil) -> thresholded signed pair graph.

    Transmitters are always estimated from all rows of a neuron; ``neuropils`` restricts the synapses that are
    counted as edges.
    """
    neurons = neuron_table(rows)
    if neuropils is not None:
        rows = rows[rows["neuropil"].isin(neuropils)]
    pairs = (rows.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False)["syn_count"].sum()
             .reset_index().rename(columns={"pre_pt_root_id": "pre", "post_pt_root_id": "post", "syn_count": "syn"}))
    pairs = pairs[(pairs["syn"] >= min_synapses) & (pairs["pre"] != pairs["post"])]
    pairs = pairs.assign(sign=neurons["sign"].reindex(pairs["pre"]).fillna(0).astype(int).to_numpy())
    pairs = pairs[pairs["sign"] != 0].reset_index(drop=True)
    total = pd.concat([pairs.groupby("pre")["syn"].sum(), pairs.groupby("post")["syn"].sum()], axis=1).sum(axis=1)
    neurons = neurons.loc[neurons.index.intersection(total.index)].copy()
    neurons["syn_total"] = total.reindex(neurons.index).astype(int)
    neurons.index.name = "root_id"
    log.info("graph: %d neurons, %d pairs, %d synapses", len(neurons), len(pairs), int(pairs["syn"].sum()))
    return Graph(pairs, neurons)


def top_subgraph(graph: Graph, n: int, name: str | None = None) -> Subgraph:
    """Induced subgraph of the ``n`` neurons with the most synapses (in + out) in ``graph``."""
    ranked = graph.neurons.sort_values(["syn_total", "nt"], ascending=[False, True], kind="stable")
    chosen = ranked.head(n).reset_index()
    chosen.insert(0, "node", np.arange(len(chosen)))
    idx = pd.Series(chosen["node"].to_numpy(), index=chosen["root_id"].to_numpy())
    p = graph.pairs[graph.pairs["pre"].isin(idx.index) & graph.pairs["post"].isin(idx.index)]
    edges = pd.DataFrame({"pre": idx.reindex(p["pre"]).to_numpy(), "post": idx.reindex(p["post"]).to_numpy(),
                          "weight": (p["sign"] * p["syn"]).to_numpy()})
    edges = edges.sort_values(["post", "pre"]).reset_index(drop=True)
    cols = ["node", "root_id", "nt", "sign", "syn_total", "top_neuropil"]
    return Subgraph(name or f"top{n}", chosen[cols], edges)


def read_rows(path: Path) -> pd.DataFrame:
    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", *NT_COLUMNS]
    rows = pd.read_feather(path, columns=cols)
    log.info("%s: %d rows", path.name, len(rows))
    return rows


def build_subgraphs(path: Path, sizes: list[int], region_size: int, out_dir: Path) -> list[Subgraph]:
    """All reservoir subgraphs used by the experiments, written as small CSV files to ``out_dir``."""
    rows = read_rows(path)
    whole = build_graph(rows)
    subs = [top_subgraph(whole, n) for n in sizes]
    for name, neuropils in REGIONS.items():
        region = build_graph(rows, neuropils=neuropils)
        subs.append(top_subgraph(region, region_size, name=name))
    stats = {"neurons": len(whole.neurons), "pairs": len(whole.pairs), "synapses": int(whole.pairs["syn"].sum()),
             "nt_share": whole.neurons["nt"].value_counts(normalize=True).round(4).to_dict()}
    for s in subs:
        s.save(out_dir)
        log.info("%s: %d nodes, %d edges, density %.4f, %.1f%% inhibitory nodes", s.name, s.n, len(s.edges),
                 len(s.edges) / s.n ** 2, 100 * (s.nodes["sign"] < 0).mean())
    pd.Series(stats, dtype=object).to_json(out_dir / "graph_stats.json", indent=2)
    return subs
