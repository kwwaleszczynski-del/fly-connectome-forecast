import numpy as np
import pandas as pd

from fly_forecast import connectome as cn


def row(pre, post, neuropil, syn, nt):
    probs = {c: 0.02 for c in cn.NT_COLUMNS}
    probs[f"{nt}_avg"] = 0.9
    return {"pre_pt_root_id": pre, "post_pt_root_id": post, "neuropil": neuropil, "syn_count": syn, **probs}


def toy_rows():
    """Neurons 1..6: 1 ACh, 2 GABA, 3 glutamate, 4 dopamine (modulatory), 5 and 6 ACh."""
    return pd.DataFrame([
        row(1, 2, "MB_CA_R", 10, "ach"),
        row(1, 2, "LH_R", 3, "ach"),        # same pair in another neuropil: summed to 13
        row(2, 1, "MB_CA_R", 8, "gaba"),
        row(3, 1, "AL_R", 6, "glut"),
        row(4, 1, "MB_VL_R", 50, "da"),     # modulatory: dropped
        row(5, 6, "AL_R", 4, "ach"),        # below MIN_SYNAPSES: dropped
        row(6, 6, "AL_R", 20, "ach"),       # autapse: dropped
        row(1, 3, "MB_VL_R", 7, "ach"),
        row(6, 1, "AL_R", 9, "ach"),
    ])


def test_build_graph_threshold_sign_and_aggregation():
    g = cn.build_graph(toy_rows())
    pairs = g.pairs.set_index(["pre", "post"])
    assert pairs.loc[(1, 2), "syn"] == 13
    assert pairs.loc[(2, 1), "sign"] == -1
    assert pairs.loc[(3, 1), "sign"] == -1            # glutamate inhibitory
    assert pairs.loc[(1, 3), "sign"] == 1
    assert (4, 1) not in pairs.index and (5, 6) not in pairs.index and (6, 6) not in pairs.index
    assert g.neurons.loc[1, "nt"] == "ACH" and 4 not in g.neurons.index    # dopamine neuron has no edges left
    assert cn.neuron_table(toy_rows()).loc[4, "nt"] == "DA"
    assert g.neurons.loc[1, "syn_total"] == 13 + 7 + 8 + 6 + 9


def test_transmitter_is_synapse_weighted_majority():
    rows = pd.DataFrame([row(1, 2, "AL_R", 30, "ach"), row(1, 3, "AL_R", 10, "gaba")])
    assert cn.neuron_table(rows).loc[1, "nt"] == "ACH"


def test_top_subgraph_matrix_orientation_and_selection():
    g = cn.build_graph(toy_rows())
    sub = cn.top_subgraph(g, 3)
    assert sub.n == 3
    assert sub.nodes["root_id"].iloc[0] == 1                       # most synapses
    W = sub.matrix().toarray()
    node = dict(zip(sub.nodes["root_id"], sub.nodes["node"]))
    assert W[node[2], node[1]] == 13                                # W[post, pre]
    assert W[node[1], node[2]] == -8
    assert np.all(np.diag(W) == 0)


def test_region_restricts_edges_but_not_transmitters():
    g = cn.build_graph(toy_rows(), neuropils=cn.REGIONS["mb_right"])
    pairs = g.pairs.set_index(["pre", "post"])
    assert pairs.loc[(1, 2), "syn"] == 10                           # LH_R synapses excluded
    assert (3, 1) not in pairs.index and (6, 1) not in pairs.index
    assert set(g.neurons.index) == {1, 2, 3}


def test_subgraph_roundtrip(tmp_path):
    sub = cn.top_subgraph(cn.build_graph(toy_rows()), 4)
    sub.save(tmp_path)
    back = cn.Subgraph.load(tmp_path, sub.name)
    assert (back.matrix() != sub.matrix()).nnz == 0
    assert list(back.nodes["nt"]) == list(sub.nodes["nt"])
