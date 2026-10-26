import numpy as np
import pytest
import scipy.sparse as sp

from fly_forecast import reservoir as rs


def signed_graph(n=120, density=0.05, seed=1):
    """Synthetic 'connectome': sparse, heavy-tailed positive counts, a sign per presynaptic node (column)."""
    rng = np.random.default_rng(seed)
    W = sp.random(n, n, density=density, random_state=seed, data_rvs=lambda k: rng.pareto(1.5, k) + 5,
                  format="lil")
    W.setdiag(0)
    W = sp.csr_matrix(W)
    W.eliminate_zeros()
    sign = np.where(rng.random(n) < 0.3, -1.0, 1.0)
    return sp.csr_matrix(W @ sp.diags(sign)), sign


def degrees(W):
    A = sp.csr_matrix(W != 0, dtype=int)
    return np.asarray(A.sum(axis=1)).ravel(), np.asarray(A.sum(axis=0)).ravel()   # in (rows), out (columns)


@pytest.mark.parametrize("rho", [0.5, 0.9, 1.2])
def test_scaled_spectral_radius(rho):
    W, _ = signed_graph()
    S = rs.scale_to_radius(W, rho)
    assert np.abs(np.linalg.eigvals(S.toarray())).max() == pytest.approx(rho, rel=1e-9)


def test_sparse_eigensolver_matches_dense(monkeypatch):
    W, _ = signed_graph(n=300, density=0.03)
    dense = np.abs(np.linalg.eigvals(W.toarray())).max()
    monkeypatch.setattr(rs, "DENSE_EIG_MAX", 50)
    assert rs.spectral_radius(W) == pytest.approx(dense, rel=1e-6)


@pytest.mark.parametrize("name", [v for v in rs.VARIANTS if v != "no_recurrence"])
def test_variants_keep_size_and_density(name):
    W, _ = signed_graph()
    V = rs.make_variant(name, W, np.random.default_rng(0))
    assert V.shape == W.shape
    assert V.nnz == W.nnz
    assert V.diagonal().sum() == 0


def test_shuffled_keeps_topology_and_weight_multiset():
    W, _ = signed_graph()
    V = rs.make_variant("shuffled", W, np.random.default_rng(0))
    assert ((V != 0) != (W != 0)).nnz == 0
    assert np.allclose(np.sort(V.data), np.sort(W.data))
    assert not np.allclose(V.toarray(), W.toarray())


def test_random_topology_keeps_weight_multiset_only():
    W, _ = signed_graph()
    V = rs.make_variant("random_topology", W, np.random.default_rng(0))
    assert np.allclose(np.sort(V.data), np.sort(W.data))
    overlap = ((V != 0).multiply(W != 0)).nnz / W.nnz
    assert overlap < 0.2


def test_random_esn_weights_uniform():
    W, _ = signed_graph()
    V = rs.make_variant("random", W, np.random.default_rng(0))
    assert V.data.min() >= -1 and V.data.max() <= 1


def test_degree_preserving_keeps_degrees_strength_and_sign():
    W, sign = signed_graph(n=200, density=0.04)
    V = rs.make_variant("degree_preserving", W, np.random.default_rng(3))
    din, dout = degrees(W)
    vin, vout = degrees(V)
    assert np.array_equal(din, vin) and np.array_equal(dout, vout)
    # out-strength and sign of each presynaptic node (column) are unchanged
    assert np.allclose(np.asarray(W.sum(axis=0)).ravel(), np.asarray(V.sum(axis=0)).ravel())
    Vc = sp.csc_matrix(V)
    for j in range(V.shape[1]):
        col = Vc.data[Vc.indptr[j]:Vc.indptr[j + 1]]
        assert np.all(np.sign(col) == sign[j])
    # no self-loops, no duplicate edges, and the wiring really changed
    assert V.diagonal().sum() == 0
    assert V.nnz == W.nnz
    assert ((V != 0).multiply(W != 0)).nnz / W.nnz < 0.5


def test_no_recurrence_is_empty():
    W, _ = signed_graph()
    V = rs.make_variant("no_recurrence", W, np.random.default_rng(0))
    assert V.nnz == 0
    assert rs.scale_to_radius(V, 0.9).nnz == 0
