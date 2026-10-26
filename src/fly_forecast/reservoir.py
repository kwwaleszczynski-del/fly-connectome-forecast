"""Reservoir matrices: the connectome subgraph and the null models it is compared with.

All matrices are scipy.sparse CSR with W[post, pre]. Every variant has the same size n and the same number of
non-zero entries E as the connectome matrix it is derived from; each is scaled to the same spectral radius before use.

Variants (topology x weights, after Costi et al. 2025, plus a degree-preserving control):
    connectome          real topology, real signed synapse counts
    random              Erdos-Renyi topology with E edges, weights U(-1, 1)            (classic ESN)
    shuffled            real topology, real weights randomly permuted over the edges
    random_topology     Erdos-Renyi topology with E edges, real weights randomly assigned
    topology_random_w   real topology, weights U(-1, 1)
    degree_preserving   real in- and out-degree of every neuron, edges rewired by double-edge swaps; each edge keeps
                        the weight of its presynaptic neuron, so out-strength and sign per neuron (Dale) are kept
    no_recurrence       W = 0: the leaky input filter alone (is recurrence needed at all?)

Usage:
    W = make_variant("degree_preserving", Wc, np.random.default_rng(0))
    W = scale_to_radius(W, 0.9)
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

log = logging.getLogger(__name__)

VARIANTS = ["connectome", "random", "shuffled", "random_topology", "topology_random_w", "degree_preserving",
            "no_recurrence"]
DENSE_EIG_MAX = 3000


def spectral_radius(W: sp.spmatrix) -> float:
    """Largest absolute eigenvalue; dense LAPACK for small matrices, ARPACK otherwise."""
    if W.nnz == 0:
        return 0.0
    if W.shape[0] <= DENSE_EIG_MAX:
        return float(np.abs(np.linalg.eigvals(W.toarray())).max())
    try:
        vals = spla.eigs(W.astype(float), k=4, which="LM", return_eigenvectors=False, maxiter=W.shape[0] * 50,
                         tol=1e-8)
        return float(np.abs(vals).max())
    except spla.ArpackNoConvergence as e:
        log.warning("ARPACK did not converge (%s), falling back to dense eigvals", e)
        return float(np.abs(np.linalg.eigvals(W.toarray())).max())


def scale_to_radius(W: sp.spmatrix, rho: float) -> sp.csr_matrix:
    r = spectral_radius(W)
    if r == 0:
        return sp.csr_matrix(W, dtype=float)
    return sp.csr_matrix(W * (rho / r), dtype=float)


def _random_positions(n: int, n_edges: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n_edges distinct off-diagonal (post, pre) positions, uniformly at random."""
    if n_edges > n * (n - 1):
        raise ValueError("more edges than off-diagonal positions")
    keys = np.empty(0, dtype=np.int64)
    while len(keys) < n_edges:
        cand = rng.integers(0, n * n, size=2 * (n_edges - len(keys)) + 16)
        cand = cand[cand // n != cand % n]
        keys = np.unique(np.concatenate([keys, cand]))
    keys = rng.permutation(keys)[:n_edges]
    return keys // n, keys % n


def _coo(W: sp.spmatrix) -> sp.coo_matrix:
    c = sp.coo_matrix(W)
    c.sum_duplicates()
    c.eliminate_zeros()
    return c


def degree_preserving_rewire(W: sp.spmatrix, rng: np.random.Generator, swaps_per_edge: float = 10.0,
                             max_rounds: int = 200) -> sp.csr_matrix:
    """Directed double-edge swaps a->b, c->d  =>  a->d, c->b (vectorised in rounds of disjoint edge pairs).

    Swaps that would create a self-loop or an existing/duplicate edge are rejected. Each edge keeps its weight with
    the presynaptic node a, so in-degree, out-degree, out-strength and the sign of every node are unchanged.
    """
    c = _coo(W)
    n = c.shape[0]
    post, pre, w = c.row.astype(np.int64), c.col.astype(np.int64), c.data.copy()
    E = len(w)
    if E < 2:
        return sp.csr_matrix(c)
    target, done = int(swaps_per_edge * E), 0
    for _ in range(max_rounds):
        if done >= target:
            break
        perm = rng.permutation(E)
        i, j = perm[0:E - E % 2:2], perm[1:E - E % 2:2]
        # new edges: pre[i] -> post[j] and pre[j] -> post[i]
        k1, k2 = post[j] * n + pre[i], post[i] * n + pre[j]
        ok = (pre[i] != post[j]) & (pre[j] != post[i]) & (k1 != k2)
        existing = post * n + pre
        ok &= ~np.isin(k1, existing) & ~np.isin(k2, existing)
        new = np.concatenate([k1[ok], k2[ok]])
        uniq, counts = np.unique(new, return_counts=True)
        dup = uniq[counts > 1]
        ok[ok] &= ~np.isin(k1[ok], dup) & ~np.isin(k2[ok], dup)
        i, j = i[ok], j[ok]
        post[i], post[j] = post[j].copy(), post[i].copy()
        done += len(i)
    if done < target:
        log.warning("degree-preserving rewiring: %d of %d swaps", done, target)
    return sp.csr_matrix((w, (post, pre)), shape=(n, n))


def make_variant(name: str, W: sp.spmatrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Variant ``name`` of the (signed, weighted) connectome matrix ``W``, before spectral scaling."""
    c = _coo(W)
    n, E = c.shape[0], c.nnz
    if name == "connectome":
        return sp.csr_matrix(c)
    if name == "no_recurrence":
        return sp.csr_matrix((n, n))
    if name == "shuffled":
        return sp.csr_matrix((rng.permutation(c.data), (c.row, c.col)), shape=(n, n))
    if name == "topology_random_w":
        return sp.csr_matrix((rng.uniform(-1, 1, E), (c.row, c.col)), shape=(n, n))
    if name in ("random", "random_topology"):
        rows, cols = _random_positions(n, E, rng)
        data = rng.uniform(-1, 1, E) if name == "random" else rng.permutation(c.data)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    if name == "degree_preserving":
        return degree_preserving_rewire(c, rng)
    raise ValueError(f"unknown variant {name}")
