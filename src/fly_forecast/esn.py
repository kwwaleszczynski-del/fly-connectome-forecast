"""Leaky echo state network with a fixed reservoir matrix and a ridge readout.

State update (Jaeger 2001; leaky form as in Costi et al. 2025), input u_t includes a constant 1 (bias):
    x_t = (1 - a) x_{t-1} + a tanh(W x_{t-1} + W_in u_t)
Readout: ridge regression of the targets on [x_t, u_t]; the penalty is alpha x (number of training rows), so alpha is
per sample and comparable between training windows of different length.

Usage:
    Win = input_weights(n=W.shape[0], d=U.shape[1], scale=0.5, rng=np.random.default_rng(0))
    X = run_reservoir(W, Win, U, leak=0.3)
    path = RidgePath.fit(features(X, U)[train], Y[train])
    Y_hat = path.predict(features(X, U)[test], alpha=1e-4)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

DENSE_ABOVE = 0.08   # use a dense matrix-vector product when the reservoir is denser than this


def input_weights(n: int, d: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1, 1, size=(n, d)) * scale


def run_reservoir(W: sp.spmatrix, Win: np.ndarray, U: np.ndarray, leak: float) -> np.ndarray:
    """Reservoir states for the whole input sequence U (T x d), starting from x = 0; returns float32 (T x n)."""
    n, T = W.shape[0], U.shape[0]
    drive = U @ Win.T
    Wop = W.toarray() if W.nnz > DENSE_ABOVE * n * n else sp.csr_matrix(W)
    X = np.empty((T, n), dtype=np.float32)
    x = np.zeros(n)
    for t in range(T):
        x = (1 - leak) * x + leak * np.tanh(Wop @ x + drive[t])
        X[t] = x
    return X


def features(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    return np.hstack([X, U.astype(np.float32)])


@dataclass
class RidgePath:
    """Ridge solutions for any alpha from one eigendecomposition of the centred Gram matrix."""
    x_mean: np.ndarray
    y_mean: np.ndarray
    Q: np.ndarray        # eigenvectors of Xc'Xc
    lam: np.ndarray      # eigenvalues
    QtXtY: np.ndarray    # Q' Xc' Yc
    rows: int

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "RidgePath":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
        Xc = X - x_mean
        lam, Q = np.linalg.eigh(Xc.T @ Xc)
        return cls(x_mean, y_mean, Q, np.clip(lam, 0, None), Q.T @ (Xc.T @ (Y - y_mean)), len(X))

    def coef(self, alpha: float | np.ndarray) -> np.ndarray:
        """Coefficients (p x k); ``alpha`` is a scalar or one value per target column."""
        a = np.broadcast_to(np.asarray(alpha, dtype=float), (self.QtXtY.shape[1],))
        return self.Q @ (self.QtXtY / (self.lam[:, None] + a[None, :] * self.rows))

    def predict(self, X: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
        # a nearly unregularised fit on a short window can overflow; callers score inf / nan as a failed forecast
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return (np.asarray(X, dtype=np.float64) - self.x_mean) @ self.coef(alpha) + self.y_mean
