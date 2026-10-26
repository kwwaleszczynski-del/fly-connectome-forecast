import numpy as np
import pytest
import scipy.sparse as sp

from fly_forecast import esn
from fly_forecast.reservoir import scale_to_radius


def reservoir(n=200, seed=0, rho=0.9):
    rng = np.random.default_rng(seed)
    W = sp.random(n, n, density=0.05, random_state=seed, data_rvs=lambda k: rng.uniform(-1, 1, k))
    return scale_to_radius(W, rho)


def test_esn_predicts_sine_one_step_ahead():
    t = np.arange(3000)
    s = np.sin(2 * np.pi * t / 37) * np.cos(2 * np.pi * t / 101)
    U = np.column_stack([np.ones_like(s), s])
    W = reservoir()
    Win = esn.input_weights(W.shape[0], 2, 0.5, np.random.default_rng(1))
    F = esn.features(esn.run_reservoir(W, Win, U, leak=0.5), U)
    y = np.roll(s, -1)[:, None]
    train, test = np.arange(200, 2000), np.arange(2000, 2999)
    pred = esn.RidgePath.fit(F[train], y[train]).predict(F[test], 1e-8)
    nrmse = np.sqrt(np.mean((pred - y[test]) ** 2)) / y[test].std()
    assert nrmse < 0.01


def test_esn_remembers_past_input():
    """Recall u[t-5] from white noise: needs the recurrent memory, the readout of u[t] alone cannot do it."""
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, 4000)
    U = np.column_stack([np.ones_like(u), u])
    W = reservoir(rho=0.95)
    Win = esn.input_weights(W.shape[0], 2, 0.3, np.random.default_rng(1))
    F = esn.features(esn.run_reservoir(W, Win, U, leak=1.0), U)
    y = np.roll(u, 5)[:, None]
    train, test = np.arange(100, 3000), np.arange(3000, 4000)
    pred = esn.RidgePath.fit(F[train], y[train]).predict(F[test], 1e-8)
    r2 = 1 - np.mean((pred - y[test]) ** 2) / y[test].var()
    assert r2 > 0.9
    no_rec = esn.features(esn.run_reservoir(sp.csr_matrix(W.shape), Win, U, leak=1.0), U)
    pred0 = esn.RidgePath.fit(no_rec[train], y[train]).predict(no_rec[test], 1e-8)
    assert 1 - np.mean((pred0 - y[test]) ** 2) / y[test].var() < 0.1


def test_dense_and_sparse_state_update_agree(monkeypatch):
    W = reservoir(n=50)
    U = np.column_stack([np.ones(100), np.sin(np.arange(100) / 5)])
    Win = esn.input_weights(50, 2, 0.5, np.random.default_rng(2))
    monkeypatch.setattr(esn, "DENSE_ABOVE", 1.1)
    sparse = esn.run_reservoir(W, Win, U, 0.3)
    monkeypatch.setattr(esn, "DENSE_ABOVE", 0.0)
    dense = esn.run_reservoir(W, Win, U, 0.3)
    assert np.allclose(sparse, dense, atol=1e-6)


def test_ridge_path_matches_closed_form():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(80, 10)), rng.normal(size=(80, 3))
    alpha = np.array([1e-3, 1e-1, 1.0])
    path = esn.RidgePath.fit(X, Y)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    for k, a in enumerate(alpha):
        beta = np.linalg.solve(Xc.T @ Xc + a * len(X) * np.eye(10), Xc.T @ Yc[:, k])
        assert np.allclose(path.coef(alpha)[:, k], beta)
    assert path.predict(X, alpha).mean(0) == pytest.approx(Y.mean(0))


def test_participation_ratio():
    from fly_forecast.experiment import participation_ratio

    rng = np.random.default_rng(0)
    assert participation_ratio(rng.normal(size=(20000, 10))) == pytest.approx(10, rel=0.02)
    rank_one = np.outer(rng.normal(size=500), rng.normal(size=8))
    assert participation_ratio(rank_one) == pytest.approx(1, rel=1e-6)
