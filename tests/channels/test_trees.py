import itertools

import numpy as np
import pytest

from castlegen.channels import train, trees
from castlegen.channels.aistargets import AISTargets
from castlegen.channels.trees import CH, Bench, Forward, Oracle, Trees, chain_logz, chain_sample


def brute_logz(U, c):
    n, S = U.shape
    tot = []
    for s in itertools.product(range(S), repeat=n):
        e = sum(U[j, s[j]] for j in range(n)) + c * sum(abs(s[j] - s[j + 1]) for j in range(n - 1))
        tot.append(-e)
    tot = np.asarray(tot)
    m = tot.max()
    return m + np.log(np.exp(tot - m).sum())


def test_chain_logz_matches_enumeration():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(4, 3))
    U[2, :] = np.inf
    U[2, 1] = 0.3
    assert chain_logz(U, 0.7) == pytest.approx(brute_logz(U, 0.7), abs=1e-10)


def test_chain_sample_marginals():
    rng = np.random.default_rng(1)
    U = np.array([[0.0, 1.0, 2.0], [0.5, 0.0, 0.5], [2.0, 1.0, 0.0]])
    c = 0.8
    n = 20000
    draws = np.stack([chain_sample(U, c, rng) for _ in range(n)])
    # exact marginal of column 0 by enumeration
    S = 3
    w = np.zeros(S)
    for s in itertools.product(range(S), repeat=3):
        e = sum(U[j, s[j]] for j in range(3)) + c * (abs(s[0] - s[1]) + abs(s[1] - s[2]))
        w[s[0]] += np.exp(-e)
    p = w / w.sum()
    emp = np.bincount(draws[:, 0], minlength=S) / n
    assert np.abs(emp - p).max() < 0.02


def test_forward_state_valid():
    T = Trees(n=6, margin=1, b=1.0)
    fw = Forward(T, seed=0)
    st = fw.run(20, 20)
    v = T.violations(fw.chans)
    assert v == dict(pin=0, fill=0, trunk=0), v
    assert 0.0 <= st["present"] <= 1.0
    # margin and other slot rows stay absent
    g = fw.tree.grid
    assert (g[:, :T.J0] == 0).all() and (g[:, T.J1:] == 0).all()
    assert (np.delete(g, T.srow, axis=0) == 0).all()
    # tiles: soil below the surface, a trunk above its column's surface
    sr = T.surface_rows(fw.surf)
    for j in np.flatnonzero(g[T.srow] > 0):
        t = g[T.srow, j]
        y, x = T.R0 + T.rec_r[t], j * T.BT + T.rec_c[t]
        assert fw.tile.grid[y, x] == trees.TRUNK and fw.tile.grid[y + 1, x] == trees.SOIL
        assert sr[x // CH] == y + 1


def test_forward_two_rows_fill_rule():
    T = Trees(n=4, margin=1, BT=16, R0=16, RB=16)
    fw = Forward(T, seed=3)
    fw.run(15, 30)
    v = T.violations(fw.chans)
    assert v == dict(pin=0, fill=0, trunk=0), v
    assert T.heights(fw.surf).max() <= T.BT


def test_oracle_sweeps_valid():
    T = Trees(n=6, margin=1)
    orc = Oracle(T, seed=0)
    orc.sweep(5)
    assert T.violations(orc.chans) == dict(pin=0, fill=0, trunk=0)
    m = orc.moments(2, 5)
    assert set(m) >= {"present", "h_hist", "p1_both", "terr_rough"}


def test_hook_matches_ais_exact_split():
    """The transfer-matrix window log Z against the AIS estimate over the
    surf sampler on the same window (the pins in p_0, smooth annealed)."""
    T = Trees(n=4, margin=1, reach=1)
    B = Bench(T, seed=0, S=(10, 20))
    B.fw.run(10, 20)
    model = B.fw.model
    J = T.J0 + 1
    cand = np.array([0, 5, 20, 40])
    tg_hook = AISTargets("surf", B.region, B.designed_e, B.repaint, hook=B.hook)
    tg_ais = AISTargets("surf", B.region, B.designed_e, B.repaint, K=64, M=64, L=30.0, seed=1)
    lz_h, _ = tg_hook.log_z(model, "tree", T.srow, J, cand)
    lz_a, se = tg_ais.log_z(model, "tree", T.srow, J, cand)
    d = (lz_h - lz_h[0]) - (lz_a - lz_a[0])
    assert np.abs(d).max() < 0.25, (lz_h, lz_a, se)


def test_fit_smoke():
    T = Trees(n=6, margin=1)
    B = Bench(T, seed=0, S=(10, 15))
    tg = AISTargets("surf", B.region, B.designed_e, B.repaint, hook=B.hook)
    theta, rep = train.fit(B.contexts, "tree", tg, B.feats, B.designed_e, iters=1, n_contexts=2, K=1,
                           after_change=B.move, repaint=B.repaint, n_pairs=1, rng=0)
    assert rep["nfeat"] == sum(int(np.prod(s)) for s in T.shapes().values())
    assert np.isfinite(rep["kl_train"][-1])
    tabs = B.tables_of(theta)
    fw = Forward(T, tables=tabs, seed=5)
    fw.run(10, 15)
    assert T.violations(fw.chans) == dict(pin=0, fill=0, trunk=0)
