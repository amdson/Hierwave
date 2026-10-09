"""Pseudo-likelihood fit (castlegen.legacy.plfit) on synthetic connectivity promise
pyramids drawn from a known refinement model."""
import numpy as np
import pytest

from castlegen.legacy import plfit
from castlegen.legacy.quantities import connectivity as C

KIDS = plfit.KIDS


def _halo_set(grid, done, py, px):
    """Halo from neighbouring children that are already assigned."""
    g0, g1 = grid.shape
    out = {}
    for dy, dx in KIDS:
        y, x = 2 * py + dy, 2 * px + dx
        for d, (sy, sx) in enumerate(C.DIRS):
            if (d == C.N and dy == 0) or (d == C.S and dy == 1) or (d == C.W and dx == 0) or (d == C.E and dx == 1):
                ny, nx = (y + sy) % g0, (x + sx) % g1
                if done[ny, nx]:
                    out[dy, dx, d] = int(C.bit(grid[ny, nx], C.OPP[d]))
    return out


def n_open(kids):
    return sum(bin(k).count("1") for k in kids)


def sample_pyramid(rng, top_side=2, levels=2, beta=0.0, h_top=64):
    """{h: grid}: top grid all-open, each level refined parent by parent with
    P(refinement) ∝ exp(-beta * open sides of the four children)."""
    grids = {h_top: np.full((top_side, top_side), 15, np.int64)}
    h = h_top
    for _ in range(levels):
        G = grids[h]
        g = np.zeros((2 * G.shape[0], 2 * G.shape[1]), np.int64)
        done = np.zeros(g.shape, bool)
        for py, px in np.ndindex(*G.shape):
            cs = C.refine(int(G[py, px]), _halo_set(g, done, py, px))
            assert cs, "no consistent refinement"
            w = np.exp(-beta * np.array([n_open(c) for c in cs], float))
            k = rng.choice(len(cs), p=w / w.sum())
            for (dy, dx), v in zip(KIDS, cs[k]):
                g[2 * py + dy, 2 * px + dx] = v
                done[2 * py + dy, 2 * px + dx] = True
        h //= 2
        grids[h] = g
        assert C.consistent(g, G)
    return grids


def corpus(seed, n, beta):
    rng = np.random.default_rng(seed)
    ps = [sample_pyramid(rng, beta=beta) for _ in range(n)]
    return {h: [p[h] for p in ps] for h in ps[0]}


def test_batch_contains_observed_refinements():
    grids = corpus(0, 3, 0.0)
    b = plfit.build(grids[16], grids[32], C.refine, plfit_halo)
    assert b.skipped == 0 and b.cands.shape[0] == 3 * 16
    assert b.mask[np.arange(len(b.obs)), b.obs].all()


def plfit_halo(grid, py, px):
    return _halo_set(grid, np.ones(grid.shape, bool), py, px)


@pytest.mark.parametrize("features", ["conn", "onehot"])
def test_fit_recovers_bias(features):
    F = plfit.FEATURES[features](16)
    train, held = corpus(1, 8, 1.5), corpus(2, 3, 1.5)
    tab, rep = plfit.fit(train, C.refine, plfit_halo, F, l2=(0.01, 0.1, 1.0), holdout=held, log=lambda *a: None)
    r = rep[16]
    assert r["skipped"] == 0
    assert r["fitted"] < r["flat"] - 0.3            # the data are far from uniform
    assert r["heldout"] < r["heldout_flat"] - 0.2   # and that generalises
    u, pair, par = tab.level(16)
    assert u.shape == (16,) and pair.shape == (2, 16, 16) and par.shape == (4, 16, 16)
    opens = np.array([bin(v).count("1") for v in range(16)])
    assert np.corrcoef(u, opens)[0, 1] > 0.5        # more open sides -> higher energy


def test_unbiased_data_does_not_overfit():
    """beta = 0 picks uniformly given the partial (raster-order) halo, which is
    close to, not exactly, uniform given the full halo; the fit must not do
    worse than flat tables on held-out pyramids."""
    F = plfit.conn_features()
    train, held = corpus(3, 8, 0.0), corpus(4, 3, 0.0)
    _, rep = plfit.fit(train, C.refine, plfit_halo, F, l2=(0.1, 1.0), holdout=held, log=lambda *a: None)
    r = rep[16]
    assert r["heldout"] <= r["heldout_flat"] + 0.05
    assert abs(r["fitted"] - r["flat"]) < 0.3
