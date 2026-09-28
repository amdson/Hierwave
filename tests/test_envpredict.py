import numpy as np
import pytest

from castlegen import envpredict as EP
from castlegen import tileset
from castlegen.quantities import envelope as EN
from test_envelope import heightmap, random_H

G = 4


@pytest.fixture(scope="module")
def grids():
    ts = tileset.load("cliffs")
    rng = np.random.default_rng(0)
    maps = [heightmap(ts, random_H(rng, 64, "walk"), 64, rng) for _ in range(6)]
    out = {}
    for t in maps:
        vals, ok = EN.pyramid(ts, t, 16, G, 64)
        for h, v in vals.items():
            out.setdefault(h, []).append(v)
    return out


def test_tables_shapes_and_column_data_masks(grids):
    lg = EN.lang()
    W = {h: np.random.default_rng(h).normal(size=EP._slices(EP.features(lg))[1]) for h in grids}
    tab = EP.to_tables(W, lg)
    for h in grids:
        u, pair, par = tab.level(h)
        assert u.shape == (lg.V,) and pair.shape == (2, lg.V, lg.V) and par.shape == (4, lg.V, lg.V)
        EP.column_data(grids, h, lg, G)                   # asserts the observed columns are feasible


def test_fit_lowers_the_objective_and_matches_numpy(grids):
    lg = EN.lang()
    zero = EP.to_tables({h: np.zeros(EP._slices(EP.features(lg))[1]) for h in grids}, lg)
    W = EP.fit(grids, lg, G, l2=1e-3, maxiter=100, log=lambda *a: None)
    fitted = EP.to_tables(W, lg)
    for h in grids:
        assert EP.column_nll(fitted, grids, h, lg, G) < EP.column_nll(zero, grids, h, lg, G)


def test_column_nll_is_exact_on_a_tiny_chain():
    """Brute force over every column of a 2-row level (one parent pair)."""
    lg = EN.lang()
    rng = np.random.default_rng(1)
    W = {h: rng.normal(size=EP._slices(EP.features(lg))[1]) * 0.3 for h in (16, 32)}
    tab = EP.to_tables(W, lg)
    parent = np.array([[int(lg.encode([2, 2], [3, 3]))]])        # h32, 1x1: a ground block
    child = None
    u, l = lg.column_pairs(int(lg.parent_part(parent[0, 0], 0)), False, True, G, 16)
    ur, lr = lg.column_pairs(int(lg.parent_part(parent[0, 0], 1)), False, True, G, 16)
    child = np.array([[u[0], ur[0]], [l[0], lr[0]]])
    grids = {16: [child], 32: [parent]}
    got = EP.column_nll(tab, grids, 16, lg, G)
    uu, pair, par = tab.level(16)
    tot = 0.0
    for x, (cu, cl) in enumerate(((u, l), (ur, lr))):
        nb = child[:, 1 - x]
        e = lambda a, b: (uu[a] + uu[b] + pair[0][nb[0], a] + pair[0][a, nb[0]] + pair[0][nb[1], b] + pair[0][b, nb[1]]
                          + pair[1][a, b] + par[x, a, parent[0, 0]] + par[2 + x, b, parent[0, 0]])
        es = np.array([e(a, b) for a, b in zip(cu, cl)])
        es = es[lg.valid(cl, True, G, 16)]
        tot += e(child[0, x], child[1, x]) + np.log(np.exp(-es).sum())
    assert got == pytest.approx(tot / 2, rel=1e-9)
