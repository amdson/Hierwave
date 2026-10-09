import itertools

import numpy as np
import pytest

from castlegen.legacy import connmetrics as cm
from castlegen.legacy import exemplar as ex
from castlegen.legacy import hier, tileset
from castlegen.legacy.quantities import connectivity as C


@pytest.fixture(scope="module")
def ts():
    return tileset.load("demo")


@pytest.fixture(scope="module")
def grids(ts):
    """Short Gibbs runs at a few temperatures (sparse and dense networks)."""
    out = []
    for seed, T in ((0, 0.6), (1, 1.0), (2, 2.0)):
        t0 = np.random.default_rng(seed).integers(0, ts.n_sig, (32, 32))
        out.append(ex.gibbs(ts, t0, 15, seed, T=T))
    return out


def _children(ts, t, y0, x0, k):
    return {(dy, dx): C.summarize(ts, t, y0 + dy * k, x0 + dx * k, k) for dy in (0, 1) for dx in (0, 1)}


def test_merge_is_exact(ts, grids):
    for t in grids:
        for k in (2, 4, 8, 16):
            for y0, x0 in itertools.product(range(0, 32, 2 * k), repeat=2):
                assert C.merge(_children(ts, t, y0, x0, k)) == C.summarize(ts, t, y0, x0, 2 * k)


def test_abstract_is_a_homomorphism(ts, grids):
    """abstract(merge) = merge_promises(abstract) on the side bits; when every
    child satisfies its promise the parent satisfies the merged promise
    exactly when the promise merge is consistent."""
    checked = 0
    for t in grids:
        for k in (1, 2, 4, 8):
            for y0, x0 in itertools.product(range(0, 32, 2 * k), repeat=2):
                ch = _children(ts, t, y0, x0, k)
                a = [C.abstract(ch[key]) for key in ((0, 0), (0, 1), (1, 0), (1, 1))]
                P, ok = C.merge_promises(*a)
                parent = C.merge(ch)
                assert C.abstract(parent) == P
                if all(C.satisfied(C.abstract(s), s) for s in ch.values()):
                    assert C.satisfied(P, parent) == ok
                    checked += 1
    assert checked > 50


def test_satisfied_is_monotone(ts, grids):
    """A promise is a lower bound: dropping open sides keeps it satisfied
    (except down to the empty promise, which also forbids nodes)."""
    for t in grids:
        for y0, x0 in itertools.product(range(0, 32, 4), repeat=2):
            s = C.summarize(ts, t, y0, x0, 4)
            a = C.abstract(s)
            if C.satisfied(a, s):
                for sub in range(1, 16):
                    if sub & ~a == 0:
                        assert C.satisfied(sub, s)


def _random_halo(rng, P):
    halo = {}
    for d, kids in C.EXT.items():
        for key in kids:
            if rng.random() < 0.6:
                halo[key + (d,)] = int(rng.integers(0, 2))
    return halo


def test_refine_matches_brute_force():
    rng = np.random.default_rng(0)
    for P in range(16):
        for halo in [None] + [_random_halo(rng, P) for _ in range(6)]:
            got = sorted(C.refine(P, halo))
            assert got == sorted(C.brute_refine(P, halo))
            for c in got:
                assert C.merge_promises(*c) == (P, True)
                kids = dict(zip(((0, 0), (0, 1), (1, 0), (1, 1)), c))
                assert all(C.bit(kids[dy, dx], d) == b for (dy, dx, d), b in (halo or {}).items())


def _promise_graph_connected(O):
    """Blocks with an open side are connected through open seams (torus)."""
    g0, g1 = O.shape
    live = [tuple(p) for p in np.argwhere(O != 0)]
    if not live:
        return True
    seen, stack = {live[0]}, [live[0]]
    while stack:
        y, x = stack.pop()
        for d, (dy, dx) in enumerate(C.DIRS):
            ny, nx = (y + dy) % g0, (x + dx) % g1
            if C.bit(O[y, x], d) and (ny, nx) not in seen:
                seen.add((ny, nx)); stack.append((ny, nx))
    return len(seen) == len(live)


@pytest.mark.parametrize("n,top", [(128, 64), (64, 64), (128, 32), (32, 32)])
def test_levels_consistent_and_connected(n, top):
    """Top grid side 2, 1, 4 and 1 (a block that is its own neighbour)."""
    for seed in range(3):
        v = C.ConnectivityPromise(K=16, h_top=top)
        ctx = hier.Ctx(seed=seed)
        hier.run([v], [], n, ctx)
        lv = {l.h: l.vars["conn"] for l in ctx.levels if "conn" in l.vars}
        assert sorted(lv) == [h for h in (16, 32, 64) if h <= top]
        for h, O in lv.items():
            assert C.consistent(O, lv.get(2 * h))
            assert _promise_graph_connected(O)
            assert (O != 0).any()


def test_init_children_is_consistent():
    rng = np.random.default_rng(1)
    v = C.ConnectivityPromise()
    for G in (1, 2, 4):
        for _ in range(20):
            bits = rng.integers(0, 2, (2, G, G))
            P = (np.roll(bits[1], 1, 0) << C.N) | (bits[0] << C.E) | (bits[1] << C.S) | (np.roll(bits[0], 1, 1) << C.W)
            kids = v.init_children(P, hier.Level(16, (2 * G, 2 * G)), None)
            assert C.consistent(kids, P)


def _random_promises(rng, g):
    bits = rng.integers(0, 2, (2, g, g))
    bits[:, rng.random((g, g)) < 0.2] = 0
    return (np.roll(bits[1], 1, 0) << C.N) | (bits[0] << C.E) | (bits[1] << C.S) | (np.roll(bits[0], 1, 1) << C.W)


@pytest.mark.parametrize("K", [8, 16])
def test_fulfil(ts, grids, K):
    """Random blocks, promises and halos: every block ends up satisfying its
    promise once all are fulfilled, and each fulfil edits only its block."""
    rng = np.random.default_rng(K)
    v = C.ConnectivityPromise(K=K)
    for t in grids:
        O = _random_promises(rng, t.shape[0] // K)
        tiles = t.copy()
        for by, bx in np.ndindex(*O.shape):
            new, e, cross = v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]), seed=3)
            diff = new != tiles
            assert diff.sum() == e
            inside = np.zeros(diff.shape, bool)
            inside[by * K:(by + 1) * K, bx * K:(bx + 1) * K] = True
            assert not (diff & ~inside).any()
            tiles = new
        for by, bx in np.ndindex(*O.shape):
            s = C.summarize(ts, tiles, by * K, bx * K, K)
            assert C.satisfied(int(O[by, bx]), s), (by, bx, O[by, bx], s)
        # a second pass changes nothing
        for by, bx in np.ndindex(*O.shape):
            assert v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]), seed=3)[1] == 0


def test_window_costs_match_states(ts, grids):
    t = grids[1]
    v = C.ConnectivityPromise()
    h = 8
    cost = v.window_costs(ts, t, h)
    for u in (0, 37, 500, 1023):
        cy, cx = divmod(u, 32)
        rolled = np.roll(t, (-(cy - h // 2) + 8, -(cx - h // 2) + 8), (0, 1))
        s = C.summarize(ts, rolled, 8, 8, h)
        assert np.array_equal(cost[u], [C.violations(o, s) for o in range(16)])
        assert all((cost[u][o] == 0) == C.satisfied(o, s) for o in range(16))
