import itertools

import numpy as np
import pytest

from castlegen import exemplar as ex
from castlegen import hier, tileset
from castlegen.quantities import flow as F


@pytest.fixture(scope="module")
def ts():
    return tileset.load("wilds")


@pytest.fixture(scope="module")
def grids(ts):
    """Short Gibbs runs from random oriented grids (lots of messy water)."""
    out = []
    for seed, T in ((0, 0.6), (1, 1.0), (2, 2.0)):
        t0 = np.random.default_rng(seed).integers(0, ts.n_sig, (32, 32))
        out.append(ex.gibbs(ts, t0, 15, seed, T=T))
    return out


def _children(ts, t, y0, x0, k):
    return {(dy, dx): F.summarize(ts, t, y0 + dy * k, x0 + dx * k, k) for dy in (0, 1) for dx in (0, 1)}


def test_tileset_flow_attribute(ts):
    rivers = [s for s in range(ts.n_sig) if ts.kinds[ts.sig_kind[s]].name == "river"]
    assert sorted(ts.flow_dir[rivers]) == [0, 1, 2, 3]                 # four rotations, four signatures
    lake = [s for s in range(ts.n_sig) if ts.kinds[ts.sig_kind[s]].name == "lake"]
    assert ts.lake[lake].all() and (ts.flow_dir[lake] == -1).all()
    node = np.asarray(ts.np_tables["is_room"])
    bridges = [s for s in range(ts.n_sig) if "bridge" in ts.kinds[ts.sig_kind[s]].tags]
    assert ts.water[bridges].all() and node[bridges].all() and (ts.flow_dir[bridges] >= 0).all()
    e = rivers[ts.flow_dir[rivers].tolist().index(1)]                  # flowing east
    assert ts.np_tables["Eh"][e, e] < ts.np_tables["Eh"][e, ts.WALL]    # into water beats into dry land


def test_merge_is_exact(ts, grids):
    for t in grids:
        for k in (2, 4, 8, 16):
            for y0, x0 in itertools.product(range(0, 32, 2 * k), repeat=2):
                assert F.merge(_children(ts, t, y0, x0, k)) == F.summarize(ts, t, y0, x0, 2 * k)


def test_abstract_is_a_homomorphism(ts, grids):
    checked = 0
    for t in grids:
        for k in (1, 2, 4, 8):
            for y0, x0 in itertools.product(range(0, 32, 2 * k), repeat=2):
                ch = _children(ts, t, y0, x0, k)
                a = [F.abstract(ch[key]) for key in F.KIDS]
                P, ok = F.merge_promises(*a)
                parent = F.merge(ch)
                if all(np.abs(F._net(s)[0]).max() <= F.FMAX for s in ch.values()):
                    assert F.abstract(parent) == P
                if all(F.satisfied(F.abstract(s), s) for s in ch.values()):
                    assert F.satisfied(P, parent) == ok
                    checked += 1
    assert checked > 50


def _full_halo(rng, P):
    """A random halo consistent with P on every external half-seam."""
    halo = {}
    for d in range(4):
        a, b = F.splits(F.flux(P)[d])[rng.integers(len(F.splits(F.flux(P)[d])))]
        halo[F.EXT[d][0] + (d,)], halo[F.EXT[d][1] + (d,)] = a, b
    return halo


def test_refine_matches_brute_force():
    rng = np.random.default_rng(0)
    for P in rng.integers(0, F.V, 12):
        P = int(P)
        for i in range(3):
            halo = _full_halo(rng, P)
            if i == 2:                                                   # leave one side open
                d = int(rng.integers(4))
                halo = {k: v for k, v in halo.items() if k[2] != d}
            got = sorted(F.refine(P, halo))
            assert got == sorted(F.brute_refine(P, halo))
            assert len(got) == F.n_refine(P, halo)
            for c in got:
                assert F.merge_promises(*c) == (P, True)
        assert len(F.refine(P)) == F.n_refine(P)


@pytest.mark.parametrize("n,top", [(128, 64), (64, 64), (32, 32)])
def test_levels_consistent(n, top):
    for seed in range(2):
        v = F.FlowPromise(K=16, h_top=top)
        ctx = hier.Ctx(seed=seed)
        hier.run([v], [], n, ctx)
        lv = {l.h: l.vars["flow"] for l in ctx.levels if "flow" in l.vars}
        for h, O in lv.items():
            assert F.consistent(O, lv.get(2 * h))


def _random_promises(rng, g):
    sm = rng.integers(-2, 3, (2, g, g))
    sm[:, rng.random((g, g)) < 0.3] = 0
    return F.encode(np.stack([-np.roll(sm[1], 1, 0), sm[0], sm[1], -np.roll(sm[0], 1, 1)], -1))


def _village(ts, tiles, y, x):
    grid = dict(ts.structures)["village"]
    for r, c in np.ndindex(*grid.shape):
        tiles[y + r, x + c] = ex.sig_by_name(ts, ts.kinds[grid[r, c]].name)


@pytest.mark.parametrize("K", [8, 16])
def test_fulfil(ts, grids, K):
    """Random blocks and promises: every block satisfies its promise once all
    are fulfilled, edits stay in the block, structures are untouched, the
    flow rule holds everywhere and a second pass is a no-op."""
    rng = np.random.default_rng(K)
    v = F.FlowPromise(K=K)
    for t in grids:
        tiles = t.copy()
        _village(ts, tiles, 5, 5)
        _village(ts, tiles, 20, 13)
        struct = F.tables(ts)["struct"][tiles]
        before = tiles.copy()
        O = _random_promises(rng, t.shape[0] // K)
        for by, bx in np.ndindex(*O.shape):
            new, e, _ = v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]), seed=3)
            diff = new != tiles
            assert diff.sum() == e
            inside = np.zeros(diff.shape, bool)
            inside[by * K:(by + 1) * K, bx * K:(bx + 1) * K] = True
            assert not (diff & ~inside).any()
            tiles = new
        assert np.array_equal(tiles[struct], before[struct])
        for by, bx in np.ndindex(*O.shape):
            assert F.satisfied(int(O[by, bx]), F.summarize(ts, tiles, by * K, bx * K, K))
        assert F.flow_violations(ts, tiles) == {"dead_ends": 0, "unfed": 0}
        for by, bx in np.ndindex(*O.shape):
            assert v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]), seed=3)[1] == 0


def test_make_valid_is_consistent_at_every_level(ts, grids):
    t = np.tile(grids[1], (2, 2))                                       # 64 x 64
    tiles, O = F.make_valid(ts, t, 16, seed=1)
    assert F.flow_violations(ts, tiles) == {"dead_ends": 0, "unfed": 0}
    S = [[F.summarize(ts, tiles, y * 16, x * 16, 16) for x in range(4)] for y in range(4)]
    assert all(F.satisfied(int(O[y, x]), S[y][x]) for y in range(4) for x in range(4))
    g = 4
    while g > 1:
        g //= 2
        S = [[F.merge({(dy, dx): S[2 * y + dy][2 * x + dx] for dy in (0, 1) for dx in (0, 1)})
              for x in range(g)] for y in range(g)]
        assert all(F.satisfied(F.abstract(s), s) for row in S for s in row)


def test_exemplar_keeps_the_flow_rule(ts):
    from castlegen import pipeline
    layout = ex.load_layout("wilds")
    tiles, clamp, _ = ex.place(ts, layout)
    dead, unfed = F._cell_status(ts, tiles)
    assert not ((dead | unfed) & clamp).any()                         # the placed rivers keep the rule
    E = pipeline.exemplar_for(dict(layout, name="wilds"), ts, log=lambda *a: None)   # cached build
    assert F.flow_violations(ts, E) == {"dead_ends": 0, "unfed": 0}


def test_window_costs_match_states(ts, grids):
    t = grids[1]
    v = F.FlowPromise()
    h = 8
    cost = v.window_costs(ts, t, h)
    for u in (0, 37, 500, 1023):
        cy, cx = divmod(u, 32)
        rolled = np.roll(t, (-(cy - h // 2) + 8, -(cx - h // 2) + 8), (0, 1))
        s = F.summarize(ts, rolled, 8, 8, h)
        assert np.array_equal(cost[u], [F.violations(o, s) for o in range(F.V)])
        assert all((cost[u][o] == 0) == F.satisfied(o, s) for o in range(0, F.V, 7))


MINI = {"name": "wilds-promise-mini", "tileset": "wilds", "size": 64, "seed": 1,
        "exemplar": {"tileset": "wilds", "size": [32, 32], "sweeps": 40, "T": 0.6, "clean": True, "flow": True,
                     "connect": {"tag": "road", "void": "lake"},
                     "place": [{"stroke": "river", "points": [[8, 0], [8, 31]], "width": 2},
                               {"stroke": "river", "points": [[0, 20], [31, 20]], "width": 1},
                               {"disc": "lake", "at": [24, 8], "radius": 2},
                               {"structure": "village", "at": [14, 4]}]},
        "stages": [{"stage": "hier", "K": 16, "h_top": 32, "tables": "flat", "quantities": ["flow", "connectivity"],
                    "r": [1, 1, 0, 0, 0, 0],
                    "connectivity": {"tag": "road", "cross_socket": "land", "void": "lake"}},
                   {"stage": "tiles"},
                   {"stage": "repair", "rounds": 5},
                   {"stage": "fulfil"},
                   {"stage": "repair", "rounds": 5, "no_new_water": True},
                   {"stage": "fulfil"}]}


def test_end_to_end_mini_wilds(tmp_path, monkeypatch):
    from castlegen import pipeline as pl
    monkeypatch.setattr(pl, "CACHE_DIR", str(tmp_path))
    ctxs = []
    ts, tiles, report = pl.run(MINI, log=lambda *a: None, ctx_out=ctxs)
    last = report[-1]
    assert (last["dead_ends"], last["unfed"]) == (0, 0)
    assert last["main_share"] == 1.0
    assert all(f == 1.0 for f in last["sat"].values())                  # bottom-up: every level keeps its promise
    ctx = ctxs[0]
    for v in ctx.cache["promises"]:                                    # promise grids equal the recomputed states
        for lv in ctx.cache["levels"]:
            if v.name in lv.vars:
                S = v.level_states(ts, tiles, lv.h)
                O = lv.vars[v.name]
                assert all(v.satisfied(int(O[y, x]), S[y][x]) for y, x in np.ndindex(*O.shape))
    _, again, _ = pl.run(MINI, log=lambda *a: None)
    assert np.array_equal(tiles, again)                                # deterministic for a seed
