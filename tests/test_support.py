import itertools

import numpy as np
import pytest

from castlegen import exemplar as ex
from castlegen import hier, tileset
from castlegen.quantities import support as S


@pytest.fixture(scope="module")
def ts():
    return tileset.load("cliffs")


@pytest.fixture(scope="module")
def grids(ts):
    """Random solid / air grids (5 material signatures) and short Gibbs runs (blobs)."""
    rng = np.random.default_rng(0)
    out = [rng.integers(0, 5, (32, 32)).astype(np.int32)]
    for seed, T in ((0, 0.9), (1, 1.2)):
        out.append(ex.gibbs(ts, rng.integers(0, 5, (32, 32)), 15, seed, T=T))
    return out


def _children(ts, t, y0, x0, k, G):
    return {(dy, dx): S.summarize(ts, t, y0 + dy * k, x0 + dx * k, k, G) for dy in (0, 1) for dx in (0, 1)}


def test_tileset_solid_and_directional_rules(ts):
    names = [ts.sig_name(s).split("@")[0] for s in range(ts.n_sig)]
    assert [names[s] for s in np.flatnonzero(ts.solid)] == ["stone", "soil", "turf", "ore", "wall"]
    air, turf, stone = (names.index(k) for k in ("air", "turf", "stone"))
    Ev = ts.np_tables["Ev"]
    assert Ev[air, turf] < Ev[turf, air]                                # grass wants sky above, not below
    assert Ev[stone, turf] > 1.9                                        # turf under matter is a hard violation


@pytest.mark.parametrize("G", [0, 4, 8])
def test_merge_is_exact(ts, grids, G):
    for t in grids:
        for k in (2, 4, 8):
            for y0, x0 in itertools.product(range(0, 32, 2 * k), repeat=2):
                assert S.merge(_children(ts, t, y0, x0, k, G)) == S.summarize(ts, t, y0, x0, 2 * k, G)
    whole = S.summarize(ts, grids[1], 0, 0, 32, 0)
    assert whole.viol + int((whole.need & ~whole.top).sum()) == S.support_violations(ts, grids[1], 0)   # + wrap seam


@pytest.mark.parametrize("G", [0, 8])
def test_abstract_is_a_homomorphism(ts, grids, G):
    checked = 0
    for t in grids + [S.fix(ts, g, 8) for g in grids]:
        for k in (2, 4, 8):
            for y0, x0 in itertools.product(range(0, 32, 2 * k), repeat=2):
                ch = _children(ts, t, y0, x0, k, G)
                a = [S.abstract(ch[key]) for key in S.KIDS]
                P, ok = S.merge_promises(*a)
                parent = S.merge(ch)
                assert S.abstract(parent) == P
                for s in list(ch.values()) + [parent]:
                    assert S.satisfied(S.abstract(s), s) == (s.viol == 0)     # blocks keep their own abstraction
                    assert S.allowed(S.abstract(s), s.ground) or s.viol > 0
                if ok and all(S.satisfied(v, ch[key]) for v, key in zip(a, S.KIDS)):
                    assert S.satisfied(P, parent)                             # soundness of the promise merge
                    checked += 1
    assert checked > 50


def test_refine_matches_brute_force():
    rng = np.random.default_rng(0)
    cases = [(3, (False, False)), (1, (False, True)), (7, (False, False)), (15, (False, False)), (0, (False, True)),
             (12, (False, True)), (int(rng.integers(16)), (False, False))]
    for P, gnd in cases:
        halos = [None, {(0, 0, S.N): int(rng.integers(16)), (0, 1, S.N): 0, (1, 0, S.S): 15, (1, 1, S.S): int(rng.integers(16))}]
        for halo in halos:
            got = sorted(S.refine(P, halo, gnd))
            assert got == sorted(S.brute_refine(P, halo, gnd))
            for c in got:
                assert S.merge_promises(*c) == (P, True)
    assert len(S.refine(3)) > 1 and len(S.refine(0)) == 1


@pytest.mark.parametrize("n,top", [(128, 64), (64, 64), (32, 32)])
def test_levels_consistent(n, top):
    for seed in range(2):
        v = S.SupportPromise(K=16, h_top=top, ground=8)
        ctx = hier.Ctx(seed=seed)
        hier.run([v], [], n, ctx)
        lv = {l.h: l.vars["support"] for l in ctx.levels if "support" in l.vars}
        for h, O in lv.items():
            assert S.consistent(O, lv.get(2 * h), S.ground_rows(n, h, 8))


def _promise_grid(n, K, G, seed):
    v = S.SupportPromise(K=K, h_top=n // 2, ground=G, T=2.0)
    ctx = hier.Ctx(seed=seed)
    hier.run([v], [], n, ctx)
    return next(l.vars["support"] for l in ctx.levels if l.h == K)


@pytest.mark.parametrize("band_solid", [True, False])
def test_fulfil(ts, grids, band_solid):
    """Consistent random promise grids on random tiles: every block satisfies
    its promise, edits stay inside the block, the band is untouched, the rule
    holds everywhere and a second pass is a no-op."""
    K, G = 8, 4
    for i, t in enumerate(grids):
        tiles = t.copy()
        tiles[-G:] = 1 if band_solid else 0
        before = tiles.copy()
        O = _promise_grid(32, K, G, i)
        v = S.SupportPromise(K=K, h_top=16, ground=G)
        for by, bx in np.ndindex(*O.shape):
            new, e, _ = v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]))
            diff = new != tiles
            assert diff.sum() == e
            inside = np.zeros(diff.shape, bool)
            inside[by * K:(by + 1) * K, bx * K:(bx + 1) * K] = True
            assert not (diff & ~inside).any()
            tiles = new
        assert np.array_equal(tiles[-G:], before[-G:])
        for by, bx in np.ndindex(*O.shape):
            assert v.satisfied(int(O[by, bx]), v.summarize(ts, tiles, by * K, bx * K, K))
        assert S.support_violations(ts, tiles, G) == 0
        for by, bx in np.ndindex(*O.shape):
            assert v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]))[1] == 0


def test_fulfil_any_valid_value(ts, grids):
    """Single blocks, every valid value: fulfil always ends satisfied."""
    K, G = 8, 4
    v = S.SupportPromise(K=K, ground=G)
    t = grids[2]
    for val in range(S.V):
        for y0 in (8, 24):
            gnd = S.ground_block(32, y0, K, G)
            if not S.allowed(val, gnd):
                continue
            new, e, _ = v.fulfil(ts, t, y0, 16, K, val)
            assert v.satisfied(val, v.summarize(ts, new, y0, 16, K))


def test_make_valid_is_consistent_at_every_level(ts, grids):
    t = np.tile(grids[1], (2, 2))                                       # 64 x 64
    tiles, O = S.make_valid(ts, t, 8, 16)
    assert S.support_violations(ts, tiles, 8) == 0
    from castlegen import corpus
    g, valid = corpus.mine(ts, tiles, 16, S, ground=8)
    for h, G_ in g.items():
        assert valid[h].all()
        assert S.consistent(G_, g.get(2 * h), S.ground_rows(64, h, 8))


def test_window_costs_match_states(ts, grids):
    t = grids[1]
    v = S.SupportPromise(K=8, ground=0)
    h = 8
    cost = v.window_costs(ts, t, h)
    for u in (0, 37, 500, 1023):
        cy, cx = divmod(u, 32)
        rolled = np.roll(t, (-(cy - h // 2) + 8, -(cx - h // 2) + 8), (0, 1))
        s = S.summarize(ts, rolled, 8, 8, h, 0)
        assert np.array_equal(cost[u], [S.violations(o, s) for o in range(S.V)])


def test_exemplar_keeps_the_support_rule(ts):
    from castlegen import pipeline
    layout = ex.load_layout("cliffs")
    E = pipeline.exemplar_for(dict(layout, name="cliffs"), ts, log=lambda *a: None)   # cached build
    assert S.support_violations(ts, E, layout["ground"]) == 0


MINI = {"name": "cliffs-promise-mini", "tileset": "cliffs", "size": 64, "seed": 1,
        "exemplar": {"tileset": "cliffs", "size": [32, 32], "sweeps": 40, "T": 0.6, "clean": True, "support": True,
                     "ground": 4, "connect": False,
                     "place": [{"fill": "stone", "at": [28, 0], "size": [4, 32]},
                               {"blob": "stone", "at": [24, 10], "radius": [6, 5], "seed": True},
                               {"blob": "stone", "at": [8, 22], "radius": [2, 4]}]},
        "stages": [{"stage": "hier", "K": 16, "h_top": 32, "tables": "flat", "quantities": ["support"],
                    "r": [1, 1, 0, 0, 0, 0]},
                   {"stage": "tiles"},
                   {"stage": "ground"},
                   {"stage": "repair", "rounds": 5},
                   {"stage": "fulfil"},
                   {"stage": "repair", "rounds": 5, "keep_solidity": True},
                   {"stage": "fulfil"}]}


def test_end_to_end_mini_cliffs(tmp_path, monkeypatch):
    from castlegen import pipeline as pl
    monkeypatch.setattr(pl, "CACHE_DIR", str(tmp_path))
    ctxs = []
    ts, tiles, report = pl.run(MINI, log=lambda *a: None, ctx_out=ctxs)
    assert report[0]["unsupported"] == 0                               # the exemplar keeps the rule
    assert report[-1]["unsupported"] == 0
    assert report[-2]["unsupported"] == 0                              # the solidity-keeping repair cannot break it
    assert all(f == 1.0 for f in report[-1]["sat"].values())
    ctx = ctxs[0]
    for v in ctx.cache["promises"]:                                    # bottom-up: every level keeps its promise
        for lv in ctx.cache["levels"]:
            if v.name in lv.vars:
                St = v.level_states(ts, tiles, lv.h)
                O = lv.vars[v.name]
                assert all(v.satisfied(int(O[y, x]), St[y][x]) for y, x in np.ndindex(*O.shape))
    _, again, _ = pl.run(MINI, log=lambda *a: None)
    assert np.array_equal(tiles, again)                                # deterministic for a seed
