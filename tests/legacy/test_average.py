import itertools

import numpy as np
import pytest

from castlegen.legacy import hier, tileset
from castlegen.legacy import refheights as RH
from castlegen.legacy.envelope_var import EnvelopePromise
from castlegen.legacy.quantities import average as AV


@pytest.fixture(scope="module")
def ts():
    return tileset.load("cliffs")


def random_H(rng, n, G, kind):
    if kind == "walk":
        H = np.cumsum(rng.integers(-4, 5, n)) + n // 2
    elif kind == "steps":
        H = np.repeat(rng.integers(G, n + 1, n // 4), 4)
    else:
        H = rng.integers(G, n + 1, n)
    return np.clip(H, G, n)


@pytest.mark.parametrize("K,G", [(4, 1), (4, 2)])
def test_seam_rule_is_exactly_realisable(K, G):
    """Brute force: for a two-block stack of one sub-column (w = K/2 columns),
    the realisable (upper, lower-ground) and (upper, lower) value pairs."""
    lg = AV.lang(K, G)
    w = lg.w
    seen_g, seen = set(), set()
    for H in itertools.product(range(G, 2 * K + 1), repeat=w):          # two stacked blocks, lower is ground
        H = np.array(H)
        tl, tu = np.clip(H, 0, K), np.clip(H - K, 0, K)
        seen_g.add((tu.sum() // K, tl.sum() // K))
    for H in itertools.product(range(0, 2 * K + 1), repeat=w):          # two ordinary blocks
        H = np.array(H)
        tl, tu = np.clip(H, 0, K), np.clip(H - K, 0, K)
        seen.add((tu.sum() // K, tl.sum() // K))
    for bu, bl in itertools.product(range(w + 1), repeat=2):
        v_u, v_l = lg.encode([bu, bu]), lg.encode([bl, bl])
        if lg.valid(v_l, True, G, K):
            assert bool(lg.seam_compat(K, True, G)[v_u, v_l]) == ((bu, bl) in seen_g), (bu, bl, "ground")
        assert bool(lg.seam_compat(K, False, G)[v_u, v_l]) == ((bu, bl) in seen), (bu, bl)


def test_pyramid_is_complete_and_construct_realises_it(ts):
    rng = np.random.default_rng(0)
    K, G, n = 16, 8, 128
    lg = AV.lang(K, G)
    for kind in ("walk", "steps", "iid"):
        for _ in range(3):
            H = random_H(rng, n, G, kind)
            x = RH.from_heights(ts, H, n, G, "stone", "stone")
            vals, ok = AV.pyramid(ts, x, K, G, 64, lg)
            for h in sorted(vals):
                assert ok[h].all()
                assert lg.consistent(vals[h], vals.get(2 * h), G, h), (kind, h)
            H2 = AV.construct(vals[K], K, G, n, lg)
            x2 = RH.from_heights(ts, H2, n, G, "stone", "stone")
            assert RH.violations(ts, x2, G) == 0
            assert np.array_equal(AV.pyramid(ts, x2, K, G, K, lg)[0][K], vals[K])


def test_sampled_pyramids_are_realisable(ts):
    K, G, n = 16, 8, 128
    lg = AV.lang(K, G)
    for seed in range(6):
        for T in (0.0, 1.0):
            v = EnvelopePromise(K, 64, None, T=T, sweeps=2, top_sweeps=3, ground=G, lang=lg)
            ctx = hier.Ctx(seed)
            hier.run([v], [], n, ctx)
            grids = {lv.h: lv.vars[v.name] for lv in ctx.levels if v.name in lv.vars}
            for h, g in grids.items():
                assert lg.consistent(g, grids.get(2 * h), G, h)
            H = AV.construct(grids[K], K, G, n, lg)
            x = RH.from_heights(ts, H, n, G, "stone", "stone")
            assert RH.violations(ts, x, G) == 0
            assert all(f == 1.0 for f in AV.satisfaction(ts, x, grids, K, G, lg).values())
