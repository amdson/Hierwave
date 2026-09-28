import itertools

import numpy as np
import pytest

from castlegen import tileset
from castlegen.quantities import envelope as EN
from castlegen.quantities import support as SP

G = 4


@pytest.fixture(scope="module")
def ts():
    return tileset.load("cliffs")


def heightmap(ts, H, n, rng):
    """Valid map: column c solid (random solid material) on rows >= n - H[c]; band stone."""
    solid = np.flatnonzero(ts.solid)
    air = int(np.flatnonzero(~ts.solid)[0])
    rows = np.arange(n)[:, None]
    t = np.where(rows >= n - np.asarray(H)[None, :], rng.choice(solid, (n, n)), air).astype(np.int32)
    return t


def random_H(rng, n, kind):
    if kind == "walk":
        H = np.cumsum(rng.integers(-3, 4, n)) + n // 2
    elif kind == "steps":
        H = np.repeat(rng.integers(G, n + 1, n // 4), 4)
    else:
        H = rng.integers(G, n + 1, n)
    return np.clip(H, G, n)


@pytest.fixture(scope="module")
def valid_maps(ts):
    rng = np.random.default_rng(0)
    return [heightmap(ts, random_H(rng, 64, kind), 64, rng) for kind in ("walk", "steps", "iid") for _ in range(4)]


def brute_state(ts, tiles, y0, x0, k, ground):
    s = EN.effective_solid(ts, tiles, ground)[y0:y0 + k, x0:x0 + k]
    t = np.array([next((i for i in range(k) if not s[k - 1 - i, c]), k) for c in range(k)])
    viol = int((s[:-1] & ~s[1:]).sum())
    return t, s[0], viol


def test_level_matches_brute_force_and_merge_is_exact(ts, valid_maps):
    rng = np.random.default_rng(1)
    grids = valid_maps[:3] + [rng.integers(0, ts.n_sig, (64, 64)).astype(np.int32) for _ in range(3)]
    for tiles in grids:
        Ls = EN.levels(ts, tiles, 8, G)
        for k, L in Ls.items():
            for by, bx in itertools.product(range(64 // k), repeat=2):
                t, top, viol = brute_state(ts, tiles, by * k, bx * k, k, G)
                assert np.array_equal(L.t[by, bx * k:(bx + 1) * k], t)
                assert np.array_equal(L.top[by, bx * k:(bx + 1) * k], top)
                assert L.viol[by, bx] == viol
    # viol summed over the top level == whole-map violations minus the (exempt) wrap seam
    for tiles in grids:
        assert EN.levels(ts, tiles, 8, G)[64].viol.sum() == SP.support_violations(ts, tiles, G)


def test_valid_maps_have_no_violations(ts, valid_maps):
    for tiles in valid_maps:
        assert SP.support_violations(ts, tiles, G) == 0
        _, ok = EN.pyramid(ts, tiles, 8, G)
        assert all(m.all() for m in ok.values())


@pytest.mark.parametrize("I,Q", [(2, 2), (2, 4), (4, 2)])
def test_abstraction_is_a_homomorphism_and_complete(ts, valid_maps, I, Q):
    lg = EN.lang(I, Q)
    K = 16 if I * Q > 4 else 8
    for tiles in valid_maps:
        vals, ok = EN.pyramid(ts, tiles, K, G, lg=lg)
        for h in sorted(vals):
            g = vals[h]
            assert lg.consistent(g, G=G, k=h), (h, "completeness: a valid map abstracts consistently")
            if 2 * h in vals:
                P, okm = lg.merge(g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2])
                assert okm.all() and np.array_equal(P, vals[2 * h])


def test_bins_nest():
    lg = EN.lang()
    for k in (8, 16, 32):
        t = np.arange(k + 1)
        b = lg.bins(t, k)
        assert np.array_equal(lg.up(b), lg.bins(t, 2 * k))                 # lower child: same height (t = k -> mid)
        assert np.array_equal(lg.up_upper(b), lg.bins(k + t, 2 * k))       # upper child over a full lower


def test_seam_rule_is_the_image_of_the_tile_rule():
    """COMPAT[u, l] iff some pair of column-height vectors with those
    abstractions obeys the per-column rule (u.t > 0 => l.t = k), k = 4, I = 2."""
    lg, k = EN.lang(), 4
    cols = list(itertools.product(range(k + 1), repeat=2))               # (t0, t1) of one interval, width 2
    seen = np.zeros((lg.V, lg.V), bool)
    iv = {}
    for a in cols:
        iv.setdefault((int(lg.bins(min(a), k)), int(lg.bins(max(a), k))), []).append(a)
    keys = list(iv)
    for (pu0, pu1) in itertools.product(keys, repeat=2):
        for (pl0, pl1) in itertools.product(keys, repeat=2):
            u = lg.encode([pu0[0], pu1[0]], [pu0[1], pu1[1]])
            l = lg.encode([pl0[0], pl1[0]], [pl0[1], pl1[1]])
            ok = all(any(all(tl == k for tu, tl in zip(a, b) if tu > 0) for a in iv[pu] for b in iv[pl])
                     for pu, pl in ((pu0, pl0), (pu1, pl1)))
            seen[u, l] = True
            assert bool(lg.COMPAT[u, l]) == ok


def test_every_valid_parent_has_a_refinement():
    lg = EN.lang()
    k = 16
    for dx, v in itertools.product((0, 1), range(lg.V)):
        part = int(lg.parent_part(v, dx))
        for gu, gl, ground_parent in ((False, False, False), (False, True, True)):
            if not lg.valid(v, ground_parent, G, 2 * k):
                continue
            u, l = lg.column_pairs(part, gu, gl, G, k)
            assert len(u) > 0, (v, dx, gu, gl)
