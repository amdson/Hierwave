"""The biome circles test's exact pieces: stamps, the support against brute
painting, the collapsed mid conditional against a per-tile brute force (on a
reduced family set) and a hand count, the top conditional against the
kernel's mask, dormancy in Forward, the tile energy identity, stamp_features
against full-grid features, symmetrise, and the forward / oracle running."""
import numpy as np

from castlegen.channels import paintpot
from castlegen.channels.circles_biome import CirclesBiome, Forward, Oracle, dcentre_finite, OFFS, OFFNAMES


def _sets_dem(C, obj):
    """Independent painter from the STAMP lists."""
    nd = np.zeros((C.H, C.W), int)
    nr = np.zeros((C.H, C.W), int)
    for i in range(obj.shape[0]):
        for j in range(obj.shape[1]):
            for y, x, s in C.STAMP[obj[i, j]]:
                Y, X = i * C.BM + y, j * C.BM + x
                if 0 <= Y < C.H and 0 <= X < C.W:
                    (nd if s == 1 else nr)[Y, X] += 1
    return np.where(nd > 0, np.where(nr > 0, 3, 1), np.where(nr > 0, 2, 0))


def _small(**kw):
    """Radius-1 disc and 1 x 2 bar at BM = 4."""
    disc = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    fams = [("disc", disc, [(cy, cx) for cy in (1, 2) for cx in (1, 2)]),
            ("bar", [(0, 0), (0, 1)], [(ry, rx) for ry in range(4) for rx in range(3)])]
    return CirclesBiome(2, 2, BM=4, families=fams, **kw)


def _random_valid(C, rng, O):
    """A conflict-free obj state with random biome (oracle sweeps from random biome)."""
    O.biome.grid[:] = rng.integers(C.P, size=O.biome.grid.shape)
    O.obj.grid[:] = 0
    C.paint_allow(O.biome, O.allow)
    for _ in range(3):
        for i in range(C.nmy):
            for j in range(C.nmx):
                O.mid_move(i, j)
    C.paint_dem(O.obj, O.dem)
    assert not (O.dem.grid == 3).any()


def test_stamps():
    C = CirclesBiome(2, 2)
    assert C.D == 33 and C.P == 4 and C.n_off == [16, 16]
    for o in range(1, C.D):
        st = C.STAMP[o]
        nd = sum(s == 1 for _, _, s in st)
        nr = sum(s == 2 for _, _, s in st)
        assert (nd, nr) == ((13, 24) if C.FAM[o] == 1 else (8, 16)), o
        assert all(0 <= y < C.BM and 0 <= x < C.BM for y, x, s in st if s == 1)
        assert all(-1 <= y <= C.BM and -1 <= x <= C.BM for y, x, s in st)
    # value numbering
    assert tuple(C.ANCH[1 + 1 * 4 + 3]) == (3, 5) and tuple(C.ANCH[17 + 4 * 3 + 3]) == (5, 4)
    assert C.FAM[16] == 1 and C.FAM[17] == 2
    f0 = -np.log1p(np.exp(-0.3))
    assert np.isclose(C.object_cost("disc"), 13 * -f0 + 24 * (0.3 - f0))
    assert np.isclose(C.object_cost("bar"), 8 * -f0 + 16 * (0.3 - f0))
    assert np.isclose(C.b[1], C.presence_bonus("bar")) and np.isclose(C.b[0], C.object_cost(0) - np.log(16))
    assert np.allclose(C.bio_u0, 4 * np.log([1, 2, 2, 3]))
    assert C.group() == [(0, 0, 0)]                       # the bar offsets are not mirror symmetric


def test_support_brute_force():
    C = CirclesBiome(2, 2)
    facs = {f.off: f.table for f in C.support_factors()}
    assert set(facs) == set(OFFS)
    g = np.zeros((3, 3), np.int32)
    for d in OFFS:
        for t in range(C.D):
            for tp in range(C.D):
                g[:] = 0
                g[1, 1] = t
                g[1 + d[0], 1 + d[1]] = tp
                bad = (C.dem_of(g) == 3).any()
                assert np.isinf(facs[d][t, tp]) == bad, (d, t, tp)
    assert np.isinf(facs[(0, 1)]).any() and np.isinf(facs[(1, 0)]).any()
    # admissible = the support over all 8 neighbours (reflections included), plus the mask
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = rng.integers(C.D, size=(3, 3)).astype(np.int32)
        ok = C.admissible(g, 1, 1)
        for t in range(C.D):
            g2 = g.copy(); g2[1, 1] = t
            dem0 = C.dem_of(np.where(np.arange(9).reshape(3, 3) == 4, 0, g))
            dem1 = C.dem_of(g2)
            assert ok[t] == (not ((dem1 == 3) & (dem0 != 3)).any()), t
    allow = np.full((3, 3), 1, np.int32)                  # discs only
    assert not C.admissible(np.zeros((3, 3), np.int32), 1, 1, allow)[17:].any()


def _brute_mid(C, O, i, j):
    out = np.zeros(C.D)
    BM = C.BM
    y0, y1, x0, x1 = max(i * BM - 1, 0), min((i + 1) * BM + 1, C.H), max(j * BM - 1, 0), min((j + 1) * BM + 1, C.W)
    a = O.allow.grid[i, j]
    for o in range(C.D):
        obj = O.obj.grid.copy()
        obj[i, j] = o
        dem = _sets_dem(C, obj)
        e = 0.0
        if o > 0:
            e -= C.b[C.FAM[o] - 1]
            if not (a >> (C.FAM[o] - 1)) & 1:
                e = np.inf
        for y in range(y0, y1):
            for x in range(x0, x1):
                Z = sum(np.exp(-(C.mu * (c == 0) + C.kt[c, dem[y, x]])) for c in (0, 1))
                e -= np.log(Z) if Z > 0 else -np.inf
        out[o] = -e
    p = np.exp(out - out.max())
    return p / p.sum()


def test_mid_probs_brute_force():
    rng = np.random.default_rng(1)
    for C in (_small(mu=0.7, b={"disc": 3.0, "bar": 1.5}), _small()):
        O = Oracle(C, seed=2)
        for trial in range(6):
            _random_valid(C, rng, O)
            for i, j in [(1, 1), (0, 0), (2, 1), (3, 3), (1, 2)]:
                p = O.mid_probs(i, j)
                q = _brute_mid(C, O, i, j)
                assert np.allclose(p, q, atol=1e-9, rtol=0)
                assert np.array_equal(p > 0, C.admissible(O.obj, i, j, O.allow))


def test_mid_probs_hand_pair():
    """Disc centre (3, 5) in the left block, bar top-left (2, 1) in the right
    one.  Disc: tiles x <= 7, ring at x = 8 on rows 2, 3, 4.  Bar: footprint
    rows 2-3, cols 9-12 (global), ring column 8 rows 1-4.  No conflict; shared
    ring tiles (2, 8), (3, 8), (4, 8): 3, each f0 - mu."""
    C = CirclesBiome(2, 2)
    disc, bar = 1 + 1 * 4 + 3, 17
    want = 3 * (C.f0 - C.mu)
    raw = C.reference(centred=False)
    assert np.isclose(raw["obj_h"][disc, bar], want)
    O = Oracle(C, seed=0)
    O.biome.grid[:] = 3
    C.paint_allow(O.biome, O.allow)
    O.obj.grid[:] = 0
    O.obj.grid[1, 1] = disc
    p = O.mid_probs(1, 2)
    assert np.isclose(np.log(p[bar] / p[0]), -(C.object_cost("bar") + want) + C.b[1])
    # and an isolated allowed slot is 50/50 absent / present
    O.obj.grid[:] = 0
    p = O.mid_probs(1, 1)
    assert np.isclose(p[0], 1 / 3) and np.isclose(p[1:17].sum(), 1 / 3)       # both families allowed
    O.biome.grid[:] = 1
    C.paint_allow(O.biome, O.allow)
    p = O.mid_probs(1, 1)
    assert np.isclose(p[0], 0.5) and p[17:].sum() == 0
    # reference: INF on the support, finite part double-centred over finite entries
    ref = C.reference()
    for n in OFFNAMES:
        assert np.array_equal(np.isinf(ref[n]), np.isinf(raw[n]))
        A = np.where(np.isinf(ref[n]), 0, ref[n])
        fin = np.isfinite(ref[n])
        assert np.allclose(A.sum(1)[fin.any(1)] / fin.sum(1)[fin.any(1)], 0, atol=1e-8)
        assert np.allclose(A.sum(0)[fin.any(0)] / fin.sum(0)[fin.any(0)], 0, atol=1e-8)
    assert np.allclose(raw["bio_u"], [0, -4 * np.log(2), -4 * np.log(2), -4 * np.log(3)])
    assert np.allclose(raw["obj_u"][1:], np.log(16)) and raw["obj_u"][0] == 0
    assert ref["bio_h"] is None and ref["bio_v"] is None
    assert np.allclose(dcentre_finite(np.arange(12.).reshape(3, 4)), 0)


def test_top_probs_brute_force():
    """Against the kernel: weight exp(-bio_u0[T]) iff every slot of the block
    has finite site energy under the designed obj factors (mask, pres)."""
    rng = np.random.default_rng(4)
    C = CirclesBiome(2, 2)
    O = Oracle(C, seed=5)
    for trial in range(10):
        O.obj.grid[:] = rng.integers(C.D, size=O.obj.grid.shape) * (rng.random(O.obj.grid.shape) < 0.5)
        for i in range(C.nty):
            for j in range(C.ntx):
                w = np.zeros(C.P)
                old = O.biome.grid[i, j]
                for T in range(C.P):
                    O.biome.grid[i, j] = T
                    C.paint_allow(O.biome, O.allow)
                    ok = all(np.isfinite(O.model.site_energies("obj", y, x)[O.obj.grid[y, x]])
                             for y in range(2 * i, 2 * i + 2) for x in range(2 * j, 2 * j + 2))
                    w[T] = np.exp(-C.bio_u0[T]) * ok
                O.biome.grid[i, j] = old
                C.paint_allow(O.biome, O.allow)
                assert np.allclose(O.top_probs(i, j), w / w.sum())


def test_dormancy_forward():
    C = CirclesBiome(2, 2)
    F = Forward(C, C.theta0(), seed=0)
    F.model = C.model(F.chans, F.theta, F.extra)
    F.biome.grid[:] = np.array([[0, 3], [1, 2]])
    F.obj.grid[:] = 5                                      # junk under the none block too
    assert F.init_obj() == 4
    assert (F.obj.grid[:2, :2] == 0).all() and F.obj.fixed[:2, :2].all() and not F.obj.fixed[2:].any()
    F.model.sweep("obj", 5, seed=1)
    assert (F.obj.grid[:2, :2] == 0).all()
    assert (C.FAM[F.obj.grid[2:, :2]] != 2).all() and (C.FAM[F.obj.grid[2:, 2:]] != 1).all()   # mask
    # a fully dormant world: the obj sweep changes nothing, energy unchanged
    F.biome.grid[:] = 0
    F.init_obj()
    e0 = F.model.energy("obj")
    F.model.sweep("obj", 3, seed=2)
    assert (F.obj.grid == 0).all() and F.model.energy("obj") == e0
    s = F.run(S_T=0, S_M=2, S_F=2, fresh=False)
    assert s["dormant"] == 1.0 and F.n_dormant == C.nmy * C.nmx


def test_tile_energy_identity():
    C = CirclesBiome(2, 2)
    O = Oracle(C, seed=5)
    O.biome.grid[:] = 3
    C.paint_allow(O.biome, O.allow)
    O.sweep(3)
    O.biome.grid[:] = 3                                    # keep objects present for the check
    tot, nviol = O.model.energy("tile")
    dem = C.dem_of(O.obj.grid)
    assert nviol == 0 and (dem > 0).any()
    assert np.isclose(tot, C.mu * (O.tile.grid == 0).sum())
    y, x = np.argwhere(dem == 1)[0] if (dem == 1).any() else np.argwhere(dem == 2)[0]
    O.tile.grid[y, x] = 1 - O.tile.grid[y, x]
    assert O.model.energy("tile")[1] == 1


def test_stamp_features_full_grid():
    rng = np.random.default_rng(7)
    C = CirclesBiome(2, 2)
    O = Oracle(C, seed=3)
    for trial in range(5):
        _random_valid(C, rng, O)
        for i, j in [(0, 0), (1, 2), (3, 3), (2, 0)]:
            cand = np.array([0, 3, 9, 17, 30, int(rng.integers(C.D))])
            X = C.stamp_features(O.obj, i, j, cand)
            g = O.obj.grid.copy()
            g[i, j] = 0
            f0 = paintpot.features(C.dem_of(g), 4)
            for k, t in enumerate(cand):
                g[i, j] = t
                assert np.array_equal(X[k], paintpot.features(C.dem_of(g), 4) - f0), (i, j, t)
            assert X.shape == (len(cand), paintpot.nfeat(4, paintpot.OFF8)) and not X[0].any()


def test_symmetrise_exact():
    """A family set with a symmetric bar range: the group is identity, the
    mirrors and the 180 rotation (no transpose: the bar is 2 x 4); stats of a
    transformed state = transformed stats."""
    disc = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if a * a + b * b <= 4]
    bar = [(a, b) for a in range(2) for b in range(4)]
    fams = [("disc", disc, [(cy, cx) for cy in range(2, 6) for cx in range(2, 6)]),
            ("bar", bar, [(ry, rx) for ry in range(1, 6) for rx in range(1, 4)])]
    C = CirclesBiome(2, 2, families=fams)
    assert sorted(C.group()) == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    rng = np.random.default_rng(9)
    O = Oracle(C, seed=8)
    _random_valid(C, rng, O)
    O.obj.grid[0, 0] = 1                                    # asymmetric
    s = C.stats(O.biome, O.obj, O.tile)
    for g in C.group():
        t = C.stats(*C.transform_state(O.biome, O.obj, O.tile, g))
        ts = C.transform_stats(s, g)
        for k in s:
            assert np.allclose(t[k], ts[k]), (g, k)
    sym = C.symmetrise(s)
    for g in C.group():
        ts = C.transform_stats(sym, g)
        for k in sym:
            assert np.allclose(ts[k], sym[k]), (g, k)


def test_forward_and_oracle_run():
    C = CirclesBiome(2, 2)
    th = C.theta0()
    th["bio_u"] = -C.bio_u0                                  # cancel the designed prior: biomes vary
    F = Forward(C, th, seed=0)
    O = Oracle(C, seed=0)
    O.sweep(3)
    for s in (F.run(S_T=3, S_M=3, S_F=3), C.stats(O.biome, O.obj, O.tile)):
        for k in OFFNAMES + ["obj_u", "bio_h", "bio_v", "bio_u", "bio_hist"]:
            assert np.all(s[k] >= 0) and np.isclose(s[k].sum(), 1), k
        assert s["obj_h"].shape == (33, 33) and s["bio_h"].shape == (4, 4)
        assert s["conflict"] == 0
        for k in ("present_disc", "present_bar", "contact", "dormant", "edge_air_top"):
            assert 0 <= s[k] <= 1, k
    assert np.array_equal(F.dem.grid, C.dem_of(F.obj.grid))
    assert np.array_equal(O.dem.grid, C.dem_of(O.obj.grid))
    assert np.array_equal(O.allow.grid, np.repeat(np.repeat(O.biome.grid, 2, 0), 2, 1))
    mo = O.moments(1, 2)
    assert set(mo) == set(s)
    xs = list(O.joint_samples(1, 1, 3))
    assert len(xs) == 3 and all(len(x) == 3 for x in xs)
    img = C.render(O.biome, O.obj, O.tile)
    assert img.shape == (C.H, C.W, 3) and img.dtype == np.uint8
    names = [f.name for f in C.learned_factors(th)]
    assert names == OFFNAMES + ["obj_u", "bio_h", "bio_v", "bio_u"]


def test_oracle_exact_on_one_block():
    """CirclesBiome(1, 1): p*(biome) and the disc fraction by enumerating the
    33^4 slot values (energy pairwise in the slots, checked against direct
    painting) against the oracle chain."""
    import itertools
    C = CirclesBiome(1, 1)
    D = C.D

    def F(g):
        d = C.dem_of(g)
        return np.inf if (d == 3).any() else C.fz[d].sum()

    z = np.zeros((2, 2), np.int32)
    F0 = F(z)
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    U = np.zeros((4, D))
    for k, p in enumerate(pos):
        for o in range(D):
            g = z.copy(); g[p] = o
            U[k, o] = F(g) - F0
    E = sum(U[k].reshape([D if a == k else 1 for a in range(4)]) for k in range(4))
    E = E + sum(C.pres_e.reshape([D if a == k else 1 for a in range(4)]) for k in range(4))
    for a, b in itertools.combinations(range(4), 2):
        T = np.zeros((D, D))
        for o in range(D):
            for q in range(D):
                g = z.copy(); g[pos[a]] = o; g[pos[b]] = q
                T[o, q] = F(g) - F0 - U[a, o] - U[b, q]
        E = E + T.reshape([D if k in (a, b) else 1 for k in range(4)])
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = rng.integers(D, size=4)
        g = z.copy()
        for k, p in enumerate(pos):
            g[p] = o[k]
        e = F(g) - F0 + C.pres_e[o].sum()
        assert (np.isinf(e) and np.isinf(E[tuple(o)])) or np.isclose(e, E[tuple(o)])
    w = np.exp(-E)
    isd = (C.FAM == 1).astype(float)
    nd = sum(isd.reshape([D if a == k else 1 for a in range(4)]) for k in range(4)) / 4
    pb, num = np.zeros(C.P), 0.0
    for t in range(C.P):
        ok = C.FAMOK[C.FAM, t]
        m = ok[:, None, None, None] & ok[None, :, None, None] & ok[None, None, :, None] & ok[None, None, None, :]
        ww = np.exp(-C.bio_u0[t]) * w * m
        pb[t] = ww.sum()
        num += (ww * nd).sum()
    O = Oracle(C, seed=1)
    n = 8000
    cnt, pd = np.zeros(C.P), 0.0
    for _ in range(n):
        O.sweep(1, tile_sweeps=0)
        cnt[O.biome.grid[0, 0]] += 1
        pd += (C.FAM[O.obj.grid] == 1).mean()
    assert np.abs(cnt / n - pb / pb.sum()).max() < 0.03, (cnt / n, pb / pb.sum())
    assert abs(pd / n - num / pb.sum()) < 0.03


# ------------------------------------------------- Forward on the Sampler
def _theta_vary(C):
    th = C.theta0()
    th["bio_u"] = -C.bio_u0                                  # biomes uniform: every family and dormancy occur
    return th


def test_forward_sampler_matches_old():
    """K = None, hb = 1: the Sampler path draws exactly what the old path draws;
    hb = BT (block order) and K = 8: the same monitors within noise."""
    C = CirclesBiome(2, 2)
    th = _theta_vary(C)
    old, new = Forward(C, th, seed=3), Forward(C, th, seed=3, use_sampler=True, hb=1)
    for _ in range(5):
        s0, s1 = old.run(5, 5, 5), new.run(5, 5, 5)
        for a, b in zip(old.chans, new.chans):
            assert np.array_equal(a.grid, b.grid), a.name
        assert new.n_dormant == old.n_dormant and new.dormant_frac == s1["dormant"]
    keys = ["present_disc", "present_bar", "contact", "dormant", "edge_air_top"]
    n = 300
    res = {}
    for name, F in (("old", Forward(C, th, seed=1)), ("hb", Forward(C, th, seed=2, use_sampler=True)),
                    ("K8", Forward(C, th, seed=4, use_sampler=True, K=8))):
        x = np.array([[F.run(4, 6, 2)[k] for k in keys] for _ in range(n)])
        res[name] = x.mean(0), x.std(0, ddof=1) / np.sqrt(n)
        assert x[:, keys.index("dormant")].max() <= 1
    for name in ("hb", "K8"):
        z = np.abs(res[name][0] - res["old"][0]) / np.sqrt(res[name][1] ** 2 + res["old"][1] ** 2 + 1e-12)
        assert z.max() < 4.0, (name, dict(zip(keys, z)))


def test_forward_sampler_dormancy():
    C = CirclesBiome(2, 2)
    F = Forward(C, C.theta0(), seed=0, use_sampler=True, K=8)
    F.build()
    F.biome.grid[:] = np.array([[0, 3], [1, 2]])
    F.obj.grid[:] = 5                                      # junk under the none block too
    C.paint_allow(F.biome, F.allow)
    S = F.S["obj"]
    assert S.init() == 0.25 and (F.obj.grid[:2, :2] == 0).all() and S.dormant[:2, :2].all()
    assert not S.dormant[2:].any() and not S.dormant[:, 2:].any()
    assert np.array_equal(S.active, [[False, True], [True, True]])
    S.sweep(10, seed=1)
    assert (F.obj.grid[:2, :2] == 0).all()
    assert (C.FAM[F.obj.grid[2:, :2]] != 2).all() and (C.FAM[F.obj.grid[2:, 2:]] != 1).all()   # mask
    F.biome.grid[:] = 0                                    # fully dormant: the sweep is a no-op
    C.paint_allow(F.biome, F.allow)
    assert S.init() == 1.0 and not S.active.any()
    e0 = S.energy()
    S.sweep(5, seed=2)
    assert (F.obj.grid == 0).all() and S.energy() == e0
    s = F.run(S_T=0, S_M=2, S_F=2, fresh=False)
    assert s["dormant"] == 1.0 == F.dormant_frac and F.n_dormant == C.nmy * C.nmx


def test_forward_sampler_relax():
    """p = 3 relaxed sweeps with the mask at lam = 3: support kept (no conflict
    ever), dormant slots kept at absent, mask violations possible but rare."""
    C = CirclesBiome(2, 2)
    th = _theta_vary(C)
    F = Forward(C, th, seed=5, use_sampler=True, p_relax=3)
    mv = []
    for _ in range(40):
        s = F.run(3, 3, 2)
        assert s["conflict"] == 0
        assert (F.obj.grid[F.S["obj"].dormant] == 0).all()
        assert np.array_equal(F.dem.grid, C.dem_of(F.obj.grid))
        mv.append(s["mask_viol"])
    assert F.soft is not None and np.isfinite(F.soft.factors[0].table).all()
    assert 0 <= np.mean(mv) < 0.2


def test_random_families():
    from castlegen.channels.circles_biome import random_families
    rng = np.random.default_rng(3)
    fams = random_families(6, rng)
    assert len({tuple(f[1]) for f in fams}) == 6
    C = CirclesBiome(2, 2, families=fams)
    assert C.D == 1 + sum(len(f[2]) for f in fams)
    for name, cells, offs in fams:
        cs = set(cells)
        assert 6 <= len(cs) <= 14 and min(a for a, _ in cs) == 0 and min(b for _, b in cs) == 0
        seen, todo = set(), [cells[0]]                     # 4-connected
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            todo += [(p[0] + u, p[1] + v) for u, v in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (p[0] + u, p[1] + v) in cs]
        assert seen == cs, name
        assert 1 <= len(offs) <= 16 and len(set(offs)) == len(offs)
    for o in range(1, C.D):
        f = C.FAM[o] - 1
        cs = {(C.ANCH[o, 0] + a, C.ANCH[o, 1] + b) for a, b in fams[f][1]}
        dirt = {(y, x) for y, x, s in C.STAMP[o] if s == 1}
        ring = {(y, x) for y, x, s in C.STAMP[o] if s == 2}
        assert dirt == cs and all(0 <= y < C.BM and 0 <= x < C.BM for y, x in dirt)
        dil = {(y + u, x + v) for y, x in cs for u in (-1, 0, 1) for v in (-1, 0, 1)}
        assert ring == dil - cs and all(-1 <= y <= C.BM and -1 <= x <= C.BM for y, x in ring)


def test_masks_random_families_run():
    from castlegen.channels.circles_biome import random_families, random_pair_masks
    rng = np.random.default_rng(5)
    fams, masks = random_families(5, rng), random_pair_masks(5, rng)
    assert masks[0] == 0 and len(masks) == 8 and all(bin(m).count("1") == 2 for m in masks[1:])
    C = CirclesBiome(2, 2, families=fams, masks=masks)
    assert C.P == 8 and np.allclose(C.bio_u0, 4 * np.log1p([0] + [2] * 7))
    for f in range(5):
        assert np.array_equal(C.FAMOK[f + 1], (np.array(masks) >> f) & 1 == 1)
    biome = C.channels()[0]
    assert np.array_equal(np.asarray(biome.views["allow_f2"]), (np.array(masks) >> 2) & 1)
    th = C.theta0()
    th["bio_u"] = -C.bio_u0
    for kw in (dict(), dict(use_sampler=True), dict(use_sampler=True, K=8)):
        F = Forward(C, th, seed=1, **kw)
        for _ in range(2):
            s = F.run(S_T=3, S_M=3, S_F=2)
            assert s["conflict"] == 0 and s["mask_viol"] == 0
            dorm = C.dormant_of(F.allow)
            assert (F.obj.grid[dorm] == 0).all() and np.array_equal(dorm, C.MASKS[F.allow.grid] == 0)
    O = Oracle(C, seed=2)
    O.sweep(3)
    s = C.stats(O.biome, O.obj, O.tile)
    assert s["conflict"] == 0 and s["mask_viol"] == 0 and s["bio_h"].shape == (8, 8)
    assert C.render(O.biome, O.obj, O.tile).shape == (C.H, C.W, 3)
    assert C.group() == [(0, 0, 0)] or (0, 0, 0) in C.group()


# ------------------------------------------------------- periodic world
def _torus_dem(C, og):
    """Independent painter on the torus: og is the (ny, nx) torus obj grid."""
    ny, nx = og.shape
    H, W = ny * C.BM, nx * C.BM
    nd, nr = np.zeros((H, W), int), np.zeros((H, W), int)
    for i in range(ny):
        for j in range(nx):
            for y, x, s in C.STAMP[og[i, j]]:
                (nd if s == 1 else nr)[(i * C.BM + y) % H, (j * C.BM + x) % W] += 1
    return np.where(nd > 0, np.where(nr > 0, 3, 1), np.where(nr > 0, 2, 0))


def test_periodic_paint_wraps():
    C = CirclesBiome(2, 2, periodic=True)
    assert (C.nty, C.nmy, C.H) == (4, 8, 64) and (C.nty_in, C.ntx_in) == (2, 2)
    rng = np.random.default_rng(0)
    biome, obj, tile, allow, dem = C.channels()
    assert np.array_equal(obj.fixed, C.ghost_mask(obj.grid.shape)) and obj.fixed.sum() == 64 - 16
    for _ in range(5):
        obj.grid[:] = rng.integers(C.D, size=obj.grid.shape) * (rng.random(obj.grid.shape) < 0.4)
        C.sync(obj)
        C.paint_dem(obj, dem)
        assert np.array_equal(C.inner(dem), _torus_dem(C, C.inner(obj)))
        assert np.array_equal(dem.grid, dem.grid[np.ix_(*[(np.arange(64) - 16) % 32 + 16] * 2)])  # ghosts = copies
    # a disc at the torus top-left slot, centre (2, 2): its ring row and column -1 land on the last torus row and column
    obj.grid[:] = 0
    obj.grid[2, 2] = 1
    C.sync(obj)
    C.paint_dem(obj, dem)
    d = C.inner(dem)
    assert d[-1, 2] == 2 and d[0, 2] == 1 and d[2, -1] == 2 and d[-1, -1] == 0 and d[-1, -2] == 0
    biome.grid[:] = rng.integers(C.P, size=biome.grid.shape)
    C.sync(biome)
    C.paint_allow(biome, allow, (1, 1))
    C.paint_allow(biome, allow)
    assert np.array_equal(C.inner(allow), np.repeat(np.repeat(C.inner(biome), 2, 0), 2, 1))


def test_periodic_oracle_energy_identity():
    """Periodic 2 x 2 world: mid_probs against direct torus painting, the
    tile energy identity, ghosts staying copies, the forward running."""
    C = CirclesBiome(2, 2, periodic=True)
    O = Oracle(C, seed=3)
    O.sweep(4)
    rng = np.random.default_rng(1)
    for _ in range(3):
        O.sweep(1)
        for c in (O.biome, O.obj, O.tile, O.allow, O.dem):
            g = c.grid.copy()
            C.sync(c)
            assert np.array_equal(g, c.grid), c.name
        tg = C.inner(O.obj).copy()
        for i, j in [(2, 2), (5, 5), (2, 5), (rng.integers(2, 6), rng.integers(2, 6))]:
            e = np.zeros(C.D)
            a = O.allow.grid[i, j]
            for o in range(C.D):
                g = tg.copy()
                g[i - 2, j - 2] = o
                dm = _torus_dem(C, g)
                e[o] = C.fz[dm].sum() + C.pres_e[o] + C.mask_e[o, a]
            p = np.exp(-(e - e[np.isfinite(e)].min()))
            assert np.allclose(O.mid_probs(i, j), p / p.sum(), atol=1e-9)
    s = C.stats(O.biome, O.obj, O.tile)
    assert s["conflict"] == 0 and s["mask_viol"] == 0 and np.isclose(s["obj_h"].sum(), 1)
    assert np.array_equal(C.inner(O.dem), _torus_dem(C, C.inner(O.obj)))
    tot, nviol = O.model.energy("tile")
    assert nviol == 0 and np.isclose(tot, C.mu * (O.tile.grid == 0).sum())
    th = C.theta0()
    F = Forward(C, th, seed=2)
    s = F.run(4, 4, 2)
    assert s["conflict"] == 0 and s["mask_viol"] == 0
    for c in F.chans:
        g = c.grid.copy()
        C.sync(c)
        assert np.array_equal(g, c.grid), c.name
    assert C.render(F.biome, F.obj, F.tile).shape == (32, 32, 3)


def test_stats_region_margin0_is_stats():
    C = CirclesBiome(2, 3)
    O = Oracle(C, seed=4)
    O.sweep(3)
    s, r = C.stats(O.biome, O.obj, O.tile), C.stats_region(O.biome, O.obj, O.tile, margin=0)
    for k in s:
        assert np.allclose(s[k], r[k]), k
