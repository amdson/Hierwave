"""The circles test's exact pieces: painters (with an independent set-based
painter), the collapsed mid conditional against a per-tile brute force, the
reference pair table against a hand count, the top conditionals against
brute force, the dihedral symmetrisation against transformed states, the
forward model and oracle running, and the kernel's tile energy against numpy."""
import itertools

import numpy as np

from castlegen.channels.circles import Circles, Forward, Oracle


def _sets_dem(C, mid):
    """Independent painter: per tile, the number of discs and rings covering it."""
    nd = np.zeros((C.H, C.W), int)
    nr = np.zeros((C.H, C.W), int)
    for i in range(C.nmy):
        for j in range(C.nmx):
            o = mid[i, j]
            if o == 0:
                continue
            cy, cx = i * C.BM + C.OY[o], j * C.BM + C.OX[o]
            for lst, cnt in ((C.DISC, nd), (C.RING, nr)):
                for dy, dx in lst:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < C.H and 0 <= x < C.W:
                        cnt[y, x] += 1
    dem = np.where(nd > 0, np.where(nr > 0, 3, 1), np.where(nr > 0, 2, 0))
    return dem, nd, nr


def _tile_Z(C, d):
    """Per-tile Z by summing the two colours with the full energy."""
    return sum(np.exp(-(C.mu * (c == 0) + C.kt[c, d])) for c in (0, 1))


def _logsumexp(a):
    a = np.asarray(a)
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def test_shapes_and_values():
    C = Circles(2, 2)
    assert C.D == 17 and len(C.DISC) == 13 and len(C.RING) == 24
    assert C.OY[1] == 2 and C.OX[1] == 2 and C.OY[16] == 5 and C.OX[16] == 5
    assert C.OY[1 + (3 - 2) * 4 + (5 - 2)] == 3 and C.OX[1 + (3 - 2) * 4 + (5 - 2)] == 5
    assert max(max(abs(a), abs(b)) for a, b in C.RING) == 3       # spills at most one tile


def test_paint_dem_conflict_and_shared():
    rng = np.random.default_rng(0)
    C = Circles(2, 2)
    for _ in range(20):
        mid = rng.integers(C.D, size=(C.nmy, C.nmx)).astype(np.int32)
        ref, nd, nr = _sets_dem(C, mid)
        assert np.array_equal(C.dem_of(mid), ref)
    # hand case: left centre (3, 5), right centre (3, 2) in the next block
    mid = np.zeros((C.nmy, C.nmx), np.int32)
    mid[0, 0] = 1 + 1 * 4 + 3
    mid[0, 1] = 1 + 1 * 4 + 0
    dem = C.dem_of(mid)
    assert dem[3, 7] == 3 and dem[3, 8] == 3                    # disc of one on the ring of the other
    for y, x in [(2, 7), (4, 7), (2, 8), (4, 8)]:                # two rings: air
        assert dem[y, x] == 2
    assert (dem == 3).sum() == 2
    # off-grid spill dropped, no error at the corner
    mid[:] = 0
    mid[0, 0] = 1
    assert C.dem_of(mid)[0, 0] == 2 and (C.dem_of(mid) > 0).sum() < 37


def _brute_mid(C, O, i, j, plain=False):
    out = np.zeros(C.D)
    s = O.slot.grid[i, j]
    BM = C.BM
    y0, y1, x0, x1 = max(i * BM - 1, 0), min((i + 1) * BM + 1, C.H), max(j * BM - 1, 0), min((j + 1) * BM + 1, C.W)
    for o in range(C.D):
        mid = O.mid.grid.copy()
        mid[i, j] = o
        dem, _, _ = _sets_dem(C, mid)
        e = C.lam * ((o > 0) != s) - C.b * (o > 0)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if plain:
                    e += C.kt[O.tile.grid[y, x], dem[y, x]]
                else:
                    e -= np.log(_tile_Z(C, dem[y, x]))
        out[o] = -e
    return np.exp(out - _logsumexp(out))


def test_mid_probs_brute_force():
    rng = np.random.default_rng(1)
    for C in (Circles(2, 2, kappa=1.7, mu=0.6, lam=1.3), Circles(2, 2, BM=4, R=1, kappa=2.5, mu=0.4)):
        O = Oracle(C, seed=2)
        for trial in range(6):
            O.mid.grid[:] = rng.integers(C.D, size=O.mid.grid.shape)
            O.top.grid[:] = rng.integers(C.P, size=O.top.grid.shape)
            O.tile.grid[:] = rng.integers(2, size=O.tile.grid.shape)
            C.paint_slot(O.top, O.slot)
            C.paint_dem(O.mid, O.dem)
            for i, j in [(1, 1), (0, 0), (2, 1), (3, 3)]:
                assert np.allclose(O.mid_probs(i, j), _brute_mid(C, O, i, j), atol=1e-9, rtol=0)
                assert np.allclose(O.mid_probs_plain(i, j), _brute_mid(C, O, i, j, plain=True), atol=1e-9, rtol=0)


def test_mid_move_keeps_dem_painted():
    C = Circles(2, 2)
    O = Oracle(C, seed=0)
    rng = np.random.default_rng(3)
    O.mid.grid[:] = rng.integers(C.D, size=O.mid.grid.shape)
    C.paint_dem(O.mid, O.dem)
    for k in range(40):
        O.mid_move(k % C.nmy, (k // 3) % C.nmx)
        assert np.array_equal(O.dem.grid, C.dem_of(O.mid.grid))
    for k in range(8):
        O.top_move(k % 2, k // 4)
        assert np.array_equal(O.dem.grid, C.dem_of(O.mid.grid))
        assert np.array_equal(O.slot.grid, C.SLOTPAT[O.top.grid].transpose(0, 2, 1, 3).reshape(O.slot.grid.shape))


def test_reference_mid_h_hand_pair():
    """Left centre (3, 5), right centre (3, 2): by hand, the right disc's
    column-8 cell (3, 8) lies on the left ring (dx = +3) and the left disc's
    (3, 7) on the right ring (dx = -3): 2 conflict tiles; the left ring's
    (2, 8), (4, 8) meet the right ring's dx = -2 cells, and the right ring's
    (2, 7), (4, 7) the left ring's dx = +2 cells: 4 shared ring tiles."""
    C = Circles(2, 2)
    f0, fd, fa, fc = C.fz
    o, op = 1 + 1 * 4 + 3, 1 + 1 * 4 + 0
    want = 2 * (fc - fd - fa + f0) + 4 * (fa - 2 * fa + f0)
    raw = C.reference(centred=False)
    assert np.isclose(raw["mid_h"][o, op], want)
    assert np.isclose(raw["mid_v"][C._sym["t"][0][o], C._sym["t"][0][op]], want)
    assert np.allclose(raw["mid_h"][0], 0) and np.allclose(raw["mid_h"][:, 0], 0)
    # far apart on the row: no shared tiles
    assert np.isclose(raw["mid_h"][1 + 4 * 1 + 0, 1 + 4 * 1 + 3], 0)
    # mid_u: every present offset costs the full footprint in an interior block
    assert np.isclose(C.object_cost(), 13 * (fd - f0) + 24 * (fa - f0))
    assert np.allclose(raw["mid_u"][1:], 13 * (fd - f0) + 24 * (fa - f0) - C.b)
    assert np.isclose(raw["mid_u"][1] - raw["mid_u"][0], np.log(16))
    # top tables: only east-west corner pairs on one row interact
    th = raw["top_h"]
    for T in range(4):
        for Tp in range(4):
            if not (T in (1, 3) and Tp == T - 1):
                assert abs(th[T, Tp]) < 1e-9, (T, Tp)
    ref = C.reference()
    assert np.allclose(ref["mid_h"].mean(0), 0) and np.allclose(ref["mid_h"].mean(1), 0)
    assert np.allclose(ref["top_u"], 0)


def _brute_top(C, O, i, j):
    """p*(T | tiles, outside objects) by enumerating the BT x BT cells' values
    with the full kappa energy over the tiles (independent painter)."""
    BT = C.BT
    sl = (slice(i * BT, (i + 1) * BT), slice(j * BT, (j + 1) * BT))
    old = O.mid.grid[sl].copy()
    ek = []
    cfgs = list(itertools.product(range(C.D), repeat=BT * BT))
    for cfg in cfgs:
        O.mid.grid[sl] = np.array(cfg).reshape(BT, BT)
        dem, _, _ = _sets_dem(C, O.mid.grid)
        ek.append(C.kt[O.tile.grid, dem].sum())
    O.mid.grid[sl] = old
    ek = np.array(ek)
    pres = np.array(cfgs) > 0
    logw = np.zeros(C.P)
    for T in range(C.P):
        lam = C.lam * (pres != C.SLOTPAT[T].ravel()).sum(1) - C.b * pres.sum(1)
        logw[T] = _logsumexp(-(ek + lam))
    p = np.exp(logw - logw.max())
    return p / p.sum()


def test_top_probs_brute_force():
    rng = np.random.default_rng(4)
    C = Circles(2, 2, BM=4, R=1, kappa=1.5, mu=0.5, lam=0.8)
    O = Oracle(C, seed=5)
    for trial in range(3):
        O.mid.grid[:] = rng.integers(C.D, size=O.mid.grid.shape)
        O.tile.grid[:] = rng.integers(2, size=O.tile.grid.shape)
        C.paint_dem(O.mid, O.dem)
        for i, j in [(0, 0), (1, 0), (1, 1)]:
            assert np.allclose(O.top_probs(i, j), _brute_top(C, O, i, j), atol=1e-9, rtol=0)
    # top_probs_mid
    for i, j in [(0, 0), (1, 1)]:
        BT = C.BT
        pres = O.mid.grid[i * BT:(i + 1) * BT, j * BT:(j + 1) * BT] > 0
        w = np.array([np.exp(-C.lam * (pres != C.SLOTPAT[T]).sum()) for T in range(C.P)])
        assert np.allclose(O.top_probs_mid(i, j), w / w.sum())


def test_top_move_invariance():
    """The Metropolis top move leaves p*(T, mid cells | tiles, outside)
    invariant: its T marginal over many moves matches top_probs."""
    rng = np.random.default_rng(6)
    C = Circles(1, 1, BM=4, R=1, kappa=1.0, mu=0.5, lam=0.7)
    O = Oracle(C, seed=7)
    O.tile.grid[:] = rng.integers(2, size=O.tile.grid.shape)
    O.mid.grid[:] = 0
    want = O.top_probs(0, 0)
    for _ in range(500):                                          # burn in (independence sampler)
        O.top_move(0, 0)
    cnt = np.zeros(C.P)
    n = 20000
    for _ in range(n):
        O.top_move(0, 0)
        cnt[O.top.grid[0, 0]] += 1
    assert 0.5 * np.abs(cnt / n - want).sum() < 0.03, (cnt / n, want)


def test_symmetrise_exact():
    C = Circles(2, 2)
    O = Oracle(C, seed=8)
    O.sweep(2)
    rng = np.random.default_rng(9)
    O.mid.grid[:] = rng.integers(C.D, size=O.mid.grid.shape)       # an asymmetric state
    s = C.stats(O.top, O.mid, O.tile)
    for g in C.GROUP:
        t = C.stats(*C.transform_state(O.top, O.mid, O.tile, g))
        ts = C.transform_stats(s, g)
        for k in s:
            assert np.allclose(t[k], ts[k]), (g, k)
    sym = C.symmetrise(s)
    for g in C.GROUP:
        ts = C.transform_stats(sym, g)
        for k in sym:
            assert np.allclose(ts[k], sym[k]), (g, k)


def test_forward_and_oracle_run():
    C = Circles(2, 2)
    F = Forward(C, C.theta0(), seed=0)
    O = Oracle(C, seed=0)
    O.sweep(2)
    for s in (F.run(S_T=3, S_M=3, S_F=3), C.stats(O.top, O.mid, O.tile)):
        for k, shape in dict(mid_h=(17, 17), mid_v=(17, 17), mid_u=(17,), top_h=(4, 4), top_v=(4, 4),
                             top_u=(4,)).items():
            assert s[k].shape == shape, k
            assert np.all(s[k] >= 0) and np.isclose(s[k].sum(), 1), k
        for k in ("present", "viol", "conflict", "contact", "slot_viol", "edge_air_top"):
            assert 0 <= s[k] <= 1, k
    assert np.array_equal(F.dem.grid, C.dem_of(F.mid.grid))
    mo = O.moments(1, 2)
    assert set(mo) == set(s)
    img = C.render(O.top, O.mid, O.tile)
    assert img.shape == (C.H, C.W, 3) and img.dtype == np.uint8
    from castlegen.channels.train import counts
    N = counts(F.model, ["mid_h", "mid_v", "mid_u", "top_h", "top_v", "top_u"])
    assert N["mid_h"].shape == (17, 17) and N["top_u"].sum() == C.nty * C.ntx


def test_tile_energy_matches_numpy():
    C = Circles(2, 2, kappa=1.0)
    O = Oracle(C, seed=5)
    O.sweep(1)
    rng = np.random.default_rng(10)
    O.tile.grid[:] = rng.integers(2, size=O.tile.grid.shape)
    tot, nviol = O.model.energy("tile")
    assert nviol == 0
    dem = C.dem_of(O.mid.grid)
    assert np.isclose(tot, C.kt[O.tile.grid, dem].sum() + C.mu * (O.tile.grid == 0).sum())


def test_presence_bonus_isolated_block():
    """Default b: with lam = 0 an isolated interior block is absent / present
    50/50 (mid_probs), and the kernel's mid conditional carries the bonus."""
    C = Circles(2, 2, lam=0.0)
    assert np.isclose(C.b, C.object_cost() - np.log(16))
    O = Oracle(C, seed=0)
    O.mid.grid[:] = 0
    C.paint_dem(O.mid, O.dem)
    p = O.mid_probs(1, 1)
    assert np.isclose(p[0], 0.5) and np.allclose(p[1:], 0.5 / 16)
    e = O.model.site_energies("mid", 1, 1)
    assert np.isclose(e[1] - e[0], -C.b)
    C2 = Circles(2, 2, b=1.25)
    assert C2.b == 1.25 and C2.lam == 3.0 and C2.kappa == 4.0 and C2.mu == 0.3
