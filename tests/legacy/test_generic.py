import numpy as np
import pytest

from castlegen.legacy import generic as GN
from castlegen.legacy import texsyn, tileset


@pytest.fixture(scope="module")
def setup():
    ts = tileset.load("cliffs")
    E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)[184:216, 96:128]                  # sky over terrain
    an = texsyn.Analysis(ts, E, n_pca=8, bounds="edge", pad=8)
    return ts, an, np.asarray(ts.solid, bool)


@pytest.fixture(scope="module")
def torus_setup():
    """A torus exemplar (Analysis(bounds=None)): no padding rows, all map sides free."""
    ts = tileset.load("cliffs")
    E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)[184:216, 96:128]
    an = texsyn.Analysis(ts, E, n_pca=8)
    return ts, an, np.asarray(ts.solid, bool)


def _setup(request, torus):
    return request.getfixturevalue("torus_setup" if torus else "setup")


@pytest.mark.parametrize("torus", [False, True])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("n", [16, 32])                                   # 4 x 4 wraps the 5-wide patch onto itself
def test_cond_energy_exact(request, n, periodic, torus):
    ts, an, solid = _setup(request, torus)
    lvl = GN.JointLevel(an, solid, 4, n, seed=0, periodic=periodic)
    rng = np.random.default_rng(1)
    U = rng.integers(0, an.my * an.m, (lvl.R, lvl.C))
    for py, px in [(0, 0), (1, 2), (lvl.R - 1, lvl.C - 1)]:
        cand = rng.integers(0, an.my * an.m, 5)
        c = lvl.cond_c(py, px, U)[cand]
        full = []
        for u in cand:
            U2 = U.copy()
            U2[py, px] = u
            full.append(lvl.energy_c(U2))
        np.testing.assert_allclose(c - c[0], np.array(full) - full[0], rtol=1e-3, atol=1e-2)


def test_free_sides_crop_costs_nothing(setup):
    """On an exemplar crop, patches clear of the top / bottom padding cost
    nothing with free sides; with periodic ones the wrap seam costs."""
    ts, an, solid = setup
    h, n = 2, 16
    U = np.array([[(an.pad + 6 + h * y) * an.m + (5 + h * x) for x in range(n // h)] for y in range(n // h)])

    def per_q(lvl):
        X = lvl._ext(U)
        return np.array([[((sum(lvl.G[s, lvl._slot(X, qy, qx, s)] for s in range(len(GN.OFF5))) - lvl.c0
                            - lvl.NE[X[qy, qx]]) ** 2).sum() for qx in range(lvl.C)]
                         for qy in range(lvl.B + 2, lvl.B + lvl.R - 2)])
    free = per_q(GN.JointLevel(an, solid, h, n, seed=0))
    wrap = per_q(GN.JointLevel(an, solid, h, n, seed=0, periodic=True))     # rows clear of the padding
    assert free.max() < 1e-3 < wrap.max()


@pytest.mark.parametrize("torus", [False, True])
@pytest.mark.parametrize("periodic", [True, False])
def test_energy_c_vectorised_matches_loop(request, periodic, torus):
    ts, an, solid = _setup(request, torus)
    lvl = GN.JointLevel(an, solid, 4, 32, seed=0, periodic=periodic)
    U = np.random.default_rng(9).integers(0, an.my * an.m, (lvl.R, lvl.C))
    np.testing.assert_allclose(lvl.energy_c(U), lvl.energy_c_ref(U), rtol=1e-4)


def test_window_classes_match_direct(setup):
    ts, an, solid = setup
    s = solid[an.E]
    for h in (2, 4, 8):
        vals = GN.window_classes(s, h)
        rng = np.random.default_rng(h)
        for u in rng.integers(0, an.my * an.m, 50):
            uy, ux = divmod(int(u), an.m)
            rows = np.clip(uy - h // 2 + np.arange(h), 0, an.my - 1)
            cols = (ux - h // 2 + np.arange(h)) % an.m
            assert vals[u] == GN.surface_class(s[rows[:, None], cols[None]])


def test_compat_is_necessary():
    """Support-satisfying maps are height fields: every vertical block pair
    of one is compatible, as are the boundaries."""
    rng = np.random.default_rng(0)
    cp = GN.surface_compat()
    n = 64
    for h in (4, 8, 16):
        for _ in range(20):
            hgt = np.clip(np.cumsum(rng.integers(-6, 7, n)) + rng.integers(0, n), 0, n)
            hgt[rng.random(n) < 0.2] = rng.integers(0, n + 1)
            s = np.arange(n)[:, None] >= n - hgt[None]
            R = n // h
            v = GN.surface_class(s.reshape(R, h, R, h).transpose(0, 2, 1, 3))
            col = np.concatenate([np.full((1, R), GN.TOP), v, np.full((1, R), GN.BOT)], 0)
            assert cp[col[:-1], col[1:]].all()


def test_cost_normalised(setup):
    ts, an, solid = setup
    vals = GN.window_classes(solid[an.E], 8)
    cost = GN.cost_table(vals, an.my, an.m, 8, alpha=1.0)
    np.testing.assert_allclose(np.exp(-cost).sum(1), 1.0, rtol=1e-9)
    seen = np.bincount(vals, minlength=GN.V) > 0
    assert np.isfinite(cost[:, seen]).all() and np.isinf(cost[:, ~seen]).all()


def test_sweeps_keep_promises_consistent(setup):
    ts, an, solid = setup
    lvl = GN.JointLevel(an, solid, 4, 32, seed=0)
    rng = np.random.default_rng(2)
    lvl.init(rng.integers(0, an.my * an.m, (lvl.R, lvl.C)))
    assert lvl.valid()
    lvl.run(2)
    assert lvl.valid() and np.isfinite(lvl.energy())


def test_pad_update_matches_energy(setup):
    ts, an, solid = setup
    lvl = GN.JointLevel(an, solid, 4, 32, seed=0)
    U = np.random.default_rng(5).integers(0, an.my * an.m, (lvl.R, lvl.C))
    for i, ey in ((0, 0), (3, lvl.R + 3)):
        e = lvl._cond_ext(ey, 2, lvl._ext(U), lvl.pad_rows[i] * an.m + np.arange(4))
        full = []
        for c in range(4):
            lvl.PX[i, 2] = c
            full.append(lvl.energy_c(U))
        np.testing.assert_allclose(e - e[0], np.array(full) - full[0], rtol=1e-3, atol=1e-2)


def test_cond_c_subset(setup):
    ts, an, solid = setup
    lvl = GN.JointLevel(an, solid, 4, 32, seed=0)
    U = np.random.default_rng(3).integers(0, an.my * an.m, (lvl.R, lvl.C))
    cand = np.array([5, 17, 400, 1000])
    np.testing.assert_allclose(lvl.cond_c(2, 3, U, cand), lvl.cond_c(2, 3, U)[cand], rtol=1e-5, atol=1e-3)


def test_padding_rows_pull_a_crop_to_the_bottom(setup):
    """With the padding's own terms, the bottom half of the extended grid
    costs nothing for a crop on the exemplar's bottom edge, and more once
    the crop is lifted clear of the edge continuation."""
    ts, an, solid = setup
    h, n = 2, 16
    lvl = GN.JointLevel(an, solid, h, n, seed=0)
    lvl.PX[:] = 5 + h * np.arange(n // h)                                   # padding columns under the crop
    crop = lambda y0: np.array([[(y0 + h * y) * an.m + (5 + h * x) for x in range(n // h)] for y in range(n // h)])
    low = an.my - 1 - 2 * h - (n // h - 1) * h                              # bottom row just above padding row R
    def bottom(U):
        X = lvl._ext(U)
        return sum(((sum(lvl.G[s, lvl._slot(X, qy, qx, s)] for s in range(len(GN.OFF5))) - lvl.c0
                     - lvl.NE[X[qy, qx]]) ** 2).sum() for qy in range(len(X) // 2, len(X)) for qx in range(lvl.C))
    assert bottom(crop(low)) < 1e-3 < bottom(crop(low - 4 * h))


def test_mh_keeps_pibar_invariant(setup):
    """One MH move at a cell (neighbours fixed) from exact draws of pibar
    leaves pibar: the moved samples are as close to it as i.i.d. ones."""
    ts, an, solid = setup
    lvl = GN.JointLevel(an, solid, 4, 32, lam_c=0.3, seed=0, knn=8, eps=0.3)
    rng = np.random.default_rng(4)
    lvl.init(rng.integers(0, an.my * an.m, (lvl.R, lvl.C)))
    py, px = 3, 4
    lp, _ = lvl._logpi(py, px, 1.0)
    p = np.exp(lp - lp.max()); p /= p.sum()
    k = 20000
    cnt = np.zeros(len(p))
    for u in rng.choice(len(p), k, p=p):
        lvl.U[py, px] = u
        lvl.update(py, px)
        cnt[lvl.U[py, px]] += 1
    iid = np.bincount(rng.choice(len(p), k, p=p), minlength=len(p))
    tv = lambda c: 0.5 * np.abs(c / k - p).sum()
    assert tv(cnt) < 1.3 * tv(iid)


@pytest.mark.parametrize("torus", [False, True])
@pytest.mark.parametrize("fast", [True, "numba"])
@pytest.mark.parametrize("n, periodic", [(32, False), (20, True), (20, False)])
def test_cond_batch_matches_reference(request, n, periodic, fast, torus):
    """Every colour's batched E_c equals the single-cell reference (map and
    padding cells, edges included), up to a per-cell constant."""
    ts, an, solid = _setup(request, torus)
    lvl = GN.JointLevel(an, solid, 4, n, seed=0, periodic=periodic, knn=4, fast=fast)
    rng = np.random.default_rng(6)
    lvl.U = rng.integers(0, an.my * an.m, (lvl.R, lvl.C))
    X = lvl._ext(lvl.U)
    EY, EX = np.divmod(np.arange(len(X) * lvl.C), lvl.C)
    for c in range(25):
        sel = (EY % 5 == c // 5) & (EX % 5 == c % 5)
        ey, ex = EY[sel], EX[sel]
        cand = rng.integers(0, an.my * an.m, (len(ey), 6))
        got = lvl._cond_batch(ey, ex, cand, X)
        for i in range(len(ey)):
            ref = lvl._cond_ext(ey[i], ex[i], X, cand[i])
            np.testing.assert_allclose(got[i] - got[i, 0], ref - ref[0], rtol=1e-3, atol=2e-2)


def test_h1_support_and_loc(setup):
    """h = 1: compat is the support rule on the tiles, E_loc batched matches
    the reference, and sweeps keep the tile map support-valid."""
    ts, an, solid = setup
    lvl = GN.JointLevel(an, solid, 1, 16, seed=0, knn=8, loc=ts, fast=True)
    rng = np.random.default_rng(7)
    lvl.init(rng.integers(0, an.my * an.m, (lvl.R, lvl.C)))
    assert lvl.valid()
    py, px = np.divmod(np.arange(lvl.R * lvl.C), lvl.C)
    cand = rng.integers(0, an.my * an.m, (len(py), 5))
    got = lvl._loc_batch(py, px, cand)
    for i in range(0, len(py), 7):
        np.testing.assert_allclose(got[i], lvl.e_loc(py[i], px[i], cand[i]))
    lvl.run(2)
    s = solid[lvl.tiles()]
    assert lvl.valid() and not (s[:-1] & ~s[1:]).any()
    assert np.array_equal(lvl.V == GN.BOT, s)


def test_mh_batch_keeps_pibar_invariant(setup):
    ts, an, solid = setup
    lvl = GN.JointLevel(an, solid, 4, 32, lam_c=0.3, seed=0, knn=8, eps=0.3, fast=True)
    rng = np.random.default_rng(8)
    lvl.init(rng.integers(0, an.my * an.m, (lvl.R, lvl.C)))
    py, px = 3, 4
    lp, _ = lvl._logpi(py, px, 1.0)
    p = np.exp(lp - lp.max()); p /= p.sum()
    k = 4000
    cnt = np.zeros(len(p))
    for u in rng.choice(len(p), k, p=p):
        lvl.U[py, px] = u
        lvl._mh_batch(np.array([py]), np.array([px]), lvl._ext(lvl.U), 1.0)
        cnt[lvl.U[py, px]] += 1
    iid = np.bincount(rng.choice(len(p), k, p=p), minlength=len(p))
    tv = lambda c: 0.5 * np.abs(c / k - p).sum()
    assert tv(cnt) < 1.3 * tv(iid)
