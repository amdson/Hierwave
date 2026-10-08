import itertools

import numpy as np

from castlegen.channels import paintpot as pp
from castlegen.channels.circles import Circles
from castlegen.channels import induce_circles as ic


def brute_features(p, V, offsets):
    H, W = p.shape
    f = np.zeros(pp.nfeat(V, offsets))
    for y in range(H):
        for x in range(W):
            f[p[y, x]] += 1
            for k, (dy, dx) in enumerate(offsets):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    f[V + k * V * V + p[y, x] * V + p[yy, xx]] += 1
    return f


def test_features_brute():
    rng = np.random.default_rng(0)
    for V, offs in ((2, pp.OFF8), (4, pp.OFF8), (3, pp.OFF2)):
        P = rng.integers(V, size=(5, 6, 7))
        X = pp.features(P, V, offs)
        for n in range(5):
            assert np.array_equal(X[n], brute_features(P[n], V, offs))
            assert np.array_equal(pp.features(P[n], V, offs), X[n])


def test_energy_is_features_dot_theta():
    rng = np.random.default_rng(1)
    V = 3
    th = rng.standard_normal(pp.nfeat(V, pp.OFF2))
    u, g = pp.unpack(th, V, pp.OFF2)
    p = rng.integers(V, size=(6, 5))
    e = u[p].sum()
    H, W = p.shape
    for k, (dy, dx) in enumerate(pp.OFF2):
        for y in range(H):
            for x in range(W):
                if 0 <= y + dy < H and 0 <= x + dx < W:
                    e += g[k, p[y, x], p[y + dy, x + dx]]
    assert abs(pp.energy(th, p, V, pp.OFF2) - e) < 1e-9
    assert np.allclose(pp.pack(u, g), th)


def test_fit_recovers_planted():
    """Exhaustive 3 x 3 binary windows inside a fixed background ring;
    noiseless targets from a planted theta: the fit reproduces every
    window difference, and the canonical gauge is recovered."""
    rng = np.random.default_rng(2)
    V = 2
    th0 = rng.standard_normal(pp.nfeat(V, pp.OFF8))
    ring = rng.integers(V, size=(5, 5))
    P = []
    for bits in itertools.product(range(V), repeat=9):
        p = ring.copy()
        p[1:4, 1:4] = np.array(bits).reshape(3, 3)
        P.append(p)
    P = np.array(P)
    ref = ring.copy()
    ref[1:4, 1:4] = 0
    X = pp.features(P, V) - pp.features(ref, V)
    y = X @ th0
    th, d = pp.fit(X, y, 1e-10)
    assert d["rel"] < 1e-6
    assert np.allclose(X @ th, y, atol=1e-6)


def test_materialise_unary_only_is_reference():
    """dem paint potential with u = fz, no pairs: the materialised mid tables
    are reference()'s mid_h / mid_v (reference gauge) and mid_u + b."""
    C = Circles(6, 6)
    raw = C.reference(centred=False)
    th = pp.pack(C.fz, np.zeros((len(pp.OFF8), 4, 4)))
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    u, g = pp.materialise(th, lambda v: ic.paint_single_mid(C, v), lambda v, w, d: ic.paint_pair_mid(C, v, w, d),
                          C.D, dirs, 4, pp.OFF8)
    assert np.abs(g[0] - raw["mid_h"]).max() < 1e-9
    assert np.abs(g[1] - raw["mid_v"]).max() < 1e-9
    assert np.abs(u + C.pres_e - raw["mid_u"]).max() < 1e-9
    assert np.abs(g[2]).max() < 1e-9 and np.abs(g[3]).max() < 1e-9


def test_mid_region_F_matches_window_hook():
    C = Circles(6, 6)
    MW = ic.MidWindows(C, 3, 3)
    rng = np.random.default_rng(3)
    r = np.zeros((3, 3), np.int32)
    for _ in range(10):
        z = ((rng.random((3, 3)) < 0.5) * rng.integers(1, C.D, (3, 3))).astype(np.int32)
        a = MW.window_free_energy_exact(z) - MW.window_free_energy_exact(r)
        b = ic.mid_window_F(C, z) - ic.mid_window_F(C, r)
        assert abs(a - b) < 1e-9
        th = pp.pack(C.fz, np.zeros((4, 4, 4)))
        halo = ((rng.random((5, 5)) < 0.5) * rng.integers(1, C.D, (5, 5))).astype(np.int32)
        assert abs(pp.energy(th, ic.paint_region_mid(C, z, halo), 4) - ic.mid_window_F(C, z, halo)) < 1e-9


def test_paint_region_top():
    C = Circles(6, 6)
    z = np.array([[0, 1], [2, 3]], np.int32)
    halo = np.full((4, 4), 3, np.int32)
    s = ic.paint_region_top(C, z, halo)
    assert s.shape == (8, 8) and s.sum() == 16
    assert s[2, 2] == 1 and s[2, 5] == 1 and s[5, 2] == 1 and s[5, 5] == 1
    assert ic.paint_pair_top(C, 1, 2, (1, -1)).sum() == 25
