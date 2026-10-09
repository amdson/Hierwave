"""induce.py: AIS log Z against brute force, the table fit against planted
tables, and the circles window hooks (exact free energy against per-tile
brute force; AIS against exact)."""
import itertools

import numpy as np

from castlegen.channels.core import Channel, Factor, Model
from castlegen.channels.circles import Circles
from castlegen.channels.induce import ais_log_z, ais_se, fit_tables, windows, materialise
from castlegen.channels.induce_circles import MidWindows


def _tiny(seed=0):
    rng = np.random.default_rng(seed)
    D = 3
    c = Channel("x", 1, D).add_view("v", np.arange(D), D)
    c.grid = np.zeros((2, 2), np.int32)
    u = rng.normal(0, 1, D)
    g = rng.normal(0, 1.5, (D, D))
    fac = [Factor.unary(("x", "v"), u, name="u"), Factor.pair(("x", "v"), ("x", "v"), (0, 1), g, name="g")]
    return Model(2, 2, [c], fac), u, g


def test_ais_tiny_brute_force():
    m, u, g = _tiny()
    c = m.chan("x")
    lz = []
    for vals in itertools.product(range(3), repeat=4):
        c.grid[:] = np.array(vals, np.int32).reshape(2, 2)
        e, nv = m.energy("x")
        lz.append(-e)
        # the energy is unary per site + one h pair per row
        a = np.array(vals).reshape(2, 2)
        assert np.isclose(e, u[a].sum() + g[a[:, 0], a[:, 1]].sum())
    lz = np.log(np.exp(np.array(lz)).sum())
    est, lw = ais_log_z(m, "x", None, K=32, M=32, seed=3)
    se = ais_se(lw)
    assert abs(est - lz) < 3 * se + 1e-6, (est, lz, se)


def test_ais_no_rest_is_exact_and_fixed_cells():
    m, u, g = _tiny()
    m.factors = [m.factors[0]]                       # unaries only: AIS = log Z_0 exactly
    c = m.chan("x")
    c.fixed = np.array([[True, False], [False, False]])
    c.grid[0, 0] = 2
    est, lw = ais_log_z(m, "x", None, K=4, M=3, seed=0)
    assert np.allclose(lw, 0)
    assert np.isclose(est, 3 * np.log(np.exp(-u).sum()) - u[2])
    assert c.grid[0, 0] == 2


def test_fit_tables_recovers_planted():
    rng = np.random.default_rng(1)
    D = 3
    u = rng.normal(0, 1, D)
    gh, gv = rng.normal(0, 1, (D, D)), rng.normal(0, 1, (D, D))
    for g in (gh, gv):                               # the reference gauge (ref = 0)
        g[0, :] = 0
        g[:, 0] = 0
    wv = np.array(list(itertools.product(range(D), repeat=4))).reshape(-1, 2, 2)
    y = u[wv].sum((1, 2)) + gh[wv[:, :, 0], wv[:, :, 1]].sum(1) + gv[wv[:, 0, :], wv[:, 1, :]].sum(1)
    fu, fh, fv, res, rms = fit_tables(wv, y, D, ridge=1e-10)
    assert res < 1e-6 and rms > 1
    assert np.allclose(fu, u - u.mean(), atol=1e-6)
    assert np.allclose(fh, gh - gh.mean(), atol=1e-6)
    assert np.allclose(fv, gv - gv.mean(), atol=1e-6)
    th = materialise({"a": 0}, fu + 1, fh, fv, ("x_u", "x_h", "x_v"))
    assert np.isclose(th["x_u"].mean(), 0) and th["a"] == 0


def test_circles_window_exact_per_tile_brute_force():
    C = Circles(2, 2, BM=4, R=1)
    assert C.D == 5
    W = MidWindows(C, 1, 1)
    for o in range(C.D):
        z = np.array([[o]], np.int32)
        m, home = W.window_fine_model(z)
        tile = m.chan(home)
        assert tile.grid.shape == (6, 6)
        # independent tiles: log Z = sum over tiles of log sum over the 2 colours
        lz = 0.0
        for y in range(6):
            for x in range(6):
                e = m.site_energies(home, y, x)
                lz += np.log(np.exp(-e).sum())
        assert np.isclose(W.window_free_energy_exact(z), -lz)
        # and the per-site energies are the full energy's differences
        rng = np.random.default_rng(o)
        tile.grid[:] = rng.integers(2, size=tile.grid.shape)
        e0, _ = m.energy(home)
        tile.grid[2, 3] ^= 1
        e1, _ = m.energy(home)
        se = m.site_energies(home, 2, 3)
        assert np.isclose(e1 - e0, se[tile.grid[2, 3]] - se[1 - tile.grid[2, 3]])
    # an absent window is all free tiles
    assert np.isclose(W.window_free_energy_exact(np.zeros((1, 1), np.int32)), 36 * C.fz[0])


def test_circles_ais_vs_exact():
    """AIS vs exact within 3 standard errors.  At K = 32, M = 16 (the budget
    first specified) this is not reliable: on 200 windows only 80% are within
    3 delta-method SE (the SE underestimates the error and the estimator is
    biased by +0.5 nats in F; stage A of notes/induce_test.md), and the 5
    windows here pass for 3 of 6 seeds.  At K = 256, M = 16, 98% are."""
    C = Circles(6, 6)
    W = MidWindows(C)
    rng = np.random.default_rng(7)
    zs = [((rng.random((2, 2)) < 0.5) * rng.integers(1, C.D, (2, 2))).astype(np.int32) for _ in range(5)]
    fe, _ = windows(W, zs, "exact")
    fa, se = windows(W, zs, "ais", K=256, M=16, seed=2)
    z = np.abs(fa - fe) / np.maximum(se, 1e-12)
    assert (np.abs(fa - fe) <= 3 * se + 1e-9).all(), (fa - fe, se, z)
