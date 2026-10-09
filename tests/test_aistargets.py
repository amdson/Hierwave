"""AISTargets on the old circles and on Potts: with an exact hook it equals
the oracles' collapsed mid conditionals; by AIS it is within the reported
standard error; the model's state is restored.

Circles: R(p) is the block plus its one-tile spill border (10 x 10), the
tiles any candidate's demand writes; the tiles are independent given dem
(no same-level tile factors), so nothing is dilated.  Oracle.mid_probs
sums fz over each candidate's footprint only, relative to absent; the
tiles of R(p) outside o's footprint keep their dem whatever o is, so their
fz terms are a constant across candidates and cancel: the two agree
exactly.  Potts: R(p) is the block alone, the region Oracle._filter's
transfer matrix integrates (the tiles around it fixed boundary)."""
import numpy as np

from castlegen.channels import circles, kernel, potts
from castlegen.channels.aistargets import AISTargets, _erest, _Packing
from castlegen.channels.core import Factor, Model


# ------------------------------------------------------------- circles
def circles_hooks(C, O):
    BM = C.BM

    def region(m, home, i, j):
        return i * BM - 1, (i + 1) * BM + 1, j * BM - 1, (j + 1) * BM + 1

    def designed_e(m, home, i, j, cand):
        return m.site_energies(home, i, j)[cand]                # lam (slot) + pres: the designed mid rows

    def after_change(m, home, i, j):
        C.paint_dem(O.mid, O.dem, region(m, home, i, j))

    def hook(m, home, i, j):
        """log Z over R(p): -sum fz[dem] (tiles independent given dem)."""
        y0, y1, x0, x1 = region(m, home, i, j)
        return -C.fz[O.dem.grid[max(y0, 0):y1, max(x0, 0):x1]].sum()

    return region, designed_e, after_change, hook


def _state(model):
    return [(c.grid.copy(), c.fixed.copy()) for c in model.channels]


def _same(model, st):
    return all(np.array_equal(c.grid, g) and np.array_equal(c.fixed, f) for c, (g, f) in zip(model.channels, st))


def test_circles_hook_exact():
    C = circles.Circles(3, 3)
    O = circles.Oracle(C, seed=3)
    region, de, ac, hook = circles_hooks(C, O)
    T = AISTargets("tile", region, de, ac, hook=hook)
    rng = np.random.default_rng(0)
    for _ in range(10):
        O.sweep(1)
        i, j = rng.integers(C.nmy), rng.integers(C.nmx)
        cand = np.sort(rng.choice(C.D, size=rng.integers(2, C.D + 1), replace=False))
        pe = O.mid_probs(i, j)[cand]
        pi = T.at(O.model, "mid", i, j, cand)
        assert abs(pi.sum() - 1) < 1e-12
        assert np.abs(pi - pe / pe.sum()).max() < 1e-9
    assert T.n_ais == 0 and T.n_hook > 0


def test_circles_ais():
    """The default p_0 holds kappa (it reads dem, which the tile sweeps never
    change) at full strength: nothing is annealed and AIS is exact here."""
    C = circles.Circles(3, 3)
    O = circles.Oracle(C, seed=4)
    O.sweep(3)
    region, de, ac, hook = circles_hooks(C, O)
    A = AISTargets("tile", region, de, ac, K=32, M=16, L=30.0, seed=1)
    E = AISTargets("tile", region, de, ac, hook=hook)
    assert A.packing(O.model).anneal_names == []
    cand = np.arange(C.D)
    rng = np.random.default_rng(1)
    for _ in range(5):
        i, j = rng.integers(C.nmy), rng.integers(C.nmx)
        lz, se = A.log_z(O.model, "mid", i, j, cand)
        lx, _ = E.log_z(O.model, "mid", i, j, cand)
        assert np.all(np.abs(lz - lx) <= 3 * se + 1e-9), (lz - lx, se)
        assert np.abs(A.at(O.model, "mid", i, j, cand) - O.mid_probs(i, j)).sum() < 0.02
    assert A.n_ais == 10 * C.D


def test_circles_hard_honour():
    """kappa = inf: hard rows stay inf in p_0; conflicting candidates get 0."""
    with np.errstate(divide="ignore"):
        C = circles.Circles(2, 2, kappa=np.inf)
    O = circles.Oracle(C, seed=5)
    O.sweep(4)
    assert (O.dem.grid != circles.CONFLICT).all()
    region, de, ac, hook = circles_hooks(C, O)
    A = AISTargets("tile", region, de, ac, K=8, M=4)
    cand = np.arange(C.D)
    for i, j in ((0, 1), (1, 2), (3, 3)):
        pe, pi = O.mid_probs(i, j), A.at(O.model, "mid", i, j, cand)
        assert np.abs(pi - pe).max() < 1e-9 and (pi == 0).sum() == (pe == 0).sum()


def test_circles_restore():
    C = circles.Circles(2, 2)
    O = circles.Oracle(C, seed=6)
    O.sweep(2)
    region, de, ac, hook = circles_hooks(C, O)
    O.tile.fixed[3:6, 2:5] = True                              # a pre-fixed patch must survive too
    st = _state(O.model)
    for T in (AISTargets("tile", region, de, ac, K=4, M=2), AISTargets("tile", region, de, ac, hook=hook),
              AISTargets("tile", region, de, ac, K=4, M=2, anneal_names=["kappa"])):
        for i, j in ((0, 0), (1, 2), (3, 3)):
            T.at(O.model, "mid", i, j, np.arange(C.D))
            assert _same(O.model, st)
    assert np.array_equal(O.dem.grid, C.dem_of(O.mid))


# --------------------------------------------------------------- potts
def potts_hooks(Pm, O):
    BM = Pm.BM

    def region(m, home, i, j):
        return i * BM, (i + 1) * BM, j * BM, (j + 1) * BM

    def designed_e(m, home, i, j, cand):
        return m.site_energies(home, i, j)[cand]                # lam

    def hook(m, home, i, j):
        return O._filter(i, j, int(O.mid.grid[i, j]), False)  # transfer matrix, boundary tiles fixed

    return region, designed_e, (lambda m, home, i, j: None), hook


def test_potts_hook_and_ais():
    Pm = potts.Potts(2, 2, J=0.1, kappa=8.0)
    O = potts.Oracle(Pm, seed=2)
    O.sweep(3)
    region, de, ac, hook = potts_hooks(Pm, O)
    E = AISTargets("tile", region, de, ac, hook=hook)
    A = AISTargets("tile", region, de, ac, K=32, M=16, seed=2)
    assert A.packing(O.model).anneal_names == ["J_h", "J_v"]   # kappa reads mid: p_0
    cand = np.arange(Pm.q)
    rng = np.random.default_rng(2)
    st = _state(O.model)
    for n in range(5):
        i, j = rng.integers(Pm.nmy), rng.integers(Pm.nmx)
        assert np.abs(E.at(O.model, "mid", i, j, cand) - O.mid_probs(i, j)).max() < 1e-9
        lz, se = A.log_z(O.model, "mid", i, j, cand)
        lx, _ = E.log_z(O.model, "mid", i, j, cand)
        sef = np.maximum(se, np.median(se))      # the delta-method se from M = 16 chains is itself noisy
        assert np.all(np.abs(lz - lx) <= 3 * sef), ((lz - lx) / se)
        assert np.abs(A.at(O.model, "mid", i, j, cand) - O.mid_probs(i, j)).sum() < 0.02
        assert _same(O.model, st)


def test_erest_window():
    """E_rest over a window (pairs and a count row) against total_energy:
    the full grid when all is free, and its changes when one patch is free."""
    Pm = potts.Potts(2, 2, J=0.3, kappa=1.0)
    O = potts.Oracle(Pm, seed=2)
    rng = np.random.default_rng(0)
    cnt = Factor.count(("tile", "col"), ("mid", "col"), rng.normal(size=(Pm.q, Pm.BM ** 2 * (Pm.q - 1) + 1)), name="cnt")
    m = Model(Pm.H, Pm.W, [O.top, O.mid, O.tile], Pm.designed_factors() + [cnt])
    pk = _Packing(m, "tile", None, 30.0)
    P = pk.P
    assert pk.anneal_names == ["J_h", "J_v", "cnt"]
    rest = tuple(t if i in set(pk.fac1[:, 7]) else np.zeros_like(t) for i, t in enumerate(pk.tabs))

    def tot():
        return kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, rest, P.cert, P.joins, P.delta)[0]

    def win(fixed, w):
        return _erest(*w, P.home, P.grids, P.hs, P.views, pk.fac1, pk.tabs, fixed)

    H, W = O.tile.grid.shape
    assert abs(win(np.zeros((H, W), bool), (0, H, 0, W)) - tot()) < 1e-9
    fixed = np.ones((H, W), bool)
    fixed[4:8, 4:8] = False
    a0, b0 = win(fixed, (3, 9, 3, 9)), tot()
    for _ in range(5):
        O.tile.grid[4:8, 4:8] = rng.integers(Pm.q, size=(4, 4))
        assert abs(win(fixed, (3, 9, 3, 9)) - a0 - (tot() - b0)) < 1e-9
