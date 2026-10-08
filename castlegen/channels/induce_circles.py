"""Window hooks of the circles toy for castlegen.channels.induce
(notes/induce_test.md).  Circles-specific; induce.py stays generic.

MidWindows  coarse = mid, fine = tiles.  A window of wy x wx mid cells, the
            exterior mid cells absent (the reference value).  The fine region
            is the window's blocks plus a one-tile ring (the furthest any ring
            spills); dem is painted from the window's objects and fixed.
            Tiles outside it are free under the field and do not depend on the
            window, so leaving them out is exact for F(z) - F(r).
TopWindows  coarse = top, fine = mid.  A window of wy x wx top cells, the
            exterior tops at the reference corner r.  The fine region is the
            window's mid cells (free) plus a one-mid-cell halo ring fixed to a
            sample of the forward mid kernel under reference tops (item =
            (z, halo index); halo None: no ring, the window's edge is a free
            boundary).  Factors: lam (painted slot), pres, and the learned
            mid_h, mid_v, mid_u."""
import numpy as np

from .circles import Circles, _paint_dem
from .core import Channel, Factor, Model


class MidWindows:
    def __init__(self, C: Circles, wy=2, wx=2):
        self.C, self.wy, self.wx = C, wy, wx
        self.Hf, self.Wf = wy * C.BM + 2, wx * C.BM + 2

    def dem(self, z):
        """(Hf, Wf) dem of the window's objects over its blocks plus the ring."""
        C, BM = self.C, self.C.BM
        mg = np.zeros((self.wy + 2, self.wx + 2), np.int32)
        mg[1:-1, 1:-1] = z
        d = np.zeros((mg.shape[0] * BM, mg.shape[1] * BM), np.int32)
        _paint_dem(mg, d, BM, C.OY, C.OX, C.SH, C.R1, 0, d.shape[0], 0, d.shape[1])
        return np.ascontiguousarray(d[BM - 1:BM + self.Hf - 1, BM - 1:BM + self.Wf - 1])

    def window_fine_model(self, z):
        C = self.C
        tile = Channel("tile", 1, 2).add_view("col", np.arange(2), 2)
        dem = Channel("dem", 1, 4).add_view("val", np.arange(4), 4)
        tile.grid = np.zeros((self.Hf, self.Wf), np.int32)
        dem.grid = self.dem(z)
        dem.fixed = np.ones(dem.grid.shape, bool)
        fac = [f for f in C.designed_factors() if f.name in ("kappa", "mu")]
        return Model(self.Hf, self.Wf, [tile, dem], fac), "tile"

    def window_free_energy_exact(self, z):
        return float(self.C.fz[self.dem(z)].sum())


class TopWindows:
    def __init__(self, C: Circles, theta, halos=None, wy=2, wx=2, ref=0):
        self.C, self.theta, self.halos, self.wy, self.wx, self.ref = C, theta, halos, wy, wx, ref
        BT = C.BT
        self.ny, self.nx = wy * BT + 2, wx * BT + 2                   # with the halo ring

    def slot(self, z):
        """(ny, nx) slot of the window's tops and the reference tops around it."""
        C, BT = self.C, self.C.BT
        tg = np.full((self.wy + 2, self.wx + 2), self.ref, np.int32)
        tg[1:-1, 1:-1] = z
        sg = np.zeros((tg.shape[0] * BT, tg.shape[1] * BT), np.int32)
        C.paint_slot(tg, sg)
        return np.ascontiguousarray(sg[BT - 1:BT - 1 + self.ny, BT - 1:BT - 1 + self.nx])

    def window_fine_model(self, item):
        z, h = item
        C = self.C
        sl = self.slot(z)
        mg = np.zeros(sl.shape, np.int32)
        fixed = np.zeros(sl.shape, bool)
        if h is None:                                                 # no halo: the inner window only
            sl, mg, fixed = (np.ascontiguousarray(a[1:-1, 1:-1]) for a in (sl, mg, fixed))
        else:
            ring = np.ones(sl.shape, bool)
            ring[1:-1, 1:-1] = False
            mg[ring] = self.halos[h][ring]
            fixed = ring
        mid = Channel("mid", 1, C.D).add_view("self", np.arange(C.D), C.D) \
            .add_view("present", (np.arange(C.D) > 0).astype(np.int64), 2)
        slot = Channel("slot", 1, 2).add_view("val", np.arange(2), 2)
        mid.grid, mid.fixed = mg, fixed
        slot.grid, slot.fixed = sl, np.ones(sl.shape, bool)
        th = self.theta
        M = ("mid", "self")
        fac = [f for f in C.designed_factors() if f.name in ("lam", "pres")] + \
              [Factor.pair(M, M, (0, 1), th["mid_h"], name="mid_h"),
               Factor.pair(M, M, (1, 0), th["mid_v"], name="mid_v"),
               Factor.unary(M, th["mid_u"], name="mid_u")]
        return Model(mg.shape[0], mg.shape[1], [mid, slot], fac), "mid"


def reference_halos(C: Circles, theta, n, wy=2, wx=2, ref=0, sweeps=30, seed=0):
    """n halo samples ((wy BT + 2) x (wx BT + 2) mid arrays; only the ring is
    used): the forward mid kernel (lam via the painted slot, pres, learned mid
    tables) under all-reference tops on a (wy + 2) x (wx + 2) top grid, from
    all-absent, `sweeps` sweeps, independent runs."""
    S = Circles(wy + 2, wx + 2, BM=C.BM, BT=C.BT, kappa=C.kappa, mu=C.mu, lam=C.lam, R=C.R, b=C.b)
    rng = np.random.default_rng(seed)
    BT = C.BT
    out = []
    for _ in range(n):
        chans = S.channels()
        top, mid, tile, slot, dem = chans
        top.grid[:] = ref
        S.paint_slot(top, slot)
        m = S.model(chans, theta)
        m.sweep("mid", sweeps, seed=int(rng.integers(1 << 30)))
        out.append(mid.grid[BT - 1:BT - 1 + wy * BT + 2, BT - 1:BT - 1 + wx * BT + 2].copy())
    return out


# ------------------------------------------------- convpot embeddings (notes/convpot_test.md)
def embed_identity(D, k, seed=0):
    """(D, k) Gaussian rows, one per value: the random table-row baseline
    (cannot generalise across values)."""
    return np.random.default_rng(seed).standard_normal((D, k))


def patch_features(P: Circles, level):
    """(D, F) painted-view features before projection.
    mid: each value's demand map on its block plus the one-tile ring
         ((BM + 2)^2 tiles, channels dirt / air, flattened; absent = zeros);
    top: each corner's BT x BT slot pattern, flattened."""
    if level == "top":
        return P.SLOTPAT.reshape(P.P, -1).astype(float)
    assert level == "mid", level
    BM = P.BM
    out = np.zeros((P.D, 2, BM + 2, BM + 2))
    for v in range(P.D):
        mg = np.zeros((3, 3), np.int32)
        mg[1, 1] = v
        d = np.zeros((3 * BM, 3 * BM), np.int32)
        _paint_dem(mg, d, BM, P.OY, P.OX, P.SH, P.R1, 0, d.shape[0], 0, d.shape[1])
        d = d[BM - 1:2 * BM + 1, BM - 1:2 * BM + 1]
        out[v, 0] = d == 1                                           # dirt (disc)
        out[v, 1] = d == 2                                           # air (ring)
    return out.reshape(P.D, -1)


def embed_patch(P: Circles, level, k, seed=0):
    """(D, k) painted-view embedding: patch_features projected by a fixed
    Gaussian matrix (entries N(0, 1 / F)).  The reference value's row is
    whatever the projection gives it (zero for absent, whose map is empty)."""
    F = patch_features(P, level)
    G = np.random.default_rng(seed).standard_normal((F.shape[1], k)) / np.sqrt(F.shape[1])
    return F @ G


# ------------------------------------------- paint-potential hooks (notes/paintpot_test.md)
# Painted regions for castlegen.channels.paintpot.  Every region holds the
# window, its halo blocks and the painter's spill; background = the
# painter's reference (mid: absent, dem free; top: corner `ref`).
def _dem_crop(P: Circles, mg, ny, nx):
    """dem of the mid grid mg (absent outer ring included by the caller),
    cropped to the blocks [1, ny + 1) x [1, nx + 1) plus a one-tile ring."""
    BM = P.BM
    d = np.zeros((mg.shape[0] * BM, mg.shape[1] * BM), np.int32)
    _paint_dem(np.ascontiguousarray(mg, dtype=np.int32), d, BM, P.OY, P.OX, P.SH, P.R1, 0, d.shape[0], 0,
               d.shape[1])
    return np.ascontiguousarray(d[BM - 1:BM + ny * BM + 1, BM - 1:BM + nx * BM + 1])


def paint_region_mid(P: Circles, zW, halo=None):
    """dem over the window + one-block halo + one-tile spill ring:
    ((wy + 2) BM + 2, (wx + 2) BM + 2).  halo: (wy + 2, wx + 2) mid values
    whose ring is used (interior ignored); None = absent.  Outside the halo
    every block is absent."""
    zW = np.asarray(zW, np.int32)
    wy, wx = zW.shape
    mg = np.zeros((wy + 4, wx + 4), np.int32)
    if halo is not None:
        mg[1:-1, 1:-1] = halo
    mg[2:-2, 2:-2] = zW
    return _dem_crop(P, mg, wy + 2, wx + 2)


def mid_window_F(P: Circles, zW, halo=None):
    """Exact F of the region of paint_region_mid: sum of fz[dem] (tiles are
    independent given dem; tiles outside the region do not depend on zW)."""
    return float(P.fz[paint_region_mid(P, zW, halo)].sum())


_CANVAS = 5          # coarse canvas for single / pair paints: centre (2, 2), every d with |d| <= 1 fits with spill


def paint_single_mid(P: Circles, v):
    mg = np.zeros((_CANVAS, _CANVAS), np.int32)
    mg[2, 2] = v
    return _dem_crop(P, np.pad(mg, 1), _CANVAS, _CANVAS)


def paint_pair_mid(P: Circles, v, vp, d):
    mg = np.zeros((_CANVAS, _CANVAS), np.int32)
    mg[2, 2] = v
    mg[2 + d[0], 2 + d[1]] = vp
    return _dem_crop(P, np.pad(mg, 1), _CANVAS, _CANVAS)


def _slot_of(P: Circles, tg):
    return P.SLOTPAT[np.asarray(tg)].transpose(0, 2, 1, 3).reshape(tg.shape[0] * P.BT, tg.shape[1] * P.BT)


def paint_region_top(P: Circles, zW, halo=None, ref=0):
    """slot at mid resolution over the window + its ring of halo top cells:
    ((wy + 2) BT, (wx + 2) BT).  halo: (wy + 2, wx + 2) top values whose
    ring is used; None = the reference corner.  slot has no spill."""
    zW = np.asarray(zW, np.int32)
    wy, wx = zW.shape
    tg = np.full((wy + 2, wx + 2), ref, np.int32) if halo is None else np.array(halo, np.int32)
    tg[1:-1, 1:-1] = zW
    return _slot_of(P, tg)


def paint_single_top(P: Circles, v, ref=0):
    tg = np.full((_CANVAS, _CANVAS), ref, np.int32)
    tg[2, 2] = v
    return _slot_of(P, tg)


def paint_pair_top(P: Circles, v, vp, d, ref=0):
    tg = np.full((_CANVAS, _CANVAS), ref, np.int32)
    tg[2, 2] = v
    tg[2 + d[0], 2 + d[1]] = vp
    return _slot_of(P, tg)
