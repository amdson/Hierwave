"""The biome circles test: hard honour, object families, a biome that masks them.

notes/circles_biome_test.md.  Tiles (h = 1) air (0) / dirt (1), field mu on
air.  Mid slots (h = BM) hold an object: absent (0) or one placement of one
family.  A family is data, (name, footprint cells relative to an anchor,
anchor offsets inside the block); its value range is contiguous, in family
order, offsets in the given order.  Default: disc (13-cell radius-2 disc,
centres {2..5}^2, values 1..16) and bar (2 x 4, top-left ry in {2..5}, rx in
{1..4}, values 17..32).  Every object demands dirt on its footprint and air on
its ring (8-neighbour dilation minus the footprint); footprints stay in their
block, rings spill at most one tile.  Top cells (h = BM * BT): biome, a
bitmask over families (D = 2^N; N = 2: none, discs, bars, both).

Painted channels (fixed, refreshed when the source changes):
    allow  h = BM, D = 2^N: the parent biome's value
    dem    h = 1,  D = 4: 0 free, 1 dirt, 2 air, 3 conflict (dirt of one object
                          on the ring of another)
Designed factors (position free):
    mask    pair (obj, fam) x (allow, val)  INF when the family's bit is clear
    pres    unary (obj, self)               -b_fam on the family's values
    honour  pair (tile, col) x (dem, val)   free 0, dirt / air demand INF on the
                                            other colour, conflict INF on both
    mu      unary (tile, col)               [mu, 0]
    bio_u0  unary (biome, pal)              BT^2 log(1 + #families admitted)
                                            (named bio_u0: the learned unary is bio_u)
Given the objects the tiles are independent, per-tile free energy fz = (f0,
0, mu, INF) for free / dirt / air / conflict, f0 = -log(1 + e^-mu).  The
support (pairs of objects that make a conflict tile) is pairwise between
slots at the 8 neighbouring offsets: support_factors() gives it as hard pair
tables on the four canonical offsets (the generic layer adds the reflections).

Provides the model, painters, support, learned factors, stamp_features (the
paint-potential features of a candidate, contract of the note), admissible,
stats, exact reference, symmetrise (the dihedral elements that map the
family stamp set to itself), render, Forward (generic kernel, dormancy) and
Oracle (exact collapsed moves)."""
import time

import numpy as np
from numba import njit

from .core import Channel, Factor, Model
from . import paintpot

AIR, DIRT = 0, 1
FREE, DDIRT, DAIR, CONFLICT = 0, 1, 2, 3
INF = np.inf
OFFS = [(0, 1), (1, 0), (1, 1), (1, -1)]            # obj_h, obj_v, obj_d1, obj_d2 (canonical: reflections by core)
OFFNAMES = ["obj_h", "obj_v", "obj_d1", "obj_d2"]
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def default_families():
    disc = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if a * a + b * b <= 4]
    bar = [(a, b) for a in range(2) for b in range(4)]
    return [("disc", disc, [(cy, cx) for cy in range(2, 6) for cx in range(2, 6)]),
            ("bar", bar, [(ry, rx) for ry in range(2, 6) for rx in range(1, 5)])]


# ------------------------------------------------------------ numba core
@njit(cache=True, inline="always")
def _dem_of(nd, nr):
    if nd > 0:
        return 3 if nr > 0 else 1
    return 2 if nr > 0 else 0


@njit(cache=True)
def _nd_nr(obj, y, x, BM, SHB, si, sj):
    """Dirt and ring counts at tile (y, x) from the objects of the blocks
    around it, slot (si, sj) skipped."""
    nmy, nmx = obj.shape
    bi = y // BM
    bj = x // BM
    nd = 0
    nr = 0
    for a in range(bi - 1, bi + 2):
        if a < 0 or a >= nmy:
            continue
        ly = y - a * BM + 1
        if ly < 0 or ly >= BM + 2:
            continue
        for b in range(bj - 1, bj + 2):
            if b < 0 or b >= nmx or (a == si and b == sj):
                continue
            o = obj[a, b]
            if o == 0:
                continue
            lx = x - b * BM + 1
            if lx < 0 or lx >= BM + 2:
                continue
            s = SHB[o, ly, lx]
            if s == 1:
                nd += 1
            elif s == 2:
                nr += 1
    return nd, nr


@njit(cache=True)
def _paint_dem(obj, dem, BM, SHB, y0, y1, x0, x1):
    H, W = dem.shape
    for y in range(max(y0, 0), min(y1, H)):
        for x in range(max(x0, 0), min(x1, W)):
            nd, nr = _nd_nr(obj, y, x, BM, SHB, -1, -1)
            dem[y, x] = _dem_of(nd, nr)


@njit(cache=True)
def _delta(obj, i, j, H, W, BM, SHB, SY, SX, SS, SN, fz, out):
    """out[o]: the tile free-energy change of putting o in slot (i, j) against
    absent there, every touched tile integrated out (INF on a new conflict)."""
    D = out.shape[0]
    out[0] = 0.0
    for o in range(1, D):
        e = 0.0
        for k in range(SN[o]):
            y = i * BM + SY[o, k]
            x = j * BM + SX[o, k]
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            nd, nr = _nd_nr(obj, y, x, BM, SHB, i, j)
            d0 = _dem_of(nd, nr)
            if d0 == 3:
                continue                                  # a conflict not made by o
            d1 = _dem_of(nd + 1, nr) if SS[o, k] == 1 else _dem_of(nd, nr + 1)
            if d1 == 3:
                e = np.inf
                break
            e += fz[d1] - fz[d0]
        out[o] = e


@njit(cache=True)
def _redraw(tile, dem, old, y0, y1, x0, x1, mu, u, only_changed):
    """Tiles of the window from their exact conditional given the demand
    (free: air with prob e^-mu / (1 + e^-mu)); only_changed: just the tiles
    whose demand differs from old (window-shaped)."""
    H, W = tile.shape
    pa = np.exp(-mu) / (1.0 + np.exp(-mu))
    k = 0
    for y in range(max(y0, 0), min(y1, H)):
        for x in range(max(x0, 0), min(x1, W)):
            d = dem[y, x]
            if only_changed and old[y - y0, x - x0] == d:
                continue
            if d == 1:
                tile[y, x] = 1
            elif d == 2:
                tile[y, x] = 0
            elif d == 0:
                tile[y, x] = 0 if u[k] < pa else 1
            k += 1


@njit(cache=True)
def _cand_dem(obj, i, j, y0, y1, x0, x1, BM, SHB, cand, out):
    """out[c] = dem over the window [y0, y1) x [x0, x1) with cand[c] at (i, j)."""
    for y in range(y0, y1):
        for x in range(x0, x1):
            nd, nr = _nd_nr(obj, y, x, BM, SHB, i, j)
            ly = y - i * BM + 1
            lx = x - j * BM + 1
            inside = 0 <= ly < BM + 2 and 0 <= lx < BM + 2
            for c in range(cand.shape[0]):
                s = SHB[cand[c], ly, lx] if inside else 0
                out[c, y - y0, x - x0] = _dem_of(nd + (s == 1), nr + (s == 2))


def _softmax(logw):
    p = np.exp(logw - logw.max())
    return p / p.sum()


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return (np.log(np.exp(a - m).sum(axis=axis, keepdims=True)) + m).squeeze(axis)


def dcentre_finite(A, iters=200, tol=1e-12):
    """Double centering over the finite entries: A minus its least-squares
    additive fit r_i + c_j on the finite entries (alternating row / column
    mean removal); INF entries stay INF.  Plain double centering when all
    entries are finite."""
    A = np.asarray(A, float)
    fin = np.isfinite(A)
    R = np.where(fin, A, 0.0)
    nr, nc = np.maximum(fin.sum(1), 1), np.maximum(fin.sum(0), 1)
    for _ in range(iters):
        R = np.where(fin, R - (R.sum(1) / nr)[:, None], 0.0)
        cm = R.sum(0) / nc
        R = np.where(fin, R - cm[None, :], 0.0)
        if np.abs(cm).max() < tol and np.abs(R.sum(1) / nr).max() < tol:
            break
    return np.where(fin, R, INF)


def _grid(c):
    return c.grid if hasattr(c, "grid") else c


# ------------------------------------------------------------------ model
class CirclesBiome:
    def __init__(self, nty, ntx, BM=8, BT=2, mu=0.3, b=None, families=None):
        fams = default_families() if families is None else families
        self.nty, self.ntx, self.BM, self.BT, self.mu = nty, ntx, BM, BT, float(mu)
        self.nmy, self.nmx = nty * BT, ntx * BT
        self.H, self.W = self.nmy * BM, self.nmx * BM
        self.NF = len(fams)
        self.names = [f[0] for f in fams]
        self.P = 1 << self.NF
        FAM, ANCH, STAMP = [0], [(-1, -1)], [[]]
        self.n_off = []
        for f, (name, cells, offs) in enumerate(fams):
            cs = {(int(a), int(c)) for a, c in cells}
            dil = {(a + u, c + v) for a, c in cs for u in (-1, 0, 1) for v in (-1, 0, 1)}
            ring = sorted(dil - cs)
            self.n_off.append(len(offs))
            for ay, ax in offs:
                st = [(ay + a, ax + c, DDIRT) for a, c in sorted(cs)] + [(ay + a, ax + c, DAIR) for a, c in ring]
                assert all(0 <= y < BM and 0 <= x < BM for y, x, s in st if s == DDIRT), (name, ay, ax)
                assert all(-1 <= y <= BM and -1 <= x <= BM for y, x, s in st), (name, ay, ax)
                FAM.append(f + 1); ANCH.append((ay, ax)); STAMP.append(st)
        self.D = len(FAM)
        self.FAM = np.array(FAM, np.int64)
        self.ANCH = np.array(ANCH, np.int64)
        self.STAMP = STAMP
        L = max(len(s) for s in STAMP)
        self.SY, self.SX, self.SS = (np.zeros((self.D, L), np.int64) for _ in range(3))
        self.SN = np.array([len(s) for s in STAMP], np.int64)
        self.SHB = np.zeros((self.D, BM + 2, BM + 2), np.int64)        # block frame, origin at (-1, -1)
        for o in range(1, self.D):
            for k, (y, x, s) in enumerate(STAMP[o]):
                self.SY[o, k], self.SX[o, k], self.SS[o, k] = y, x, s
                self.SHB[o, y + 1, x + 1] = s
        m = self.mu
        self.f0 = -np.log1p(np.exp(-m))
        self.fz = np.array([self.f0, 0.0, m, INF])                       # per-tile -log Z by demand
        self.kt = np.array([[0.0, INF, 0.0, INF],                        # air:  free, dirt-dem, air-dem, conflict
                            [0.0, 0.0, INF, INF]])                       # dirt
        bits = np.arange(self.P)
        # FAMOK[f, a]: family f (0 = absent) admitted by allow value a
        self.FAMOK = np.ones((self.NF + 1, self.P), bool)
        for f in range(self.NF):
            self.FAMOK[f + 1] = (bits >> f) & 1 == 1
        self.mask_t = np.where(self.FAMOK, 0.0, INF)                     # (N + 1, P)
        self.mask_e = self.mask_t[self.FAM]                              # (D, P)
        bd = {} if b is None else dict(b)
        self.b = np.array([bd.get(n, self.presence_bonus(n)) for n in self.names])
        self.pres_e = np.concatenate([[0.0], -self.b[self.FAM[1:] - 1]])
        nadm = np.array([bin(v).count("1") for v in range(self.P)])
        self.bio_u0 = BT * BT * np.log1p(nadm)
        self._canvas()
        self._sym = self._group()

    # ---------------------------------------------------------- stamps
    def _fam_index(self, fam):
        return self.names.index(fam) if isinstance(fam, str) else int(fam)

    def object_cost(self, fam):
        """F_fam: sum over one object's footprint of fz[dem] - f0 (interior
        block, neighbours empty; the same for every offset).  fam: name or
        0-based family index."""
        f = self._fam_index(fam)
        o = int(np.nonzero(self.FAM == f + 1)[0][0])
        s = self.SS[o, :self.SN[o]]
        return float((self.fz[s] - self.f0).sum())

    def presence_bonus(self, fam):
        """F_fam - log(#offsets): an isolated allowed slot is 50/50 absent / present."""
        f = self._fam_index(fam)
        return self.object_cost(f) - np.log(self.n_off[f])

    def _canvas(self):
        """Stamp masks on a 3 x 3 block canvas (plus a one-tile margin): the
        exact pair tables, the support and the contact tables, all by
        counting tiles."""
        BM, D = self.BM, self.D
        S = 3 * BM + 2
        def masks(by, bx):
            M = np.zeros((3, D, S, S), bool)          # 0 dirt, 1 ring, 2 any (dilated by 4-nbrs: 3)
            for o in range(1, D):
                for y, x, s in self.STAMP[o]:
                    M[s - 1, o, 1 + by * BM + y, 1 + bx * BM + x] = True
            M[2] = M[0] | M[1]
            return M.reshape(3, D, -1)
        A = masks(1, 1)
        anyA = A[2].reshape(D, S, S)
        tch = anyA.copy()
        tch[:, 1:] |= anyA[:, :-1]; tch[:, :-1] |= anyA[:, 1:]
        tch[:, :, 1:] |= anyA[:, :, :-1]; tch[:, :, :-1] |= anyA[:, :, 1:]
        tch = tch.reshape(D, -1).astype(float)
        fz = self.fz
        dem = lambda nd, nr: 3 if (nd and nr) else (1 if nd else (2 if nr else 0))
        # per-tile pair contribution c[sA, sB] (s = 0 none, 1 dirt, 2 ring)
        c = np.zeros((3, 3))
        for sa in range(3):
            for sb in range(3):
                nd, nr = (sa == 1) + (sb == 1), (sa == 2) + (sb == 2)
                if dem(nd, nr) == 3:
                    c[sa, sb] = INF
                else:
                    c[sa, sb] = fz[dem(nd, nr)] - fz[dem(sa == 1, sa == 2)] - fz[dem(sb == 1, sb == 2)] + fz[0]
        self.PAIR, self.CONF, self.CONTACT = {}, {}, {}
        Af = A.astype(float)
        for d in OFFS:
            B = masks(1 + d[0], 1 + d[1]).astype(float)
            raw = np.zeros((D, D))
            conf = np.zeros((D, D), bool)
            for sa in (1, 2):
                for sb in (1, 2):
                    n = Af[sa - 1] @ B[sb - 1].T
                    if np.isinf(c[sa, sb]):
                        conf |= n > 0
                    else:
                        raw += n * c[sa, sb]
            self.PAIR[d] = np.where(conf, INF, raw)
            self.CONF[d] = conf
            self.CONTACT[d] = (tch @ B[2].T) > 0
        # directed support over the 8 neighbours: CONF8[k][t, t'] = t at p, t' at p + DIR8[k]
        self.CONF8 = np.zeros((8, D, D), bool)
        for k, (dy, dx) in enumerate(DIR8):
            if (dy, dx) in self.CONF:
                self.CONF8[k] = self.CONF[(dy, dx)]
            else:
                self.CONF8[k] = self.CONF[(-dy, -dx)].T

    # --------------------------------------------------------- channels
    def channels(self):
        D, P = self.D, self.P
        biome = Channel("biome", self.BM * self.BT, P).add_view("pal", np.arange(P), P)
        for f, n in enumerate(self.names):
            biome.add_view("allow_" + n, (np.arange(P) >> f) & 1, 2)
        obj = Channel("obj", self.BM, D).add_view("self", np.arange(D), D) \
            .add_view("present", (self.FAM > 0).astype(np.int64), 2).add_view("fam", self.FAM, self.NF + 1)
        tile = Channel("tile", 1, 2).add_view("col", np.arange(2), 2)
        allow = Channel("allow", self.BM, P).add_view("val", np.arange(P), P)
        dem = Channel("dem", 1, 4).add_view("val", np.arange(4), 4)
        biome.grid = np.zeros((self.nty, self.ntx), np.int32)
        for c in (obj, allow):
            c.grid = np.zeros((self.nmy, self.nmx), np.int32)
        for c in (tile, dem):
            c.grid = np.zeros((self.H, self.W), np.int32)
        for c in (biome, obj, tile, allow, dem):
            c.fixed = np.zeros(c.grid.shape, bool)
        allow.fixed[:] = True
        dem.fixed[:] = True
        self.paint_allow(biome, allow)
        self.paint_dem(obj, dem)
        return biome, obj, tile, allow, dem

    def soft_mask_factor(self, lam=3.0):
        """The mask with lam in place of INF (post-relaxation: honour loosened)."""
        return Factor.pair(("obj", "fam"), ("allow", "val"), (0, 0), np.where(self.FAMOK, 0.0, float(lam)), name="mask")

    def designed_factors(self, soft_mask=None):
        """soft_mask = lam: the mask softened (soft_mask_factor) instead of hard."""
        mask = Factor.pair(("obj", "fam"), ("allow", "val"), (0, 0), self.mask_t, name="mask") \
            if soft_mask is None else self.soft_mask_factor(soft_mask)
        return [mask,
                Factor.unary(("obj", "self"), self.pres_e, name="pres"),
                Factor.pair(("tile", "col"), ("dem", "val"), (0, 0), self.kt, name="honour"),
                Factor.unary(("tile", "col"), np.array([self.mu, 0.0]), name="mu"),
                Factor.unary(("biome", "pal"), self.bio_u0, name="bio_u0")]

    def support_factors(self):
        """Hard pair tables on obj at the four canonical offsets (0,1), (1,0),
        (1,1), (1,-1): INF iff the two stamps make a conflict tile.  The
        generic layer packs each with its reflection, so the 8 neighbours are
        covered."""
        S = ("obj", "self")
        return [Factor.pair(S, S, d, np.where(self.CONF[d], INF, 0.0), name="sup_" + n[4:])
                for d, n in zip(OFFS, OFFNAMES)]

    def learned_factors(self, theta):
        """Tabular learned factors; a key missing from theta is left out."""
        M, T = ("obj", "self"), ("biome", "pal")
        mk = {n: (lambda t, d=d, n=n: Factor.pair(M, M, d, t, name=n)) for d, n in zip(OFFS, OFFNAMES)}
        mk.update(obj_u=lambda t: Factor.unary(M, t, name="obj_u"),
                  bio_h=lambda t: Factor.pair(T, T, (0, 1), t, name="bio_h"),
                  bio_v=lambda t: Factor.pair(T, T, (1, 0), t, name="bio_v"),
                  bio_u=lambda t: Factor.unary(T, t, name="bio_u"))
        return [f(theta[k]) for k, f in mk.items() if k in theta]

    def theta0(self):
        D, P = self.D, self.P
        th = {n: np.zeros((D, D)) for n in OFFNAMES}
        th.update(obj_u=np.zeros(D), bio_h=np.zeros((P, P)), bio_v=np.zeros((P, P)), bio_u=np.zeros(P))
        return th

    def model(self, chans, theta=None, extra=(), soft_mask=None):
        """All five channels (allow, dem fixed everywhere); designed factors
        (mask softened to lam = soft_mask when given), the learned ones when
        theta is given, plus `extra`."""
        for c in chans:
            if c.name in ("allow", "dem"):
                c.fixed = np.ones(c.grid.shape, bool)
        fac = self.designed_factors(soft_mask) + (self.learned_factors(theta) if theta is not None else []) + list(extra)
        return Model(self.H, self.W, list(chans), fac)

    # ---------------------------------------------------------- painters
    def paint_allow(self, biome, allow, region=None):
        """allow[slot] = its parent's biome; region = (I, J): that top block only."""
        bg, ag = _grid(biome), _grid(allow)
        BT = self.BT
        if region is None:
            ag[:] = np.repeat(np.repeat(bg, BT, 0), BT, 1)
        else:
            I, J = region
            ag[I * BT:(I + 1) * BT, J * BT:(J + 1) * BT] = bg[I, J]

    def paint_dem(self, obj, dem, window=None):
        """Union of the objects' demands; window = (y0, y1, x0, x1) repaints those tiles."""
        og, dg = _grid(obj), _grid(dem)
        y0, y1, x0, x1 = (0, dg.shape[0], 0, dg.shape[1]) if window is None else window
        _paint_dem(og, dg, self.BM, self.SHB, y0, y1, x0, x1)

    def dem_of(self, obj):
        og = _grid(obj)
        d = np.zeros((og.shape[0] * self.BM, og.shape[1] * self.BM), np.int32)
        self.paint_dem(og, d)
        return d

    def dormant_of(self, allow):
        """Slots whose allow admits no family (only absent is admissible)."""
        return ~self.FAMOK[1:, _grid(allow)].any(0)

    def admissible(self, obj, i, j, allow=None):
        """(D,) bool: values at slot (i, j) with no support conflict against the
        other slots (and the mask, when allow is given)."""
        og = _grid(obj)
        ok = np.ones(self.D, bool)
        for k, (dy, dx) in enumerate(DIR8):
            y, x = i + dy, j + dx
            if 0 <= y < og.shape[0] and 0 <= x < og.shape[1]:
                ok &= ~self.CONF8[k][:, og[y, x]]
        if allow is not None:
            ok &= self.FAMOK[self.FAM, _grid(allow)[i, j]]
        return ok

    def stamp_features(self, obj, i, j, cand, offsets=paintpot.OFF8):
        """(len(cand), nfeat): paintpot.features of the dem paint with cand[k]
        at (i, j) minus the same with absent there, over the block plus a
        two-tile border (every pair touching a changed tile; equal to the
        full-grid difference)."""
        og = _grid(obj)
        BM = self.BM
        y0, y1 = max(i * BM - 2, 0), min((i + 1) * BM + 2, og.shape[0] * BM)
        x0, x1 = max(j * BM - 2, 0), min((j + 1) * BM + 2, og.shape[1] * BM)
        c = np.concatenate([[0], np.asarray(cand, np.int64)])
        out = np.empty((len(c), y1 - y0, x1 - x0), np.int64)
        _cand_dem(og, i, j, y0, y1, x0, x1, BM, self.SHB, c, out)
        X = paintpot.features(out, 4, offsets)
        return X[1:] - X[0]

    # ------------------------------------------------------- statistics
    @staticmethod
    def _pairs(g, n):
        out = []
        for a, b in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :]), (g[:-1, :-1], g[1:, 1:]), (g[:-1, 1:], g[1:, :-1])):
            t = np.zeros((n, n))
            np.add.at(t, (a.ravel(), b.ravel()), 1)
            out.append(t / max(t.sum(), 1))
        return out

    def stats(self, biome, obj, tile):
        D, P, BM, BT = self.D, self.P, self.BM, self.BT
        bg, og, fg = _grid(biome), _grid(obj), _grid(tile)
        oh, ov, od1, od2 = self._pairs(og, D)
        bh, bv, _, _ = self._pairs(bg, P)
        bu = np.bincount(bg.ravel(), minlength=P) / bg.size
        dem = self.dem_of(og)
        pres = og > 0
        ph, pv = pres[:, :-1] & pres[:, 1:], pres[:-1] & pres[1:]
        nc = self.CONTACT[(0, 1)][og[:, :-1], og[:, 1:]][ph].sum() + self.CONTACT[(1, 0)][og[:-1], og[1:]][pv].sum()
        npair = ph.sum() + pv.sum()
        B = BM * BT
        yy, xx = np.mgrid[:self.H, :self.W]
        band = ((xx % B == B - 1) & (xx < self.W - 1)) | ((xx % B == 0) & (xx > 0)) | \
               ((yy % B == B - 1) & (yy < self.H - 1)) | ((yy % B == 0) & (yy > 0))
        allow = np.repeat(np.repeat(bg, BT, 0), BT, 1)
        s = dict(obj_h=oh, obj_v=ov, obj_d1=od1, obj_d2=od2, obj_u=np.bincount(og.ravel(), minlength=D) / og.size,
                 bio_h=bh, bio_v=bv, bio_u=bu)
        fam = self.FAM[og]
        for f, n in enumerate(self.names):
            s["present_" + n] = float((fam == f + 1).mean())
        s.update(conflict=float((dem == CONFLICT).mean()),
                 contact=float(nc / npair) if npair else 0.0,
                 bio_hist=bu.copy(),
                 dormant=float(self.dormant_of(allow).mean()),
                 mask_viol=float((~self.FAMOK[fam, allow]).mean()),
                 edge_air_top=float((fg[band] == AIR).mean()) if band.any() else 0.0)
        return s

    # ---------------------------------------------------- symmetrisation
    GENS = dict(x=((1, 0), (0, -1)), y=((-1, 0), (0, 1)), t=((0, 1), (1, 0)))   # matrices on (dy, dx)
    GROUP = [(fx, fy, tr) for fx in (0, 1) for fy in (0, 1) for tr in (0, 1)]

    def _pt(self, g, y, x):
        """Block-frame point (y, x) under g = (flip x, flip y, transpose), in that order."""
        L = self.BM - 1
        if g[0]:
            x = L - x
        if g[1]:
            y = L - y
        if g[2]:
            y, x = x, y
        return y, x

    def _group(self):
        """{g: (value map (D,), biome map (P,), linear map on offsets)} for the
        dihedral elements that map the stamp set to itself (family of the image
        stamp may differ: families are permuted, biome bits with them)."""
        key = {frozenset(map(tuple, s)): o for o, s in enumerate(self.STAMP)}
        out = {}
        for g in self.GROUP:
            m = np.zeros(self.D, np.int64)
            ok = True
            for o in range(1, self.D):
                img = frozenset((*self._pt(g, y, x), s) for y, x, s in self.STAMP[o])
                if img not in key:
                    ok = False
                    break
                m[o] = key[img]
            if not ok:
                continue
            fperm = np.zeros(self.NF + 1, np.int64)
            for o in range(1, self.D):
                fperm[self.FAM[o]] = self.FAM[m[o]]
            if len(set(fperm[1:])) != self.NF:
                continue
            bm = np.zeros(self.P, np.int64)
            for v in range(self.P):
                for f in range(self.NF):
                    if (v >> f) & 1:
                        bm[v] |= 1 << (fperm[f + 1] - 1)
            M = np.eye(2, dtype=np.int64)
            for flag, k in zip(g, "xyt"):
                if flag:
                    M = np.array(self.GENS[k]) @ M
            out[g] = (m, bm, M, fperm)
        return out

    def group(self):
        """The dihedral elements kept by symmetrise (g = (flip x, flip y, transpose))."""
        return list(self._sym)

    def transform_stats(self, s, g):
        m, bm, M, fperm = self._sym[g]
        out = dict(s)
        for maps, keys in ((m, dict(zip(OFFNAMES, OFFS))), (bm, {"bio_h": (0, 1), "bio_v": (1, 0)})):
            vecs = {k: np.array(d) for k, d in keys.items()}
            for k, d in keys.items():
                gd = M @ np.array(d)
                for k2, d2 in vecs.items():
                    if np.array_equal(gd, d2):
                        t = np.zeros_like(s[k]); t[np.ix_(maps, maps)] = s[k]; out[k2] = t
                    elif np.array_equal(gd, -d2):
                        t = np.zeros_like(s[k]); t[np.ix_(maps, maps)] = s[k].T; out[k2] = t
        for k, mp in (("obj_u", m), ("bio_u", bm), ("bio_hist", bm)):
            t = np.zeros_like(s[k]); t[mp] = s[k]; out[k] = t
        for f, n in enumerate(self.names):
            out["present_" + self.names[fperm[f + 1] - 1]] = s["present_" + n]
        return out

    def transform_state(self, biome, obj, tile, g):
        m, bm, M, fperm = self._sym[g]
        bg, og, fg = _grid(biome).copy(), _grid(obj).copy(), _grid(tile).copy()
        for flag, k in zip(g, "xyt"):
            if flag:
                f = (lambda a: a[:, ::-1]) if k == "x" else (lambda a: a[::-1]) if k == "y" else (lambda a: a.T)
                bg, og, fg = f(bg), f(og), f(fg)
        return (np.ascontiguousarray(bm[bg]).astype(np.int32), np.ascontiguousarray(m[og]).astype(np.int32),
                np.ascontiguousarray(fg))

    def symmetrise(self, stats):
        """Average over group() (exact on the stats; a symmetry of p* when
        nty == ntx or the group has no transpose).  Default families: the bar
        offsets (ry 2..5, rx 1..4) are not mirror symmetric in the block, so
        the group is the identity alone."""
        out = None
        for g in self._sym:
            t = self.transform_stats(stats, g)
            out = t if out is None else {k: out[k] + t[k] for k in out}
        return {k: v / len(self._sym) for k, v in out.items()}

    # ------------------------------------------------------- reference
    def reference(self, centred=True):
        """Exact mid tables (interior blocks, the other blocks empty), counted
        over tiles: obj_h / obj_v / obj_d1 / obj_d2 = F(o, o') - F(o, -) - F(-, o')
        + F(-, -), INF on the support; obj_u = F(o) - F(-) - b_fam(o);
        bio_u at zeroth order (slots independent): -BT^2 log sum_o
        exp(-obj_u(o)) over the admitted values (0 at none); bio_h, bio_v
        None (oracle only).  centred: pair tables double-centred over their
        finite entries (dcentre_finite), unaries centred."""
        ref = {n: self.PAIR[d].copy() for d, n in zip(OFFS, OFFNAMES)}
        u = np.zeros(self.D)
        for o in range(1, self.D):
            u[o] = (self.fz[self.SS[o, :self.SN[o]]] - self.f0).sum() + self.pres_e[o]
        ref["obj_u"] = u
        ok = self.FAMOK[self.FAM]                                   # (D, P)
        w = np.exp(-u)[:, None] * ok
        ref["bio_u"] = -self.BT ** 2 * np.log(w.sum(0))
        ref["bio_h"] = ref["bio_v"] = None
        if centred:
            for n in OFFNAMES:
                ref[n] = dcentre_finite(ref[n])
            for k in ("obj_u", "bio_u"):
                ref[k] = ref[k] - ref[k].mean()
        return ref

    # ------------------------------------------------------------ render
    FCOL = [(215, 40, 40), (40, 90, 220), (40, 170, 70), (200, 150, 20), (150, 50, 190), (20, 170, 180)]

    def render(self, biome, obj, tile):
        """(H, W, 3) uint8: dirt brown, air light; footprint (dirt) tiles of each
        object tinted by family colour, its anchor dark; slot edges thin, top
        edges thick; dormant (biome none) blocks hatched."""
        bg, og, fg = _grid(biome), _grid(obj), _grid(tile)
        base = np.array([(232, 236, 242), (130, 88, 50)], np.float64)
        img = base[fg].copy()
        BM, B = self.BM, self.BM * self.BT
        rng = np.random.default_rng(12345)
        fcol = [self.FCOL[f] if f < len(self.FCOL) else tuple(rng.integers(40, 220, 3)) for f in range(self.NF)]
        for i, j in zip(*np.nonzero(og)):
            o = og[i, j]
            c = np.array(fcol[self.FAM[o] - 1], float)
            for y, x, s in self.STAMP[o]:
                Y, X = i * BM + y, j * BM + x
                if s == DDIRT and fg[Y, X] == DIRT:
                    img[Y, X] = 0.4 * img[Y, X] + 0.6 * c
            img[i * BM + self.ANCH[o, 0], j * BM + self.ANCH[o, 1]] = 0.5 * c
        yy, xx = np.mgrid[:self.H, :self.W]
        dorm = (bg == 0)[yy // B, xx // B]
        hatch = dorm & ((yy + xx) % 4 == 0)
        img[dorm] = 0.75 * img[dorm] + 0.25 * np.array([120, 120, 140])
        img[hatch] *= 0.55
        mid_edge = (yy % BM == 0) | (xx % BM == 0)
        top_edge = (yy % B == 0) | (xx % B == 0) | (yy % B == B - 1) | (xx % B == B - 1)
        img[mid_edge] *= 0.8
        img[top_edge] *= 0.5 / np.where(mid_edge[top_edge], 0.8, 1.0)[:, None]
        return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------- forward model
class Forward:
    """Biome sweeps, paint allow, obj init (dormancy) + sweeps, paint dem, tile
    sweeps, on the generic kernel with designed + learned (+ support) factors.

    use_sampler: the chain on sampler.Sampler, one per channel (biome, obj with
    hb = BT so a dormant top block is skipped whole, tile): obj init() fixes
    the slots whose hard parent rows admit absent alone (dormant fraction in
    self.dormant_frac), S_M sweeps with the candidate cap K, then p_relax
    sweeps under the same channels with the mask softened to lam (relax; the
    support stays hard, dormant slots stay fixed).  K = None, hb = 1 draws
    exactly what the old path draws.  Wall times per stage in self.times."""

    def __init__(self, C: CirclesBiome, theta: dict, seed=0, extra=(), support=True, use_sampler=False, K=None,
                 p_relax=0, lam=3.0, hb=None):
        self.C, self.theta = C, theta
        self.extra = list(extra) + (C.support_factors() if support else [])
        self.chans = C.channels()
        self.biome, self.obj, self.tile, self.allow, self.dem = self.chans
        self.model = None
        self.n_dormant = 0
        self.rng = np.random.default_rng(seed)
        self.use_sampler, self.K, self.p_relax, self.lam = use_sampler, K, int(p_relax), lam
        self.hb = C.BT if hb is None else hb
        self.dormant_frac = 0.0
        self.times = {}

    def init_obj(self):
        """Paint allow; a slot whose allow admits no family is set to absent and
        fixed (dormant), the others freed.  Returns the dormant count."""
        C = self.C
        C.paint_allow(self.biome, self.allow)
        dorm = C.dormant_of(self.allow)
        self.obj.grid[dorm] = 0
        self.obj.fixed[:] = dorm
        self.n_dormant = int(dorm.sum())
        return self.n_dormant

    def build(self):
        """Model (and samplers) at the current theta."""
        from .sampler import Sampler
        C = self.C
        self.model = C.model(self.chans, self.theta, self.extra)
        if self.use_sampler:
            self.soft = C.model(self.chans, self.theta, self.extra, soft_mask=self.lam) if self.p_relax else None
            self.S = dict(biome=Sampler(self.model, "biome"), obj=Sampler(self.model, "obj", K=self.K, hb=self.hb),
                          tile=Sampler(self.model, "tile"))

    def run(self, S_T=30, S_M=30, S_F=20, fresh=True):
        C = self.C
        self.build()
        if fresh:
            self.biome.grid[:] = 0
            self.obj.grid[:] = 0
            self.tile.grid[:] = 0
        if self.use_sampler:
            return self._run_sampler(S_T, S_M, S_F)
        if S_T:
            self.model.sweep("biome", S_T, seed=int(self.rng.integers(1 << 30)))
        self.init_obj()
        if S_M:
            self.model.sweep("obj", S_M, seed=int(self.rng.integers(1 << 30)))
        C.paint_dem(self.obj, self.dem)
        if S_F:
            self.model.sweep("tile", S_F, seed=int(self.rng.integers(1 << 30)))
        return C.stats(self.biome, self.obj, self.tile)

    def _run_sampler(self, S_T, S_M, S_F):
        C, S, tm = self.C, self.S, time.perf_counter
        t0 = tm()
        if S_T:
            S["biome"].sweep(S_T, seed=int(self.rng.integers(1 << 30)))
        C.paint_allow(self.biome, self.allow)
        self.obj.fixed[:] = False
        t1 = tm()
        self.dormant_frac = S["obj"].init()
        self.n_dormant = int(S["obj"].dormant.sum())
        t1b = tm()
        if S_M:
            S["obj"].sweep(S_M, seed=int(self.rng.integers(1 << 30)))
        t2 = tm()
        if self.p_relax:
            S["obj"].relax(self.p_relax, self.soft, seed=int(self.rng.integers(1 << 30)))
        t3 = tm()
        C.paint_dem(self.obj, self.dem)
        if S_F:
            S["tile"].sweep(S_F, seed=int(self.rng.integers(1 << 30)))
        t4 = tm()
        self.times = dict(biome=t1 - t0, obj_init=t1b - t1, obj=t2 - t1b, relax=t3 - t2, tile=t4 - t3)
        return C.stats(self.biome, self.obj, self.tile)


# ---------------------------------------------------- oracle: two-way
class Oracle:
    """Exact sampler of p* by collapsed moves: biome by p*(T | its slots), each
    slot by p*(o | allow, other slots) with every touched tile integrated
    out (then the tiles whose demand changed redrawn), tiles by the kernel."""

    def __init__(self, C: CirclesBiome, seed=0):
        self.C = C
        self.rng = np.random.default_rng(seed)
        self.chans = C.channels()
        self.biome, self.obj, self.tile, self.allow, self.dem = self.chans
        self.biome.grid[:] = self.rng.integers(C.P, size=self.biome.grid.shape)
        C.paint_allow(self.biome, self.allow)
        C.paint_dem(self.obj, self.dem)
        _redraw(self.tile.grid, self.dem.grid, self.dem.grid, 0, C.H, 0, C.W, C.mu, self.rng.random(C.H * C.W), False)
        self.model = C.model(self.chans)
        self._e = np.empty(C.D)

    # -------------------------------------------------------------- top
    def top_probs(self, i, j):
        """p*(T | its BT x BT slots): exp(-bio_u0[T]) on the values whose mask
        admits every present family."""
        C = self.C
        BT = C.BT
        f = np.unique(C.FAM[self.obj.grid[i * BT:(i + 1) * BT, j * BT:(j + 1) * BT]])
        ok = C.FAMOK[f].all(0)
        return _softmax(np.where(ok, -C.bio_u0, -INF))

    def top_move(self, i, j):
        p = self.top_probs(i, j)
        self.biome.grid[i, j] = int(np.searchsorted(np.cumsum(p), self.rng.random() * p.sum(), side="right"))
        self.C.paint_allow(self.biome, self.allow, (i, j))

    # -------------------------------------------------------------- mid
    def mid_energies(self, i, j):
        C = self.C
        _delta(self.obj.grid, i, j, C.H, C.W, C.BM, C.SHB, C.SY, C.SX, C.SS, C.SN, C.fz, self._e)
        return self._e + C.pres_e + C.mask_e[:, self.allow.grid[i, j]]

    def mid_probs(self, i, j):
        """p*(o | allow, the other slots), footprint and spilled ring tiles
        integrated out (closed form); 0 on inadmissible values."""
        return _softmax(-self.mid_energies(i, j))

    def mid_move(self, i, j):
        C = self.C
        p = self.mid_probs(i, j)
        o = min(int(np.searchsorted(np.cumsum(p), self.rng.random(), side="right")), C.D - 1)
        while p[o] == 0:                                   # round-off at the top of the cumsum
            o -= 1
        if o == self.obj.grid[i, j]:
            return
        self.obj.grid[i, j] = o
        BM = C.BM
        y0, y1, x0, x1 = i * BM - 1, (i + 1) * BM + 1, j * BM - 1, (j + 1) * BM + 1
        ya, xa = max(y0, 0), max(x0, 0)
        old = self.dem.grid[ya:y1, xa:x1].copy()
        C.paint_dem(self.obj, self.dem, (y0, y1, x0, x1))
        _redraw(self.tile.grid, self.dem.grid, old, ya, y1, xa, x1, C.mu, self.rng.random((BM + 2) ** 2), True)

    # ------------------------------------------------------------ sweeps
    def tile_sweep(self, n=1):
        self.model.sweep("tile", n, seed=int(self.rng.integers(1 << 30)))

    def sweep(self, n=1, tile_sweeps=2):
        C = self.C
        for _ in range(n):
            for k in self.rng.permutation(C.nty * C.ntx):
                self.top_move(k // C.ntx, k % C.ntx)
            dorm = C.dormant_of(self.allow).ravel()
            for k in self.rng.permutation(C.nmy * C.nmx):
                if not dorm[k]:
                    self.mid_move(k // C.nmx, k % C.nmx)
            self.tile_sweep(tile_sweeps)

    def moments(self, burn, sweeps, symmetrise=True):
        self.sweep(burn)
        acc = None
        for _ in range(sweeps):
            self.sweep(1)
            s = self.C.stats(self.biome, self.obj, self.tile)
            acc = s if acc is None else {k: acc[k] + s[k] for k in acc}
        out = {k: v / sweeps for k, v in acc.items()}
        return self.C.symmetrise(out) if symmetrise else out

    def joint_samples(self, burn, every, n):
        """Yield n (biome, obj, tile) grid copies, `every` sweeps apart after `burn`."""
        self.sweep(burn)
        for k in range(n):
            if k:
                self.sweep(every)
            yield self.biome.grid.copy(), self.obj.grid.copy(), self.tile.grid.copy()
