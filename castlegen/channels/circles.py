"""The circles test: induced repulsion between objects, two levels deep.

Tiles (h = 1) are air (0) or dirt (1), with a field mu on air.  Mid cells
(h = BM) hold an object, absent (0) or a disc centre at an offset inside the
block; the object demands dirt on its disc (Euclidean radius R) and air on
its ring (the disc's 8-neighbour dilation minus the disc, spilling at most
one tile into the neighbouring blocks), cost kappa per violated tile.  Top
cells (h = BM * BT) choose a corner T; the mid cell at that corner should
hold an object and the other mid cells of the top block should be empty,
cost lam per disagreeing mid cell.

Both honour terms depend on where a cell sits inside its parent, so they
are carried by deterministic painted channels (fixed everywhere, refreshed
whenever their source changes):
    slot  h = BM, D = 2: 1 iff the mid cell is the corner its top chose
    dem   h = 1,  D = 4: 0 free, 1 dirt (a disc), 2 air (one or more rings),
                         3 conflict (a disc of one object and a ring of another)
and the designed factors are position free:
    lam    pair (mid, present) x (slot, val)  lam [present != slot]
    pres   unary (mid, present)               [0, -b]  (presence bonus)
    kappa  pair (tile, col) x (dem, val)      kt[col, dem]
    mu     unary (tile, col)                  [mu, 0]
Given the objects the tiles are independent, so every induced coupling is a
sum over tiles of the per-tile free energy fz[dem] = -log Z(dem).  Note that
dem has one "air" value however many rings cover a tile, so the per-tile
free energy of an air tile does not depend on the ring count (the spec's
f_a(n) reduces to f_a(1); a shared ring tile gains f0 - f_a).

The presence bonus b defaults to presence_bonus() = F_obj - log(n_offsets),
F_obj the free-energy cost of one object in an interior block with empty
neighbours (about 26.95 at kappa = 4, mu = 0.3) and n_offsets = (BM - 2R)^2:
an isolated block without the top is then absent / present 50/50, and lam
only tilts it, so every conditional stays soft.

This module provides the model, its channels, designed and learned
factors, painters, stats (with the dihedral symmetrisation), exact
reference tables, render, the forward model on the generic kernel, and the
oracle: a two-way sampler of p* with collapsed moves (mid: exact, tiles of
the block's footprint integrated out; top: an independence Metropolis move
on (T, the BT x BT mid cells) whose proposal is the factorised collapsed
conditional, so the chain is exact).  See notes/circles_test.md."""
import numpy as np
from numba import njit

from .core import Channel, Factor, Model

AIR, DIRT = 0, 1
FREE, DDIRT, DAIR, CONFLICT = 0, 1, 2, 3


# ------------------------------------------------------------ numba core
@njit(cache=True, inline="always")
def _dem_of(nd, nr):
    if nd > 0:
        return 3 if nr > 0 else 1
    return 2 if nr > 0 else 0


@njit(cache=True)
def _nd_nr(mid, y, x, BM, OY, OX, SH, R1, si0, si1, sj0, sj1):
    """Disc and ring counts at tile (y, x) from the objects of the mid blocks
    around it, skipping the blocks in [si0, si1) x [sj0, sj1)."""
    nmy, nmx = mid.shape
    bi = y // BM
    bj = x // BM
    nd = 0
    nr = 0
    S = SH.shape[0]
    for a in range(bi - 1, bi + 2):
        if a < 0 or a >= nmy:
            continue
        for b in range(bj - 1, bj + 2):
            if b < 0 or b >= nmx:
                continue
            if si0 <= a < si1 and sj0 <= b < sj1:
                continue
            o = mid[a, b]
            if o == 0:
                continue
            dy = y - (a * BM + OY[o]) + R1
            dx = x - (b * BM + OX[o]) + R1
            if 0 <= dy < S and 0 <= dx < S:
                s = SH[dy, dx]
                if s == 1:
                    nd += 1
                elif s == 2:
                    nr += 1
    return nd, nr


@njit(cache=True)
def _paint_dem(mid, dem, BM, OY, OX, SH, R1, y0, y1, x0, x1):
    H, W = dem.shape
    for y in range(max(y0, 0), min(y1, H)):
        for x in range(max(x0, 0), min(x1, W)):
            nd, nr = _nd_nr(mid, y, x, BM, OY, OX, SH, R1, 0, 0, 0, 0)
            dem[y, x] = _dem_of(nd, nr)


@njit(cache=True)
def _delta(mid, tile, i, j, si0, si1, sj0, sj1, BM, OY, OX, FPY, FPX, FPS, SH, R1, fz, kt, plain, out):
    """out[o] = the change of the tile energy when object o is put in block
    (i, j), relative to no object there, the blocks in the skip range
    removed.  plain = 0: collapsed, sum over o's footprint of fz[dem with o]
    - fz[dem without o]; plain = 1: tiles fixed, kt[col, ...] instead."""
    H, W = tile.shape
    D = out.shape[0]
    out[0] = 0.0
    for o in range(1, D):
        cy = i * BM + OY[o]
        cx = j * BM + OX[o]
        e = 0.0
        for k in range(FPY.shape[0]):
            y = cy + FPY[k]
            x = cx + FPX[k]
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            nd, nr = _nd_nr(mid, y, x, BM, OY, OX, SH, R1, si0, si1, sj0, sj1)
            d0 = _dem_of(nd, nr)
            if FPS[k] == 1:
                d1 = _dem_of(nd + 1, nr)
            else:
                d1 = _dem_of(nd, nr + 1)
            if plain:
                c = tile[y, x]
                e += kt[c, d1] - kt[c, d0]
            else:
                e += fz[d1] - fz[d0]
        out[o] = e


@njit(cache=True)
def _redraw(tile, dem, y0, y1, x0, x1, kt, mu, u):
    """Tiles of the window from their exact conditional given the demand."""
    H, W = tile.shape
    k = 0
    for y in range(max(y0, 0), min(y1, H)):
        for x in range(max(x0, 0), min(x1, W)):
            d = dem[y, x]
            wa = np.exp(-mu - kt[0, d])
            wd = np.exp(-kt[1, d])
            tile[y, x] = 0 if u[k] * (wa + wd) < wa else 1
            k += 1


@njit(cache=True)
def _win_kappa(mid, tile, y0, y1, x0, x1, si0, si1, sj0, sj1, BM, OY, OX, SH, R1, kt):
    """Sum over the window of kt[col, dem(all)] - kt[col, dem(skip blocks removed)]."""
    H, W = tile.shape
    e = 0.0
    for y in range(max(y0, 0), min(y1, H)):
        for x in range(max(x0, 0), min(x1, W)):
            nd, nr = _nd_nr(mid, y, x, BM, OY, OX, SH, R1, 0, 0, 0, 0)
            nd0, nr0 = _nd_nr(mid, y, x, BM, OY, OX, SH, R1, si0, si1, sj0, sj1)
            c = tile[y, x]
            e += kt[c, _dem_of(nd, nr)] - kt[c, _dem_of(nd0, nr0)]
    return e


@njit(cache=True)
def _top_exact(e, cd, cr, nd0, nr0, col, kt, pc):
    """log sum over the 4 mid cells' values (BT = 2) of exp(-(sum_c e[T, c, o_c]
    + corr(o))), corr the seam-tile correction of the factorised energy."""
    P, C, D = e.shape
    S = nd0.shape[0]
    base0 = 0.0
    for s in range(S):
        base0 += kt[col[s], _dem_of(nd0[s], nr0[s])]
    mx = np.full(P, -np.inf)
    acc = np.zeros(P)
    da = np.empty(S, np.int64)
    ra = np.empty(S, np.int64)
    db = np.empty(S, np.int64)
    rb = np.empty(S, np.int64)
    dc = np.empty(S, np.int64)
    rc = np.empty(S, np.int64)
    for o0 in range(D):
        for s in range(S):
            da[s] = nd0[s] + cd[0, o0, s]
            ra[s] = nr0[s] + cr[0, o0, s]
        for o1 in range(D):
            for s in range(S):
                db[s] = da[s] + cd[1, o1, s]
                rb[s] = ra[s] + cr[1, o1, s]
            for o2 in range(D):
                for s in range(S):
                    dc[s] = db[s] + cd[2, o2, s]
                    rc[s] = rb[s] + cr[2, o2, s]
                for o3 in range(D):
                    ex = 0.0
                    for s in range(S):
                        ex += kt[col[s], _dem_of(dc[s] + cd[3, o3, s], rc[s] + cr[3, o3, s])]
                    corr = ex - base0 - (pc[0, o0] + pc[1, o1] + pc[2, o2] + pc[3, o3])
                    for T in range(P):
                        v = -(corr + e[T, 0, o0] + e[T, 1, o1] + e[T, 2, o2] + e[T, 3, o3])
                        if v > mx[T]:
                            acc[T] = acc[T] * np.exp(mx[T] - v) + 1.0
                            mx[T] = v
                        else:
                            acc[T] += np.exp(v - mx[T])
    return mx + np.log(acc)


def _softmax(logw):
    p = np.exp(logw - logw.max())
    return p / p.sum()


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return (np.log(np.exp(a - m).sum(axis=axis, keepdims=True)) + m).squeeze(axis)


def _dcentre(A):
    return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()


def _grid(c):
    return c.grid if hasattr(c, "grid") else c


# ------------------------------------------------------------------ model
class Circles:
    def __init__(self, nty, ntx, BM=8, BT=2, kappa=4.0, mu=0.3, lam=3.0, R=2, b=None):
        assert BM - 2 * R >= 1, "the disc must fit the block"
        b_arg = b                                       # (b is reused as a loop variable below)
        self.nty, self.ntx, self.BM, self.BT, self.R = nty, ntx, BM, BT, R
        self.q, self.P = 2, 4
        self.kappa, self.mu, self.lam = float(kappa), float(mu), float(lam)
        self.nmy, self.nmx = nty * BT, ntx * BT
        self.H, self.W = self.nmy * BM, self.nmx * BM
        n = BM - 2 * R                                  # centres R .. BM-1-R per axis
        self.D = 1 + n * n
        self.OY = np.full(self.D, -1, np.int64)
        self.OX = np.full(self.D, -1, np.int64)
        for v in range(1, self.D):
            self.OY[v], self.OX[v] = R + (v - 1) // n, R + (v - 1) % n
        R1 = R + 1
        self.R1 = R1
        rr = np.arange(-R1, R1 + 1)
        dy, dx = np.meshgrid(rr, rr, indexing="ij")
        disc = dy ** 2 + dx ** 2 <= R * R
        dil = np.zeros_like(disc)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                dil |= np.roll(np.roll(disc, a, 0), b, 1)
        ring = dil & ~disc
        self.SH = (disc * 1 + ring * 2).astype(np.int64)            # (2R+3)^2: 0 none, 1 disc, 2 ring
        self.DISC = [(int(a), int(b)) for a, b in zip(dy[disc], dx[disc])]
        self.RING = [(int(a), int(b)) for a, b in zip(dy[ring], dx[ring])]
        fp = disc | ring
        self.FPY, self.FPX, self.FPS = dy[fp].astype(np.int64), dx[fp].astype(np.int64), self.SH[fp].astype(np.int64)
        k, m = self.kappa, self.mu
        self.kt = np.array([[0.0, k, 0.0, k],          # air:  free, dirt-demand, air-demand, conflict
                            [0.0, 0.0, k, k]])         # dirt
        self.fz = -np.log(np.exp(-m - self.kt[0]) + np.exp(-self.kt[1]))     # (4,) per-tile -log Z
        cs = np.array([[0, 0], [0, BT - 1], [BT - 1, 0], [BT - 1, BT - 1]])   # NW, NE, SW, SE
        self.CORNER = cs
        # slot pattern of each corner over the BT x BT cells
        self.SLOTPAT = np.zeros((self.P, BT, BT), np.int32)
        for T in range(self.P):
            self.SLOTPAT[T, cs[T, 0], cs[T, 1]] = 1
        self.contact_h, self.contact_v = self._contact_tables()
        self.b = self.presence_bonus() if b_arg is None else float(b_arg)
        self.pres_e = np.where(np.arange(self.D) > 0, -self.b, 0.0)   # (D,) the pres factor per mid value
        self._sym = self._sym_maps()

    def object_cost(self):
        """F_obj: sum over one object's footprint of fz[dem] - fz[free] (the
        same for every offset in an interior block, neighbours empty)."""
        return float((self.fz[self.FPS] - self.fz[0]).sum())

    def presence_bonus(self):
        """F_obj - log(n_offsets): absent / present 50/50 in an isolated block."""
        return self.object_cost() - np.log(self.D - 1)

    # --------------------------------------------------------- channels
    def channels(self):
        D, P = self.D, self.P
        top = Channel("top", self.BM * self.BT, P).add_view("pal", np.arange(P), P)
        mid = Channel("mid", self.BM, D).add_view("self", np.arange(D), D) \
            .add_view("present", (np.arange(D) > 0).astype(np.int64), 2)
        tile = Channel("tile", 1, 2).add_view("col", np.arange(2), 2)
        slot = Channel("slot", self.BM, 2).add_view("val", np.arange(2), 2)
        dem = Channel("dem", 1, 4).add_view("val", np.arange(4), 4)
        top.grid = np.zeros((self.nty, self.ntx), np.int32)
        mid.grid = np.zeros((self.nmy, self.nmx), np.int32)
        tile.grid = np.zeros((self.H, self.W), np.int32)
        slot.grid = np.zeros((self.nmy, self.nmx), np.int32)
        dem.grid = np.zeros((self.H, self.W), np.int32)
        for c in (top, mid, tile, slot, dem):
            c.fixed = np.zeros(c.grid.shape, bool)
        slot.fixed[:] = True
        dem.fixed[:] = True
        self.paint_slot(top, slot)
        self.paint_dem(mid, dem)
        return top, mid, tile, slot, dem

    def designed_factors(self):
        return [Factor.pair(("mid", "present"), ("slot", "val"), (0, 0), self.lam * (1 - np.eye(2)), name="lam"),
                Factor.pair(("tile", "col"), ("dem", "val"), (0, 0), self.kt, name="kappa"),
                Factor.unary(("tile", "col"), np.array([self.mu, 0.0]), name="mu"),
                Factor.unary(("mid", "present"), np.array([0.0, -self.b]), name="pres")]

    def learned_factors(self, theta):
        """The tabular learned factors of theta; a key missing from theta is
        left out (its role taken by an extra factor, e.g. a convpot)."""
        M, T = ("mid", "self"), ("top", "pal")
        mk = dict(mid_h=lambda t: Factor.pair(M, M, (0, 1), t, name="mid_h"),
                  mid_v=lambda t: Factor.pair(M, M, (1, 0), t, name="mid_v"),
                  mid_u=lambda t: Factor.unary(M, t, name="mid_u"),
                  top_h=lambda t: Factor.pair(T, T, (0, 1), t, name="top_h"),
                  top_v=lambda t: Factor.pair(T, T, (1, 0), t, name="top_v"),
                  top_u=lambda t: Factor.unary(T, t, name="top_u"))
        return [f(theta[k]) for k, f in mk.items() if k in theta]

    def theta0(self):
        D, P = self.D, self.P
        return dict(mid_h=np.zeros((D, D)), mid_v=np.zeros((D, D)), mid_u=np.zeros(D),
                    top_h=np.zeros((P, P)), top_v=np.zeros((P, P)), top_u=np.zeros(P))

    def model(self, chans, theta=None, extra=()):
        """All five channels (slot and dem fixed everywhere); designed factors,
        plus the learned ones when theta is given, plus `extra` factors."""
        for c in chans:
            if c.name in ("slot", "dem"):
                c.fixed = np.ones(c.grid.shape, bool)
        fac = self.designed_factors() + (self.learned_factors(theta) if theta is not None else []) + list(extra)
        return Model(self.H, self.W, list(chans), fac)

    # ---------------------------------------------------------- painters
    def paint_slot(self, top, slot, region=None):
        """slot[i, j] = 1 iff mid cell (i, j) is the corner its top parent chose.
        region = (I, J): repaint that top block only."""
        tg, sg = _grid(top), _grid(slot)
        BT = self.BT
        if region is None:
            pat = self.SLOTPAT[tg]                                   # (nty, ntx, BT, BT)
            sg[:] = pat.transpose(0, 2, 1, 3).reshape(sg.shape)
        else:
            I, J = region
            sg[I * BT:(I + 1) * BT, J * BT:(J + 1) * BT] = self.SLOTPAT[tg[I, J]]

    def paint_dem(self, mid, dem, window=None):
        """The union of the objects' demands (0 free, 1 dirt, 2 air, 3 conflict);
        window = (y0, y1, x0, x1) repaints those tiles only."""
        mg, dg = _grid(mid), _grid(dem)
        y0, y1, x0, x1 = (0, dg.shape[0], 0, dg.shape[1]) if window is None else window
        _paint_dem(mg, dg, self.BM, self.OY, self.OX, self.SH, self.R1, y0, y1, x0, x1)

    def dem_of(self, mid):
        d = np.zeros((self.H, self.W), np.int32)
        self.paint_dem(mid, d)
        return d

    # ------------------------------------------------------- statistics
    @staticmethod
    def _pairs(g, n):
        h = np.zeros((n, n)); v = np.zeros((n, n))
        np.add.at(h, (g[:, :-1].ravel(), g[:, 1:].ravel()), 1)
        np.add.at(v, (g[:-1, :].ravel(), g[1:, :].ravel()), 1)
        return h / max(h.sum(), 1), v / max(v.sum(), 1)

    def _contact_tables(self):
        """(D, D) bool: footprints of o (left / upper) and o' (right / lower,
        one block on) overlap or are 4-adjacent."""
        D, BM = self.D, self.BM
        fp = [None] + [{(self.OY[o] + a, self.OX[o] + b) for a, b in self.DISC + self.RING} for o in range(1, D)]

        def touch(A, B):
            for (y, x) in A:
                for (dy, dx) in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (y + dy, x + dx) in B:
                        return True
            return False

        ch = np.zeros((D, D), bool)
        cv = np.zeros((D, D), bool)
        for o in range(1, D):
            for p in range(1, D):
                ch[o, p] = touch(fp[o], {(y, x + BM) for y, x in fp[p]})
                cv[o, p] = touch(fp[o], {(y + BM, x) for y, x in fp[p]})
        return ch, cv

    def stats(self, top, mid, tile):
        D, P, BM, BT = self.D, self.P, self.BM, self.BT
        tg, mg, fg = _grid(top), _grid(mid), _grid(tile)
        mid_h, mid_v = self._pairs(mg, D)
        top_h, top_v = self._pairs(tg, P)
        dem = self.dem_of(mg)
        demanded = dem > 0
        bad = self.kt[fg, dem] > 0
        pres = mg > 0
        slot = self.SLOTPAT[tg].transpose(0, 2, 1, 3).reshape(mg.shape)
        ph = pres[:, :-1] & pres[:, 1:]
        pv = pres[:-1] & pres[1:]
        nc = self.contact_h[mg[:, :-1], mg[:, 1:]][ph].sum() + self.contact_v[mg[:-1], mg[1:]][pv].sum()
        npair = ph.sum() + pv.sum()
        B = BM * BT
        yy, xx = np.mgrid[:self.H, :self.W]
        band = ((xx % B == B - 1) & (xx < self.W - 1)) | ((xx % B == 0) & (xx > 0)) | \
               ((yy % B == B - 1) & (yy < self.H - 1)) | ((yy % B == 0) & (yy > 0))
        return dict(mid_h=mid_h, mid_v=mid_v, mid_u=np.bincount(mg.ravel(), minlength=D) / mg.size,
                    top_h=top_h, top_v=top_v, top_u=np.bincount(tg.ravel(), minlength=P) / tg.size,
                    present=float(pres.mean()),
                    viol=float(bad[demanded].mean()) if demanded.any() else 0.0,
                    conflict=float((dem == CONFLICT).mean()),
                    contact=float(nc / npair) if npair else 0.0,
                    slot_viol=float((pres != (slot > 0)).mean()),
                    edge_air_top=float((fg[band] == AIR).mean()) if band.any() else 0.0)

    # ---------------------------------------------------- symmetrisation
    def _sym_maps(self):
        """Value maps of the three generators (flip x, flip y, transpose) on
        mid values (offsets) and top values (corners)."""
        BM, D = self.BM, self.D
        n = BM - 2 * self.R

        def val(cy, cx):
            return 1 + (cy - self.R) * n + (cx - self.R)

        mx, my, mt = np.zeros(D, np.int64), np.zeros(D, np.int64), np.zeros(D, np.int64)
        for o in range(1, D):
            cy, cx = self.OY[o], self.OX[o]
            mx[o], my[o], mt[o] = val(cy, BM - 1 - cx), val(BM - 1 - cy, cx), val(cx, cy)
        tx = np.array([1, 0, 3, 2])          # NW<->NE, SW<->SE
        ty = np.array([2, 3, 0, 1])          # NW<->SW, NE<->SE
        tt = np.array([0, 2, 1, 3])          # NE<->SW
        return dict(x=(mx, tx), y=(my, ty), t=(mt, tt))

    @staticmethod
    def _op_stats(s, g, mmap, tmap):
        """One generator on the fitted tables (a new dict; scalars kept)."""
        out = dict(s)
        for pre, m in (("mid", mmap), ("top", tmap)):
            h, v, u = s[pre + "_h"], s[pre + "_v"], s[pre + "_u"]
            nh, nv, nu = np.zeros_like(h), np.zeros_like(v), np.zeros_like(u)
            nu[m] = u
            if g == "x":          # left/right swap order
                nh[np.ix_(m, m)] = h.T
                nv[np.ix_(m, m)] = v
            elif g == "y":
                nh[np.ix_(m, m)] = h
                nv[np.ix_(m, m)] = v.T
            else:                 # transpose: horizontal pairs become vertical ones
                nh[np.ix_(m, m)] = v
                nv[np.ix_(m, m)] = h
            out[pre + "_h"], out[pre + "_v"], out[pre + "_u"] = nh, nv, nu
        return out

    GROUP = [(fx, fy, tr) for fx in (0, 1) for fy in (0, 1) for tr in (0, 1)]

    def transform_stats(self, s, g):
        """Apply group element g = (flip x, flip y, transpose), in that order."""
        for flag, k in zip(g, "xyt"):
            if flag:
                s = self._op_stats(s, k, *self._sym[k])
        return s

    def transform_state(self, top, mid, tile, g):
        """(top, mid, tile) arrays of the transformed state (same order as
        transform_stats)."""
        tg, mg, fg = _grid(top).copy(), _grid(mid).copy(), _grid(tile).copy()
        for flag, k in zip(g, "xyt"):
            if not flag:
                continue
            mm, tm = self._sym[k]
            if k == "x":
                tg, mg, fg = tg[:, ::-1], mg[:, ::-1], fg[:, ::-1]
            elif k == "y":
                tg, mg, fg = tg[::-1], mg[::-1], fg[::-1]
            else:
                tg, mg, fg = tg.T, mg.T, fg.T
            tg, mg, fg = tm[tg].astype(np.int32), mm[mg].astype(np.int32), np.ascontiguousarray(fg)
        return tg, mg, fg

    def symmetrise(self, stats):
        """Average over the dihedral group (exact on the stats: the fitted
        tables are permuted / transposed, the monitors are invariant).  An
        exact symmetry of p* when nty == ntx."""
        out = None
        for g in self.GROUP:
            t = self.transform_stats(stats, g)
            out = t if out is None else {k: out[k] + t[k] for k in out}
        return {k: v / len(self.GROUP) for k, v in out.items()}

    # ------------------------------------------------------- reference
    def _F(self, mg):
        """sum over tiles of -log Z(dem) for the objects mg (a small grid)."""
        BM = self.BM
        d = np.zeros((mg.shape[0] * BM, mg.shape[1] * BM), np.int32)
        _paint_dem(mg.astype(np.int32), d, BM, self.OY, self.OX, self.SH, self.R1, 0, d.shape[0], 0, d.shape[1])
        return self.fz[d].sum()

    def _pair_raw(self, shape, p1, p2, vals1, vals2):
        """F(o, o') - F(o, -) - F(-, o') + F(-, -) over the given values."""
        mg = np.zeros(shape, np.int32)
        f00 = self._F(mg)
        f1 = np.zeros(len(vals1)); f2 = np.zeros(len(vals2))
        for a, o in enumerate(vals1):
            mg[:] = 0; mg[p1] = o; f1[a] = self._F(mg)
        for b, o in enumerate(vals2):
            mg[:] = 0; mg[p2] = o; f2[b] = self._F(mg)
        out = np.zeros((len(vals1), len(vals2)))
        for a, o in enumerate(vals1):
            for b, op in enumerate(vals2):
                mg[:] = 0; mg[p1] = o; mg[p2] = op
                out[a, b] = self._F(mg) - f1[a] - f2[b] + f00
        return out, f1 - f00, f2 - f00

    def reference(self, centred=True):
        """Exact tables (interior blocks, the other blocks empty):
        mid_h / mid_v  the induced pair term F(o,o') - F(o,-) - F(-,o') + F(-,-),
        mid_u          F(o) - F(-) - b [o present] (the same for every present
                       offset in an interior block: only absent vs present
                       differs; at the default b the gap is log n_offsets),
        top_h / top_v  at lam -> inf: -log sum_{o,o' present} exp(-F) of the two
                       (approximate at finite lam: at lam = 3 presence is only
                       tilted by the top, not forced)
                       chosen corners' objects (zero when they are more than
                       one block apart, before centering),
        top_u          at lam -> inf: -log sum_o exp(-u(o)), constant (0 centred).
        centred: pair tables double-centred, unaries centred."""
        D, BT = self.D, self.BT
        allv = list(range(D))
        mid_h, _, _ = self._pair_raw((3, 4), (1, 1), (1, 2), allv, allv)
        mid_v, _, _ = self._pair_raw((4, 3), (1, 1), (2, 1), allv, allv)
        mg = np.zeros((3, 3), np.int32)
        f0 = self._F(mg)
        mid_u = np.zeros(D)
        for o in range(1, D):
            mg[1, 1] = o
            mid_u[o] = self._F(mg) - f0 - self.b
        pres = list(range(1, D))
        top_u = np.full(self.P, -_logsumexp(-mid_u[1:]))

        def top_tab(horiz):
            out = np.zeros((self.P, self.P))
            shape = (BT + 2, 2 * BT + 2) if horiz else (2 * BT + 2, BT + 2)
            for T in range(self.P):
                for Tp in range(self.P):
                    a, b = self.CORNER[T]
                    ap, bp = self.CORNER[Tp]
                    p1 = (1 + a, 1 + b)
                    p2 = (1 + ap, 1 + BT + bp) if horiz else (1 + BT + ap, 1 + bp)
                    pair, u1, u2 = self._pair_raw(shape, p1, p2, pres, pres)
                    e = u1[:, None] + u2[None, :] + pair
                    # relative to the non-interacting value, so far pairs are exactly 0
                    out[T, Tp] = -_logsumexp(-e) + _logsumexp(-u1) + _logsumexp(-u2)
            return out

        top_h, top_v = top_tab(True), top_tab(False)
        ref = dict(mid_h=mid_h, mid_v=mid_v, mid_u=mid_u, top_h=top_h, top_v=top_v, top_u=top_u)
        if centred:
            for k in ("mid_h", "mid_v", "top_h", "top_v"):
                ref[k] = _dcentre(ref[k])
            for k in ("mid_u", "top_u"):
                ref[k] = ref[k] - ref[k].mean()
        return ref

    # ------------------------------------------------------------ render
    def render(self, top, mid, tile):
        """(H, W, 3) uint8: dirt brown, air light; mid-block edges thin
        (first row / column of each block darkened), top-block edges thicker
        (both sides darker); disc centres marked red."""
        base = np.array([(232, 236, 242), (130, 88, 50)], np.float64)
        fg, mg = _grid(tile), _grid(mid)
        img = base[fg].copy()
        BM, BT = self.BM, self.BM * self.BT
        yy, xx = np.mgrid[:self.H, :self.W]
        mid_edge = (yy % BM == 0) | (xx % BM == 0)
        top_edge = (yy % BT == 0) | (xx % BT == 0) | (yy % BT == BT - 1) | (xx % BT == BT - 1)
        img[mid_edge] *= 0.75
        img[top_edge] *= 0.45 / np.where(mid_edge[top_edge], 0.75, 1.0)[:, None]
        ii, jj = np.nonzero(mg)
        img[ii * BM + self.OY[mg[ii, jj]], jj * BM + self.OX[mg[ii, jj]]] = (230, 30, 30)
        return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------- forward model
class Forward:
    """Top, paint slot, mid, paint dem, tiles; all on the generic kernel with
    designed + learned factors (the model is rebuilt from self.theta at
    every run)."""

    def __init__(self, C: Circles, theta: dict, seed=0, extra=()):
        self.C, self.theta, self.extra = C, theta, list(extra)
        self.chans = C.channels()
        self.top, self.mid, self.tile, self.slot, self.dem = self.chans
        self.model = None
        self.rng = np.random.default_rng(seed)

    def run(self, S_T=30, S_M=30, S_F=20, fresh=True):
        C = self.C
        self.model = C.model(self.chans, self.theta, self.extra)
        if fresh:
            self.top.grid[:] = 0
            self.mid.grid[:] = 0
            self.tile.grid[:] = 0
        self.model.sweep("top", S_T, seed=int(self.rng.integers(1 << 30)))
        C.paint_slot(self.top, self.slot)
        self.model.sweep("mid", S_M, seed=int(self.rng.integers(1 << 30)))
        C.paint_dem(self.mid, self.dem)
        self.model.sweep("tile", S_F, seed=int(self.rng.integers(1 << 30)))
        return C.stats(self.top, self.mid, self.tile)


# ---------------------------------------------------- oracle: two-way
class Oracle:
    """Two-way sampler of p* with collapsed moves (exact chain)."""

    def __init__(self, C: Circles, seed=0):
        self.C = C
        self.rng = np.random.default_rng(seed)
        self.chans = C.channels()
        self.top, self.mid, self.tile, self.slot, self.dem = self.chans
        self.top.grid[:] = self.rng.integers(C.P, size=self.top.grid.shape)
        C.paint_slot(self.top, self.slot)
        # start near p*: an object (random offset) on each chosen corner, the rest empty
        self.mid.grid[:] = self.slot.grid * self.rng.integers(1, C.D, size=self.mid.grid.shape)
        C.paint_dem(self.mid, self.dem)
        _redraw(self.tile.grid, self.dem.grid, 0, C.H, 0, C.W, C.kt, C.mu, self.rng.random(C.H * C.W))
        self.model = C.model(self.chans)
        self.n_top, self.n_acc, self.n_change = 0, 0, 0
        self._e = np.empty(C.D)

    # -------------------------------------------------------------- mid
    def _mid_e(self, i, j, plain):
        C = self.C
        _delta(self.mid.grid, self.tile.grid, i, j, i, i + 1, j, j + 1, C.BM, C.OY, C.OX, C.FPY, C.FPX, C.FPS,
               C.SH, C.R1, C.fz, C.kt, plain, self._e)
        s = self.slot.grid[i, j]
        pres = (np.arange(C.D) > 0).astype(np.int64)
        return self._e + C.lam * (pres != s) + C.pres_e

    def mid_probs(self, i, j):
        """p*(o | slot, the other objects), the tiles of every candidate's
        footprint integrated out exactly (the other tiles do not depend on o)."""
        return _softmax(-self._mid_e(i, j, 0))

    def mid_probs_plain(self, i, j):
        """The one-site conditional with the tiles fixed."""
        return _softmax(-self._mid_e(i, j, 1))

    def _block_window(self, i, j):
        BM = self.C.BM
        return i * BM - 1, (i + 1) * BM + 1, j * BM - 1, (j + 1) * BM + 1

    def mid_move(self, i, j):
        """Exact Gibbs move of (o, the tiles of the block's extended window):
        o from mid_probs, repaint dem, redraw the window's tiles given it."""
        C = self.C
        o = int(self.rng.choice(C.D, p=self.mid_probs(i, j)))
        self.mid.grid[i, j] = o
        w = self._block_window(i, j)
        C.paint_dem(self.mid, self.dem, w)
        _redraw(self.tile.grid, self.dem.grid, *w, C.kt, C.mu, self.rng.random((C.BM + 2) ** 2))

    # -------------------------------------------------------------- top
    def top_probs_mid(self, i, j):
        """p*(T | mid): prod over the BT x BT mid cells of exp(-lam [present != slot(T, cell)])
        (the pres bonus does not depend on T and cancels)."""
        C = self.C
        BT = C.BT
        pres = (self.mid.grid[i * BT:(i + 1) * BT, j * BT:(j + 1) * BT] > 0).astype(np.int64)
        e = C.lam * (pres[None] != C.SLOTPAT).sum((1, 2))
        return _softmax(-e)

    def top_move_mid(self, i, j):
        t = int(self.rng.choice(self.C.P, p=self.top_probs_mid(i, j)))
        self.top.grid[i, j] = t
        self.C.paint_slot(self.top, self.slot, (i, j))

    def _top_fact(self, i, j):
        """(P, BT*BT, D) factorised energies: lam[present != slot(T, c)] - b [present] + the
        plain kappa change of o in cell c against the objects outside the top
        block (all BT x BT cells removed)."""
        C = self.C
        BT = C.BT
        pres = (np.arange(C.D) > 0).astype(np.int64)
        e = np.empty((C.P, BT * BT, C.D))
        for c in range(BT * BT):
            a, b = divmod(c, BT)
            _delta(self.mid.grid, self.tile.grid, i * BT + a, j * BT + b, i * BT, (i + 1) * BT, j * BT,
                   (j + 1) * BT, C.BM, C.OY, C.OX, C.FPY, C.FPX, C.FPS, C.SH, C.R1, C.fz, C.kt, 1, self._e)
            for T in range(C.P):
                e[T, c] = self._e + C.lam * (pres != C.SLOTPAT[T, a, b]) + C.pres_e
        return e

    def _top_window(self, i, j):
        B = self.C.BM * self.C.BT
        return i * B - 1, (i + 1) * B + 1, j * B - 1, (j + 1) * B + 1

    def _corr(self, i, j, e_kappa_fact):
        """Exact kappa energy of the top block's objects (relative to none)
        minus the factorised sum."""
        C = self.C
        BT = C.BT
        y0, y1, x0, x1 = self._top_window(i, j)
        ex = _win_kappa(self.mid.grid, self.tile.grid, y0, y1, x0, x1, i * BT, (i + 1) * BT, j * BT, (j + 1) * BT,
                        C.BM, C.OY, C.OX, C.SH, C.R1, C.kt)
        return ex - e_kappa_fact

    def top_probs_fact(self, i, j):
        """The factorised collapsed conditional: prod over cells of sum_o
        exp(-lam[..] - kappa change(o)) (seam interactions between the
        block's own objects ignored).  The top move's proposal."""
        e = self._top_fact(i, j)
        return _softmax(_logsumexp(-e, axis=2).sum(1))

    def top_probs(self, i, j):
        """p*(T | tiles, the objects outside the top block), the BT x BT mid
        cells summed out exactly (enumeration of D^4 with the seam-tile
        correction; BT = 2)."""
        C = self.C
        BT, BM = C.BT, C.BM
        assert BT == 2
        e = self._top_fact(i, j)
        y0, y1, x0, x1 = self._top_window(i, j)
        y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, C.H), min(x1, C.W)
        # footprint masks per cell, seam tiles = touched by two or more cells
        touch = np.zeros((BT * BT, y1 - y0, x1 - x0), bool)
        cells = []
        for c in range(BT * BT):
            a, b = divmod(c, BT)
            ci, cj = i * BT + a, j * BT + b
            lst = []
            for o in range(1, C.D):
                ys, xs = ci * BM + C.OY[o] + C.FPY, cj * BM + C.OX[o] + C.FPX
                ok = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
                touch[c, ys[ok] - y0, xs[ok] - x0] = True
                lst.append((ys[ok], xs[ok], C.FPS[ok]))
            cells.append(lst)
        seam = touch.sum(0) >= 2
        sy, sx = np.nonzero(seam)
        S = len(sy)
        idx = -np.ones(seam.shape, np.int64)
        idx[sy, sx] = np.arange(S)
        cd = np.zeros((BT * BT, C.D, S), np.int64)
        cr = np.zeros((BT * BT, C.D, S), np.int64)
        for c in range(BT * BT):
            for o in range(1, C.D):
                ys, xs, ss = cells[c][o - 1]
                k = idx[ys - y0, xs - x0]
                m = k >= 0
                cd[c, o, k[m & (ss == 1)]] = 1
                cr[c, o, k[m & (ss == 2)]] = 1
        nd0 = np.zeros(S, np.int64); nr0 = np.zeros(S, np.int64)
        for s in range(S):
            nd0[s], nr0[s] = _nd_nr(self.mid.grid, y0 + sy[s], x0 + sx[s], BM, C.OY, C.OX, C.SH, C.R1,
                                    i * BT, (i + 1) * BT, j * BT, (j + 1) * BT)
        col = self.tile.grid[y0 + sy, x0 + sx].astype(np.int64)
        dem0 = np.array([0, 1, 2, 3])[np.where(nd0 > 0, np.where(nr0 > 0, 3, 1), np.where(nr0 > 0, 2, 0))]
        pc = np.zeros((BT * BT, C.D))
        for c in range(BT * BT):
            for o in range(C.D):
                d1 = np.where(nd0 + cd[c, o] > 0, np.where(nr0 + cr[c, o] > 0, 3, 1),
                              np.where(nr0 + cr[c, o] > 0, 2, 0))
                pc[c, o] = (C.kt[col, d1] - C.kt[col, dem0]).sum()
        logw = _top_exact(e, cd, cr, nd0, nr0, col, C.kt, pc)
        return _softmax(logw)

    def top_move(self, i, j):
        """Independence Metropolis move of (T, the BT x BT mid cells) given the
        tiles: propose T from top_probs_fact and each cell from its factor,
        accept with the seam-tile correction; repaint slot and dem."""
        C = self.C
        BT = C.BT
        e = self._top_fact(i, j)
        sl = (slice(i * BT, (i + 1) * BT), slice(j * BT, (j + 1) * BT))
        old_m = self.mid.grid[sl].copy()
        old_T = int(self.top.grid[i, j])
        cs = np.arange(BT * BT)
        lam_part = lambda T, m: C.lam * ((m.ravel() > 0) != C.SLOTPAT[T].ravel()).sum() + C.pres_e[m.ravel()].sum()
        corr_old = self._corr(i, j, e[old_T, cs, old_m.ravel()].sum() - lam_part(old_T, old_m))
        lw = _logsumexp(-e, axis=2).sum(1)
        T = int(self.rng.choice(C.P, p=_softmax(lw)))
        new = np.array([self.rng.choice(C.D, p=_softmax(-e[T, c])) for c in range(BT * BT)], np.int32)
        new_m = new.reshape(BT, BT)
        self.mid.grid[sl] = new_m
        corr_new = self._corr(i, j, e[T, cs, new].sum() - lam_part(T, new_m))
        self.n_top += 1
        if np.log(self.rng.random()) < corr_old - corr_new:
            self.n_acc += 1
            self.n_change += int(T != old_T)
            self.top.grid[i, j] = T
            C.paint_slot(self.top, self.slot, (i, j))
            C.paint_dem(self.mid, self.dem, self._top_window(i, j))
        else:
            self.mid.grid[sl] = old_m

    # ------------------------------------------------------------ sweeps
    def tile_sweep(self, n=1):
        self.model.sweep("tile", n, seed=int(self.rng.integers(1 << 30)))

    def sweep(self, n=1, tile_sweeps=2):
        C = self.C
        for _ in range(n):
            for k in self.rng.permutation(C.nty * C.ntx):
                self.top_move(k // C.ntx, k % C.ntx)
            for k in self.rng.permutation(C.nmy * C.nmx):
                self.mid_move(k // C.nmx, k % C.nmx)
            self.tile_sweep(tile_sweeps)

    def moments(self, burn, sweeps, symmetrise=True):
        self.sweep(burn)
        acc = None
        for _ in range(sweeps):
            self.sweep(1)
            s = self.C.stats(self.top, self.mid, self.tile)
            acc = s if acc is None else {k: acc[k] + s[k] for k in acc}
        out = {k: v / sweeps for k, v in acc.items()}
        return self.C.symmetrise(out) if symmetrise else out
