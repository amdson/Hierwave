"""Legacy (G1b, coarse-to-fine promises era); superseded by castlegen/channels. See notes/history/coarse_to_fine.tex.

Support as local height envelopes (notes/history/coarse_to_fine.tex; the redesign
of quantities/support.py's language).  Side view, gravity +y.

The rule (same as support.py): a solid cell outside the ground band (bottom G
rows, 0 < G < K) needs a solid cell or a band cell directly below.  Band
cells count as solid ("effective solid") and never need support.  Every valid
map is a heightmap: each column is air above a height H(c) >= G and solid
below it.

Exact block state (k x k block, `State`, or the vectorised `Level`):
  t     (k,) per column: the effective-solid run length from the block's
        bottom row upward (local height, 0..k);
  top   (k,) top-row effective-solid mask;
  viol  number of cells inside the block that break the rule (rows 0..k-2).
merge (upper u over lower l, per column):  t = l.t if l.t < k else k + u.t;
top = u.top; viol = sum + #(u.t > 0 & ~l.top).  Exact.

Promise (value v): per column interval i (I intervals of width k/I) the bins
of (min t, max t), lo_i <= hi_i.  Bins of a local height t in 0..k, Q even:
    0            t = 0        (interval empty)
    1..Q         0 < t < k,   ceil(t Q / k)
    F = Q + 1    t = k        (full)
Everything is block-local.  Encoding: interval envelopes are indexed by
PAIRS (C = Q + 2 codes, P = C (C + 1) / 2 pairs), v = sum_i pair_i P^i,
V = P^I (100 at I = Q = 2).  LO, HI: (V, I) bin tables.

Seam rule (upper u directly above lower l, per interval i), exact image of
the tile rule: u.hi_i > 0 => l.hi_i = F, and u.lo_i > 0 => l.lo_i = F.
The torus wrap seam (ground row above row 0) is exempt: the ground block's
bottom row is band.  Vertical seams carry no hard constraint.
Validity: a ground block (bottom block row) has lo_i >= bin(G) everywhere.

Promise merge (exact on consistent pairs): per child column, stacked
children map to parent bins by
    lo = up(l.lo)             if l.lo < F else  mid + up(u.lo)
    hi = mid + up(u.hi)       if u.hi > 0 else  up(l.hi)
with up(b) = ceil(b / 2) for interior bins (the parent's bins are twice as
wide), up(0) = 0, up(F) = mid = Q / 2 (the bin of t = k in the parent), and
mid + up(F) read as F;
then side by side: lo = min, hi = max over the two child intervals a parent
interval covers.

Guarantee.  The tile-level contract is: the map is valid (support_violations
== 0) and every K-block's abstraction equals its promise.  Completeness:
every valid map abstracts to a consistent pyramid (seam rule, validity,
merge) - so a corpus of valid maps needs no projection.  Realisability: in
any consistent stack the lo (hi) of every block is attained at one common
column, the interval's global minimum (maximum) of H, so any consistent
level-K grid has a completion when intervals are >= 2 columns wide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

I_DEFAULT = 2
Q_DEFAULT = 2
KIDS = ((0, 0), (0, 1), (1, 0), (1, 1))


# ------------------------------------------------------------------ language
class Lang:
    """Encoding tables for one (I, Q).  Use `lang(I, Q)` (cached)."""

    def __init__(self, I=I_DEFAULT, Q=Q_DEFAULT):
        assert I % 2 == 0 and Q % 2 == 0 and Q >= 2
        self.I, self.Q = I, Q
        self.F = Q + 1
        self.C = Q + 2
        self.mid = Q // 2
        self.PAIRS = np.array([(lo, hi) for lo in range(self.C) for hi in range(lo, self.C)], np.int64)
        self.P = len(self.PAIRS)
        self.V = self.P ** I
        v = np.arange(self.V)
        idx = np.stack([(v // self.P ** i) % self.P for i in range(I)], 1)        # (V, I) pair index per interval
        self.LO, self.HI = self.PAIRS[idx, 0], self.PAIRS[idx, 1]
        self._pair_index = {tuple(p): j for j, p in enumerate(self.PAIRS.tolist())}
        # seam rule, (V, V) [upper, lower]
        F = self.F
        LOu, HIu = self.LO[:, None, :], self.HI[:, None, :]
        LOl, HIl = self.LO[None, :, :], self.HI[None, :, :]
        self.COMPAT = (((HIu == 0) | (HIl == F)) & ((LOu == 0) | (LOl == F))).all(-1)
        # interfaces: what a value presents across a horizontal seam
        self.IFACE_UP = ((self.LO > 0).astype(np.int64) + 2 * (self.HI > 0)) @ (4 ** np.arange(I))       # as upper
        self.IFACE_DOWN = ((self.LO == F).astype(np.int64) + 2 * (self.HI == F)) @ (4 ** np.arange(I))   # as lower
        self._cols = {}

    # ---- encoding
    def encode(self, lo, hi):
        """lo, hi: (..., I) bins -> values (...)."""
        lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        pidx = self._pidx(lo, hi)
        return (pidx * self.P ** np.arange(self.I)).sum(-1)

    def _pidx(self, lo, hi):
        # pair index of (lo, hi): row-major over lo, hi >= lo
        C = self.C
        return lo * C - lo * (lo - 1) // 2 + (hi - lo)

    def bins(self, t, k):
        """Bin of local height(s) t in a block of side k."""
        t = np.asarray(t, np.int64)
        return np.where(t == 0, 0, np.where(t >= k, self.F, (t * self.Q + k - 1) // k))

    def valid(self, v, ground_row=False, G=0, k=None):
        """Validity of value(s) v; a ground block needs lo >= bin(G) everywhere."""
        v = np.asarray(v)
        if not ground_row:
            return np.ones(v.shape, bool)
        return (self.LO[v] >= self.bins(G, k)).all(-1)

    # ---- promise merge
    def up(self, b):
        """Child bin -> parent bin of the same local height (lower child;
        a full child, t = k, is the parent's bin mid)."""
        b = np.asarray(b)
        return np.where(b == self.F, self.mid, (b + 1) // 2)

    def up_upper(self, b):
        """Child bin of the upper child -> parent bin of k + t."""
        b = np.asarray(b)
        return np.where(b == self.F, self.F, self.mid + (b + 1) // 2)

    def stack(self, u, l):
        """Vertical merge per interval: values u (upper), l (lower) ->
        (lo, hi) parent-scale bins (..., I) of the column the pair covers."""
        F = self.F
        LOu, HIu, LOl, HIl = self.LO[u], self.HI[u], self.LO[l], self.HI[l]
        lo = np.where(LOl < F, self.up(LOl), self.up_upper(LOu))
        hi = np.where(HIu > 0, self.up_upper(HIu), self.up(HIl))
        return lo, hi

    def column_part(self, lo, hi):
        """Parent-scale (lo, hi) of one child column (..., I child intervals)
        -> the parent's I/2 intervals over that column, as (lo, hi) (..., I/2)."""
        lo = np.minimum(lo[..., 0::2], lo[..., 1::2])
        hi = np.maximum(hi[..., 0::2], hi[..., 1::2])
        return lo, hi

    def merge(self, c00, c01, c10, c11):
        """-> (parent values, ok): ok = both internal seams obey the seam rule."""
        c = [np.asarray(x) for x in (c00, c01, c10, c11)]
        ok = self.COMPAT[c[0], c[2]] & self.COMPAT[c[1], c[3]]
        parts = [self.column_part(*self.stack(c[dx], c[2 + dx])) for dx in (0, 1)]
        lo = np.concatenate([parts[0][0], parts[1][0]], -1)
        hi = np.concatenate([parts[0][1], parts[1][1]], -1)
        return self.encode(lo, hi), ok

    def parent_part(self, v, dx):
        """The parent's I/2 intervals over child column dx, as an index into
        the P^(I/2) column-part space."""
        v = np.asarray(v)
        h = self.I // 2
        pidx = self._pidx(self.LO[v][..., dx * h:(dx + 1) * h], self.HI[v][..., dx * h:(dx + 1) * h])
        return (pidx * self.P ** np.arange(h)).sum(-1)

    def column_pairs(self, part, g_upper=False, g_lower=False, G=0, k=None):
        """(u, l) arrays: every (upper, lower) child pair of one child column
        that is valid, obeys the seam rule and merges to column part `part`
        (from `parent_part`).  k: child block side (for the ground bin)."""
        key = (int(part), bool(g_upper), bool(g_lower), int(G), k)
        if key not in self._cols:
            u, l = np.meshgrid(np.arange(self.V), np.arange(self.V), indexing="ij")
            u, l = u.ravel(), l.ravel()
            m = self.COMPAT[u, l]
            m &= self.valid(u, g_upper, G, k) & self.valid(l, g_lower, G, k)
            lo, hi = self.column_part(*self.stack(u, l))
            h = self.I // 2
            pidx = (self._pidx(lo, hi) * self.P ** np.arange(h)).sum(-1)
            m &= pidx == part
            self._cols[key] = (u[m], l[m])
        return self._cols[key]

    def pair_mask(self, part, g_upper=False, g_lower=False, G=0, k=None):
        """(V, V) bool [upper, lower] version of column_pairs."""
        u, l = self.column_pairs(part, g_upper, g_lower, G, k)
        M = np.zeros((self.V, self.V), bool)
        M[u, l] = True
        return M

    def consistent(self, grid, parents=None, G=0, k=None):
        """A level grid (rows, cols) obeys validity (last row = ground row
        when G > 0) and the seam rule on every non-wrap horizontal seam; with
        `parents`, every 2x2 group merges to its parent."""
        g = np.asarray(grid)
        if not self.COMPAT[g[:-1], g[1:]].all():
            return False
        if G and not self.valid(g[-1], True, G, k).all():
            return False
        if parents is not None:
            P_, ok = self.merge(g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2])
            if not (ok.all() and np.array_equal(P_, parents)):
                return False
        return True


_LANGS = {}


def lang(I=I_DEFAULT, Q=Q_DEFAULT):
    if (I, Q) not in _LANGS:
        _LANGS[(I, Q)] = Lang(I, Q)
    return _LANGS[(I, Q)]


# ---------------------------------------------------------------- exact state
def effective_solid(ts, tiles, ground):
    """Solid or band, (n, n) bool."""
    s = ts.solid[np.asarray(tiles)].copy()
    if ground:
        s[-ground:] = True
    return s


@dataclass
class Level:
    """Exact states of every block of side k, vectorised.
    t, top: (n/k, n) per block row and global column; viol: (n/k, n/k)."""
    k: int
    t: np.ndarray
    top: np.ndarray
    viol: np.ndarray


def level_K(ts, tiles, K, ground):
    """Exact states of the K-blocks of `tiles` (n x n, n % K == 0)."""
    s = effective_solid(ts, tiles, ground)
    n = s.shape[0]
    R = n // K
    b = s.reshape(R, K, n)
    t = np.cumprod(b[:, ::-1, :], axis=1).sum(1)
    top = b[:, 0, :].copy()
    bad = b[:, :-1, :] & ~b[:, 1:, :]                         # solid over non-solid inside the block
    viol = bad.sum(1).reshape(R, n // K, K).sum(-1)
    return Level(K, t.astype(np.int64), top, viol.astype(np.int64))


def merge_level(L):
    """States of the blocks of side 2k from those of side k."""
    k = L.k
    tu, tl = L.t[0::2], L.t[1::2]
    t = np.where(tl < k, tl, k + tu)
    seam = (tu > 0) & ~L.top[1::2]
    n = L.t.shape[1]
    seam_blocks = seam.reshape(seam.shape[0], n // (2 * k), 2 * k).sum(-1)
    v = L.viol
    viol = v[0::2, 0::2] + v[0::2, 1::2] + v[1::2, 0::2] + v[1::2, 1::2] + seam_blocks
    return Level(2 * k, t, L.top[0::2].copy(), viol)


def levels(ts, tiles, K, ground, h_top=None):
    """{h: Level} for h = K .. h_top (default n)."""
    n = np.asarray(tiles).shape[0]
    h_top = n if h_top is None else h_top
    L = level_K(ts, tiles, K, ground)
    out = {K: L}
    while L.k < h_top:
        L = merge_level(L)
        out[L.k] = L
    return out


def abstract_level(L, lg=None):
    """(n/k, n/k) promise values of a Level (meaningful where viol == 0)."""
    lg = lg or lang()
    k, I = L.k, lg.I
    R, n = L.t.shape
    t = L.t.reshape(R, n // k, I, k // I)
    lo, hi = lg.bins(t.min(-1), k), lg.bins(t.max(-1), k)
    return lg.encode(lo, hi)


def pyramid(ts, tiles, K, ground, h_top=None, lg=None):
    """-> ({h: values}, {h: viol == 0 mask}) for h = K .. h_top."""
    Ls = levels(ts, tiles, K, ground, h_top)
    return ({h: abstract_level(L, lg) for h, L in Ls.items()}, {h: L.viol == 0 for h, L in Ls.items()})


def satisfaction(ts, tiles, grids, K, ground, lg=None):
    """{h: fraction of blocks whose exact state has viol 0 and abstracts to
    the value held in grids[h]}."""
    h_top = max(grids)
    vals, ok = pyramid(ts, tiles, K, ground, h_top, lg)
    return {h: float(((vals[h] == np.asarray(grids[h])) & ok[h]).mean()) for h in grids}
