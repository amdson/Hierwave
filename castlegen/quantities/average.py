"""Support as hierarchical average heights (an alternative to envelope.py,
same guarantee: valid maps are heightmaps H(c) >= G; see envelope.py).

Every block has m = 2 sub-columns, whatever its side, so a sub-column of a
block of side k is k/2 columns wide.  Values count "full-column equivalents":

  level K:  b = floor(s / K),  s = sum over the sub-column's w = K/2 columns of
            the local height t = clip(H - offset, 0, K);  b in 0..w.
  level 2h: b = floor((u_0 + u_1 + l_0 + l_1) / 4), the four child
            sub-column values under the parent sub-column (upper and lower
            child of the same child column), b in 0..w again.

So b / w is a (hierarchically rounded) fill fraction.  The coarse value is a
defined function of its children - the merge is exact by definition - and a
deterministic function of x, but only approximately the true coarse average.
A block value is the pair (b_0, b_1); V = (w + 1)^2 (81 at K = 16).

Seam rule at level K (upper value over lower value, per sub-column).  Upper
matter needs full columns below it: with s_u in its bin the upper block has
at least ceil(s_u / K) columns with matter, and the lower block has at most
fmax(b_l) full columns, where fmax(b) = b for an ordinary block (a sum in
[bK, (b+1)K) has at most b full columns) and, for a ground block (every
column at least G, the band), the most full columns any sum in its bin allows.
Choosing s = bK (b full columns, the rest empty) makes an ordinary block both
as easy to stand on and as light as possible at once, so a column of blocks
is realisable iff every adjacent pair is:

      b_u == 0  or  b_u <= fmax(b_l, ground_l).

Validity at level K: a ground block needs a sum >= w G in its bin.  Above K
the seam relation and validity are not written down: the level sampler
computes the realisable pairs bottom-up (envelope_var.feasible), so here
they are all-true.

Tile contract at level K: every K-block's sub-column sum lies in its bin,
[bK, (b+1)K - 1] (exactly wK for b = w).  With H a heightmap the support rule
holds automatically; the bins only constrain sums, so single-column moves are
blocked only at a bin edge.
"""
from __future__ import annotations

import numpy as np

from castlegen.quantities import envelope as EN

M = 2                                       # sub-columns per block


class AvgLang:
    """Same interface as envelope.Lang where envelope_var uses it: V, P, I,
    COMPAT, valid, parent_part, column_pairs, pair_mask, consistent, plus
    seam_compat(h, ground_lower) (level-dependent base seam relation)."""

    def __init__(self, K=16):
        assert K % M == 0
        self.K, self.w = K, K // M
        self.P = self.w + 1                  # values per sub-column
        self.I = M                           # envelope_var reads P ** (I // 2) = column parts
        self.V = self.P ** M
        v = np.arange(self.V)
        self.SUB = np.stack([(v // self.P ** j) % self.P for j in range(M)], 1)      # (V, 2)
        self._cols, self._pm = {}, {}
        self.COMPAT = self.seam_compat(K, False)

    def encode(self, b):
        b = np.asarray(b, np.int64)
        return (b * self.P ** np.arange(M)).sum(-1)

    # ---- level-K arithmetic
    def fmax(self, b, ground, G):
        """Most full columns a sub-column in bin b can hold (level K)."""
        K, w = self.K, self.w
        if b == w:
            return w
        if not ground:
            return b
        best = -1
        for s in range(b * K, (b + 1) * K):
            if s < w * G:
                continue
            f = min(w, (s - w * G) // (K - G)) if K > G else w
            while f >= 0 and not (f * K + (w - f) * G <= s <= f * K + (w - f) * (K - 1)):
                f -= 1
            best = max(best, f)
        return best                           # -1: the bin holds no valid ground sum

    def seam_compat(self, h, ground_lower, G=None):
        """(V, V) [upper, lower] base seam relation at level h (all-true above K)."""
        G = self._G if G is None else G
        if h != self.K:
            return np.ones((self.V, self.V), bool)
        key = ("seam", ground_lower, G)
        if key not in self._pm:
            fm = np.array([self.fmax(b, ground_lower, G) for b in range(self.P)])
            ok1 = (np.arange(self.P)[:, None] == 0) | (np.arange(self.P)[:, None] <= fm[None, :])   # [b_u, b_l]
            S = self.SUB
            self._pm[key] = ok1[S[:, None, 0], S[None, :, 0]] & ok1[S[:, None, 1], S[None, :, 1]]
        return self._pm[key]

    _G = 0

    def set_ground(self, G):
        self._G = int(G)
        self.COMPAT = self.seam_compat(self.K, False, self._G)
        return self

    def valid(self, v, ground_row=False, G=0, k=None):
        v = np.asarray(v)
        if not ground_row or k != self.K:
            return np.ones(v.shape, bool)
        okb = np.array([self.fmax(b, True, G) >= 0 for b in range(self.P)])
        return okb[self.SUB[v]].all(-1)

    # ---- merge
    def parent_part(self, v, dx):
        return self.SUB[np.asarray(v)][..., dx]

    def column_pairs(self, part, g_upper=False, g_lower=False, G=0, k=None):
        """(u, l): child pairs of one child column (side k) that are valid,
        obey the level-k seam relation and merge to the parent's sub-column
        value `part`."""
        key = (int(part), bool(g_upper), bool(g_lower), int(G), k)
        if key not in self._cols:
            u, l = np.meshgrid(np.arange(self.V), np.arange(self.V), indexing="ij")
            u, l = u.ravel(), l.ravel()
            m = self.seam_compat(k, g_lower, G)[u, l]
            m &= self.valid(u, g_upper, G, k) & self.valid(l, g_lower, G, k)
            m &= (self.SUB[u].sum(1) + self.SUB[l].sum(1)) // 4 == part
            self._cols[key] = (u[m], l[m])
        return self._cols[key]

    def pair_mask(self, part, g_upper=False, g_lower=False, G=0, k=None):
        u, l = self.column_pairs(part, g_upper, g_lower, G, k)
        Mk = np.zeros((self.V, self.V), bool)
        Mk[u, l] = True
        return Mk

    def merge(self, c00, c01, c10, c11):
        c = [self.SUB[np.asarray(x)] for x in (c00, c01, c10, c11)]
        p0 = (c[0].sum(-1) + c[2].sum(-1)) // 4
        p1 = (c[1].sum(-1) + c[3].sum(-1)) // 4
        return self.encode(np.stack([p0, p1], -1))

    def consistent(self, grid, parents=None, G=0, k=None):
        g = np.asarray(grid)
        R = g.shape[0]
        for y in range(R - 1):
            if not self.seam_compat(k, bool(G) and y + 1 == R - 1, G)[g[y], g[y + 1]].all():
                return False
        if G and not self.valid(g[-1], True, G, k).all():
            return False
        if parents is not None:
            if not np.array_equal(self.merge(g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2]), parents):
                return False
        return True


_LANGS = {}


def lang(K=16, G=0):
    if (K, G) not in _LANGS:
        _LANGS[(K, G)] = AvgLang(K).set_ground(G)
    return _LANGS[(K, G)]


# ---------------------------------------------------------------- bottom-up
def sums_K(ts, tiles, K, ground):
    """(R, R, 2) sub-column sums of local heights of every K-block, and the
    (R, R) violation counts (envelope.level_K)."""
    L = EN.level_K(ts, tiles, K, ground)
    R, n = L.t.shape
    s = L.t.reshape(R, n // K, M, K // M).sum(-1)
    return s, L.viol


def pyramid(ts, tiles, K, ground, h_top=None, lg=None):
    """-> ({h: values}, {h: ok}) for h = K .. h_top (ok: no violation below)."""
    lg = lg or lang(K, ground)
    n = np.asarray(tiles).shape[0]
    h_top = n if h_top is None else h_top
    s, viol = sums_K(ts, tiles, K, ground)
    g = lg.encode(s // K)
    ok = viol == 0
    vals, oks, h = {K: g}, {K: ok}, K
    while h < h_top:
        g = lg.merge(g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2])
        ok = ok[0::2, 0::2] & ok[0::2, 1::2] & ok[1::2, 0::2] & ok[1::2, 1::2]
        h *= 2
        vals[h], oks[h] = g, ok
    return vals, oks


def satisfaction(ts, tiles, grids, K, ground, lg=None):
    h_top = max(grids)
    vals, ok = pyramid(ts, tiles, K, ground, h_top, lg)
    return {h: float(((vals[h] == np.asarray(grids[h])) & ok[h]).mean()) for h in grids}


def intervals(P, K, lg):
    """(lo, hi) (R, R, 2) allowed sub-column sums of every K-block for the
    level-K value grid P."""
    b = lg.SUB[np.asarray(P)]
    lo = b * K
    hi = np.where(b == lg.w, lg.w * K, (b + 1) * K - 1)
    return lo, hi


def construct(P, K, ground, n, lg):
    """A heightmap (n,) whose level-K values equal P (the concentrated
    arrangement: b full columns per sub-column, the ground block topped up)."""
    P = np.asarray(P)
    R = P.shape[0]
    b = lg.SUB[P]                                       # (R, C, 2)
    w = lg.w
    H = np.zeros(n, np.int64)
    for bx in range(P.shape[1]):
        for j in range(M):
            cols = bx * K + j * w + np.arange(w)
            bb = b[:, bx, j]                            # values top to bottom
            need = bb[R - 2] if R > 1 else 0            # full columns the ground block must offer
            # ground block: f full columns, the rest at least G, sum in its bin
            s_lo, s_hi = bb[R - 1] * K, (bb[R - 1] * K + K - 1 if bb[R - 1] < w else w * K)
            t = None
            for f in range(max(need, 0), w + 1):
                lo_s, hi_s = f * K + (w - f) * ground, f * K + (w - f) * (K - 1) if f < w else w * K
                s = max(lo_s, s_lo)
                if s <= min(hi_s, s_hi):
                    rest = s - f * K
                    tt = np.full(w, ground if f < w else K)
                    tt[:f] = K
                    extra = rest - (w - f) * ground
                    for c in range(f, w):
                        add = min(extra, K - 1 - ground)
                        tt[c] += add
                        extra -= add
                    t = tt
                    break
            assert t is not None, "value grid not realisable (ground block)"
            h_col = t.copy()                            # local height in the ground block
            for r in range(R - 2, -1, -1):              # rows above: full iff column index < b_r
                full_below = h_col == (R - 1 - r) * K
                h_col = np.where(full_below & (np.arange(w) < bb[r]), h_col + K, h_col)
            H[cols] = h_col
    return H
