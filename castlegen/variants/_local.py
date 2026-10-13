"""Local conditionals over known cells, generic over the declarations
(castlegen.bench.problem).  Shared helper for variants; not frozen.

cond(problem, state, region, chan, Y, X) -> (soft, nviol), each (n, D): for
every candidate value of `chan` at the sites (Y, X), the summed finite
entries and the number of inf entries of every factor instance that touches
the site and whose other cells are all known (state value >= 0; unknown is
-1).  Instances with an unknown cell are skipped (no lookahead)."""
from __future__ import annotations

import numpy as np


def read_view(problem, state, cv, Y, X, region, pad):
    """View values of channel/view cv at level-coordinates (Y, X), and whether
    the cell exists (or is padded) and is known."""
    c = problem.chan[cv[0]]
    arr = state[c.name]
    h, w = arr.shape
    ok = np.ones(np.shape(Y), bool)
    if region.periodic:
        Xr = np.mod(X, w)
    else:
        ok &= (X >= 0) & (X < w)
        Xr = np.clip(X, 0, w - 1)
    off = (Y < 0) | (Y >= h)
    raw = arr[np.clip(Y, 0, h - 1), Xr]
    view = c.views[cv[1]]
    v = view[np.maximum(raw, 0)]
    known = raw >= 0
    if pad >= 0:
        v = np.where(off, pad, v)
        known = known | off
    else:
        ok &= ~off
    return v, ok & known


def _within(f, la, Ya, Xa, lb, Yb, Xb):
    if f.within is None:
        return True
    L = f.within
    return ((Ya << la) >> L == (Yb << lb) >> L) & ((Xa << la) >> L == (Xb << lb) >> L)


def _add(soft, nv, e, ok):
    inf = np.isinf(e)
    soft += np.where(inf | ~ok[:, None], 0.0, e)
    nv += (inf & ok[:, None]).astype(np.int64)


def cond(problem, state, region, chan, Y, X):
    c = problem.chan[chan]
    l, D = c.level, c.D
    n = Y.size
    soft = np.zeros((n, D))
    nv = np.zeros((n, D), np.int64)
    for f in problem.factors:
        la = problem.chan[f.a[0]].level
        if f.kind == "unary":
            if f.a[0] == chan:
                _add(soft, nv, np.broadcast_to(f.table[c.views[f.a[1]]], (n, D)).copy(), np.ones(n, bool))
            continue
        lb = problem.chan[f.b[0]].level
        dy, dx = f.off
        if f.kind == "pair":
            if f.a[0] == chan:                          # site is a
                ta = c.views[f.a[1]]
                Yb, Xb = (Y + dy, X + dx) if lb == la else ((Y >> 1) + dy, (X >> 1) + dx)
                vb, ok = read_view(problem, state, f.b, Yb, Xb, region, f.pad_b)
                ok &= _within(f, la, Y, X, lb, Yb, Xb)
                if f.by_pos:
                    e = f.table[(2 * (Y & 1) + (X & 1))[:, None], ta[None, :], vb[:, None]]
                else:
                    e = f.table[ta[None, :], vb[:, None]]
                _add(soft, nv, e, ok)
            if f.b[0] == chan:                          # site is b
                tb = c.views[f.b[1]]
                if lb == la:
                    Ya, Xa = Y - dy, X - dx
                    va, ok = read_view(problem, state, f.a, Ya, Xa, region, f.pad_a)
                    ok &= _within(f, la, Ya, Xa, lb, Y, X)
                    _add(soft, nv, f.table[va[:, None], tb[None, :]], ok)
                else:                                   # site is the parent: every child of (Y - dy, X - dx)
                    Yp, Xp = Y - dy, X - dx
                    hp = problem.H >> lb
                    for i in (0, 1):
                        for j in (0, 1):
                            Ya, Xa = 2 * Yp + i, 2 * Xp + j
                            va, ok = read_view(problem, state, f.a, Ya, Xa, region, -1)
                            ok &= (Yp >= 0) & (Yp < hp)
                            ok &= _within(f, la, Ya, Xa, lb, Y, X)
                            if f.by_pos:
                                e = f.table[2 * i + j, va[:, None], tb[None, :]]
                            else:
                                e = f.table[va[:, None], tb[None, :]]
                            _add(soft, nv, e, ok)
        elif f.kind == "count":
            if f.a[0] == chan:                          # site is a child
                ta = c.views[f.a[1]]
                k = 2 * (Y & 1) + (X & 1)
                Yq, Xq = Y >> 1, X >> 1
                vb, ok = read_view(problem, state, f.b, Yq, Xq, region, -1)
                ok &= np.isin(k, f.subset)
                s = np.zeros(n, np.int64)
                for kk in f.subset:
                    Yc, Xc = 2 * Yq + (kk >> 1), 2 * Xq + (kk & 1)
                    vc, okc = read_view(problem, state, f.a, Yc, Xc, region, -1)
                    mine = k == kk
                    ok &= okc | mine
                    s += np.where(mine, 0, vc)
                idx = np.minimum(s[:, None] + ta[None, :], f.table.shape[1] - 1)   # rows not in the subset are masked
                e = f.table[vb[:, None], idx]
                _add(soft, nv, e, ok)
            if f.b[0] == chan:                          # site is the parent
                tb = c.views[f.b[1]]
                ok = np.ones(n, bool)
                s = np.zeros(n, np.int64)
                for kk in f.subset:
                    vc, okc = read_view(problem, state, f.a, 2 * Y + (kk >> 1), 2 * X + (kk & 1), region, -1)
                    ok &= okc
                    s += vc
                _add(soft, nv, f.table[tb[None, :], s[:, None]], ok)
    return soft, nv


def factor_reach(problem):
    """Largest |dx| any factor reaches at its own level (at least 1)."""
    r = 1
    for f in problem.factors:
        if f.kind == "pair":
            r = max(r, abs(f.off[1]) + 1)
    return r
