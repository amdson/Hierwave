"""Dependency-cone returns for the coloured pass (cfast).

A draw i's return may sum the costs of any conservative superset of its
read-graph descendants D_i (the draws whose inputs, hence value or cost, can
change when a_i changes).  We derive one from the schedule alone:

1. Reach.  Every draw of a (level l, channel c) step reads only cells whose
   fine footprint lies within its own footprint widened by (ay, by) rows
   above / below and (ax, bx) columns left / right.  The reads are listed from
   the declarations: the actor window (same level around the parent block
   corner, parent level around the parent), and every factor touching c with
   the lookahead (same cells as the factor instances: pair partners, parent
   or children, count parent, siblings and children).  Known flags are fixed
   by the schedule, so only values propagate.

2. Hull.  Starting from i's footprint, walk the later steps in order.  At a
   step, the sites whose widened footprint meets the current hull may change
   (a box of the step's colour lattice); the hull grows to cover their
   footprints.  The hull always contains every possibly changed cell, so the
   step's box contains every draw of that step in D_i.  x is unwrapped: on a
   periodic strip the changed set is the union of the cones of i's periodic
   images, which is the wrap of the unwrapped cone (reachability is a union
   over sources); a box at least as wide as the lattice covers it all.

3. Sums.  G_i = sum over later steps of the step's box sum of per-site
   costs, by integral images (the lattice tiled twice in x for the wrap)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._cpass_core import NCOL, P, key_of


# ----------------------------------------------------------------- reach
def _reads(problem, l, cname, win, Y, X):
    """Cells (level, Y, X) read by a draw of channel cname at level l, site (Y, X)."""
    T = problem.T
    out = [(l, Y, X)]
    for dy, dx in win.same:
        out.append((l, 2 * (Y >> 1) + dy, 2 * (X >> 1) + dx))
    if l < T:
        for dy, dx in win.parent:
            out.append((l + 1, (Y >> 1) + dy, (X >> 1) + dx))
    for f in problem.factors:
        if f.kind == "unary":
            continue
        la, lb = problem.chan[f.a[0]].level, problem.chan[f.b[0]].level
        dy, dx = f.off
        if f.kind == "pair":
            if f.a[0] == cname:
                if lb == la:
                    out.append((lb, Y + dy, X + dx))
                else:
                    out.append((lb, (Y >> 1) + dy, (X >> 1) + dx))
            if f.b[0] == cname:
                if lb == la:
                    out.append((la, Y - dy, X - dx))
                else:
                    for i in (0, 1):
                        for j in (0, 1):
                            out.append((la, 2 * (Y - dy) + i, 2 * (X - dx) + j))
        elif f.kind == "count":
            if f.a[0] == cname:
                out.append((lb, Y >> 1, X >> 1))
                for kk in range(4):
                    out.append((la, 2 * (Y >> 1) + (kk >> 1), 2 * (X >> 1) + (kk & 1)))
            if f.b[0] == cname:
                for kk in range(4):
                    out.append((la, 2 * Y + (kk >> 1), 2 * X + (kk & 1)))
    return out


def fine_reach(problem, l, cname, win):
    """(ay, by, ax, bx): fine rows above / below and columns left / right of a
    level-l site's footprint that its draw may read."""
    base = 1 << (problem.T + 4)
    ay = by = ax = bx = 0
    for py in (0, 1):
        for px in (0, 1):
            Y, X = base + py, base + px
            for lc, Yc, Xc in _reads(problem, l, cname, win, Y, X):
                ay = max(ay, (Y << l) - (Yc << lc))
                by = max(by, ((Yc + 1) << lc) - ((Y + 1) << l))
                ax = max(ax, (X << l) - (Xc << lc))
                bx = max(bx, ((Xc + 1) << lc) - ((X + 1) << l))
    return ay, by, ax, bx


# ----------------------------------------------------------------- cones
def _fdiv(a, b):
    return np.floor_divide(a, b)


def _cdiv(a, b):
    return -np.floor_divide(-a, b)


class Cone:
    """Cone boxes for every recorded draw of a periodic layout (x0 = 0), and
    the gather tables for the cone returns."""

    def __init__(self, lay, win):
        assert lay.periodic and lay.x0 == 0
        p = lay.p
        self.lay = lay
        # step table: (level, colour, channel index, channel name)
        steps = [None] * lay.nsteps
        for l in lay.levels:
            for k in range(NCOL):
                for ci, c in enumerate(lay.chans[l]):
                    steps[lay.step(l, k, ci)] = (l, k, ci, c.name)
        self.steps = steps
        self.reach = [fine_reach(p, l, cn, win) for (l, k, ci, cn) in steps]
        # integral-image layout: one leading zero, then each step's (ny + 1, 2 nx + 1)
        self.ii_off, off = {}, 1
        for s, (l, k, ci, cn) in enumerate(steps):
            ny, nx = lay.nyx[l]
            self.ii_off[s] = off
            off += (ny + 1) * (2 * nx + 1)
        self.ii_size = off
        self.idx, self.first = {}, {}
        self.boxes = {}
        for l in lay.levels:
            for ci, c in enumerate(lay.chans[l]):
                key = key_of(l, c.name)
                self.first[key] = lay.base[l]
                self.boxes[key], self.idx[key] = self._tables(l, ci)

    def _tables(self, l0, ci0):
        """boxes (16, n, S, 4) int64 [iy0, iy1, ix0, ix1] (iy0 > iy1: empty; x
        unwrapped) and flat gather indices (16, n, S, 4) int32 into the
        integral images, for steps base[l0] .. nsteps - 1 (S of them)."""
        lay = self.lay
        n = lay.nsite(l0)
        s_first = lay.base[l0]
        S = lay.nsteps - s_first
        boxes = np.zeros((NCOL, n, S, 4), np.int64)
        boxes[..., 0], boxes[..., 1] = 1, 0                       # empty
        idx = np.zeros((NCOL, n, S, 4), np.int32)
        k0 = np.repeat(np.arange(NCOL), n)                          # (16 n,)
        ny0, nx0 = lay.nyx[l0]
        iy, ix = np.divmod(np.tile(np.arange(n), NCOL), nx0)
        Y0 = k0 // P + P * iy
        X0 = k0 % P + P * ix
        s0 = np.array([lay.step(l0, k, ci0) for k in range(NCOL)])[k0]
        hy0, hy1 = Y0 << l0, ((Y0 + 1) << l0) - 1
        hx0, hx1 = X0 << l0, ((X0 + 1) << l0) - 1
        bx_ = boxes.reshape(NCOL * n, S, 4)
        ix_ = idx.reshape(NCOL * n, S, 4)
        for s in range(s_first, lay.nsteps):
            l, k, ci, cn = self.steps[s]
            ay, by, ax, bx = self.reach[s]
            h, w = lay.hw[l]
            ny, nx = lay.nyx[l]
            cy, cx = k // P, k % P
            u = 1 << l
            Ylo = np.maximum(_fdiv(hy0 - by, u), 0)
            Yhi = np.minimum(_fdiv(hy1 + ay, u), h - 1)
            iy0 = np.maximum(_cdiv(Ylo - cy, P), 0)
            iy1 = _fdiv(Yhi - cy, P)
            Xlo = _fdiv(hx0 - bx, u)
            Xhi = _fdiv(hx1 + ax, u)
            ix0 = _cdiv(Xlo - cx, P)
            ix1 = _fdiv(Xhi - cx, P)
            full = ix1 - ix0 + 1 >= nx
            ix0 = np.where(full, 0, ix0)
            ix1 = np.where(full, nx - 1, ix1)
            on = (s > s0) & (iy0 <= iy1) & (ix0 <= ix1)
            j = s - s_first
            bx_[on, j] = np.stack([iy0, iy1, ix0, ix1], 1)[on]
            # hull
            hy0 = np.where(on, np.minimum(hy0, (cy + P * iy0) << l), hy0)
            hy1 = np.where(on, np.maximum(hy1, ((cy + P * iy1 + 1) << l) - 1), hy1)
            hx0 = np.where(on, np.minimum(hx0, (cx + P * ix0) << l), hx0)
            hx1 = np.where(on, np.maximum(hx1, ((cx + P * ix1 + 1) << l) - 1), hx1)
            # gather indices: box rows iy0..iy1, columns a..b with a in [0, nx)
            a = np.mod(ix0, nx)
            b = a + (ix1 - ix0)
            Wd = 2 * nx + 1
            o = self.ii_off[s]
            r0, r1, c0, c1 = iy0, iy1 + 1, a, b + 1
            g = np.stack([r1 * Wd + c1, r0 * Wd + c1, r1 * Wd + c0, r0 * Wd + c0], 1) + o
            ix_[on, j] = g[on]
        return boxes, idx

    # ------------------------------------------------------------- returns
    def integral(self, costs):
        """costs: {key: (B, 16, n)} per-site costs (0 at invalid sites) ->
        (B, ii_size) concatenated integral images."""
        lay = self.lay
        Bn = next(iter(costs.values())).shape[0]
        parts = [None] * lay.nsteps
        for key, c in costs.items():
            l = int(key[1:key.index("_")])
            ny, nx = lay.nyx[l]
            img = c.reshape(Bn, NCOL, ny, nx)
            img = jnp.concatenate([img, img], -1)
            ii = jnp.cumsum(jnp.cumsum(img, -2), -1)
            ii = jnp.pad(ii, ((0, 0), (0, 0), (1, 0), (1, 0)))
            ci = [c_.name for c_ in lay.chans[l]].index(key[key.index("_") + 1:])
            for k in range(NCOL):
                parts[lay.step(l, k, ci)] = ii[:, k].reshape(Bn, -1)
        return jnp.concatenate([jnp.zeros((Bn, 1), jnp.float32)] + parts, -1)

    def returns(self, costs, idx=None):
        """{key: (B, 16, n)} cone returns G_i (sum of D_i's costs)."""
        ii = self.integral(costs)
        idx = idx if idx is not None else {k: jnp.asarray(v) for k, v in self.idx.items()}
        G = {}
        for key, ix in idx.items():
            v = ii[:, ix]                                            # (B, 16, n, S, 4)
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        return G

    def member(self, key, k, j):
        """{key2: (16, n2) bool} draws in the cone of draw (key, colour k, site j)."""
        lay = self.lay
        bx = self.boxes[key][k, j]                                  # (S, 4)
        out = {}
        for l in lay.levels:
            for ci, c in enumerate(lay.chans[l]):
                key2 = key_of(l, c.name)
                ny, nx = lay.nyx[l]
                m = np.zeros((NCOL, ny, nx), bool)
                for kk in range(NCOL):
                    s = lay.step(l, kk, ci)
                    jj = s - self.first[key]
                    if jj < 0:
                        continue
                    iy0, iy1, ix0, ix1 = bx[jj]
                    if iy0 > iy1:
                        continue
                    cols = np.mod(np.arange(ix0, ix1 + 1), nx)
                    m[kk, iy0:iy1 + 1][:, cols] = True
                out[key2] = m.reshape(NCOL, -1)
        return out
