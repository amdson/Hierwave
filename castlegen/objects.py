"""Rigid objects: an extra per-cell channel that makes multi-cell structures
exact without a tile type per cell (a pyramid of 17 x 17 uses six stone
kinds, not 289).

Channel: o[p] = -1 (no object) or a code c = base[k] + dy w_k + dx, cell
(dy, dx) of object k's h_k x w_k template (exemplar.object_grid).  Tables:
    tmpl (NC,)       the template's tile at code c
    nbr  (NC + 1, 4) the code one step toward side d inside the same object
                     (-1 off the template); row NC = -1 (for o = -1)
Rule, per neighbour pair p, q = p + e_d:
    o[p] >= 0 and nbr[o[p], d] >= 0  =>  o[q] = nbr[o[p], d]   (and from q's side)
and t[p] = tmpl[o[p]] wherever o[p] >= 0.  Satisfied pairs compose along any
path, so a region with no broken pair is one translated copy of the template,
and since the rule fires from whichever side lies inside, a copy cannot end
before the template does: objects are whole or absent.  A cell off the map
counts as o = -1.

Coarse levels see the rule through exemplar coordinates: a level-h cell with
coordinate u holds the exemplar window centred on u, so the pairs across a
seam between two cells are fixed by their two coordinates
(coarse_term: lvl.extra for generic.JointLevel).  The tile level samples
(t, o) jointly (blockconn.map_sweep)."""
from __future__ import annotations

import numpy as np

from castlegen import exemplar as ex

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPP = (2, 3, 0, 1)


def tables(ts, layout):
    """-> (names, grids, base, tmpl, nbr) for layout["objects"]."""
    names = list(layout.get("objects", {}))
    grids = [ex.object_grid(ts, layout, n) for n in names]
    base = np.cumsum([0] + [g.size for g in grids])
    NC = int(base[-1])
    tmpl = np.zeros(NC, np.int64)
    nbr = np.full((NC + 1, 4), -1, np.int64)
    for k, g in enumerate(grids):
        h, w = g.shape
        tmpl[base[k]:base[k + 1]] = g.ravel()
        for y in range(h):
            for x in range(w):
                for d, (dy, dx) in enumerate(DIRS):
                    if 0 <= y + dy < h and 0 <= x + dx < w:
                        nbr[base[k] + y * w + x, d] = base[k] + (y + dy) * w + x + dx
    return names, grids, base[:-1], tmpl, nbr


def channel(ts, layout, tabs):
    """(H, W) int64 object channel of the layout's "object" items (wrapped)."""
    names, grids, base, _, _ = tabs
    H, W = layout["size"]
    O = np.full((H, W), -1, np.int64)
    for item in layout["place"]:
        if "object" in item:
            k = names.index(item["object"])
            h, w = grids[k].shape
            y0, x0 = item["at"]
            ys, xs = (y0 + np.arange(h)) % H, (x0 + np.arange(w)) % W
            O[np.ix_(ys, xs)] = base[k] + np.arange(h * w).reshape(h, w)
    return O


def broken(O, nbr):
    """Number of broken (pair, side) rule checks on the map (not wrapped)."""
    Op = np.pad(O, 1, constant_values=-1)
    H, W = O.shape
    n = 0
    for d, (dy, dx) in enumerate(DIRS):
        q = Op[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        want = nbr[O, d]
        n += int(((want >= 0) & (q != want)).sum())
    return n


def census(O, T, tabs):
    """{name: (complete, cells of the object outside a complete copy)}, plus
    "tiles off template": cells whose tile differs from tmpl[o]."""
    names, grids, base, tmpl, _ = tabs
    H, W = O.shape
    out = {}
    for k, n in enumerate(names):
        h, w = grids[k].shape
        codes = base[k] + np.arange(h * w).reshape(h, w)
        cover = np.zeros((H, W), bool)
        done = 0
        for y, x in zip(*np.nonzero(O == base[k])):
            if y + h <= H and x + w <= W and np.array_equal(O[y:y + h, x:x + w], codes):
                done += 1
                cover[y:y + h, x:x + w] = True
        out[n] = (done, int(((O >= base[k]) & (O < base[k] + h * w) & ~cover).sum()))
    on = O >= 0
    out["tiles off template"] = int((on & (T != tmpl[np.maximum(O, 0)])).sum())
    return out


def _edge(u, h, m, d):
    """(..., h) flat exemplar coordinates of side d's edge cells of the h x h
    window centred on each coordinate u (torus exemplar of side m)."""
    uy, ux = np.divmod(np.asarray(u)[..., None], m)
    i = np.arange(h)
    y0, x0 = uy - h // 2, ux - h // 2
    ly, lx = ((0 * i, i), (i, 0 * i + h - 1), (0 * i + h - 1, i), (i, 0 * i))[d]
    return ((y0 + ly) % m) * m + (x0 + lx) % m


def coarse_term(lvl, Oex, nbr, w):
    """lvl.extra for a JointLevel on a torus exemplar: w times the rule checks
    broken across cell p's four seams with u_p = each candidate (neighbours
    as they are; off the map: o = -1)."""
    Of = Oex.ravel()
    h, m = lvl.h, lvl.m

    def extra(py, px, allc):
        e = np.zeros(allc.shape)
        for d, (dy, dx) in enumerate(DIRS):
            a = Of[_edge(allc, h, m, d)]                                     # (k, M, h) p's side d
            ny, nx = py + dy, px + dx
            inm = (ny >= 0) & (ny < lvl.R) & (nx >= 0) & (nx < lvl.C)
            uq = lvl.U[np.clip(ny, 0, lvl.R - 1), np.clip(nx, 0, lvl.C - 1)]
            b = np.where(inm[:, None], Of[_edge(uq, h, m, OPP[d])], -1)[:, None, :]   # (k, 1, h) q's facing side
            wa, wb = nbr[a, d], nbr[b, OPP[d]]
            e += ((wa >= 0) & (b != wa)).sum(-1) + ((wb >= 0) & (a != wb)).sum(-1)
        return w * e
    return extra
