"""dimers: perfect matchings of the triangular lattice (notes/bench.md).

The square grid plus the (1, 1) diagonal: every cell has six neighbours.
T = 0, H = 32 rows (the strip has a top and a bottom edge), x periodic or a
window.  One channel p (D = 6): the direction of the cell's partner,
    0 E (0, +1)   1 W (0, -1)   2 S (+1, 0)   3 N (-1, 0)   4 SE (+1, +1)   5 NW (-1, -1)
in (dy, dx), rows growing downward.  Hard: for each forward offset
o in {E, S, SE}, (p[c] == o) == (p[c + o] == -o); a designed channel edge
(0 middle, 1 top row, 2 bottom row) forbids N / NW in the top row and S / SE
in the bottom row.  No soft terms: the target is uniform over perfect
matchings (a gapped, short-correlated target on the triangular lattice).
"""
from __future__ import annotations

import numpy as np

from .. import noise
from ..problem import INF, Chan, Problem, pair

T, H, W = 0, 32, 64
E, Wd, S, N, SE, NW = range(6)
DIRS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1]], np.int64)
OPP = np.array([Wd, E, N, S, NW, SE], np.int64)
TAG_PROBE = noise.tag_of("dimers/probe")


def edge_designed(seed, Y, X, period):
    Y = np.asarray(Y, np.int64)
    return np.where(Y == 0, 1, np.where(Y == H - 1, 2, 0)).astype(np.int64)


def _factors():
    fs = []
    v = np.arange(6)
    for o, name in ((E, "E"), (S, "S"), (SE, "SE")):
        tab = np.where((v[:, None] == o) != (v[None, :] == OPP[o]), INF, 0.0)
        fs.append(pair(("p", "id"), ("p", "id"), tuple(DIRS[o]), tab, name=f"match{name}"))
    tab = np.zeros((6, 3))
    tab[[N, NW], 1] = INF
    tab[[S, SE], 2] = INF
    fs.append(pair(("p", "id"), ("edge", "id"), (0, 0), tab, name="edge"))
    return fs


# ------------------------------------------------------------------ stats
def _orient(p):
    return np.asarray(p, np.int64) // 2              # 0 horizontal, 1 vertical, 2 diagonal


def stats(x):
    """Orientation fractions (per cell = per dimer), and flippable rhombi.

    A rhombus is two lattice triangles sharing an edge; anchored at cell u it
    is u, u + a, u + b, u + a + b for a pair {a, b} of the forward directions
    {E, S, SE} (three rhombi per anchor).  It is flippable when two dimers lie
    on one pair of its opposite sides: (u, u+a) and (u+b, u+a+b) both dimers,
    or (u, u+b) and (u+a, u+a+b).  flip = flippable rhombi per cell;
    flip_cells = fraction of cells covered by at least one flippable rhombus.
    x wraps (periodic regions); rhombi leaving the strip vertically do not
    exist."""
    p = np.asarray(x["p"], np.int64)
    h, w = p.shape
    o = _orient(p)
    out = {"horizontal": float((o == 0).mean()), "vertical": float((o == 1).mean()),
           "diagonal": float((o == 2).mean())}

    def sh(arr, d, fill):
        dy, dx = int(d[0]), int(d[1])
        r = np.roll(arr, -dx, 1)
        res = np.full_like(arr, fill)
        if dy >= 0:
            res[:h - dy] = r[dy:]
        else:
            res[-dy:] = r[:h + dy]
        return res

    nflip = 0
    cover = np.zeros((h, w), bool)
    for a, b in ((E, S), (E, SE), (S, SE)):
        f1 = (p == a) & (sh(p, DIRS[b], -1) == a)
        f2 = (p == b) & (sh(p, DIRS[a], -1) == b)
        f = f1 | f2
        nflip += int(f.sum())
        for d in ((0, 0), DIRS[a], DIRS[b], DIRS[a] + DIRS[b]):
            dy, dx = int(d[0]), int(d[1])
            g = np.roll(f, dx, 1)
            if dy > 0:
                g = np.concatenate([np.zeros((dy, w), bool), g[:h - dy]], 0)
            cover |= g
    out["flip"] = nflip / (h * w)
    out["flip_cells"] = float(cover.mean())
    return out


def render(x, scale=4):
    """Cells coloured by their dimer's orientation, the forward end (E, S, SE)
    lighter, with a gap between cells that are not partners."""
    p = np.asarray(x["p"], np.int64)
    h, w = p.shape
    base = np.array([[214, 96, 77], [70, 130, 180], [96, 170, 90]], np.float64)
    col = base[p // 2] * np.where(p % 2 == 0, 1.0, 0.75)[..., None]
    img = np.repeat(np.repeat(col, scale, 0), scale, 1)
    gap = np.full((h * scale, w * scale), False)
    # darken the right / bottom border of a cell unless its partner is there
    rb = np.zeros((scale, scale), bool)
    rb[:, -1] = True
    bb = np.zeros((scale, scale), bool)
    bb[-1, :] = True
    gap |= np.kron(p != E, rb).astype(bool)
    gap |= np.kron(p != S, bb).astype(bool)
    img[gap] *= 0.45
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ probes
def _put(p, y, x, d, w):
    dy, dx = DIRS[d]
    p[y, x % w] = d
    p[y + dy, (x + dx) % w] = OPP[d]


def _band_squares(p, y0, w, u):
    """2 x 2 squares on rows y0, y0+1 (offset u[0]), each two horizontal or
    two vertical dimers."""
    c = int(u[0] * 2)
    for j in range(w // 2):
        x = 2 * j + c
        if u[1 + j] < 0.5:
            _put(p, y0, x, E, w)
            _put(p, y0 + 1, x, E, w)
        else:
            _put(p, y0, x, S, w)
            _put(p, y0, x + 1, S, w)


def _band_rhombi(p, y0, w, u):
    """{E, SE} rhombi (y0, x), (y0, x+1), (y0+1, x+1), (y0+1, x+2), each two
    horizontal or two diagonal dimers."""
    c = int(u[0] * 2)
    for j in range(w // 2):
        x = 2 * j + c
        if u[1 + j] < 0.5:
            _put(p, y0, x, E, w)
            _put(p, y0 + 1, x + 1, E, w)
        else:
            _put(p, y0, x, SE, w)
            _put(p, y0, x + 1, SE, w)


def _tiling(seed, k, region, mode):
    """mode 0: squares; 1: rhombi; 2: each 2-row band squares or rhombi."""
    w = region.W
    p = np.full((H, w), -1, np.int64)
    for band in range(H // 2):
        u = noise.uniform(np, seed, TAG_PROBE, k, np.full(w // 2 + 2, band), np.arange(w // 2 + 2))
        kind = mode if mode < 2 else int(u[-1] < 0.5)
        (_band_squares if kind == 0 else _band_rhombi)(p, 2 * band, w, u)
    assert p.min() >= 0
    return p


def probes(seed, region, n=4):
    """Valid tilings by construction; needs a periodic region (the bands
    wrap) or a window, where pairs crossing the window edge do not exist."""
    out = []
    for k, mode in enumerate((0, 1, 2, 2)[:n]):
        p = _tiling(seed, k, region, mode)
        Y = np.arange(H)[:, None] + np.zeros((1, region.W), np.int64)
        out.append({"p": p, "edge": edge_designed(seed, Y, Y * 0, None)})
    return out


def build_dimers():
    return Problem(
        name="dimers", T=T, H=H,
        chans=[Chan("edge", 0, 3, designed=edge_designed), Chan("p", 0, 6)],
        factors=_factors(), W_train=W, W_eval=W, stats=stats, render=render, probes=probes,
        # exact orientation fractions of the 32-row bench strip (Kasteleyn/Pfaffian, reportables/2026-10-15_deq;
        # user decision 2026-10-16). The earlier pre-check target 0.5/0.25/0.25 was not an enumeration.
        exact={"horizontal": 0.351, "vertical": 0.324, "diagonal": 0.324},
        tol={"horizontal": 0.02, "vertical": 0.02, "diagonal": 0.02, "flip": 0.03, "flip_cells": 0.03},
        notes="triangular-lattice dimers (square grid + (1,1) diagonal): partner direction per cell, "
              "hard matching consistency, strip edges by a designed edge channel; uniform over "
              "perfect matchings.")
