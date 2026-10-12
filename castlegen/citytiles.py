"""City tile set for wave function collapse: macro-scale 2D city map.

Dual-grid Wang tiles.  A tile's four CORNERS are city blocks (a zone label
each) and its four EDGES carry what crosses them centre to centre (a street
network on the dual grid: the tile centre is an intersection, the corners are
the four blocks around it).  Neighbours agree on the two corners and the
feature of their shared edge.  The tile set is every (corners, edges) tuple
that passes the city rules below, so rotations and mirrors come for free.

Zones (corner labels)
  P park, R residential, C commercial, D downtown, I industrial, W water,
  Hrc / Vrc airport blocks: a 2 x 3 (H) or 3 x 2 (V) footprint, each corner
  labelled with its position in the footprint.
Features (edge labels)
  0 none (the two blocks on either side merge), s street, a avenue, t rail,
  h highway.

Rules (all local to one tile)
  topology   street: degree 2-4 (dead end only among four R blocks);
             avenue: straight or a turn, a straight one may take street
             crossings or a street T; rail: straight or a turn, a straight
             one may take a level crossing (street or avenue); highway:
             straight only, crossed only straight through (overpass or
             interchange).  No other mixes.
  zoning     at most two land zones per tile (no three districts meet at a
             crossing; water does not count); every pair of corners in a tile is compatible (gradient
             D - C - R, I never meets R or D, airport only with I, P, C, W).
  districts  two different land zones on one edge are split by a feature: no
             district (or park) boundary without a road; shores are free.
  nuisance   no R block on a rail or highway tile.
  shores     nothing crosses an edge between land and water: waterfront
             blocks are served from the land side, a road that reaches the
             water bridges it.
  bridges    a feature crossing an all-water edge goes straight through the
             tile and is its only feature.
  airport    corners inside one footprint sit at consistent positions, the
             footprint is closed (a footprint corner whose neighbour position
             is inside the footprint has that label there, one outside has a
             non-airport there), no feature inside it, a feature on every
             land perimeter edge.

Infinite tilings are trivial (all park, all water, a residential street
grid), but none of the long-range structure a city needs is visible to arc
consistency: highways are infinite straight lines, district boundaries must
close into road-bounded regions, the zoning gradient forces buffer rings, and
footprints and bridges interlock.  Unguided WFC contradicts on large patches;
see notes/city_tiles.md.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

BASE_ZONES = ["P", "R", "C", "D", "I", "W"]
AIR_H = [f"H{r}{c}" for r in range(2) for c in range(3)]   # 2 rows x 3 cols
AIR_V = [f"V{r}{c}" for r in range(3) for c in range(2)]   # 3 rows x 2 cols
ZONES = BASE_ZONES + AIR_H + AIR_V
FEATS = ["0", "s", "a", "t", "h"]
Z = {z: i for i, z in enumerate(ZONES)}
F = {f: i for i, f in enumerate(FEATS)}
BUILT = {"R", "C", "D", "I"}

# corner order NW, NE, SE, SW; edge order N, E, S, W
# edge k lies between corners EDGE_CORNERS[k]
EDGE_CORNERS = [(0, 1), (1, 2), (3, 2), (0, 3)]   # N:(NW,NE) E:(NE,SE) S:(SW,SE) W:(NW,SW)
CORNER_POS = [(0, 0), (0, 1), (1, 1), (1, 0)]     # (row, col) of each corner in the tile

COMPAT = {   # unordered pairs of zones allowed in one tile (P, W go with anything)
    ("R", "R"), ("C", "C"), ("D", "D"), ("I", "I"), ("A", "A"),
    ("R", "C"), ("C", "D"), ("C", "I"), ("A", "I"), ("A", "C"),
}

ZONE_COLOR = {"P": "#7fbf6a", "R": "#e8d9a8", "C": "#e3a35f", "D": "#c4574a",
              "I": "#9a8fb0", "W": "#4f86c6", "A": "#b9bfc4"}
FEAT_STYLE = {"s": ("#f4f4f4", 2), "a": ("#ffffff", 4), "t": ("#2b2b2b", 2),
              "h": ("#f2b632", 5)}


def zclass(z):
    return "A" if z[0] in "HV" else z


def air(z):
    """(kind, row, col, rows, cols) of an airport label, else None."""
    if z[0] == "H":
        return "H", int(z[1]), int(z[2]), 2, 3
    if z[0] == "V":
        return "V", int(z[1]), int(z[2]), 3, 2
    return None


def compatible(a, b):
    a, b = zclass(a), zclass(b)
    if a in "PW" or b in "PW":
        return True
    return (a, b) in COMPAT or (b, a) in COMPAT


def corners_ok(cs):
    if len({zclass(c) for c in cs} - {"W"}) > 2:   # at most two land zones meet at a tile
        return False
    if any(not compatible(cs[i], cs[j]) for i, j in itertools.combinations(range(4), 2)):
        return False
    for i in range(4):                       # airport footprints exact
        ai = air(cs[i])
        if ai is None:
            continue
        k, r, c, R, C = ai
        for j in range(4):
            if j == i:
                continue
            dr = CORNER_POS[j][0] - CORNER_POS[i][0]
            dc = CORNER_POS[j][1] - CORNER_POS[i][1]
            rr, cc = r + dr, c + dc
            inside = 0 <= rr < R and 0 <= cc < C
            want = f"{k}{rr}{cc}" if inside else None
            if inside and cs[j] != want:
                return False
            if not inside and air(cs[j]) is not None:
                return False
    return True


def feats_ok(fs, cs):
    n = {f: [k for k in range(4) if fs[k] == f] for f in FEATS}
    deg = {f: len(n[f]) for f in FEATS}
    straight = lambda ks: len(ks) == 2 and (ks[1] - ks[0]) == 2
    turn = lambda ks: len(ks) == 2 and (ks[1] - ks[0]) in (1, 3)
    others = lambda f: [fs[k] for k in range(4) if fs[k] != f]
    if deg["h"]:
        if deg["h"] == 4:
            return True                          # interchange
        o = others("h")
        return straight(n["h"]) and len(o) == 2 and o[0] == o[1]
    if deg["t"]:
        if straight(n["t"]):
            o = others("t")
            return o in (["0", "0"], ["s", "s"], ["a", "a"])
        return turn(n["t"]) and others("t") == ["0", "0"]
    if deg["a"]:
        if straight(n["a"]):
            return all(f in "0s" for f in others("a"))
        return turn(n["a"]) and others("a") == ["0", "0"]
    if deg["s"] == 1:
        return all(c == "R" for c in cs)
    return True


def tile_ok(cs, fs):
    if not feats_ok(fs, cs):
        return False
    for k, (i, j) in enumerate(EDGE_CORNERS):
        a, b = cs[i], cs[j]
        ca, cb = zclass(a), zclass(b)
        if ca != cb and "W" not in (ca, cb) and fs[k] == "0":
            return False                         # district boundary needs a road
        if (ca == "A") != (cb == "A") and "W" not in (ca, cb) and fs[k] == "0":
            return False                         # airport perimeter road
        if (ca == "W") != (cb == "W") and fs[k] != "0":
            return False                         # nothing runs along a shore
        if ca == "A" and cb == "A" and fs[k] != "0":
            return False                         # nothing crosses the airfield
        if a == "W" and b == "W" and fs[k] != "0":   # bridge: straight, alone
            f = fs[k]
            if fs[(k + 2) % 4] != f or any(fs[m] != "0" for m in ((k + 1) % 4, (k + 3) % 4)):
                return False
    if any(f in "th" for f in fs) and "R" in cs:
        return False
    return True


# ---- weights: what a plain WFC samples from --------------------------------
ZONE_W = {"P": 0.6, "R": 1.0, "C": 0.8, "D": 0.5, "I": 0.6, "W": 0.35, "A": 0.25}
BOUNDARY_W = 0.12
BRIDGE_W = 0.1
FEAT_W = {"0": 1.0, "s": 1.2, "a": 0.35, "t": 0.15, "h": 0.06}


def weight(cs, fs):
    w = 1.0
    for c in cs:
        w *= ZONE_W[zclass(c)] ** 0.25
    for f in fs:
        w *= FEAT_W[f] ** 0.5
    built = [zclass(c) in BUILT for c in cs]
    if all(f == "0" for f in fs) and all(built):
        w *= 0.15                                # superblock interiors are rare
    for k, (i, j) in enumerate(EDGE_CORNERS):
        if zclass(cs[i]) != zclass(cs[j]):
            w *= BOUNDARY_W                      # districts, not speckle
        if cs[i] == "W" and cs[j] == "W" and fs[k] != "0":
            w *= BRIDGE_W ** 0.5                 # bridges are expensive (each edge is shared)
    return w


def build():
    corner_sets = [cs for cs in itertools.product(ZONES, repeat=4) if corners_ok(cs)]
    feat_sets = list(itertools.product(FEATS, repeat=4))
    tiles = []
    for cs in corner_sets:
        for fs in feat_sets:
            if tile_ok(cs, fs):
                tiles.append((cs, fs))
    return tiles


class TileSet:
    """Arrays for a sampler.  T tiles; sockets are (corner, corner, feature)
    triples read left-to-right (horizontal edges) or top-to-bottom
    (vertical edges), so a tile's E socket must equal its right neighbour's W
    socket and its S socket its lower neighbour's N socket."""

    def __init__(self, tiles=None):
        self.tiles = tiles if tiles is not None else build()
        T = len(self.tiles)
        self.corners = np.array([[Z[c] for c in cs] for cs, _ in self.tiles], np.int32)
        self.feats = np.array([[F[f] for f in fs] for _, fs in self.tiles], np.int32)
        self.w = np.array([weight(cs, fs) for cs, fs in self.tiles], np.float64)
        nz, nf = len(ZONES), len(FEATS)
        enc = lambda a, b, f: (a * nz + b) * nf + f
        c, f = self.corners, self.feats
        self.sN = enc(c[:, 0], c[:, 1], f[:, 0])
        self.sE = enc(c[:, 1], c[:, 2], f[:, 1])
        self.sS = enc(c[:, 3], c[:, 2], f[:, 2])
        self.sW = enc(c[:, 0], c[:, 3], f[:, 3])
        self.n_sock = nz * nz * nf
        self.T = T

    def to_json(self, path):
        out = {
            "doc": __doc__,
            "zones": ZONES, "features": FEATS,
            "corner_order": ["NW", "NE", "SE", "SW"], "edge_order": ["N", "E", "S", "W"],
            "match": "E of left == W of right as (NE,SE,E) == (NW,SW,W); "
                     "S of top == N of bottom as (SW,SE,S) == (NW,NE,N)",
            "tiles": [{"corners": list(cs), "edges": list(fs), "weight": round(float(w), 5)}
                      for (cs, fs), w in zip(self.tiles, self.w)],
        }
        with open(path, "w") as fh:
            json.dump(out, fh, indent=None, separators=(",", ":"))


# ---- rendering --------------------------------------------------------------
def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def render(ts, grid, px=16, fail=None):
    """grid: (H, W) tile ids, -1 undecided.  Returns an (H*px, W*px, 3) uint8 image."""
    H, W = grid.shape
    img = np.full((H * px, W * px, 3), 40, np.uint8)
    h2 = px // 2
    quads = [(slice(0, h2), slice(0, h2)), (slice(0, h2), slice(h2, px)),
             (slice(h2, px), slice(h2, px)), (slice(h2, px), slice(0, h2))]
    zc = {i: np.array(_rgb(ZONE_COLOR[zclass(z)])) for i, z in enumerate(ZONES)}
    for y in range(H):
        for x in range(W):
            t = grid[y, x]
            cell = img[y * px:(y + 1) * px, x * px:(x + 1) * px]
            if t < 0:
                if fail is not None and fail[y, x]:
                    cell[:] = (255, 0, 255)
                continue
            for k, (qy, qx) in enumerate(quads):
                cell[qy, qx] = zc[ts.corners[t, k]]
            for k in range(4):
                f = FEATS[ts.feats[t, k]]
                if f == "0":
                    continue
                col, wd = FEAT_STYLE[f]
                col = np.array(_rgb(col))
                lo = h2 - (wd + 1) // 2
                hi = lo + wd
                if k == 0:
                    cell[:h2 + 1, lo:hi] = col
                elif k == 2:
                    cell[h2 - 1:, lo:hi] = col
                elif k == 1:
                    cell[lo:hi, h2 - 1:] = col
                else:
                    cell[lo:hi, :h2 + 1] = col
    return img


if __name__ == "__main__":
    ts = TileSet()
    path = os.path.join(os.path.dirname(__file__), "citytiles.json")
    ts.to_json(path)
    print(ts.T, "tiles ->", path)
