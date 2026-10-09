"""A stochastic split grammar compiled to a block pyramid.

Nodes are fixed-width int records: type, box [y0, y1) x [x0, x1) in world
tiles, a 63-bit seed hashed from the node's path, then type attributes.
Every choice a node makes is a pure function of its record (its own rng is
seeded from it), so the derivation is position-hashed and needs no state.

Compilation: levels 32 and 8 hold, per block, a fixed number of slots
(K = 64) of records.  A node type declares `split`, the level at which it
is replaced by its children.  Refinement is one generic step: a block's
slots are its parent block's slots, with every record whose split level is
>= the block's level replaced by those of its children that overlap the
block, repeatedly.  A record is copied into every block it covers
(broadcast), so replicas agree by construction and no seam is ever
checked.  Leaves are painted at level 1 from the level-8 slots, in
(z, seed) order.

The only state that is not a pure function of position lives at the owner
level (the grammar's roots): variables that interact with neighbours,
sampled by Gibbs with position-hashed noise and a fixed number of sweeps,
so that a root's value depends only on a bounded light cone.

Caches (children, rasters) are memoisation of pure functions; `clear()`
drops them so a chunk can be rendered from nothing."""
from __future__ import annotations

import numpy as np

R = 48
TYPE, Y0, X0, Y1, X1, SEED = range(6)
A = 6                                       # first attribute
K = 64                                      # slots per block
LEVELS = (32, 8)
_M = (1 << 64) - 1


def mix(*xs) -> int:
    """63-bit hash of a tuple of ints (splitmix64 rounds)."""
    h = 0x9E3779B97F4A7C15
    for x in xs:
        h = (h ^ (int(x) & _M)) * 0xBF58476D1CE4E5B9 & _M
        h ^= h >> 31
        h = (h + 0x9E3779B97F4A7C15) & _M
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9 & _M
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB & _M
    return (h ^ (h >> 31)) >> 1


def rng(*xs):
    return np.random.default_rng(mix(*xs))


def hash_grid(salt, ys, xs):
    """Uniform [0, 1) per (y, x), vectorised, a pure function of position."""
    with np.errstate(over="ignore"):
        h = (np.asarray(ys, np.int64).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) \
            ^ (np.asarray(xs, np.int64).astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)) \
            ^ np.uint64(mix(salt))
        h ^= h >> np.uint64(33)
        h *= np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(33)
        h *= np.uint64(0xC4CEB9FE1A85EC53)
        h ^= h >> np.uint64(33)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def record(t, y0, x0, y1, x1, seed, attrs=()):
    r = np.zeros(R, np.int64)
    r[:A] = (t, y0, x0, y1, x1, seed)
    r[A:A + len(attrs)] = attrs
    return r


def overlaps(r, y0, x0, y1, x1):
    return r[Y0] < y1 and r[Y1] > y0 and r[X0] < x1 and r[X1] > x0


class NodeType:
    split = 0                               # level at which children replace the node; 0 = leaf
    z = 0

    def children(self, rec):
        return []

    def raster(self, rec):
        """Leaf: (types (h, w) int, rgb (h, w, 3) uint8) over its box; type -1 is transparent."""
        raise NotImplementedError


class Canvas:
    """A leaf's drawing surface in absolute coordinates."""

    def __init__(self, rec):
        self.y0, self.x0 = int(rec[Y0]), int(rec[X0])
        h, w = int(rec[Y1] - rec[Y0]), int(rec[X1] - rec[X0])
        self.T = np.full((h, w), -1, np.int16)
        self.C = np.zeros((h, w, 3), np.uint8)

    def put(self, y, x, t, rgb, jitter=0.08, salt=0):
        """Write tile type t with colour rgb at absolute (y, x) (scalars or
        arrays); colour jittered by a position hash."""
        y, x = np.broadcast_arrays(np.asarray(y), np.asarray(x))
        yy, xx = y - self.y0, x - self.x0
        m = (yy >= 0) & (yy < self.T.shape[0]) & (xx >= 0) & (xx < self.T.shape[1])
        y, x, yy, xx = y[m], x[m], yy[m], xx[m]
        self.T[yy, xx] = t
        f = 1.0 + jitter * (2 * hash_grid(salt * 7919 + t, y, x) - 1)
        self.C[yy, xx] = np.clip(np.asarray(rgb, float)[None, :] * f[:, None], 0, 255).astype(np.uint8)

    def rect(self, y0, x0, y1, x1, t, rgb, **kw):
        ys, xs = np.mgrid[y0:y1, x0:x1]
        self.put(ys, xs, t, rgb, **kw)

    def sprite(self, rows, y_bottom, x0, t, palette, **kw):
        h = len(rows)
        for i, row in enumerate(rows):
            for j, ch in enumerate(row):
                if ch != " ":
                    self.put(y_bottom - (h - 1 - i), x0 + j, t, palette[ch], **kw)


class Grammar:
    """Subclass: H, types (list of NodeType, index = type id), roots(x0, x1),
    paint_field(x0, x1, T, C)."""
    H = 0
    types: list = []

    def __init__(self):
        self._kids, self._rast = {}, {}

    def clear(self):
        self._kids.clear()
        self._rast.clear()

    def children(self, rec):
        key = rec.tobytes()
        if key not in self._kids:
            self._kids[key] = self.types[rec[TYPE]].children(rec)
        return self._kids[key]

    def raster(self, rec):
        key = rec.tobytes()
        if key not in self._rast:
            self._rast[key] = self.types[rec[TYPE]].raster(rec)
        return self._rast[key]

    def expand(self, recs, level, box):
        out, stack = [], [r for r in recs if overlaps(r, *box)]
        while stack:
            r = stack.pop()
            if self.types[r[TYPE]].split >= level:
                stack.extend(c for c in self.children(r) if overlaps(c, *box))
            else:
                out.append(r)
        return out

    def render(self, x0, x1, stats=None):
        """(types (H, x1 - x0), rgb (H, x1 - x0, 3)) of the world columns
        [x0, x1), x0 and x1 multiples of 8."""
        assert x0 % 8 == 0 and x1 % 8 == 0
        H, W = self.H, x1 - x0
        T = np.full((H, W), -1, np.int16)
        C = np.zeros((H, W, 3), np.uint8)
        self.paint_field(x0, x1, T, C)
        roots = [r for r in self.roots(x0, x1) if overlaps(r, 0, x0, H, x1)]
        b32, b8 = LEVELS
        for by in range(H // b32):
            for bx in range(x0 // b32, -(-x1 // b32)):
                box = (by * b32, bx * b32, (by + 1) * b32, (bx + 1) * b32)
                s32 = self.expand(roots, b32, box)
                assert len(s32) <= K, len(s32)
                if stats is not None:
                    stats[b32] = max(stats.get(b32, 0), len(s32))
                for cy in range(box[0], box[2], b8):
                    for cx in range(max(box[1], x0), min(box[3], x1), b8):
                        box8 = (cy, cx, cy + b8, cx + b8)
                        s8 = self.expand(s32, b8, box8)
                        assert len(s8) <= K, len(s8)
                        if stats is not None:
                            stats[b8] = max(stats.get(b8, 0), len(s8))
                        for r in sorted(s8, key=lambda r: (self.types[r[TYPE]].z, int(r[SEED]))):
                            self._paint(r, box8, x0, T, C)
        return T, C

    def _paint(self, r, box, x0, T, C):
        lt, lc = self.raster(r)
        ya, xa = max(box[0], r[Y0]), max(box[1], r[X0])
        yb, xb = min(box[2], r[Y1]), min(box[3], r[X1])
        if ya >= yb or xa >= xb:
            return
        st = lt[ya - r[Y0]:yb - r[Y0], xa - r[X0]:xb - r[X0]]
        sc = lc[ya - r[Y0]:yb - r[Y0], xa - r[X0]:xb - r[X0]]
        m = st >= 0
        dt = T[ya:yb, xa - x0:xb - x0]
        dc = C[ya:yb, xa - x0:xb - x0]
        dt[m] = st[m]
        dc[m] = sc[m]
