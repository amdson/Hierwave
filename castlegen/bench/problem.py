"""Problem declarations for the coloured-pass benchmark (notes/bench.md).

FROZEN: variants read this module, never edit it.

World.  A strip of H fine rows, infinite in x.  Levels l = 0 (fine) .. T
(coarsest); a level-l cell (Y, X) covers fine cells [Y 2^l, (Y+1) 2^l) x
[X 2^l, (X+1) 2^l).  Its parent is (Y >> 1, X >> 1) at l + 1 and its child
position in that parent is k = 2 (Y & 1) + (X & 1).  Rows grow downward.

Region.  Either a periodic strip of circumference W fine cells (closed: used
for training, scoring and most metrics) or the window [x0, x0 + W) of the
infinite strip (open: used to test finite support).  W and x0 are multiples
of 2^T.

Sample.  dict channel name -> int array (H >> l, W >> l) over the region at
that channel's level.  Designed channels are included by the bench.

Channels.  Chan(name, level, D, views): values 0..D-1; views[name] is a (D,)
int array mapping a value to the view's value (view "id" always exists).
A designed channel has designed(seed, Y, X, period) -> int array: a fixed
input, a deterministic function of absolute cell coordinates (period is the
strip circumference at that level, or None on the infinite strip).

Factors (tables over views; inf entries are hard, finite entries soft).
A factor is homed on the level of its `a` channel, which is the finer side.
    unary  E[va(a[p])]
    pair   E[va(a[p]), vb(b[q])]
             same level:   q = p + off
             parent level: q = parent(p) + off; by_pos tables are (4, Va, Vb),
                           indexed first by p's child position
           rows of q off the strip read pad_b (-1: no instance); rows of p off
           the strip with q on it read pad_a (-1: no instance).
           within = L: an instance exists only if p and q (unwrapped) share a
           level-L ancestor.
    count  E[vb(b[q]), sum_{k in subset} va(a[child_k(q)])], b one level
           coarser than a; table (Vb, len(subset) * (Va - 1) + 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

INF = np.inf


@dataclass(frozen=True)
class Region:
    W: int
    periodic: bool = True
    x0: int = 0

    def __post_init__(self):
        assert not (self.periodic and self.x0 != 0), "a periodic strip starts at x0 = 0"


@dataclass
class Chan:
    name: str
    level: int
    D: int
    views: dict = field(default_factory=dict)
    designed: Callable | None = None

    def __post_init__(self):
        self.views = {k: np.asarray(v, np.int64) for k, v in self.views.items()}
        self.views.setdefault("id", np.arange(self.D, dtype=np.int64))
        for k, v in self.views.items():
            assert v.shape == (self.D,) and v.min() >= 0, (self.name, k)

    def nview(self, view):
        return int(self.views[view].max()) + 1


@dataclass
class Factor:
    kind: str                         # "unary" | "pair" | "count"
    a: tuple                          # (channel, view): the finer / home side
    table: np.ndarray
    b: tuple | None = None            # (channel, view)
    off: tuple = (0, 0)
    by_pos: bool = False
    subset: tuple = (0, 1, 2, 3)
    pad_a: int = -1
    pad_b: int = -1
    within: int | None = None
    name: str = ""

    def __post_init__(self):
        self.table = np.asarray(self.table, np.float64)
        self.off = tuple(int(o) for o in self.off)
        self.subset = tuple(self.subset)

    @property
    def hard(self):
        return bool(np.isinf(self.table).any())


def unary(a, table, name=""):
    return Factor("unary", a, table, name=name)


def pair(a, b, off, table, by_pos=False, pad_a=-1, pad_b=-1, within=None, name=""):
    return Factor("pair", a, table, b=b, off=off, by_pos=by_pos, pad_a=pad_a, pad_b=pad_b,
                  within=within, name=name)


def count(a, b, table, subset=(0, 1, 2, 3), name=""):
    return Factor("count", a, table, b=b, subset=subset, name=name)


@dataclass
class Problem:
    name: str
    T: int                            # coarsest level
    H: int                            # strip height in fine cells, a multiple of 2^T
    chans: list                       # declared order: coarse to fine, sampling order within a level
    factors: list
    W_train: int = 128                # suggested training circumference
    W_eval: int = 128                 # evaluation circumference / window
    stats: Callable | None = None     # stats(sample) -> {name: float}, per world
    render: Callable | None = None    # render(sample) -> (h, w, 3) uint8
    probes: Callable | None = None    # probes(seed, region) -> list of valid samples
    exact: dict = field(default_factory=dict)    # stat name -> exact value under the target
    tol: dict = field(default_factory=dict)      # stat name -> absolute tolerance
    version: int = 1
    notes: str = ""

    def __post_init__(self):
        assert self.H % (1 << self.T) == 0
        self.chan = {c.name: c for c in self.chans}
        assert len(self.chan) == len(self.chans), "channel names must be unique"
        for c in self.chans:
            assert 0 <= c.level <= self.T
        for f in self.factors:
            self._check(f)

    def _check(self, f):
        ca = self.chan[f.a[0]]
        va = ca.nview(f.a[1])
        if f.kind == "unary":
            assert f.table.shape == (va,), (f.name, f.table.shape)
            return
        cb = self.chan[f.b[0]]
        vb = cb.nview(f.b[1])
        if f.kind == "pair":
            assert cb.level in (ca.level, ca.level + 1), f.name
            shape = (4, va, vb) if f.by_pos else (va, vb)
            assert not f.by_pos or cb.level == ca.level + 1, f.name
            assert f.table.shape == shape, (f.name, f.table.shape, shape)
            assert -1 <= f.pad_a < va and -1 <= f.pad_b < vb, f.name
        elif f.kind == "count":
            assert cb.level == ca.level + 1, f.name
            shape = (vb, len(f.subset) * (va - 1) + 1)
            assert f.table.shape == shape, (f.name, f.table.shape, shape)
        else:
            raise ValueError(f.kind)

    # ---------------------------------------------------------------- helpers
    def level_chans(self, l):
        return [c for c in self.chans if c.level == l]

    def sampled(self):
        return [c for c in self.chans if c.designed is None]

    def shape(self, chan, region):
        l = self.chan[chan].level
        return (self.H >> l, region.W >> l)

    def designed_values(self, seed, region):
        """{name: array} for every designed channel over the region."""
        out = {}
        for c in self.chans:
            if c.designed is None:
                continue
            h, w = self.H >> c.level, region.W >> c.level
            Y, X = np.meshgrid(np.arange(h), np.arange(w) + (region.x0 >> c.level), indexing="ij")
            period = (region.W >> c.level) if region.periodic else None
            v = np.asarray(c.designed(seed, Y, X, period), np.int64)
            assert v.shape == (h, w) and v.min() >= 0 and v.max() < c.D, c.name
            out[c.name] = v
        return out

    def check_sample(self, x, region):
        for c in self.chans:
            assert c.name in x, f"missing channel {c.name}"
            a = np.asarray(x[c.name])
            assert a.shape == self.shape(c.name, region), (c.name, a.shape)
            assert a.min() >= 0 and a.max() < c.D, (c.name, a.min(), a.max())


# ------------------------------------------------------------- designed inputs
def value_noise_1d(seed, tag, X, cell, period=None):
    """Smooth value noise in [0, 1] along x: lattice values every `cell` cells,
    cubic Hermite in between.  X absolute (level) coordinates; period (cells)
    makes it periodic and must be a multiple of `cell`."""
    from . import noise
    X = np.asarray(X, np.int64)
    i0 = np.floor_divide(X, cell)
    t = (X - i0 * cell) / cell
    i1 = i0 + 1
    if period is not None:
        assert period % cell == 0, (period, cell)
        n = period // cell
        i0, i1 = i0 % n, i1 % n
    v0 = noise.uniform(np, seed, tag, 0, np.zeros_like(i0), i0)
    v1 = noise.uniform(np, seed, tag, 0, np.zeros_like(i1), i1)
    s = t * t * (3 - 2 * t)
    return v0 * (1 - s) + v1 * s
