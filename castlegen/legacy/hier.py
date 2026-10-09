"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Hierarchical variables: one level sampler for every variable type.

A level is indexed by its cell side h in fine cells; its grid has side n/h and
every cell holds one value per variable type that exists at h (notes/history/promises.md
sections 3.2, 3.3, 3.6).  `run` scans h = h_max .. 1, coarse to fine: each
variable is initialised from its parent level (`init_children`) or, at its
coarsest level, from `top`; then a few sweeps update it over the four
checkerboard subpasses (0,0), (1,1), (1,0), (0,1), each vectorised over the
whole subgrid of that colour:

    C = v.candidates(level, colour, ctx)          # (a, b, nC, *value_shape)
    e = v.energy(C, level, colour, ctx)           # (a, b, nC)
      + sum of coupling energies involving v
    value = pick(C, e, v.T(h), noise)             # argmin at T = 0, Gumbel-max otherwise

A variable with `group = 2` (promises) is updated per parent instead: the
colours run over the grid of 2x2 sibling groups, candidates are whole
refinements, and `assign` writes the picked one back (notes/history/promises.md section 5).

A variable with `chain = True` samples its own levels instead (e.g. an exact
block sampler, castlegen/legacy/envelope_var.py): `init_level(parent_grid, level,
ctx, couplings)` replaces top/init_children (parent_grid is None at the
variable's coarsest level) and `sweep_level(level, s, ctx, couplings)` is
called once per sweep s < sweeps(h); `couplings` are those involving it.
Within a level, chain variables are initialised first and swept first in
every sweep, before the colour loop of the other variables (which skips
them), so e.g. the coordinate variable sees the promise grid of its level.

All randomness comes from `noise`, a pure function of (seed, level, step,
colour, y, x, slot), so any window of any level can be regenerated alone.
The concrete coordinate variable (texture synthesis) is texsyn.CoordVar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COLOURS = ((0, 0), (1, 1), (1, 0), (0, 1))      # texsyn's subpass order
INIT_STEP = 0xFFFFFFFF                          # noise `step` used for initialisation (jitter)

_M32 = np.uint64(0xFFFFFFFF)


# --------------------------------------------------------------------- noise
def _fmix32(x):
    x = x ^ (x >> np.uint64(16))
    x = (x * np.uint64(0x85EBCA6B)) & _M32
    x = x ^ (x >> np.uint64(13))
    x = (x * np.uint64(0xC2B2AE35)) & _M32
    return x ^ (x >> np.uint64(16))


def _u32(v):
    return np.asarray(v).astype(np.uint64) & _M32


def noise(seed, level, step, colour, y, x, slot):
    """Uniform in (0, 1), float32, shape = broadcast of the integer inputs.
    numpy port of core.noise (bit-identical), murmur3 finaliser on
    (seed, level, step, colour, y, x, slot)."""
    h = _fmix32((_u32(seed) * np.uint64(0x9E3779B1) + _u32(level)) & _M32)
    h = _fmix32(h ^ ((_u32(step) * np.uint64(0x85EBCA6B) + _u32(colour)) & _M32))
    h = _fmix32(h ^ ((_u32(y) * np.uint64(0xC2B2AE35) + _u32(x)) & _M32))
    h = _fmix32(h ^ _u32(slot))
    return ((h >> np.uint64(8)).astype(np.float32) + np.float32(0.5)) / np.float32(16777216.0)


def gumbel(u):
    return -np.log(-np.log(u))


def level_index(h):
    """Noise key of the level with cell side h (log2 h)."""
    return int(h).bit_length() - 1


def colour_coords(shape, colour):
    """Row and column indices (a, 1), (1, b) of the cells of one subpass colour."""
    i, j = colour
    return np.arange(i, shape[0], 2)[:, None], np.arange(j, shape[1], 2)[None, :]


# -------------------------------------------------------------------- levels
@dataclass
class Level:
    h: int                                       # cell side in fine cells
    shape: tuple                                 # (n // h, n // h)
    vars: dict = field(default_factory=dict)     # name -> array of shape + value_shape


@dataclass
class Ctx:
    seed: int = 0
    levels: list = field(default_factory=list)  # every Level of the last run, coarse to fine
    data: dict = field(default_factory=dict)    # shared scratch for variable types and couplings


class VarType:
    """One hierarchical variable.  Methods take a whole level (and a subpass
    colour) and return arrays; see the module docstring for shapes."""
    name = "var"
    hard = False          # infeasible candidates get +inf energy
    h_min, h_max = 1, 1   # range of cell sides h at which the variable exists
    salt = 0              # separates this variable's noise stream from others'
    group = 1             # update unit: 1 = one cell; 2 = a parent's 2x2 children jointly
                          # (colours then run over the grid of groups, see `assign`)
    chain = False         # True: the variable samples whole levels itself (init_level, sweep_level)

    def exists(self, h):
        return self.h_min <= h <= self.h_max

    def top(self, level, ctx):
        """Values at the variable's coarsest level, shape level.shape + value_shape."""
        raise NotImplementedError

    def init_children(self, parent, level, ctx):
        """Initial values at `level` from the parent level's grid `parent`."""
        raise NotImplementedError

    def sweeps(self, h):
        return 0

    def T(self, h):
        return 0.0

    def candidates(self, level, colour, ctx):
        """(a, b, nC, *value_shape) candidates for the cells of one colour."""
        raise NotImplementedError

    def energy(self, cands, level, colour, ctx):
        """(a, b, nC) local energy of each candidate."""
        raise NotImplementedError

    def assign(self, level, colour, values, ctx):
        """Write the picked candidates (a, b, *value_shape) back.  With group
        g > 1, (a, b) index the groups of that colour and the variable decides
        which cells a candidate sets."""
        i, j = colour
        level.vars[self.name][i::2, j::2] = values


class Coupling:
    """An energy term between variables of the same cell (or its ancestors).
    Added to the energy of whichever involved variable is being updated."""
    vars = ()

    def involves(self, var_name):
        return var_name in self.vars

    def energy(self, var, cands, level, colour, ctx):
        raise NotImplementedError


# ------------------------------------------------------------------- sampler
def pick(C, e, T, u=None):
    """Choose one candidate per cell: argmin of e at T = 0, else Gumbel-max
    of -e/T with uniforms u (same shape as e)."""
    if T > 0:
        with np.errstate(invalid="ignore"):
            # an infeasible candidate (+inf) stays unpickable even when u rounds to 1.0 (gumbel = +inf)
            e = np.where(np.isfinite(e), e / T - gumbel(u), np.inf)
    best = np.argmin(e, 2)
    idx = best.reshape(best.shape + (1,) * (C.ndim - 2))
    return np.take_along_axis(C, idx, 2)[:, :, 0]


def run(types, couplings, n, ctx=None):
    """Sample every level from h = max h_max down to 1.  Returns the finest
    Level; all levels are left in ctx.levels (coarse to fine)."""
    ctx = Ctx() if ctx is None else ctx
    ctx.levels = []
    couplings = list(couplings)
    h = max(v.h_max for v in types)
    assert n % h == 0, "output side must be a multiple of the coarsest cell side"
    prev = None
    while h >= 1:
        lv = Level(h, (n // h, n // h))
        live = [v for v in types if v.exists(h)]
        chains = [v for v in live if getattr(v, "chain", False)]
        cells = [v for v in live if not getattr(v, "chain", False)]
        for v in chains:
            parent = None if prev is None or not v.exists(2 * h) else prev.vars[v.name]
            lv.vars[v.name] = v.init_level(parent, lv, ctx, [cp for cp in couplings if cp.involves(v.name)])
        for v in cells:
            if prev is None or not v.exists(2 * h):
                lv.vars[v.name] = v.top(lv, ctx)
            else:
                lv.vars[v.name] = v.init_children(prev.vars[v.name], lv, ctx)
        for s in range(max([v.sweeps(h) for v in live], default=0)):
            for v in chains:
                if s < v.sweeps(h):
                    v.sweep_level(lv, s, ctx, [cp for cp in couplings if cp.involves(v.name)])
            for ci, colour in enumerate(COLOURS):
                i, j = colour
                for v in cells:
                    gshape = (lv.shape[0] // v.group, lv.shape[1] // v.group)
                    if s >= v.sweeps(h) or i >= gshape[0] or j >= gshape[1]:
                        continue
                    C = v.candidates(lv, colour, ctx)
                    e = v.energy(C, lv, colour, ctx)
                    for cp in couplings:
                        if cp.involves(v.name):
                            e = e + cp.energy(v, C, lv, colour, ctx)
                    T = v.T(h)
                    u = None
                    if T > 0:
                        ys, xs = colour_coords(gshape, colour)
                        k = np.arange(e.shape[2]) + (v.salt << 20)
                        u = noise(ctx.seed, level_index(h), s, ci, ys[..., None], xs[..., None], k)
                    v.assign(lv, colour, pick(C, e, T, u), ctx)
        ctx.levels.append(lv)
        prev = lv
        h //= 2
    return prev
