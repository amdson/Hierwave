"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Support promise (notes/history/promises.md section 4.3, simplified): side view, gravity +y.

The rule, the guarantee: every solid cell (`ts.solid`) has a solid cell
directly below it, except cells of the ground band (the bottom G rows of the
map, exempt).  Ground-band cells also count as support for the cell above
them, whatever they hold (the pipeline paints the band solid anyway).  The
rule is per cell, so vertical block seams carry no constraint; only
horizontal seams do.  The map stays a torus; with G > 0 the wrap makes the
band the ceiling of row 0, which is harmless because band cells never need
support.  `support_violations` checks the rule on a whole grid.

State (exact, `State`): the block's bottom-row mask of solid cells that need
support from the block below (`need`: not exempt, row below not exempt), its
top-row solid mask (`top`), and the number of cells inside the block that
break the rule (`viol`, rows 0..k-2).  `merge` concatenates masks and adds
the internal horizontal seams' violations (upper.need & ~lower.top); exact.

Promise, I intervals per side (default 2, each child-sized):
  bh_i  "the bottom row has a solid needing support in interval i"   (upper bound on what the block needs)
  tf_i  "the top row is entirely solid on interval i"               (lower bound on what the block offers)
encoded v = sum_i bh_i 2**i + tf_i 2**(I+i), V = 4**I.  abstract(state) =
(any need per interval, all top per interval).  satisfied(v, state): viol ==
0, need_i => bh_i, tf_i => top all solid on i.  Every block with viol == 0
satisfies its own abstraction.

Validity per position: a ground block (its bottom row, or the row below it,
is in the band) has bh = 0; any other block has tf_i => bh_i (a solid top
interval needs its columns solid down to the bottom row).  Consistency
across a horizontal seam: upper.bh_i => lower.tf_i; vertical seams always
consistent.  Promise merge (exact): parent.bh_j = OR of the lower child's bh
bits on half j, parent.tf_j = AND of the upper child's tf bits on half j;
internal horizontal seams must be consistent.  Soundness: if the promise
grid is consistent and every block satisfies its promise, a bottom-row cell
needing support lies in an interval with bh = 1, so the block below has that
whole interval solid on its top row; with viol = 0 inside every block, no
cell anywhere breaks the rule.  A consistent parent level makes every
refinement's external seams consistent automatically (bh_j = 1 upstairs
forces every tf bit of the lower parent's child j), so a parent's
refinement never depends on its neighbours.

The language is sound but not complete: a valid grid can have inconsistent
abstractions (an upper need supported by a partly solid lower top row), and
consistency at K does not imply consistency of the merged level.  Corpus
grids are therefore projected top down (`project`) and fulfilled, as flow's.

Fulfil (per K-block, edits only inside it, ground band never edited, always
succeeds for a valid value): fill every column under a tf interval solid
down to the bottom row (or the band); clear bottom-row solids in bh = 0
intervals and cascade the clearing upward; then sweep bottom-up and, for
each unsupported solid, either fill below it down to the next solid (a new
bottom-row solid only where bh = 1) or clear it and the solid run above it,
whichever edits fewer cells.  Filled cells then take the solid material that
fits their neighbours best (greedy, within the solid class, so support is
unchanged); turf under a solid is re-picked the same way.
"""
from __future__ import annotations

import itertools

import numpy as np

from castlegen.legacy import hier
from castlegen.legacy.promise import PromiseVar

N, E, S, W = 0, 1, 2, 3
KIDS = ((0, 0), (0, 1), (1, 0), (1, 1))
TOP_COLOUR = 7
I_DEFAULT = 2
V = 4 ** I_DEFAULT                      # domain size at the default interval count


# ------------------------------------------------------------------- encoding
def nvals(I=I_DEFAULT):
    return 4 ** I


def encode(bh, tf, I=I_DEFAULT):
    bh, tf = np.asarray(bh, np.int64), np.asarray(tf, np.int64)
    w = 1 << np.arange(I)
    return (bh * w).sum(-1) + ((tf * w).sum(-1) << I)


_BITS = {}


def bits(I=I_DEFAULT):
    """(BH, TF): (V, I) bool tables of every value's bits."""
    if I not in _BITS:
        v = np.arange(nvals(I))[:, None]
        i = np.arange(I)[None, :]
        _BITS[I] = ((v >> i) & 1).astype(bool), ((v >> (I + i)) & 1).astype(bool)
    return _BITS[I]


def decode(v, I=I_DEFAULT):
    BH, TF = bits(I)
    return BH[int(v)], TF[int(v)]


def allowed(v, ground, I=I_DEFAULT):
    """Validity of value(s) v for a ground / ordinary block (array-friendly)."""
    BH, TF = bits(I)
    v = np.asarray(v)
    if ground:
        return ~BH[v].any(-1)
    return ~(TF[v] & ~BH[v]).any(-1)


# ------------------------------------------------------------------- the rule
def exempt_rows(n, ground):
    return np.arange(n) >= n - ground


def support_status(ts, tiles, ground=0):
    """Bool grid of solid cells breaking the rule (torus)."""
    s = ts.solid[np.asarray(tiles)]
    ex = exempt_rows(s.shape[0], ground)[:, None]
    below = np.roll(s, -1, 0) | np.roll(ex, -1, 0)
    return s & ~ex & ~below


def support_violations(ts, tiles, ground=0):
    """Number of unsupported solid cells on the torus."""
    return int(support_status(ts, tiles, ground).sum())


def has_support(ts):
    return getattr(ts, "solid", None) is not None and bool(ts.solid.any())


def ground_block(n, y0, k, ground):
    """True when the block's bottom row or the row below it is in the band."""
    lo = n - ground
    return ground > 0 and (y0 + k - 1 >= lo or (y0 + k < n and y0 + k >= lo))


def ground_rows(n, h, ground):
    return np.array([ground_block(n, r * h, h, ground) for r in range(n // h)])


# ---------------------------------------------------------------------- state
class State:
    __slots__ = ("y0", "x0", "k", "need", "top", "viol", "ground")

    def __init__(self, y0, x0, k, need, top, viol, ground):
        self.y0, self.x0, self.k = y0, x0, k
        self.need, self.top = np.asarray(need, bool), np.asarray(top, bool)
        self.viol, self.ground = int(viol), bool(ground)

    def key(self):
        return (self.y0, self.x0, self.k, self.need.tobytes(), self.top.tobytes(), self.viol, self.ground)

    def __eq__(self, other):
        return isinstance(other, State) and self.key() == other.key()

    def __repr__(self):
        return (f"State({self.y0},{self.x0},k={self.k}, need={self.need.astype(int)}, top={self.top.astype(int)}, "
                f"viol={self.viol}, ground={self.ground})")


def summarize(ts, tiles, y0, x0, k, ground=0):
    """Exact state of the k x k block at (y0, x0) (inside the grid)."""
    tiles = np.asarray(tiles)
    n = tiles.shape[0]
    s = ts.solid[tiles[y0:y0 + k, x0:x0 + k]]
    ex = exempt_rows(n, ground)[np.arange(y0, y0 + k + 1) % n]          # rows of the block and the row below
    sup = s[1:] | ex[1:k, None]                                           # support of rows 0..k-2
    viol = int((s[:-1] & ~ex[:k - 1, None] & ~sup).sum())
    need = s[k - 1] & ~ex[k - 1] & ~ex[k]
    return State(y0, x0, k, need, s[0], viol, ground_block(n, y0, k, ground))


def merge(children):
    c00, c01, c10, c11 = (children[key] for key in KIDS)
    seam = int((c00.need & ~c10.top).sum() + (c01.need & ~c11.top).sum())
    viol = c00.viol + c01.viol + c10.viol + c11.viol + seam
    return State(c00.y0, c00.x0, 2 * c00.k, np.concatenate([c10.need, c11.need]),
                 np.concatenate([c00.top, c01.top]), viol, c10.ground)


def _intervals(mask, I):
    return np.asarray(mask).reshape(I, -1)


def abstract(state, I=I_DEFAULT):
    return int(encode(_intervals(state.need, I).any(1), _intervals(state.top, I).all(1), I))


def satisfied(value, state, I=I_DEFAULT):
    bh, tf = decode(value, I)
    need = _intervals(state.need, I).any(1)
    full = _intervals(state.top, I).all(1)
    return state.viol == 0 and not (need & ~bh).any() and not (tf & ~full).any()


def violations(value, state, I=I_DEFAULT):
    """Proxy fulfil cost: 0 if satisfied, else 1 + mismatched bits + (viol > 0)."""
    need = _intervals(state.need, I).any(1)
    full = _intervals(state.top, I).all(1)
    return float(_costs(need, full, state.viol, I)[int(value)])


def _costs(need, full, viol, I=I_DEFAULT):
    """(V,) cost of every promise for a block with per-interval need / full bits."""
    BH, TF = bits(I)
    bad = (np.asarray(need)[None] & ~BH).sum(1) + (TF & ~np.asarray(full)[None]).sum(1) + (viol > 0)
    return np.where(bad == 0, 0, 1 + bad).astype(np.float32)


# ------------------------------------------------------------- promise merge
def _half(I):
    """Parent interval j -> (child column c, child interval indices)."""
    h = I // 2
    return [(j // h, [2 * (j % h), 2 * (j % h) + 1]) for j in range(I)]


def _merge_arrays(c00, c01, c10, c11, I=I_DEFAULT):
    """Vectorised merge_promises -> (parent, ok): ok = internal horizontal seams consistent."""
    BH, TF = bits(I)
    c = [np.asarray(x) for x in (c00, c01, c10, c11)]
    ok = ~(BH[c[0]] & ~TF[c[2]]).any(-1) & ~(BH[c[1]] & ~TF[c[3]]).any(-1)
    bh, tf = [], []
    for col, idx in _half(I):
        bh.append(BH[c[2 + col]][..., idx].any(-1))
        tf.append(TF[c[col]][..., idx].all(-1))
    return encode(np.stack(bh, -1), np.stack(tf, -1), I), ok


def merge_promises(c00, c01, c10, c11, I=I_DEFAULT):
    P, ok = _merge_arrays(c00, c01, c10, c11, I)
    return int(P), bool(ok)


def consistent(grid, parents=None, gnd=None, I=I_DEFAULT):
    """Every horizontal seam consistent (torus), every value valid for its
    row (gnd: (rows,) bool ground flags, optional), and with `parents` every
    2x2 group merges to its parent."""
    BH, TF = bits(I)
    g = np.asarray(grid)
    if (BH[g] & ~TF[np.roll(g, -1, 0)]).any():
        return False
    if gnd is not None:
        for r, gr in enumerate(gnd):
            if not allowed(g[r], bool(gr), I).all():
                return False
    if parents is not None:
        P, ok = _merge_arrays(g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2], I)
        return bool(ok.all() and np.array_equal(P, parents))
    return True


_PAIRS = {}


def _column_pairs(I, gu, gl):
    """(upper, lower) value pairs of one child column: both valid, seam consistent."""
    key = (I, gu, gl)
    if key not in _PAIRS:
        BH, TF = bits(I)
        V = nvals(I)
        a, b = np.meshgrid(np.arange(V), np.arange(V), indexing="ij")
        a, b = a.ravel(), b.ravel()
        m = allowed(a, gu, I) & allowed(b, gl, I) & ~(BH[a] & ~TF[b]).any(-1)
        _PAIRS[key] = a[m], b[m]
    return _PAIRS[key]


def _gnd(halo, gnd):
    if gnd is not None:
        return tuple(bool(x) for x in gnd)
    return tuple(halo.get("gnd", (False, False))) if halo else (False, False)


_REFINE = {}


def refine(parent, halo=None, gnd=None, I=I_DEFAULT):
    """Consistent refinements of `parent`: [(c00, c01, c10, c11)].  The two
    child columns are independent (no vertical-seam constraint), so this is a
    product of per-column (upper, lower) pairs.  halo: {(0, dx, N): bh bits of
    the block above child (0, dx), (1, dx, S): tf bits of the block below
    child (1, dx)} (ints), plus "gnd": (upper children ground, lower children
    ground); `gnd` overrides the halo's."""
    halo = halo or {}
    gu, gl = _gnd(halo, gnd)
    hk = tuple(sorted((k, v) for k, v in halo.items() if k != "gnd"))
    key = (int(parent), hk, gu, gl, I)
    if key in _REFINE:
        return _REFINE[key]
    BH, TF = bits(I)
    Pbh, Ptf = decode(parent, I)
    up, lo = _column_pairs(I, gu, gl)
    cols = []
    for dx in (0, 1):
        m = np.ones(len(up), bool)
        for j, (col, idx) in enumerate(_half(I)):
            if col != dx:
                continue
            m &= TF[up][:, idx].all(1) == Ptf[j]
            m &= BH[lo][:, idx].any(1) == Pbh[j]
        if (0, dx, N) in halo:                               # block above's needs lie on our full top
            m &= ~(BH[halo[0, dx, N]][None] & ~TF[up]).any(1)
        if (1, dx, S) in halo:
            m &= ~(BH[lo] & ~TF[halo[1, dx, S]][None]).any(1)
        cols.append(list(zip(up[m].tolist(), lo[m].tolist())))
    out = [(a0, a1, b0, b1) for (a0, b0), (a1, b1) in itertools.product(cols[0], cols[1])]
    _REFINE[key] = out
    return out


def n_refine(parent, halo=None, gnd=None, I=I_DEFAULT):
    return len(refine(parent, halo, gnd, I))


def brute_refine(parent, halo=None, gnd=None, I=I_DEFAULT):
    """Reference for tests: every child tuple filtered by validity, the halo
    and merge_promises."""
    halo = halo or {}
    gu, gl = _gnd(halo, gnd)
    BH, TF = bits(I)
    V = nvals(I)
    ok_u = [v for v in range(V) if allowed(v, gu, I)]
    ok_l = [v for v in range(V) if allowed(v, gl, I)]
    out = []
    for c00, c01 in itertools.product(ok_u, repeat=2):
        if any((0, dx, N) in halo and (BH[halo[0, dx, N]] & ~TF[c]).any() for dx, c in ((0, c00), (1, c01))):
            continue
        for c10, c11 in itertools.product(ok_l, repeat=2):
            if any((1, dx, S) in halo and (BH[c] & ~TF[halo[1, dx, S]]).any() for dx, c in ((0, c10), (1, c11))):
                continue
            if merge_promises(c00, c01, c10, c11, I) == (int(parent), True):
                out.append((c00, c01, c10, c11))
    return out


def halo(grid, py, px, ground=True):
    """{(0, dx, N): value above child (0, dx), (1, dx, S): value below child
    (1, dx)} of parent (py, px) read from `grid` (torus), plus "gnd": with
    `ground`, the lower children of the last parent row are ground blocks
    (the band lies in the last block row at every level)."""
    g0, g1 = grid.shape
    out = {}
    for dx in (0, 1):
        x = 2 * px + dx
        out[0, dx, N] = int(grid[(2 * py - 1) % g0, x % g1])
        out[1, dx, S] = int(grid[(2 * py + 2) % g0, x % g1])
    out["gnd"] = (False, bool(ground and 2 * py + 1 == g0 - 1))
    return out


# -------------------------------------------------------------- tile editing
def _materials(ts):
    """(air signature, solid material signatures): the first non-solid
    signature that is not the gate, and every solid one except the wall."""
    air = next(s for s in range(ts.n_sig) if not ts.solid[s] and s != ts.GATE)
    mats = np.array([s for s in range(ts.n_sig) if ts.solid[s] and s != ts.WALL], np.int64)
    return air, mats


def _refit(ts, win, todo, passes=2):
    """Greedy: every `todo` cell of win[1:-1, 1:-1] (all solid) takes the
    solid material with the lowest unary + pair energy against its current
    neighbours.  Solidity is unchanged, so support is too."""
    _, mats = _materials(ts)
    Eh, Ev, logz = ts.np_tables["Eh"], ts.np_tables["Ev"], ts.np_tables["logz"]
    for _ in range(passes):
        for y, x in zip(*np.nonzero(todo)):
            Y, X = y + 1, x + 1
            e = (-logz[mats] + Ev[win[Y - 1, X], mats] + Ev[mats, win[Y + 1, X]]
                 + Eh[win[Y, X - 1], mats] + Eh[mats, win[Y, X + 1]])
            win[Y, X] = mats[int(np.argmin(e))]


def _bad_turf(ts, win):
    """Solid cells of win[1:-1, 1:-1] whose pair with the solid above is a hard violation (turf under matter)."""
    Ev = ts.np_tables["Ev"]
    s = ts.solid[win]
    hot = Ev[win[:-2, 1:-1], win[1:-1, 1:-1]] > 1.9
    return hot & s[1:-1, 1:-1] & s[:-2, 1:-1]


def _apply(ts, win, s_new, editable):
    """Write the solid mask s_new into win[1:-1, 1:-1]: new solids get stone
    then a refit, cleared cells air; turf left under matter is refitted."""
    air, mats = _materials(ts)
    blk = win[1:-1, 1:-1]
    s_old = ts.solid[blk]
    filled, cleared = s_new & ~s_old, s_old & ~s_new
    blk[cleared] = air
    blk[filled] = mats[0]
    _refit(ts, win, (filled | _bad_turf(ts, win)) & editable)


def _fulfil_mask(s, exr, bh_col, tf_col):
    """The fill / clear projection of one block's solid mask s (k, k), in
    place.  exr (k + 1,): rows of the block and the row below it that are in
    the band; bh_col / tf_col (k,): the promise bit of each column's interval."""
    k = s.shape[0]
    bottom_needs = not exr[k - 1] and not exr[k]
    for x in np.flatnonzero(tf_col):                       # 1. full top intervals: solid columns
        for r in range(k):
            if exr[r]:
                break
            s[r, x] = True
    if bottom_needs:                                       # 2. no support below: clear, cascade up
        for x in np.flatnonzero(~bh_col & s[k - 1]):
            r = k - 1
            while r >= 0 and s[r, x]:
                s[r, x] = False
                r -= 1
    for r in range(k - 2, -1, -1):                         # 3. bottom-up: fill below or clear above
        if exr[r] or exr[r + 1]:
            continue
        for x in np.flatnonzero(s[r] & ~s[r + 1]):
            rr, fill = r + 1, []
            while rr < k and not s[rr, x] and not exr[rr]:
                fill.append(rr)
                rr += 1
            ok = rr < k or not bottom_needs or bool(bh_col[x])
            run = 0
            while r - run >= 0 and s[r - run, x]:
                run += 1
            if ok and len(fill) <= run:
                s[fill, x] = True
            else:
                s[r - run + 1:r + 1, x] = False
    return s


def fix(ts, tiles, ground, keep=None):
    """Global projection onto the rule (torus, ground > 0): bottom-up, each
    unsupported solid either gets the air below it filled down to the next
    solid or the band, or is cleared with the solid run above it, whichever
    edits fewer cells and touches no `keep` cell (clear if neither is free).
    Filled cells and turf left under matter take the best-fitting solid."""
    assert ground > 0, "the global projection needs a ground band"
    tiles = np.array(tiles, np.int32)
    n = tiles.shape[0]
    keep = np.zeros(tiles.shape, bool) if keep is None else keep
    s = ts.solid[tiles].copy()
    ex = exempt_rows(n, ground)
    for r in range(n - ground - 2, -1, -1):                # the row above the band rests on it
        for x in np.flatnonzero(s[r] & ~s[r + 1]):
            rr, fill = r + 1, []
            while not s[rr, x] and not ex[rr]:
                fill.append(rr)
                rr += 1
            run = 0
            while r - run >= 0 and s[r - run, x]:
                run += 1
            cf = len(fill) if not keep[fill, x].any() else np.inf
            cc = run if not keep[r - run + 1:r + 1, x].any() else np.inf
            if cf <= cc and cf < np.inf:
                s[fill, x] = True
            else:
                s[r - run + 1:r + 1, x] = False
    win = np.pad(tiles, 1, mode="wrap")
    editable = np.ones(tiles.shape, bool)
    editable[ex] = False
    _apply(ts, win, s, editable)
    return win[1:-1, 1:-1].copy()


# ------------------------------------------------------------------ variable
class SupportPromise(PromiseVar):
    name = "support"
    salt = 7
    V = 16

    def __init__(self, K=16, h_top=64, tables=None, T=1.0, sweeps=3, top_sweeps=20, ground=8, I=I_DEFAULT):
        assert I % 2 == 0 and K % I == 0
        assert ground < K, "the band must lie inside the last row of K-blocks (a block's top row is never band)"
        self.I, self.ground = I, ground
        self.V = nvals(I)
        super().__init__(K, h_top, tables, T, sweeps)
        self.top_sweeps = top_sweeps
        self._n = None

    # bottom-up, bound to the band and the interval count
    def summarize(self, ts, tiles, y0, x0, k):
        return summarize(ts, tiles, y0, x0, k, self.ground)

    merge = staticmethod(merge)

    def abstract(self, state):
        return abstract(state, self.I)

    def satisfied(self, value, state):
        return satisfied(value, state, self.I)

    def violations(self, value, state):
        return violations(value, state, self.I)

    def merge_promises(self, *c):
        return merge_promises(*c, I=self.I)

    def refine(self, parent, halo=None, gnd=None):
        return refine(parent, halo, gnd, self.I)

    def gnd_rows(self, level):
        return ground_rows(level.shape[0] * level.h, level.h, self.ground)

    # ---- top-down
    def top(self, level, ctx):
        """Single-site Gibbs over block values at the top level, restricted
        to values valid for the block's row and consistent with the current
        blocks above and below (torus; a block that is its own vertical
        neighbour must be consistent with itself).  Starts from all zeros
        (consistent), so every step keeps the grid consistent.  No global
        check: any consistent grid can be fulfilled."""
        G0, G1 = level.shape
        u, pair, _ = self.tables.level(level.h)
        BH, TF = bits(self.I)
        gnd = self.gnd_rows(level)
        grid = np.zeros((G0, G1), np.int64)
        cand = np.arange(self.V)
        lvl = hier.level_index(level.h)
        for s_ in range(self.top_sweeps):
            for y, x in itertools.product(range(G0), range(G1)):
                ya, yb, xl, xr = (y - 1) % G0, (y + 1) % G0, (x - 1) % G1, (x + 1) % G1
                nb = lambda yy, xx: cand if (yy, xx) == (y, x) else grid[yy, xx]
                m = allowed(cand, bool(gnd[y]), self.I)
                m &= ~(BH[nb(ya, x)] & ~TF[cand]).any(-1)
                m &= ~(BH[cand] & ~TF[nb(yb, x)]).any(-1)
                e = (u[cand] + pair[0][nb(y, xl), cand] + pair[0][cand, nb(y, xr)]
                     + pair[1][nb(ya, x), cand] + pair[1][cand, nb(yb, x)])
                g = hier.gumbel(hier.noise(ctx.seed, lvl, s_, TOP_COLOUR, y, x, cand + (self.salt << 20)))
                e = np.where(m, e / max(self._T, 1e-9) - (g if self._T > 0 else 0), np.inf)
                grid[y, x] = int(np.argmin(e))
        return grid

    def init_children(self, parent, level, ctx):
        """The first enumerated refinement of every parent (a consistent
        parent level makes it consistent with the neighbours' children)."""
        gnd = self.gnd_rows(level)
        g0, g1 = level.shape
        out = np.zeros((g0, g1), np.int64)
        for py, px in np.ndindex(*parent.shape):
            c = self.refine(int(parent[py, px]), gnd=(gnd[2 * py], gnd[2 * py + 1]))[0]
            for (dy, dx), v in zip(KIDS, c):
                out[2 * py + dy, 2 * px + dx] = v
        return out

    def candidates(self, level, colour, ctx):
        self._gnd = self.gnd_rows(level)
        return super().candidates(level, colour, ctx)

    def proposals(self, grid, parents, py, px):
        """The parent's own four children, every consistent refinement (the
        halo is checked but never binds when the parent level is consistent)."""
        y0, x0 = 2 * py, 2 * px
        cells = [(y0 + dy, x0 + dx) for dy, dx in KIDS]
        vals = self.refine(int(parents[py, px]), halo(grid, py, px, ground=False),
                           gnd=(self._gnd[y0], self._gnd[y0 + 1]))
        return cells, np.asarray(vals, np.int64).reshape(-1, 4)

    # ---- coupling table
    def window_costs(self, ts, E_, h):
        """(m*m, V) proxy cost of every promise for the h x h exemplar window
        centred on each coordinate (torus exemplar, no band)."""
        s = ts.solid[np.asarray(E_)].astype(np.int64)
        m = s.shape[0]
        I, w = self.I, h // self.I
        vm = (s & (1 - np.roll(s, -1, 0))).astype(np.int64)

        def run(a, L, axis):
            c = np.zeros_like(a)
            for i in range(L):
                c += np.roll(a, -i, axis)
            return c
        box = run(run(vm, h - 1, 0), h, 1)                 # rows ty .. ty+h-2, columns tx .. tx+h-1
        rw = run(s, w, 1)                                  # w consecutive cells of a row
        c0 = (np.arange(m) - h // 2) % m                   # window top-left for a centre index
        ty, tx = np.meshgrid(c0, c0, indexing="ij")
        ty, tx = ty.ravel(), tx.ravel()
        by = (ty + h - 1) % m
        need = np.stack([rw[by, (tx + i * w) % m] > 0 for i in range(I)], 1)
        full = np.stack([rw[ty, (tx + i * w) % m] == w for i in range(I)], 1)
        viol = box[ty, tx]
        BH, TF = bits(I)
        bad = ((need[:, None] & ~BH[None]).sum(2) + (TF[None] & ~full[:, None]).sum(2) + (viol > 0)[:, None])
        return np.where(bad == 0, 0, 1 + bad).astype(np.float32)

    # ---- fulfil
    def fulfil(self, ts, tiles, y0, x0, k, value, halo=None, seed=0, budget=None, protect=None):
        """Make the k x k block at (y0, x0) keep `value` (module docstring).
        Edits only inside the block, never in the band.  -> (tiles, edits, [])."""
        tiles = np.array(tiles, np.int32)
        if satisfied(value, summarize(ts, tiles, y0, x0, k, self.ground), self.I):
            return tiles, 0, []
        n0, n1 = tiles.shape
        win = tiles[np.ix_(np.arange(y0 - 1, y0 + k + 1) % n0, np.arange(x0 - 1, x0 + k + 1) % n1)].copy()
        old = win[1:-1, 1:-1].copy()
        exr = exempt_rows(n0, self.ground)[np.arange(y0, y0 + k + 1) % n0]
        bh, tf = decode(value, self.I)
        col = np.arange(k) // (k // self.I)
        s = _fulfil_mask(ts.solid[old].copy(), exr, bh[col], tf[col])
        editable = np.broadcast_to(~exr[:k, None], (k, k))
        _apply(ts, win, s, editable)
        tiles[y0:y0 + k, x0:x0 + k] = win[1:-1, 1:-1]
        return tiles, int((win[1:-1, 1:-1] != old).sum()), []


# ------------------------------------------------------------------- corpus
def project(ts, tiles, K, ground, I=I_DEFAULT):
    """A K-level promise grid, consistent and valid at every level up to the
    whole torus, that follows the grid's own abstractions: top down, every
    parent takes the refinement nearest (bit Hamming distance) to its
    children's raw abstractions."""
    tiles = np.asarray(tiles)
    n = tiles.shape[0]
    BH, TF = bits(I)
    g = n // K
    St = [[summarize(ts, tiles, y * K, x * K, K, ground) for x in range(g)] for y in range(g)]
    raw = {}
    h = K
    while True:
        raw[h] = np.array([[abstract(s, I) for s in row] for row in St], np.int64)
        if g == 1:
            break
        g //= 2
        St = [[merge({(dy, dx): St[2 * y + dy][2 * x + dx] for dy in (0, 1) for dx in (0, 1)})
               for x in range(g)] for y in range(g)]
        h *= 2
    bh, tf = BH[raw[n][0, 0]].copy(), TF[raw[n][0, 0]].copy()
    if ground_block(n, 0, n, ground):
        bh[:] = False
    tf &= bh | ground_block(n, 0, n, ground)
    cur = encode(bh, tf, I).reshape(1, 1)
    while h > K:
        h //= 2
        G = n // h
        gnd = ground_rows(n, h, ground)
        nxt = np.zeros((G, G), np.int64)
        R = raw[h]
        for py, px in np.ndindex(*cur.shape):
            want = [R[2 * py + dy, 2 * px + dx] for dy, dx in KIDS]
            cands = refine(int(cur[py, px]), gnd=(gnd[2 * py], gnd[2 * py + 1]), I=I)
            d = [sum(int((BH[c] != BH[w_]).sum() + (TF[c] != TF[w_]).sum()) for c, w_ in zip(cs, want)) for cs in cands]
            best = cands[int(np.argmin(d))]
            for (dy, dx), v in zip(KIDS, best):
                nxt[2 * py + dy, 2 * px + dx] = v
        cur = nxt
    return cur


def make_valid(ts, tiles, ground, K=16, seed=0, I=I_DEFAULT):
    """-> (tiles, promise grid): the global projection (`fix`), then every
    K-block fulfilled to the top-down projection of its own abstractions."""
    tiles = fix(ts, tiles, ground)
    O = project(ts, tiles, K, ground, I)
    v = SupportPromise(K, ground=ground, I=I)
    for by, bx in np.ndindex(*O.shape):
        tiles = v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]), seed=seed)[0]
    return tiles, O
