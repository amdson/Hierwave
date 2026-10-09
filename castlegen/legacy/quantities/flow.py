"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Flow promise (notes/history/promises.md section 4.2): oriented water, torus maps.

Water is every signature with a flow attribute (tileset.py): flowing tiles
(rivers, springs, bridges) point to one side, lakes are still.  The flow
rule, the guarantee:
  fed      every flowing tile that is not a spring is pointed into by a
           neighbouring water tile or touches a lake;
  drains   every flowing tile points into a water tile.
Lakes are sources and sinks and exempt; loops satisfy the rule.
`flow_violations` checks it on a whole grid.

A **crossing** of a block side is a water cell of the block pointing across
it (out) or a water cell across it pointing into the block (in).

State (exact, `State`): the block's crossings (absolute cell, side, sign)
and the number of cells of the block that break the rule, judged with the
true (torus) neighbours.  `merge` keeps the children's crossings whose
partner cell lies outside the parent and adds the counts; exact.

Promise: the net outward flux per side, f_d = #out - #in in {-2..2}, encoded
v = sum_d (f_d + 2) * 5**d (625 values).  The net internal source sum_d f_d
is implied (a block balances by definition; lakes absorb or emit any
amount, springs emit).  A state satisfies v when no cell of the block
breaks the rule and side d carries exactly |f_d| crossings, all in the
direction of sign(f_d) (crossing convention: never both directions on one
side).  abstract(state) = the clipped net flux per side (a projection).

Promise merge: internal seams cancel (child A's E flux = - child B's W
flux), a parent side's flux is the sum of its two children's, which must
have the same sign and |sum| <= 2 (so the parent side also carries crossings
in one direction only).  Soundness: if the four children satisfy their
promises and merge consistently, the parent's crossings are the union of
the children's outer crossings (same count and sign per side) and no cell
breaks the rule, so the parent satisfies the merged promise; the rule is
per cell, so every K-block satisfying its promise means no cell anywhere
breaks it.  No global check is needed at the top.

Fulfil (per K-block, edits inside the block only): |f| crossing cells per
side at boundary offsets hashed from the seam, skipping cells within one
cell of a structure on either side of the seam (the same choice from both
blocks, since fulfil never moves structures).  Out-crossings point across,
in-crossing cells receive the neighbour's flow.  Each out-crossing is fed by
a shortest-path carve of oriented river from an unused in-crossing, a lake
or a spring (created if nothing else is reachable); each remaining
in-crossing is drained into a routed cell or a lake (created if needed).
Routes avoid the block rim except at crossing cells, so the neighbour never
sees our water except at crossings, keep a one-cell margin around
structures, and cross road tiles as bridges.  Every other flowing cell is
filled with land (bridges become roads), then the land around the edits is
re-fitted to its neighbours (shores next to water).
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from castlegen.legacy import hier
from castlegen.legacy.promise import PromiseVar

N, E, S, W = 0, 1, 2, 3
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPP = (2, 3, 0, 1)
FMAX = 2
NF = 2 * FMAX + 1                       # flux values per side
V = NF ** 4
SEAM_STEP = 0xFFFFFFE0                  # noise `step` for flow crossing offsets (connectivity uses ...F0)
TOP_COLOUR = 6
EXT = {N: ((0, 0), (0, 1)), E: ((0, 1), (1, 1)), S: ((1, 0), (1, 1)), W: ((0, 0), (1, 0))}
KIDS = ((0, 0), (0, 1), (1, 0), (1, 1))

FLUX = np.array([[(v // NF ** d) % NF - FMAX for d in range(4)] for v in range(V)], np.int64)   # (V, 4)


def encode(f):
    """Side fluxes (..., 4) -> value."""
    f = np.asarray(f, np.int64)
    return ((f[..., 0] + FMAX) + NF * (f[..., 1] + FMAX) + NF ** 2 * (f[..., 2] + FMAX)
            + NF ** 3 * (f[..., 3] + FMAX))


def flux(v):
    return tuple(int(x) for x in FLUX[int(v)])


# -------------------------------------------------------------------- the rule
def _cell_status(ts, tiles, torus=True):
    """-> (dead, unfed) bool grids.  Off-grid neighbours (torus=False) are dry."""
    tiles = np.asarray(tiles)
    return _status(ts.water[tiles], ts.lake[tiles], ts.source[tiles], ts.flow_dir[tiles], torus)


def _status(water, lake, src, f, torus=True):
    def nb(a, d, fill):
        dy, dx = DIRS[d]
        if torus:
            return np.roll(a, (-dy, -dx), (0, 1))
        p = np.pad(a, 1, constant_values=fill)
        return p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
    flowing = water & ~lake
    drains = np.zeros(water.shape, bool)
    fed = np.zeros(water.shape, bool)
    for d in range(4):
        drains |= (f == d) & nb(water, d, False)
        fed |= (nb(f, d, -1) == OPP[d]) | nb(lake, d, False)
    return flowing & ~drains, flowing & ~src & ~fed


def flow_violations(ts, tiles, torus=True):
    """{"dead_ends": n, "unfed": n} over the grid (torus by default)."""
    dead, unfed = _cell_status(ts, tiles, torus)
    return {"dead_ends": int(dead.sum()), "unfed": int(unfed.sum())}


def has_flow(ts):
    return ts.water is not None and bool(ts.water.any())


# ---------------------------------------------------------------------- state
class State:
    __slots__ = ("y0", "x0", "k", "cross", "viol")

    def __init__(self, y0, x0, k, cross, viol):
        self.y0, self.x0, self.k = y0, x0, k
        self.cross = cross          # sorted [(y, x, d, sign)]: block cell, side, +1 out / -1 in
        self.viol = viol            # cells of the block breaking the rule

    def key(self):
        return (self.y0, self.x0, self.k, tuple(self.cross), self.viol)

    def __eq__(self, other):
        return isinstance(other, State) and self.key() == other.key()

    def __repr__(self):
        return f"State({self.y0},{self.x0},k={self.k}, cross={self.cross}, viol={self.viol})"

    def counts(self):
        """(4, 2) crossings per side: [:, 0] out, [:, 1] in."""
        c = np.zeros((4, 2), int)
        for _, _, d, s in self.cross:
            c[d, 0 if s > 0 else 1] += 1
        return c


def summarize(ts, tiles, y0, x0, k):
    """Exact state of the k x k block at (y0, x0); neighbours read on the torus."""
    tiles = np.asarray(tiles)
    n0, n1 = tiles.shape
    ys, xs = np.arange(y0 - 1, y0 + k + 1) % n0, np.arange(x0 - 1, x0 + k + 1) % n1
    w = tiles[np.ix_(ys, xs)]
    dead, unfed = _cell_status(ts, w, torus=False)
    viol = int((dead | unfed)[1:-1, 1:-1].sum())
    f = ts.flow_dir[w]
    cross = []
    for d in range(4):
        dy, dx = DIRS[d]
        if d == N:
            cells = [(0, j) for j in range(k)]
        elif d == S:
            cells = [(k - 1, j) for j in range(k)]
        elif d == W:
            cells = [(i, 0) for i in range(k)]
        else:
            cells = [(i, k - 1) for i in range(k)]
        for i, j in cells:
            if f[i + 1, j + 1] == d:
                cross.append((y0 + i, x0 + j, d, 1))
            if f[i + 1 + dy, j + 1 + dx] == OPP[d]:
                cross.append((y0 + i, x0 + j, d, -1))
    return State(y0, x0, k, sorted(cross), viol)


def merge(children):
    c0 = children[(0, 0)]
    k, Y0, X0 = c0.k, c0.y0, c0.x0
    side = 2 * k
    cross, viol = [], 0
    for key in KIDS:
        s = children[key]
        viol += s.viol
        for y, x, d, sg in s.cross:
            ty, tx = y + DIRS[d][0], x + DIRS[d][1]
            if not (Y0 <= ty < Y0 + side and X0 <= tx < X0 + side):
                cross.append((y, x, d, sg))
    return State(Y0, X0, side, sorted(cross), viol)


def _net(state):
    c = state.counts()
    return c[:, 0] - c[:, 1], (c[:, 0] > 0) & (c[:, 1] > 0)


def abstract(state):
    net, _ = _net(state)
    return int(encode(np.clip(net, -FMAX, FMAX)))


def satisfied(value, state):
    net, mixed = _net(state)
    return state.viol == 0 and not mixed.any() and bool(np.array_equal(net, FLUX[int(value)]))


def violations(value, state):
    """Proxy fulfil cost: 0 if satisfied, else 1 + crossing count mismatches + broken cells present."""
    c = state.counts()
    return float(_costs(c[:, 0], c[:, 1], state.viol)[int(value)])


def _costs(out, inn, viol):
    """(V,) cost of every promise for a block with per-side out / in counts (4,)."""
    want_out, want_in = np.maximum(FLUX, 0), np.maximum(-FLUX, 0)
    miss = np.abs(want_out - np.asarray(out)[None]).sum(1) + np.abs(want_in - np.asarray(inn)[None]).sum(1)
    bad = miss + (viol > 0)
    return np.where(bad == 0, 0, 1 + bad).astype(np.float32)


# ------------------------------------------------------------- promise merge
def _merge_arrays(c00, c01, c10, c11):
    """Vectorised merge_promises on value arrays -> (parent, ok)."""
    f = [FLUX[np.asarray(c)] for c in (c00, c01, c10, c11)]
    agree = ((f[0][..., E] == -f[1][..., W]) & (f[2][..., E] == -f[3][..., W])
             & (f[0][..., S] == -f[2][..., N]) & (f[1][..., S] == -f[3][..., N]))
    ok = agree
    P = []
    for d in range(4):
        a, b = (f[2 * i + j][..., d] for i, j in EXT[d])
        s = a + b
        ok = ok & (a * b >= 0) & (np.abs(s) <= FMAX)
        P.append(np.clip(s, -FMAX, FMAX))
    return encode(np.stack(P, -1)), ok


def merge_promises(c00, c01, c10, c11):
    P, ok = _merge_arrays(c00, c01, c10, c11)
    return int(P), bool(ok)


def consistent(grid, parents=None):
    g = np.asarray(grid)
    f = FLUX[g]
    if not (np.array_equal(f[..., E], -FLUX[np.roll(g, -1, 1)][..., W])
            and np.array_equal(f[..., S], -FLUX[np.roll(g, -1, 0)][..., N])):
        return False
    if parents is not None:
        P, ok = _merge_arrays(g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2])
        return bool(ok.all() and np.array_equal(P, parents))
    return True


def splits(F):
    """Same-sign splits (a, b) of a parent side flux F over its two children."""
    s = 1 if F >= 0 else -1
    return [(s * a, s * (abs(F) - a)) for a in range(abs(F) + 1)] if F else [(0, 0)]


def refine(parent, halo=None):
    """Consistent refinements of `parent`: [(c00, c01, c10, c11)].  halo:
    {(dy, dx, d): flux} fixes external half-seams."""
    halo = halo or {}
    Fp = flux(parent)
    opts = {}
    for d in range(4):
        opts[d] = [s for s in splits(Fp[d]) if all(halo.get(EXT[d][i] + (d,), s[i]) == s[i] for i in (0, 1))]
    rng = range(-FMAX, FMAX + 1)
    out = []
    for sn, se, ss, sw in itertools.product(opts[N], opts[E], opts[S], opts[W]):
        for h0, h1, v0, v1 in itertools.product(rng, repeat=4):
            c = {key: [0, 0, 0, 0] for key in KIDS}
            for sd, d in ((sn, N), (se, E), (ss, S), (sw, W)):
                for i, key in enumerate(EXT[d]):
                    c[key][d] = sd[i]
            c[0, 0][E], c[0, 1][W] = h0, -h0
            c[1, 0][E], c[1, 1][W] = h1, -h1
            c[0, 0][S], c[1, 0][N] = v0, -v0
            c[0, 1][S], c[1, 1][N] = v1, -v1
            out.append(tuple(int(encode(c[key])) for key in KIDS))
    return out


def n_refine(parent, halo=None):
    """len(refine(parent, halo)) without enumerating."""
    halo = halo or {}
    Fp = flux(parent)
    n = NF ** 4
    for d in range(4):
        n *= len([s for s in splits(Fp[d]) if all(halo.get(EXT[d][i] + (d,), s[i]) == s[i] for i in (0, 1))])
    return n


def brute_refine(parent, halo=None):
    """Reference for tests: filter child tuples (every child value compatible
    with the halo, joined on the internal seams) by merge_promises."""
    halo = halo or {}
    allv = np.arange(V)
    cand = {}
    for key in KIDS:
        m = np.ones(V, bool)
        for (dy, dx, d), b in halo.items():
            if (dy, dx) == key:
                m &= FLUX[:, d] == b
        cand[key] = allv[m]
    out = []
    for a in cand[0, 0]:
        fa = FLUX[a]
        B = cand[0, 1][FLUX[cand[0, 1], W] == -fa[E]]
        Cc = cand[1, 0][FLUX[cand[1, 0], N] == -fa[S]]
        for b in B:
            for c in Cc:
                D = cand[1, 1][(FLUX[cand[1, 1], W] == -FLUX[c, E]) & (FLUX[cand[1, 1], N] == -FLUX[b, S])]
                if not len(D):
                    continue
                P, ok = _merge_arrays(np.full(len(D), a), np.full(len(D), b), np.full(len(D), c), D)
                for dd in D[ok & (P == parent)]:
                    out.append((int(a), int(b), int(c), int(dd)))
    return out


def halo(grid, py, px):
    """{(dy, dx, d): flux} of the external half-seams of parent (py, px),
    read from the neighbouring children in `grid` (torus)."""
    g0, g1 = grid.shape
    out = {}
    for dy, dx in KIDS:
        y, x = 2 * py + dy, 2 * px + dx
        for d, (sy, sx) in enumerate(DIRS):
            if (d == N and dy == 0) or (d == S and dy == 1) or (d == W and dx == 0) or (d == E and dx == 1):
                out[dy, dx, d] = -int(FLUX[grid[(y + sy) % g0, (x + sx) % g1], OPP[d]])
    return out


# ------------------------------------------------------------ tile lookups
_TABLES = {}


def tables(ts):
    """Signature lookups for fulfil and bridges (cached per tile set)."""
    key = id(ts)
    if key in _TABLES:
        return _TABLES[key]
    door = ts.sockets.index("door") if "door" in ts.sockets else -1
    S_ = ts.n_sig
    dm = np.zeros(S_, np.int64)
    for d in range(4):
        dm |= (ts.sig_sockets[:, d] == door).astype(np.int64) << d
    node = np.array(ts.np_tables["is_room"], bool)
    skinds = {int(k) for _, g in ts.structures for k in g.ravel()}
    struct = np.isin(ts.sig_kind, list(skinds))
    bridge_by = -np.ones((16, 4), np.int64)
    dry_by = -np.ones(16, np.int64)
    for s_ in range(S_):
        f = ts.flow_dir[s_]
        if node[s_] and f >= 0 and not ts.source[s_] and bridge_by[dm[s_], f] < 0:
            bridge_by[dm[s_], f] = s_
    for s_ in range(S_):                                  # dry node with the same doors, roads first
        if node[s_] and not ts.water[s_] and not struct[s_] and dm[s_] and \
                (dry_by[dm[s_]] < 0 or ("road" in ts.kinds[ts.sig_kind[s_]].tags
                                        and "road" not in ts.kinds[ts.sig_kind[dry_by[dm[s_]]]].tags)):
            dry_by[dm[s_]] = s_

    def first(m):
        hits = np.flatnonzero(m)
        return int(hits[0]) if len(hits) else -1
    plain = ~ts.water & (dm == 0) & ~struct
    plain[[ts.WALL, ts.GATE]] = False
    t = dict(doormask=dm, node=node, struct=struct, bridge_by=bridge_by, dry_by=dry_by,
             over=np.where(dm[:, None] > 0, bridge_by[dm], -1),
             river=[first(~node & (ts.flow_dir == d) & ~ts.source & (dm == 0)) for d in range(4)],
             spring=[first(~node & (ts.flow_dir == d) & ts.source) for d in range(4)],
             lake=first(ts.lake & ~node & (np.arange(S_) != ts.WALL)),
             plain=np.flatnonzero(plain),
             flowing=ts.water & ~ts.lake)
    _TABLES[key] = t
    return t


def _dilate8(m):
    out = m.copy()
    out[1:] |= m[:-1]; out[:-1] |= m[1:]
    o2 = out.copy()
    o2[:, 1:] |= out[:, :-1]; o2[:, :-1] |= out[:, 1:]
    return o2


def block_ok(ts, tiles, y0, x0, k, loc):
    """The block-local contract, judged with a virtual ring: dry everywhere
    except the declared crossings (an in-crossing's partner points in, an
    out-crossing's partner is water that feeds nothing).  loc: [(y, x, d,
    sign)] block-local.  True when no cell breaks the rule, the block's cells
    point across a side exactly at its out-crossings, and every in-crossing
    cell is water.  Compliant neighbours only add feeding, so this implies
    the true state satisfies the promise."""
    blk = np.asarray(tiles)[y0:y0 + k, x0:x0 + k]
    f = -np.ones((k + 2, k + 2), np.int64)
    water = np.zeros((k + 2, k + 2), bool)
    lake = np.zeros((k + 2, k + 2), bool)
    src = np.zeros((k + 2, k + 2), bool)
    f[1:-1, 1:-1], water[1:-1, 1:-1] = ts.flow_dir[blk], ts.water[blk]
    lake[1:-1, 1:-1], src[1:-1, 1:-1] = ts.lake[blk], ts.source[blk]
    outs = set()
    for y, x, d, sg in loc:
        ry, rx = y + 1 + DIRS[d][0], x + 1 + DIRS[d][1]
        water[ry, rx] = True
        if sg < 0:
            f[ry, rx] = OPP[d]
            if not ts.water[blk[y, x]]:
                return False
        else:
            outs.add((y, x, d))
    dead, unfed = _status(water, lake, src, f, torus=False)
    if (dead | unfed)[1:-1, 1:-1].any():
        return False
    fb = f[1:-1, 1:-1]
    rim = {N: [(0, j) for j in range(k)], S: [(k - 1, j) for j in range(k)],
           W: [(i, 0) for i in range(k)], E: [(i, k - 1) for i in range(k)]}
    got = {(y, x, d) for d in range(4) for y, x in rim[d] if fb[y, x] == d}
    return got == outs


def prune(ts, tiles, keep=None, torus=True, max_rounds=1000):
    """Dry every flowing cell that breaks the rule (bridges become the dry
    node with the same doors, other cells the most common plain land tile),
    repeatedly, until none is left outside `keep`.  Rivers lose their stubs;
    loops and fed rivers into lakes stay."""
    t = tables(ts)
    tiles = np.array(tiles, np.int32)
    keep = np.zeros(tiles.shape, bool) if keep is None else keep
    plain = np.isin(tiles, t["plain"])
    land = int(np.bincount(tiles[plain]).argmax()) if plain.any() else int(t["plain"][0])
    for _ in range(max_rounds):
        dead, unfed = _cell_status(ts, tiles, torus)
        bad = (dead | unfed) & ~keep
        if not bad.any():
            break
        m = t["doormask"][tiles]
        tiles = np.where(bad, np.where(m > 0, np.maximum(t["dry_by"][m], 0), land), tiles).astype(np.int32)
    return tiles


# ------------------------------------------------------------------ variable

class FlowPromise(PromiseVar):
    name = "flow"
    salt = 5
    V = V

    def __init__(self, K=16, h_top=64, tables=None, T=1.0, sweeps=3, top_sweeps=20):
        super().__init__(K, h_top, tables, T, sweeps)
        self.top_sweeps = top_sweeps

    summarize = staticmethod(summarize)
    merge = staticmethod(merge)
    abstract = staticmethod(abstract)
    satisfied = staticmethod(satisfied)
    violations = staticmethod(violations)
    merge_promises = staticmethod(merge_promises)
    consistent = staticmethod(consistent)
    refine = staticmethod(refine)

    # ---- top-down
    def top(self, level, ctx):
        """Gibbs over the seam fluxes of the top grid (a block's value is read
        off its four seams, so pair consistency holds by construction, also
        when a block is its own neighbour).  No global check: any consistent
        flux grid can be fulfilled."""
        G0, G1 = level.shape
        u, pair, _ = self.tables.level(level.h)
        seams = np.zeros((2, G0, G1), np.int64)          # [0]: flux out of E of (y, x); [1]: out of S

        def values(sm):
            f = np.stack([-np.roll(sm[1], 1, 0), sm[0], sm[1], -np.roll(sm[0], 1, 1)], -1)
            return encode(f)

        def energy(sm):
            v = values(sm)
            return float(u[v].sum() + pair[0][v, np.roll(v, -1, 1)].sum() + pair[1][v, np.roll(v, -1, 0)].sum())
        lvl = hier.level_index(level.h)
        opts = np.arange(-FMAX, FMAX + 1)
        for s_ in range(self.top_sweeps):
            for o, y, x in itertools.product((0, 1), range(G0), range(G1)):
                es = []
                for fv in opts:
                    seams[o, y, x] = fv
                    es.append(energy(seams))
                g = hier.gumbel(hier.noise(ctx.seed, lvl, s_, TOP_COLOUR, y, x,
                                           np.arange(NF) + NF * o + (self.salt << 20)))
                e = np.array(es) / max(self._T, 1e-9) - (g if self._T > 0 else 0)
                seams[o, y, x] = opts[int(np.argmin(e))]
        return values(seams).astype(np.int64)

    def init_children(self, parent, level, ctx):
        """External half-seams split the parent side's flux (first child
        takes the odd unit; the split is odd in F, so both sides of a parent
        seam agree), internal seams carry nothing."""
        g0, g1 = level.shape
        ys, xs = np.arange(g0)[:, None], np.arange(g1)[None, :]
        Fp = FLUX[parent[ys // 2, xs // 2]]                           # (g0, g1, 4)
        dy, dx = ys % 2 + 0 * xs, xs % 2 + 0 * ys
        first = {N: dx == 0, S: dx == 0, W: dy == 0, E: dy == 0}
        ext = {N: dy == 0, S: dy == 1, W: dx == 0, E: dx == 1}
        f = np.zeros((g0, g1, 4), np.int64)
        for d in range(4):
            F = Fp[..., d]
            a = np.sign(F) * ((np.abs(F) + 1) // 2)
            b = np.sign(F) * (np.abs(F) // 2)
            f[..., d] = np.where(ext[d], np.where(first[d], a, b), 0)
        return encode(f).astype(np.int64)

    def proposals(self, grid, parents, py, px):
        """Own children's four internal seams x the splits of the parent's E
        and S sides (written into the neighbours' facing children).  Every
        proposal keeps all parents consistent by construction."""
        g0, g1 = grid.shape
        Fp = FLUX[int(parents[py, px])]
        y0, x0 = 2 * py, 2 * px
        own = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
        efc = [(y0, (x0 + 2) % g1), (y0 + 1, (x0 + 2) % g1)]
        sfc = [((y0 + 2) % g0, x0), ((y0 + 2) % g0, x0 + 1)]
        cells = own + efc + sfc
        rows = {}
        for c in cells:
            rows.setdefault(c, len(rows))
        sE, sS = np.array(splits(Fp[E])), np.array(splits(Fp[S]))
        r = np.arange(-FMAX, FMAX + 1)
        ie, iS, h0, h1, v0, v1 = (a.ravel() for a in np.meshgrid(np.arange(len(sE)), np.arange(len(sS)), r, r, r, r,
                                                                 indexing="ij"))
        nC = len(ie)
        F = np.broadcast_to(FLUX[[int(grid[c]) for c in rows]], (nC, len(rows), 4)).copy()

        def put(c, d, val):
            F[:, rows[c], d] = val
        put(own[0], E, h0); put(own[1], W, -h0)
        put(own[2], E, h1); put(own[3], W, -h1)
        put(own[0], S, v0); put(own[2], N, -v0)
        put(own[1], S, v1); put(own[3], N, -v1)
        for i in (0, 1):
            put((y0 + i, x0 + 1), E, sE[ie, i]); put(efc[i], W, -sE[ie, i])
            put((y0 + 1, x0 + i), S, sS[iS, i]); put(sfc[i], N, -sS[iS, i])
        vals = encode(F)[:, [rows[c] for c in cells]]
        _, first = np.unique(vals, axis=0, return_index=True)
        return cells, vals[np.sort(first)]

    # ---- coupling table
    def window_costs(self, ts, E_, h):
        """(m*m, V) proxy cost of every promise for the h x h exemplar window
        centred on each coordinate (torus exemplar)."""
        E_ = np.asarray(E_)
        m = E_.shape[0]
        dead, unfed = _cell_status(ts, E_, torus=True)
        bad = (dead | unfed).astype(np.int64)
        f = ts.flow_dir[E_]
        outc = [(f == d).astype(np.int64) for d in range(4)]
        inc = [(np.roll(f, (-DIRS[d][0], -DIRS[d][1]), (0, 1)) == OPP[d]).astype(np.int64) for d in range(4)]

        def run(a, axis):                     # sum of h consecutive cells starting at each index (torus)
            c = np.zeros_like(a)
            for i in range(h):
                c += np.roll(a, -i, axis)
            return c
        y0 = (np.arange(m) - h // 2) % m                # window top-left for centre index
        box = run(run(bad, 0), 1)
        rowN = {d: run(outc[d], 1) for d in (N, S)}
        rowNi = {d: run(inc[d], 1) for d in (N, S)}
        colE = {d: run(outc[d], 0) for d in (E, W)}
        colEi = {d: run(inc[d], 0) for d in (E, W)}
        out = np.zeros((m * m, V), np.float32)
        for u in range(m * m):
            cy, cx = divmod(u, m)
            ty, tx = y0[cy], y0[cx]
            by_, rx = (ty + h - 1) % m, (tx + h - 1) % m
            o = [rowN[N][ty, tx], colE[E][ty, rx], rowN[S][by_, tx], colE[W][ty, tx]]
            i_ = [rowNi[N][ty, tx], colEi[E][ty, rx], rowNi[S][by_, tx], colEi[W][ty, tx]]
            out[u] = _costs(o, i_, box[ty, tx])
        return out

    # ---- fulfil
    def crossing_cells(self, ts, tiles, y0, x0, k, value, seed):
        """[(y, x, d, sign)] absolute: |f_d| cells on each side d at boundary
        offsets in [1, k-2] ordered by a hash of the seam, skipping offsets
        where either cell of the seam is within one cell of a structure
        (symmetric: fulfil never moves structures)."""
        n0, n1 = tiles.shape
        nb0, nb1 = n0 // k, n1 // k
        by, bx = y0 // k, x0 // k
        lvl = hier.level_index(k)
        struct = tables(ts)["struct"]

        def near(y, x):
            return bool(struct[tiles[np.ix_(np.arange(y - 1, y + 2) % n0, np.arange(x - 1, x + 2) % n1)]].any())
        F = flux(value)
        out = []
        for d in range(4):
            f = F[d]
            if not f:
                continue
            if d == N:
                key, cell = (1, by, bx), lambda o: (y0, x0 + o)
            elif d == S:
                key, cell = (1, (by + 1) % nb0, bx), lambda o: (y0 + k - 1, x0 + o)
            elif d == W:
                key, cell = (0, by, bx), lambda o: (y0 + o, x0)
            else:
                key, cell = (0, by, (bx + 1) % nb1), lambda o: (y0 + o, x0 + k - 1)
            offs = np.arange(1, k - 1)
            order = offs[np.argsort(hier.noise(seed, lvl, SEAM_STEP, key[0], key[1], key[2], offs), kind="stable")]
            free = [int(o) for o in order if not near(*cell(o))
                    and not near(cell(o)[0] + DIRS[d][0], cell(o)[1] + DIRS[d][1])]
            pick = (free + [int(o) for o in order if o not in free])[:abs(f)]
            out += [cell(o) + (d, 1 if f > 0 else -1) for o in sorted(pick)]
        return out

    def protect_mask(self, ts, tiles):
        """Cells later quantities must keep: water and structures."""
        t = tables(ts)
        return ts.water[tiles] | t["struct"][tiles]

    def fulfil(self, ts, tiles, y0, x0, k, value, halo=None, seed=0, budget=None, protect=None):
        """Make the k x k block at (y0, x0) keep `value` given that every
        neighbour does the same; edits only inside the block, never on
        structures.  -> (tiles, edits, crossing cells (absolute, [(y, x)]))."""
        tiles = np.array(tiles, np.int32)
        cross = self.crossing_cells(ts, tiles, y0, x0, k, value, seed)
        loc = [(y - y0, x - x0, d, sg) for y, x, d, sg in cross]
        cells = [(y, x) for y, x, _, _ in cross]
        if block_ok(ts, tiles, y0, x0, k, loc):
            return tiles, 0, cells
        n0, n1 = tiles.shape
        win = tiles[np.ix_(np.arange(y0 - 1, y0 + k + 1) % n0, np.arange(x0 - 1, x0 + k + 1) % n1)].copy()
        old = win[1:-1, 1:-1].copy()
        _route(ts, win, loc, seed * 7919 + y0 * 131 + x0)
        tiles[y0:y0 + k, x0:x0 + k] = win[1:-1, 1:-1]
        if not block_ok(ts, tiles, y0, x0, k, loc):                # cannot happen; keep the guarantee anyway
            win[1:-1, 1:-1] = old
            _trivial(ts, win, loc)
            tiles[y0:y0 + k, x0:x0 + k] = win[1:-1, 1:-1]
        return tiles, int((tiles[y0:y0 + k, x0:x0 + k] != old).sum()), cells


# -------------------------------------------------------------------- routing
def _trivial(ts, win, loc):
    """Last resort: in-crossing cells become lakes, out-crossing cells springs,
    every other flowing cell dry land."""
    t = tables(ts)
    blk = win[1:-1, 1:-1]
    fixed = {(y, x) for y, x, _, _ in loc}
    for y, x in zip(*np.nonzero(t["flowing"][blk])):
        if (y, x) not in fixed:
            blk[y, x] = t["dry_by"][t["doormask"][blk[y, x]]] if t["doormask"][blk[y, x]] else t["plain"][0]
    for y, x, d, sg in loc:
        blk[y, x] = t["lake"] if sg < 0 else t["spring"][d]


def _route(ts, win, loc, seed, w_water=0.3, w_bridge=2.0):
    """Carve oriented routes in the block (win[1:-1, 1:-1], edited in place;
    the ring is read-only) for the crossings `loc`, then dry every other
    flowing cell and re-fit the land around the edits."""
    t = tables(ts)
    k = win.shape[0] - 2
    blk = win[1:-1, 1:-1]
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    jitter = 1e-3 * rng.random((k, k))
    margin = _dilate8(t["struct"][win])[1:-1, 1:-1]
    interior = np.zeros((k, k), bool)
    interior[1:-1, 1:-1] = True
    lake = ts.lake[blk].copy()
    dm = t["doormask"][blk]
    base_ok = interior & ~margin & ~lake & ~t["struct"][blk]
    routable = base_ok & (dm == 0)
    doored = base_ok & (dm > 0)
    cost = np.where(t["flowing"][blk], w_water, 1.0) + jitter
    route = -np.ones((k, k), np.int64)          # flow side of every routed cell
    springs, lakes = set(), set()
    ins = [(y, x) for y, x, d, sg in loc if sg < 0]
    outs = [(y, x, d) for y, x, d, sg in loc if sg > 0]
    crossing = {(y, x) for y, x, _, _ in loc}
    for y, x, d in outs:
        route[y, x] = d
    unused = {c for c in ins if not lake[c]}
    inside = lambda y, x: 0 <= y < k and 0 <= x < k

    def free(c):
        return bool(routable[c]) and route[c] < 0 and c not in crossing and c not in lakes

    def lake_side(c):
        return next((e for e, (dy, dx) in enumerate(DIRS) if inside(c[0] + dy, c[1] + dx)
                     and lake[c[0] + dy, c[1] + dx]), None)

    def search(start, backward):
        """Dijkstra from `start`; backward walks upstream to an origin (unused
        in-crossing, spring, lake-fed cell), forward walks downstream to a
        drain (routed cell or lake).  A doored cell (road) is crossed straight
        as a bridge.  -> (chain from source to sink or None, visited cells)."""
        dist, par, seen = {start: 0.0}, {}, []
        heap = [(0.0, start)]

        def chain(end):
            cells, c = [end], end
            while c != start:
                a, via = par[c]
                cells += ([via] if via is not None else []) + [a]
                c = a
            return cells if backward else cells[::-1]
        while heap:
            du, a = heapq.heappop(heap)
            if du > dist[a]:
                continue
            if a != start:
                seen.append(a)
                if backward and (a in unused or ts.source[blk[a]] or lake_side(a) is not None):
                    return chain(a), seen
            for e, (dy, dx) in enumerate(DIRS):
                b = (a[0] + dy, a[1] + dx)
                if not inside(*b):
                    continue
                side = OPP[e] if backward else e              # flow side of the cells on this step
                via, c = None, b
                if doored[b] and t["over"][blk[b], side] >= 0:
                    via, c = b, (b[0] + dy, b[1] + dx)
                    if not inside(*c):
                        continue
                if not backward and (lake[c] or route[c] >= 0):
                    par[c] = (a, via)
                    return chain(c), seen
                if not (free(c) or (backward and c in unused)):
                    continue
                nd = du + cost[c] + (w_bridge if via else 0.0)
                if nd < dist.get(c, np.inf):
                    dist[c] = nd
                    par[c] = (a, via)
                    heapq.heappush(heap, (nd, c))
        return None, seen

    def lay(cells):
        for a, b in zip(cells, cells[1:]):
            route[a] = DIRS.index((b[0] - a[0], b[1] - a[1]))

    def pick_far(seen, start):
        """A visited free cell about k/3 steps away (a new spring or lake)."""
        cand = [c for c in seen if free(c)]
        return min(cand, key=lambda c: (abs(abs(c[0] - start[0]) + abs(c[1] - start[1]) - k // 3), c)) \
            if cand else None

    for oy, ox, d in outs:                              # every out-crossing gets its own source
        o = (oy, ox)
        if lake_side(o) is not None:
            continue
        cells, seen = search(o, backward=True)
        if cells is None:
            src = pick_far(seen, o)
            if src is None:
                springs.add(o)
                continue
            springs.add(src)                            # make it a spring and search again
            blk_src = blk[src]
            blk[src] = t["spring"][0]
            cells, _ = search(o, backward=True)
            blk[src] = blk_src
            if cells is None or cells[0] != src:
                springs.discard(src)
                springs.add(o)
                continue
        else:
            if ts.source[blk[cells[0]]]:
                springs.add(cells[0])
        unused.discard(cells[0])
        lay(cells)
    for c in sorted(unused):                            # every other in-crossing drains somewhere
        side = lake_side(c)
        if side is not None:
            route[c] = side
            continue
        cells, seen = search(c, backward=False)
        if cells is None:
            L = pick_far(seen, c)
            if L is None:
                lakes.add(c)
                continue
            lakes.add(L)
            lake[L] = True
            cells, _ = search(c, backward=False)
            if cells is None or cells[-1] != L:
                lakes.discard(L)
                lake[L] = False
                lakes.add(c)
                continue
            for dy, dx in DIRS:                         # widen the new lake where free
                b = (L[0] + dy, L[1] + dx)
                if inside(*b) and free(b) and b not in cells:
                    lakes.add(b)
                    lake[b] = True
        lay(cells)
    # write the routes, then dry every other flowing cell and re-fit the land
    for y, x in zip(*np.nonzero(route >= 0)):
        c, d = (int(y), int(x)), int(route[y, x])
        if c in springs:
            blk[c] = t["spring"][d]
        elif dm[c] and t["over"][blk[c], d] >= 0:
            blk[c] = t["over"][blk[c], d]
        else:
            blk[c] = t["river"][d]
    for c in lakes:
        blk[c] = t["lake"]
    changed = np.zeros((k, k), bool)
    for y, x in zip(*np.nonzero(t["flowing"][blk] & (route < 0))):
        if (int(y), int(x)) in lakes:
            continue
        m = t["doormask"][blk[y, x]]
        blk[y, x] = t["dry_by"][m] if m and t["dry_by"][m] >= 0 else t["plain"][0]
        changed[y, x] = t["doormask"][blk[y, x]] == 0
    _refit(ts, win, changed)


def _refit(ts, win, changed, passes=3):
    """Greedy socket fit of the plain land cells of the block that were
    dried (`changed`), touch water or touch a dried cell: each takes the
    plain land signature with the lowest unary + pair energy against its
    current neighbours."""
    t = tables(ts)
    Eh, Ev, logz = ts.np_tables["Eh"], ts.np_tables["Ev"], ts.np_tables["logz"]
    P = t["plain"]
    wet = ts.water[win]
    near = wet[:-2, 1:-1] | wet[2:, 1:-1] | wet[1:-1, :-2] | wet[1:-1, 2:]
    c = np.pad(changed, 1)
    near |= c[:-2, 1:-1] | c[2:, 1:-1] | c[1:-1, :-2] | c[1:-1, 2:]
    todo = changed | (near & np.isin(win[1:-1, 1:-1], P))
    for _ in range(passes):
        for y, x in zip(*np.nonzero(todo)):
            Y, X = y + 1, x + 1
            e = -logz[P] + Ev[win[Y - 1, X], P] + Ev[P, win[Y + 1, X]] + Eh[win[Y, X - 1], P] + Eh[P, win[Y, X + 1]]
            win[Y, X] = P[int(np.argmin(e))]


# ------------------------------------------------------------------- corpus
def project(ts, tiles, K):
    """A K-level promise grid, consistent at every level up to the whole
    torus, that follows the grid's own net seam fluxes: top-down, a seam on a
    coarser block boundary splits its flux (same sign) as close as possible
    to its halves' raw nets, a seam internal to the coarser block takes its
    raw net clipped to [-2, 2]."""
    tiles = np.asarray(tiles)
    n = tiles.shape[0]
    g = n // K
    net = np.array([[_net(summarize(ts, tiles, y * K, x * K, K))[0] for x in range(g)] for y in range(g)])
    rawE, rawS = net[..., E], net[..., S]                       # (g, g) flux out of E / S of each K-block
    levels = []
    h = K
    while h <= n:
        levels.append(h)
        h *= 2
    e = {n: np.array([[int(np.clip(rawE[:, g - 1].sum(), -FMAX, FMAX))]])}
    s = {n: np.array([[int(np.clip(rawS[g - 1, :].sum(), -FMAX, FMAX))]])}
    for h in levels[-2::-1]:
        G, r = n // h, h // K
        re = rawE.reshape(G, r, G, r)[:, :, :, r - 1].sum(1)     # raw net of the E seam of each h-block
        rs = rawS.reshape(G, r, r, G)[:, r - 1].sum(1)
        eh, sh = np.zeros((G, G), np.int64), np.zeros((G, G), np.int64)
        for Y, X in np.ndindex(G, G):
            if X % 2:                                           # half of a coarser seam
                F_ = e[2 * h][Y // 2, X // 2]
                want = np.clip([re[2 * (Y // 2), X], re[2 * (Y // 2) + 1, X]], -FMAX, FMAX)
                a, b = min(splits(F_), key=lambda ab: (abs(ab[0] - want[0]) + abs(ab[1] - want[1]), ab))
                eh[Y, X] = a if Y % 2 == 0 else b
            else:
                eh[Y, X] = np.clip(re[Y, X], -FMAX, FMAX)
            if Y % 2:
                F_ = s[2 * h][Y // 2, X // 2]
                want = np.clip([rs[Y, 2 * (X // 2)], rs[Y, 2 * (X // 2) + 1]], -FMAX, FMAX)
                a, b = min(splits(F_), key=lambda ab: (abs(ab[0] - want[0]) + abs(ab[1] - want[1]), ab))
                sh[Y, X] = a if X % 2 == 0 else b
            else:
                sh[Y, X] = np.clip(rs[Y, X], -FMAX, FMAX)
        e[h], s[h] = eh, sh
    f = np.stack([-np.roll(s[K], 1, 0), e[K], s[K], -np.roll(e[K], 1, 1)], -1)
    return encode(f).astype(np.int64)


def make_valid(ts, tiles, K=16, seed=0):
    """-> (tiles, promise grid): the grid fulfilled block by block to its own
    projected promises (`project`), so it keeps the flow rule."""
    O = project(ts, tiles, K)
    v = FlowPromise(K)
    tiles = np.array(tiles, np.int32)
    for by, bx in np.ndindex(*O.shape):
        tiles = v.fulfil(ts, tiles, by * K, bx * K, K, int(O[by, bx]), seed=seed)[0]
    return tiles, O
