"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Connectivity promise (notes/history/promises.md section 4.1), torus maps, no gate.

Nodes are `is_room` tiles (and the gate, if any); two edge-adjacent nodes are
joined when their facing sockets connect (Dh / Dv).  A **crossing** of a
block is a boundary node joined to the node across the block edge.

State (exact, `State`): the block's crossings (y, x, d), their partition by
reachability inside the block, the number of components with no crossing
(`closed`) and the node count.  `merge` is conn.py's union-find merge with
crossings as ports: a child crossing whose target lies in a sibling is joined
to the sibling's facing crossing and disappears; a class left with no
crossing becomes closed.  Exact: merge of the children's states equals the
parent's state (tests/legacy/test_connectivity.py).

Promise: a 4-bit side set O (bit d for side d in N, E, S, W).  A state
satisfies O when
  O = {}   the block holds no node;
  O != {}  all the block's nodes form ONE component inside the block (no
           closed component, one class), and it crosses every side in O.
Crossings on closed sides are allowed (a promise is a lower bound).  One
component per block is what makes the guarantee compose: with several, two
blocks could hold a pair of components joined only to each other.
`abstract(state)` = the sides with a crossing; it is a projection (a state
with several components abstracts to the sides it crosses and does not
satisfy that value).

Promise merge: internal seams agree, parent side bit = OR of its two
children's bits, the children with an open side are connected by the open
internal seams, and a parent with no open side has four empty children.  If the four children satisfy their promises, the
parent's state satisfies the merged promise; the top level adds a global
check that the blocks with an open side are connected on the torus, so all
nodes form one network.

Sampling a level (promise.PromiseVar): a parent's joint update sets its four
children's internal seams and the split of its E and S sides (it owns those
two seams, and writes the facing bits into the neighbours' children); N and W
splits belong to the neighbours.  Every proposal keeps the three parents it
touches consistent.

Fulfil (per K-block, edits only inside the block): one crossing cell per open
side at a boundary offset hashed from the seam (the same from both blocks),
opened towards the seam (or made a `tag` piece with a door there); then
exemplar.connect joins every other component to the crossings' component or
deletes it; a hub path from each crossing to the block centre is the fallback
when connect cannot reach.  A block with O = {} loses its nodes.
"""
from __future__ import annotations

import itertools

import numpy as np

from castlegen.legacy import connmetrics as cm
from castlegen.legacy import exemplar as ex
from castlegen.legacy import hier
from castlegen.legacy.conn import _UF
from castlegen.legacy.promise import PromiseVar

N, E, S, W = 0, 1, 2, 3
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPP = (2, 3, 0, 1)
SEAM_STEP = 0xFFFFFFF0          # noise `step` for seam crossing offsets
TOP_COLOUR = 7                  # noise `colour` for the top-level Gibbs
V = 16                          # promise domain size


def bit(v, d):
    return (v >> d) & 1


# ------------------------------------------------------------------ crossings
def crossings(ts, tiles):
    """(4, H, W) bool, torus: cell is a node joined to the node on side d."""
    tiles = np.asarray(tiles)
    node = cm._node(ts)[tiles]
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    r, d = np.roll(tiles, -1, 1), np.roll(tiles, -1, 0)
    e = node & np.roll(node, -1, 1) & (Dh[tiles, r] > 0)
    s = node & np.roll(node, -1, 0) & (Dv[tiles, d] > 0)
    return np.stack([np.roll(s, 1, 0), e, s, np.roll(e, 1, 1)])


def EDGES(k):
    """Index of side d's boundary cells in a k x k block."""
    return ((0, slice(None)), (slice(None), k - 1), (k - 1, slice(None)), (slice(None), 0))


def _ring_crossings(ts, tiles, y0, x0, k):
    """Crossings of the block's cells only, from its window plus a one-cell
    ring (wrapped): (4, k, k)."""
    n0, n1 = tiles.shape
    ys, xs = np.arange(y0 - 1, y0 + k + 1) % n0, np.arange(x0 - 1, x0 + k + 1) % n1
    w = np.asarray(tiles)[np.ix_(ys, xs)]
    node = cm._node(ts)[w]
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    e = node[:, :-1] & node[:, 1:] & (Dh[w[:, :-1], w[:, 1:]] > 0)       # (k+2, k+1): col c joined to c+1
    s = node[:-1, :] & node[1:, :] & (Dv[w[:-1, :], w[1:, :]] > 0)       # (k+1, k+2)
    return np.stack([s[:-1, 1:-1], e[1:-1, 1:], s[1:, 1:-1], e[1:-1, :-1]])


# ---------------------------------------------------------------------- state
class State:
    __slots__ = ("y0", "x0", "k", "ports", "labels", "n_classes", "closed", "nodes")

    def __init__(self, y0, x0, k, ports, labels, n_classes, closed, nodes):
        self.y0, self.x0, self.k = y0, x0, k
        self.ports = ports              # sorted [(y, x, d)]: crossings, absolute cell coords
        self.labels = labels            # class per port, numbered by first appearance
        self.n_classes, self.closed, self.nodes = n_classes, closed, nodes

    def key(self):
        return (self.y0, self.x0, self.k, tuple(self.ports), tuple(self.labels), self.n_classes, self.closed,
                self.nodes)

    def __eq__(self, other):
        return isinstance(other, State) and self.key() == other.key()

    def __repr__(self):
        return (f"State({self.y0},{self.x0},k={self.k}, {len(self.ports)} ports, classes={self.n_classes}, "
                f"closed={self.closed}, nodes={self.nodes})")

    def class_masks(self):
        """Side mask per class."""
        m = [0] * self.n_classes
        for (_, _, d), c in zip(self.ports, self.labels):
            m[c] |= 1 << d
        return m


def _canonical(y0, x0, k, members, n_components, nodes):
    """members: [((y, x, d), class id)] -> State with classes renumbered by
    first appearance in sorted port order."""
    members = sorted(members)
    ids, labels = {}, []
    for _, c in members:
        ids.setdefault(c, len(ids))
        labels.append(ids[c])
    return State(y0, x0, k, [p for p, _ in members], labels, len(ids), n_components - len(ids), nodes)


def summarize(ts, tiles, y0, x0, k):
    """Exact state of the k x k block at (y0, x0) (no wrap inside the block;
    crossings read the wrapped neighbours)."""
    win = np.asarray(tiles)[y0:y0 + k, x0:x0 + k]
    lab, nc = cm.labels(ts, win, torus=False)
    cr = _ring_crossings(ts, tiles, y0, x0, k)
    members = []
    for d in range(4):
        m = np.zeros((k, k), bool)
        m[EDGES(k)[d]] = True
        for y, x in zip(*np.nonzero(m & cr[d])):
            members.append(((y0 + int(y), x0 + int(x), d), int(lab[y, x])))
    return _canonical(y0, x0, k, members, nc, int((lab >= 0).sum()))


def merge(children):
    """children: {(dy, dx): State} of four siblings -> the parent's State."""
    c0 = children[(0, 0)]
    k, Y0, X0 = c0.k, c0.y0, c0.x0
    side = 2 * k
    port_of, offs, n, closed, nodes = {}, {}, 0, 0, 0
    for key in ((0, 0), (0, 1), (1, 0), (1, 1)):
        s = children[key]
        for p, lab in zip(s.ports, s.labels):
            port_of[p] = n + lab
        closed += s.closed
        nodes += s.nodes
        n += s.n_classes
    uf = _UF(n)
    outer = []
    for (y, x, d), g in port_of.items():
        ty, tx = y + DIRS[d][0], x + DIRS[d][1]
        if Y0 <= ty < Y0 + side and X0 <= tx < X0 + side:
            uf.union(g, port_of[ty, tx, OPP[d]])
        else:
            outer.append(((y, x, d), g))
    roots = {uf.find(g) for g in range(n)}
    members = [(p, uf.find(g)) for p, g in outer]
    return _canonical(Y0, X0, side, members, closed + len(roots), nodes)


def abstract(state):
    """Sides with a crossing."""
    o = 0
    for _, _, d in state.ports:
        o |= 1 << d
    return o


def satisfied(value, state):
    if value == 0:
        return state.nodes == 0
    return state.closed == 0 and state.n_classes == 1 and (value & ~abstract(state)) == 0


def _costs(masks, closed, nodes):
    """(16,) proxy fulfil cost of every promise for a state given by its
    class side masks: 0 if satisfied, else 1 + violated conditions (open
    sides not all crossed by one class; other components; nodes in a closed block)."""
    O = np.arange(16)
    masks = np.asarray(masks, int)
    cover = np.array([bool(((masks & o) == o).any()) if len(masks) else False for o in O])
    extra = len(masks) + closed > 1 or closed > 0
    sat = np.where(O == 0, not nodes, cover & (not extra))
    viol = np.where(O == 0, int(bool(nodes)), (~cover).astype(int) + int(extra))
    return np.where(sat, 0, 1 + viol).astype(np.float32)


def violations(value, state):
    return float(_costs(state.class_masks(), state.closed, state.nodes)[value])


# ------------------------------------------------------------- promise merge
_TAB = {}


def _tables():
    """Over all 2^16 child tuples t = c00 | c01 << 4 | c10 << 8 | c11 << 12:
    the merged parent value and whether the tuple is consistent."""
    if not _TAB:
        t = np.arange(1 << 16)
        c = [(t >> (4 * i)) & 15 for i in range(4)]               # c00, c01, c10, c11
        b = lambda v, d: (v >> d) & 1
        agree = (b(c[0], E) == b(c[1], W)) & (b(c[2], E) == b(c[3], W)) \
            & (b(c[0], S) == b(c[2], N)) & (b(c[1], S) == b(c[3], N))
        P = (b(c[0], N) | b(c[1], N)) << N | (b(c[1], E) | b(c[3], E)) << E \
            | (b(c[2], S) | b(c[3], S)) << S | (b(c[0], W) | b(c[2], W)) << W
        edges = [(0, 1, b(c[0], E)), (2, 3, b(c[2], E)), (0, 2, b(c[0], S)), (1, 3, b(c[1], S))]
        lab = [np.full(t.shape, i) for i in range(4)]
        for _ in range(3):
            for i, j, e in edges:
                m = np.minimum(lab[i], lab[j])
                lab[i] = np.where(e == 1, m, lab[i])
                lab[j] = np.where(e == 1, m, lab[j])
        live = [ci != 0 for ci in c]
        ref = np.min([np.where(live[i], lab[i], 9) for i in range(4)], 0)
        conn = np.all([~live[i] | (lab[i] == ref) for i in range(4)], 0)
        empty = (P != 0) | ~np.any(live, 0)                 # a closed parent holds nothing
        _TAB["P"], _TAB["ok"] = P.astype(np.int64), agree & conn & empty
    return _TAB["P"], _TAB["ok"]


def pack(c00, c01, c10, c11):
    return int(c00) | int(c01) << 4 | int(c10) << 8 | int(c11) << 12


def unpack(t):
    return tuple((int(t) >> (4 * i)) & 15 for i in range(4))


def merge_promises(c00, c01, c10, c11):
    """-> (parent value, consistent)."""
    P, ok = _tables()
    t = pack(c00, c01, c10, c11)
    return int(P[t]), bool(ok[t])


def consistent(grid, parents=None):
    """Every seam of the torus grid agrees; with parents, every 2x2 group
    merges consistently to its parent."""
    g = np.asarray(grid)
    if not (np.array_equal((g >> E) & 1, (np.roll(g, -1, 1) >> W) & 1)
            and np.array_equal((g >> S) & 1, (np.roll(g, -1, 0) >> N) & 1)):
        return False
    if parents is not None:
        P, ok = _tables()
        t = g[0::2, 0::2] | g[0::2, 1::2] << 4 | g[1::2, 0::2] << 8 | g[1::2, 1::2] << 12
        return bool(ok[t].all() and np.array_equal(P[t], parents))
    return True


EXT = {N: ((0, 0), (0, 1)), E: ((0, 1), (1, 1)), S: ((1, 0), (1, 1)), W: ((0, 0), (1, 0))}


def refine(parent, halo=None):
    """Consistent refinements of `parent`: [(c00, c01, c10, c11)].  halo:
    {(dy, dx, d): bit} fixes external half-seams (a child's side on the
    parent boundary), e.g. from the neighbours' current children.
    Enumerates the split of every parent side and the four internal seams,
    then keeps the connected ones."""
    halo = halo or {}
    _, ok = _tables()
    split = {}
    for d in range(4):
        opts = [(0, 0)] if not bit(parent, d) else [(0, 1), (1, 0), (1, 1)]
        split[d] = [o for o in opts if all(halo.get(EXT[d][i] + (d,), o[i]) == o[i] for i in (0, 1))]
    out = []
    for sn, se, ss, sw in itertools.product(split[N], split[E], split[S], split[W]):
        for h0, h1, v0, v1 in itertools.product((0, 1), repeat=4):
            c = {}
            for key in ((0, 0), (0, 1), (1, 0), (1, 1)):
                c[key] = 0
            for sd, d in ((sn, N), (se, E), (ss, S), (sw, W)):
                for i, key in enumerate(EXT[d]):
                    c[key] |= sd[i] << d
            c[0, 0] |= h0 << E | v0 << S
            c[0, 1] |= h0 << W | v1 << S
            c[1, 0] |= h1 << E | v0 << N
            c[1, 1] |= h1 << W | v1 << N
            t = pack(c[0, 0], c[0, 1], c[1, 0], c[1, 1])
            if ok[t]:
                out.append(unpack(t))
    return out


def brute_refine(parent, halo=None):
    """Reference for tests: filter all 2^16 child tuples."""
    halo = halo or {}
    P, ok = _tables()
    t = np.flatnonzero(ok & (P == parent))
    keep = []
    for x in t:
        c = dict(zip(((0, 0), (0, 1), (1, 0), (1, 1)), unpack(x)))
        if all(bit(c[dy, dx], d) == b for (dy, dx, d), b in halo.items()):
            keep.append(unpack(x))
    return keep


# ------------------------------------------------------------------ variable
class ConnectivityPromise(PromiseVar):
    name = "conn"
    salt = 3
    V = 16

    def __init__(self, K=16, h_top=64, tables=None, T=1.0, sweeps=3, top_sweeps=20, tag="hallway",
                 cross_socket="door", void=None):
        """cross_socket: the socket a crossing cell must show its seam ("door";
        "land" in the wilds, where land joins land).  void: signature name
        deleted nodes become (default the wall)."""
        super().__init__(K, h_top, tables, T, sweeps)
        self.top_sweeps, self.tag = top_sweeps, tag
        self.cross_socket, self.void = cross_socket, void

    # ---- bottom-up
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
        """Gibbs over the seam bits of the top grid (a block's value is read
        off its four seams, so pair consistency holds by construction, also
        when a block is its own neighbour); a flip is allowed only if the
        blocks with an open side stay connected and at least one is open."""
        G0, G1 = level.shape
        u, pair, _ = self.tables.level(level.h)
        bits = np.ones((2, G0, G1), np.int64)            # [0]: E seam of (y, x); [1]: S seam of (y, x)

        def values(b):
            return (b[1][(np.arange(G0) - 1) % G0][:, :] << N) | (b[0] << E) | (b[1] << S) \
                | (b[0][:, (np.arange(G1) - 1) % G1] << W)

        def ok(b):
            v = values(b)
            live = list(zip(*np.nonzero(v)))
            if not live:
                return False
            seen, stack = {live[0]}, [live[0]]
            while stack:
                y, x = stack.pop()
                for nb, open_ in ((((y), (x + 1) % G1), b[0][y, x]), ((y, (x - 1) % G1), b[0][y, (x - 1) % G1]),
                                  (((y + 1) % G0, x), b[1][y, x]), (((y - 1) % G0, x), b[1][(y - 1) % G0, x])):
                    if open_ and nb not in seen:
                        seen.add(nb); stack.append(nb)
            return len(seen) == len(live)

        def energy(b):
            v = values(b)
            return float(u[v].sum() + pair[0][v, np.roll(v, -1, 1)].sum() + pair[1][v, np.roll(v, -1, 0)].sum())

        lvl = hier.level_index(level.h)
        for s in range(self.top_sweeps):
            for o, y, x in itertools.product((0, 1), range(G0), range(G1)):
                es = []
                for bv in (0, 1):
                    bits[o, y, x] = bv
                    es.append(energy(bits) if ok(bits) else np.inf)
                g = hier.gumbel(hier.noise(ctx.seed, lvl, s, TOP_COLOUR, y, x, np.arange(2) + 2 * o + (self.salt << 20)))
                e = np.array(es) / max(self._T, 1e-9) - (g if self._T > 0 else 0)
                bits[o, y, x] = int(np.argmin(e))
        return values(bits).astype(np.int64)

    def init_children(self, parent, level, ctx):
        """Every external half-seam copies the parent's side bit; internal
        seams are open inside nonempty parents.  Consistent across parents
        (both sides of a parent seam make the same choice)."""
        g0, g1 = level.shape
        out = np.zeros((g0, g1), np.int64)
        ys, xs = np.arange(g0)[:, None], np.arange(g1)[None, :]
        p = parent[ys // 2, xs // 2]
        dy, dx = ys % 2, xs % 2
        ext = {N: dy == 0, S: dy == 1, W: dx == 0, E: dx == 1}
        for d in range(4):
            out |= np.where(ext[d], (p >> d) & 1, (p != 0).astype(np.int64)) << d
        return out

    def proposals(self, grid, parents, py, px):
        g0, g1 = grid.shape
        G0, G1 = parents.shape
        p = int(parents[py, px])
        y0, x0 = 2 * py, 2 * px
        own = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
        efc = [(y0, (x0 + 2) % g1), (y0 + 1, (x0 + 2) % g1)]
        sfc = [((y0 + 2) % g0, x0), ((y0 + 2) % g0, x0 + 1)]
        cells = own + efc + sfc
        affected = {(py, px), (py, (px + 1) % G1), ((py + 1) % G0, px)}
        P, okt = _tables()
        sE = [(0, 0)] if not bit(p, E) else [(0, 1), (1, 0), (1, 1)]
        sS = [(0, 0)] if not bit(p, S) else [(0, 1), (1, 0), (1, 1)]
        base = {c: int(grid[c]) for c in cells}
        out = []
        for se, ss in itertools.product(sE, sS):
            for h0, h1, v0, v1 in itertools.product((0, 1), repeat=4):
                w = dict(base)

                def put(c, d, b):
                    w[c] = (w[c] & ~(1 << d)) | (b << d)
                put((y0, x0), E, h0); put((y0, x0 + 1), W, h0)
                put((y0 + 1, x0), E, h1); put((y0 + 1, x0 + 1), W, h1)
                put((y0, x0), S, v0); put((y0 + 1, x0), N, v0)
                put((y0, x0 + 1), S, v1); put((y0 + 1, x0 + 1), N, v1)
                for i in (0, 1):
                    put((y0 + i, x0 + 1), E, se[i]); put(efc[i], W, se[i])
                    put((y0 + 1, x0 + i), S, ss[i]); put(sfc[i], N, ss[i])
                good = True
                for qy, qx in affected:
                    ch = [w.get((2 * qy + a, 2 * qx + b), None) for a in (0, 1) for b in (0, 1)]
                    ch = [int(grid[2 * qy + a, 2 * qx + b]) if v is None else v
                          for v, (a, b) in zip(ch, ((0, 0), (0, 1), (1, 0), (1, 1)))]
                    t = pack(*ch)
                    if not okt[t] or P[t] != parents[qy, qx]:
                        good = False
                        break
                if good:
                    out.append(tuple(w[c] for c in cells))
        return cells, list(dict.fromkeys(out))

    # ---- coupling table
    def window_costs(self, ts, E_, h):
        """(m*m, 16) proxy cost of every promise for the h x h exemplar
        window centred on each coordinate (torus exemplar)."""
        E_ = np.asarray(E_)
        m = E_.shape[0]
        cr = crossings(ts, E_)
        out = np.zeros((m * m, 16), np.float32)
        for u in range(m * m):
            cy, cx = divmod(u, m)
            ys, xs = np.arange(cy - h // 2, cy - h // 2 + h) % m, np.arange(cx - h // 2, cx - h // 2 + h) % m
            lab, nc = cm.labels(ts, E_[np.ix_(ys, xs)], torus=False)
            masks = np.zeros(nc, int)
            for d, sl in enumerate(EDGES(h)):
                c = cr[d][np.ix_(ys, xs)][sl]
                np.bitwise_or.at(masks, lab[sl][c], 1 << d)
            ported = masks != 0
            out[u] = _costs(masks[ported], int((~ported).sum()), nc > 0)
        return out

    # ---- fulfil
    def crossing_cells(self, n, y0, x0, k, value, seed, blocked=None):
        """[(y, x, d)] absolute: the crossing cell for each open side, at a
        boundary offset in [1, k-2] hashed from the seam (same from both
        blocks).  blocked (absolute mask, e.g. water and structures): the
        first offset in hash order whose two seam cells are both unblocked
        (still symmetric, since no fulfil changes the blocked cells)."""
        nb = n // k
        by, bx = y0 // k, x0 // k
        lvl = hier.level_index(k)

        def off(orient, a, b, cell):
            o0 = 1 + int(hier.noise(seed, lvl, SEAM_STEP, orient, a, b, 0) * (k - 2))
            if blocked is None:
                return o0
            rest = np.arange(1, k - 1)
            rest = rest[np.argsort(hier.noise(seed, lvl, SEAM_STEP, orient, a, b, rest), kind="stable")]
            for o in [o0] + [int(r) for r in rest if r != o0]:
                (ya, xa), (yb, xb) = cell(o)
                if not blocked[ya % n, xa % n] and not blocked[yb % n, xb % n]:
                    return o
            return o0
        out = []
        if bit(value, N):
            out.append((y0, x0 + off(1, by, bx, lambda o: ((y0, x0 + o), (y0 - 1, x0 + o))), N))
        if bit(value, E):
            out.append((y0 + off(0, by, (bx + 1) % nb, lambda o: ((y0 + o, x0 + k - 1), (y0 + o, x0 + k))), x0 + k - 1, E))
        if bit(value, S):
            out.append((y0 + k - 1, x0 + off(1, (by + 1) % nb, bx, lambda o: ((y0 + k - 1, x0 + o), (y0 + k, x0 + o))), S))
        if bit(value, W):
            out.append((y0 + off(0, by, bx, lambda o: ((y0 + o, x0), (y0 + o, x0 - 1))), x0, W))
        return out

    def block_ok(self, ts, blk, value, cross):
        """Block-local contract: no nodes if value = 0; else one component,
        holding every crossing cell with a door facing its seam."""
        lab, nc = cm.labels(ts, blk, torus=False)
        if value == 0:
            return nc == 0
        sock = ts.sockets.index(self.cross_socket)
        if nc != 1:
            return False
        return all(lab[y, x] >= 0 and ts.sig_sockets[blk[y, x], d] == sock for y, x, d in cross)

    def fulfil(self, ts, tiles, y0, x0, k, value, halo=None, seed=0, budget=None, protect=None):
        """Make the k x k block at (y0, x0) satisfy `value` given that every
        neighbour does the same; edits only inside the block.  protect
        (absolute mask): cells never overwritten (water may still be bridged
        with its flow kept) and never chosen as crossing cells.
        -> (tiles, edits, crossing cells (absolute, [(y, x)]))."""
        tiles = np.array(tiles, np.int32)
        n = tiles.shape[0]
        blk = tiles[y0:y0 + k, x0:x0 + k].copy()
        old = blk.copy()
        cross = [(y - y0, x - x0, d) for y, x, d in self.crossing_cells(n, y0, x0, k, value, seed, protect)]
        keep = None if protect is None else protect[y0:y0 + k, x0:x0 + k].copy()
        if not self.block_ok(ts, blk, value, cross):
            blk = self._fulfil_block(ts, blk, value, cross, seed * 7919 + y0 * 131 + x0, keep=keep)
        tiles[y0:y0 + k, x0:x0 + k] = blk
        return tiles, int((blk != old).sum()), [(y + y0, x + x0) for y, x, _ in cross]

    def _fulfil_block(self, ts, blk, value, cross, seed, rounds=6, keep=None):
        node = cm._node(ts)
        void = None if self.void is None else ex.sig_by_name(ts, self.void)
        if value == 0:
            return _delete(ts, blk, node[blk], void, keep)
        OPEN = _open(ts, self.tag)
        sock = ts.sockets.index(self.cross_socket)
        for y, x, d in cross:
            t = blk[y, x]
            if self.cross_socket == "door":
                blk[y, x] = OPEN[t, d] if node[t] and OPEN[t, d] >= 0 else ex.sig_by_doors(ts, self.tag, {d})
            elif not (node[t] and ts.sig_sockets[t, d] == sock):
                blk[y, x] = _plain_node(ts, sock)
        clamp = np.zeros(blk.shape, bool) if keep is None else keep.copy()
        for y, x, _ in cross:
            clamp[y, x] = True
        cy, cx, _ = cross[0]
        for r in range(rounds + 1):
            if r == rounds:                                     # fallback: hub paths from every crossing
                blk = self._hub(ts, blk, cross, clamp)
            lab, nc = cm.labels(ts, blk, torus=False)
            if nc == 1:
                return blk
            sizes = np.bincount(lab[lab >= 0], minlength=nc)
            main = int(np.argmax(sizes))
            others = [c for c in range(nc) if c != main]
            # cheap joins: exemplar.connect's carve (walls and door openings only)
            blk, _ = ex.connect(ts, blk, clamp, torus=False, seed=seed + r, tag=self.tag,
                                sources=[main], targets=others, delete=False)
            lab, nc = cm.labels(ts, blk, torus=False)
            if nc == 1:
                return blk
            # the rest: forced path (may overwrite what it crosses) or delete, whichever edits less
            sizes = np.bincount(lab[lab >= 0], minlength=nc)
            main = int(np.argmax(sizes))
            blk = _force_join(ts, blk, lab, nc, main, clamp, OPEN, self.tag, sizes, may_delete=r < rounds,
                              void=void, keep=keep)
        lab, nc = cm.labels(ts, blk, torus=False)               # after the hub every crossing is joined
        return _delete(ts, blk, (lab >= 0) & (lab != lab[cy, cx]), void, keep)

    def _hub(self, ts, blk, cross, clamp):
        """Carve a tag path from each crossing straight in to the centre row or
        column, then along it to the block centre (overwrites what it crosses)."""
        k = blk.shape[0]
        c = k // 2
        sides = {}
        for y, x, d in cross:
            pts = [(y, x), (c, x) if d in (N, S) else (y, c), (c, c)]
            cells = ex._polyline(pts)
            sides.setdefault((y, x), set()).add(d)
            for a, b in zip(cells, cells[1:]):
                dd = ex.STEP.index((b[0] - a[0], b[1] - a[1]))
                sides.setdefault(a, set()).add(dd)
                sides.setdefault(b, set()).add(OPP[dd])
        wet = ts.water[blk] if ts.water is not None else np.zeros(blk.shape, bool)
        ends = {(y, x) for y, x, _ in cross}
        for (yy, xx), ds in sides.items():
            if clamp[yy, xx] and (yy, xx) not in ends:           # protected: bridge water keeping its flow, else skip
                if wet[yy, xx] and ts.flow_dir[blk[yy, xx]] >= 0 and not ts.source[blk[yy, xx]]:
                    from castlegen.legacy.quantities import flow as F
                    b = F.tables(ts)["bridge_by"][sum(1 << d for d in ds), ts.flow_dir[blk[yy, xx]]]
                    if b >= 0:
                        blk[yy, xx] = b
                continue
            blk[yy, xx] = ex.sig_by_doors(ts, self.tag, ds)
            clamp[yy, xx] = True
        return blk


_OPEN = {}


def _delete(ts, blk, gone, void=None, keep=None):
    """Set the cells in `gone` to `void` (default wall), with the solids next
    to them that no remaining node touches (trees left standing in a deleted
    courtyard); never water or `keep` cells."""
    node = cm._node(ts)
    void = ts.WALL if void is None else void
    keep = np.zeros(blk.shape, bool) if keep is None else keep
    if ts.water is not None:
        keep = keep | ts.water[blk]
    blk[gone & ~keep] = void
    solid = ~node[blk] & (blk != ts.WALL) & (blk != void) & ~keep
    blk[ex.dilate(gone, 1, torus=False) & solid & ~ex.dilate(node[blk], 1, torus=False)] = void
    return blk


def _plain_node(ts, sock):
    """A node signature showing `sock` on every side (e.g. grass)."""
    node = cm._node(ts)
    for s in range(ts.n_sig):
        if node[s] and (ts.sig_sockets[s] == sock).all() and not (ts.water is not None and ts.water[s]):
            return s
    raise KeyError(f"no node with {ts.sockets[sock]!r} on every side")


def _force_join(ts, blk, lab, nc, main, clamp, OPEN, tag, sizes, may_delete=True, w_over=3.0, void=None, keep=None):
    """Shortest paths from the main component to every other one, where a
    step may also overwrite a node or solid with a `tag` piece (cost w_over,
    against 1 for a wall or a door opening).  A component is joined along its
    path if that edits fewer cells than deleting it (always, if it holds a
    clamped cell), else deleted.  Overwritten nodes keep their door joins."""
    import heapq
    node = cm._node(ts)
    H, W = blk.shape
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    door = ts.sockets.index("door")

    def joined(y, x, d, ny, nx):
        a, b = blk[y, x], blk[ny, nx]
        return bool(node[a] and node[b] and (Dh[a, b] if d == E else Dh[b, a] if d == W else
                                              Dv[a, b] if d == S else Dv[b, a]))

    def enter(ny, nx, back):
        t = blk[ny, nx]
        if t == ts.WALL and not clamp[ny, nx]:
            return 1.0
        if node[t] and OPEN[t, back] >= 0:
            return 1.0
        return np.inf if clamp[ny, nx] else w_over

    def leave(y, x, d):
        t = blk[y, x]
        if not node[t] or OPEN[t, d] >= 0:
            return 0.0
        return np.inf if clamp[y, x] else w_over - 1.0

    wet = ts.water[blk] if ts.water is not None else np.zeros((H, W), bool)
    jumps = {}
    dist = np.full((H, W), np.inf)
    parent = {}
    heap = [(0.0, int(y), int(x)) for y, x in zip(*np.nonzero(lab == main))]
    for _, y, x in heap:
        dist[y, x] = 0.0
    heapq.heapify(heap)
    while heap:
        du, y, x = heapq.heappop(heap)
        if du > dist[y, x]:
            continue
        for d, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if wet[ny, nx] and not node[blk[ny, nx]]:           # water: only a straight bridge keeping its flow
                jb = ex.bridge_jump(ts, blk, y, x, d, torus=False)
                if jb is None:
                    continue
                r, (ly, lx), _ = jb
                c = leave(y, x, d) + r + enter(ly, lx, OPP[d])
                nd = du + c + 1e-6 * (ly * W + lx)
                if nd < dist[ly, lx]:
                    dist[ly, lx] = nd
                    parent[ly, lx] = (y, x, d)
                    jumps[ly, lx] = jb
                    heapq.heappush(heap, (nd, ly, lx))
                continue
            c = 0.0 if joined(y, x, d, ny, nx) else leave(y, x, d) + enter(ny, nx, OPP[d])
            nd = du + c + 1e-6 * (ny * W + nx)
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                parent[ny, nx] = (y, x, d)
                jumps.pop((ny, nx), None)
                heapq.heappush(heap, (nd, ny, nx))
    need, gone, done, bridges = {}, np.zeros((H, W), bool), set(), {}
    for c in range(nc):
        if c == main:
            continue
        cells = np.argwhere(lab == c)
        dc = dist[cells[:, 0], cells[:, 1]]
        if may_delete and not clamp[lab == c].any() and (not np.isfinite(dc.min()) or dc.min() >= sizes[c]):
            gone |= lab == c
            continue
        if not np.isfinite(dc.min()):
            continue
        y, x = map(int, cells[np.argmin(dc)])
        while (y, x) in parent and (y, x) not in done:
            done.add((y, x))
            py, px, d = parent[y, x]
            if (y, x) in jumps:
                for i, b in enumerate(jumps[y, x][2], 1):
                    bridges[py + i * DIRS[d][0], px + i * DIRS[d][1]] = b
                need.setdefault((py, px), set()).add(d)
                need.setdefault((y, x), set()).add(OPP[d])
            elif not joined(py, px, d, y, x):
                need.setdefault((py, px), set()).add(d)
                need.setdefault((y, x), set()).add(OPP[d])
            y, x = py, px
    new = blk.copy()
    for (y, x), ds in need.items():
        t = blk[y, x]
        if node[t]:
            s = t
            for d in ds:
                s = OPEN[s, d] if s >= 0 else s
            if s >= 0:
                new[y, x] = s
                continue
            ds = ds | {d for d, (dy, dx) in enumerate(DIRS)          # keep its door joins
                       if 0 <= y + dy < H and 0 <= x + dx < W and ts.sig_sockets[t, d] == door
                       and joined(y, x, d, y + dy, x + dx)}
        new[y, x] = ex.sig_by_doors(ts, tag, ds)
    for (y, x), b in bridges.items():
        new[y, x] = b
    return _delete(ts, new, gone & ~clamp, void, keep) if gone.any() else new


def _open(ts, tag):
    key = (id(ts), tag)
    if key not in _OPEN:
        _OPEN[key] = ex.open_table(ts, tag)
    return _OPEN[key]
