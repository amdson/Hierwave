"""Channels, views, factors, and the packing a kernel reads.

Channel(name, h, D, views): a grid of (H / h, W / h) values in 0..D-1 and
views[name] = (D,) int array, the view's value of each domain value.
Factor: a table over views.
    pair   E[va(z_a[p]), vb(z_b[q])],  q = (p * h_a) // h_b + off  (b's cells)
           pad_b: vb when q is off the grid (-1: the term vanishes there);
           pad_a: the same for a when the factor is read from b's side
    count  E[vb(z_b[q]), sum_{p' in block(q)} va(z_a[p'])], b coarser than a
    unary  E[va(z_a[p])]
    convpot  a learned 3 x 3 energy over value embeddings E[va(z)] of one
           channel (notes/convpot_test.md, "Definition"; numpy reference
           in convref.py): per site a . e_p + sum_{d != 0} e_p^T A_d e_{p+d}
           + v . softplus(sum_d W_d e_{p+d} + b); off-grid neighbours read
           E[pad] (pad >= 0) or nothing.  Same-level only.
Hard entries are inf.  A factor is homed on its a side, which is the finer
(or equal) level: top-down only, a coarser channel never reads a finer one.
Certificate: the one computed factor (the tree rule of notes/history/channels.tex, 14):
every cell of the tile channel with mass > 0 holds d < INF and is either a
trunk with d = 0 or has a 4-neighbour q with mass_q >= mass_p and d_q < d_p.

Model.compile(chan) packs, for the kernel of `chan`, the factors homed on it
plus the reflections of same-level pair factors (transposed table, negated
offset), and nothing homed below it.  compile(chan, below=True) also packs
the factors homed on strictly finer channels that read `chan` (pairs at
offset (0, 0) and counts), so the candidate energies are the full
conditional of the joint (bidirectional).  Kinds is a tiny tile-set stand-in
(name, tags, colour) so the demo does not depend on castlegen.legacy.tileset."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INF_E = np.inf
PAIR, COUNT, UNARY, BPAIR, BCOUNT, CONVPOT = 0, 1, 2, 3, 4, 5


# ------------------------------------------------------------------ kinds
@dataclass
class Kinds:
    """A tile domain: kind i has a name, tags and a colour.  Views are built
    from tags (view_of_tags)."""
    names: list
    tags: list              # list of frozenset
    colours: list           # (r, g, b)

    def __len__(self):
        return len(self.names)

    def index(self, name):
        return self.names.index(name)

    @staticmethod
    def concat(*ks):
        return Kinds(sum([k.names for k in ks], []), sum([k.tags for k in ks], []),
                     sum([k.colours for k in ks], []))


def view_of_tags(kinds: Kinds, table: dict, default=0):
    """(D,) int: the value of the first tag (in table order) a kind carries;
    a kind's own name counts as a tag.  default for kinds with none."""
    out = np.full(len(kinds), default, np.int64)
    for i, (name, tags) in enumerate(zip(kinds.names, kinds.tags)):
        for tag, val in table.items():
            if tag == name or tag in tags:
                out[i] = val
                break
    return out


# --------------------------------------------------------------- channels
@dataclass
class Channel:
    name: str
    h: int
    D: int
    views: dict = field(default_factory=dict)
    grid: np.ndarray = None      # (H / h, W / h) int32, set by Model
    fixed: np.ndarray = None     # bool, same shape: clamped cells (kernel skips them)
    view_n: dict = field(default_factory=dict)   # values a view can take (tables are this wide)

    def add_view(self, name, arr, n=None):
        arr = np.asarray(arr, np.int64)
        assert arr.shape == (self.D,), (name, arr.shape, self.D)
        self.views[name] = arr
        self.view_n[name] = int(arr.max()) + 1 if n is None else int(n)
        return self

    def nvals(self, view):
        return self.view_n[view]


@dataclass
class Factor:
    kind: int
    a: tuple                      # (channel name, view name)
    b: tuple = None               # (channel name, view name) for pair / count
    off: tuple = (0, 0)
    table: np.ndarray = None      # pair: (Va, Vb); count: (Vb, maxsum + 1); unary: (Va,)
    pad_b: int = -1
    pad_a: int = -1
    name: str = ""
    conv: dict = None             # convpot: {"E": (Va, k), "a": (k,), "A": (9, k, k), "W": (9, k, m), "b", "v": (m,)}

    @staticmethod
    def pair(a, b, off, table, pad_b=-1, pad_a=-1, name=""):
        return Factor(PAIR, a, b, tuple(off), np.asarray(table, np.float64), pad_b, pad_a, name)

    @staticmethod
    def count(a, b, table, name=""):
        return Factor(COUNT, a, b, (0, 0), np.asarray(table, np.float64), name=name)

    @staticmethod
    def unary(a, table, name=""):
        return Factor(UNARY, a, None, (0, 0), np.asarray(table, np.float64), name=name)

    @staticmethod
    def convpot(a, E, a_vec, A, W, b, v, pad=-1, name=""):
        """Conv potential on channel/view a (notes/convpot_test.md).  E (Va, k)
        indexed by the view value; A (9, k, k) (centre ignored), W (9, k, m),
        b, v (m,), m = 0 allowed (no head).  Offsets row-major over (dy, dx)
        in (-1, 0, 1)^2, index 4 the centre.  pad: the view value read off
        the grid (-1: off-grid neighbours contribute nothing)."""
        E = np.ascontiguousarray(E, np.float64)
        k = E.shape[1]
        W = np.ascontiguousarray(W, np.float64)
        m = W.shape[2] if W.ndim == 3 else 0          # m = 0: no head (W may be given as an empty array)
        conv = dict(E=E, a=np.ascontiguousarray(a_vec, np.float64).reshape(k),
                    A=np.ascontiguousarray(A, np.float64), W=W.reshape(9, k, m),
                    b=np.ascontiguousarray(b, np.float64).reshape(m),
                    v=np.ascontiguousarray(v, np.float64).reshape(m))
        return Factor(CONVPOT, a, None, (0, 0), None, -1, int(pad), name, conv)

    def conv_flat(self):
        """The packed convpot vector [a | A | W | b | v | E] (core.Packed)."""
        c = self.conv
        return np.ascontiguousarray(np.concatenate([c["a"].ravel(), c["A"].ravel(), c["W"].ravel(),
                                                    c["b"].ravel(), c["v"].ravel(), c["E"].ravel()]))


@dataclass
class Certificate:
    tile: str                     # level-1 channel holding the tiles
    mass: str                     # its view: 0 = not part of the structure
    trunk: str                    # its view: 1 = a root of the structure (d = 0)
    cert: str                     # the d channel (same level, D = Dmax + 2, INF = Dmax + 1)
    Dmax: int
    delta: float = 0.0            # soft cost per unit of d
    ports: tuple = None           # 4 views (N, E, S, W), 1 = a port on that side; None: every pair joined
    tree: bool = False            # the join graph is a forest: exactly one parent (mass >=, smaller d) per
                                  # root cell, every other join to a child (mass <=, larger d)
    joins: np.ndarray = None      # explicit (4, D, D) bool join table instead of ports (any level)


@dataclass
class Packed:
    """What a kernel reads for one home channel.

    fac rows (kind, bchan, aview, bview, oy, ox, pad, tab):
      0 pair    tab[aview(home cell), bview(bchan at q)], aview the home view
      1 count   tab[bview(bchan block), block sum of aview over home]
      2 unary   tab[aview(home cell), 0]
      3 below-pair   bchan = the finer channel, aview its view, bview the home
                view, oy = ox = 0, pad = -1: sum over the fine cells s of the
                home cell's block of tab[aview(z_s), bview(home cell)]
      4 below-count  bchan, aview, bview as for 3: tab[bview(home cell),
                block sum of aview over bchan]
      5 convpot (5, home, aview, -1, k, m, pad, ci): convs[ci] is one flat
                C-contiguous float64 vector [a (k) | A (9 k k) | W (9 k m) |
                b (m) | v (m) | E (Va k)] (Factor.conv_flat), unpacked by
                kernel._conv_unpack; m = 0 is the bilinear model.  Only
                convpots homed on the home channel are packed.
    Tables of kinds 3 and 4 are the factor's own (not transposed).
    src[f]: index into model.factors of row f; transposed[f]: row f reads
    the transpose of that factor's table (a reflected same-level pair)."""
    home: int
    grids: tuple
    hs: np.ndarray
    views: tuple
    fac: np.ndarray               # (F, 8) int64: kind, bchan, aview, bview, oy, ox, pad, tab
    tabs: tuple
    fixed: np.ndarray
    colours: np.ndarray           # (rows, cols) int64 colour class per site
    ncol: int
    cert: np.ndarray              # (6,) int64: massview, trunkview, certchan, Dmax, 1/0 present, 1/0 tree
    joins: np.ndarray             # (4, D, D) bool: t at p joined to t' across side d
    delta: float
    radius: int
    src: np.ndarray = None        # (F,) int64
    transposed: np.ndarray = None  # (F,) bool
    convs: tuple = None           # flat float64 convpot vectors (kind 5); (zeros(1),) when none
    nhard: int = 0                # compile(hard_first=True): rows [0, nhard) hold a +inf entry


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, H, W, channels, factors=(), certs=()):
        self.H, self.W = H, W
        self.channels = list(channels)
        self.by_name = {c.name: c for c in self.channels}
        self.factors = list(factors)
        self.certs = list(certs)
        for c in self.channels:
            assert H % c.h == 0 and W % c.h == 0, (c.name, c.h)
            if c.grid is None:
                c.grid = np.zeros((H // c.h, W // c.h), np.int32)
            if c.fixed is None:
                c.fixed = np.zeros((H // c.h, W // c.h), bool)
        self._check()

    def chan(self, name):
        return self.by_name[name]

    def _check(self):
        for f in self.factors:
            a = self.chan(f.a[0])
            assert f.a[1] in a.views, f"{f.name}: {f.a}"
            if f.kind == UNARY:
                assert f.table.shape == (a.nvals(f.a[1]),), f.name
                continue
            if f.kind == CONVPOT:
                c = f.conv
                V, k = c["E"].shape
                m = c["W"].shape[2]
                assert V == a.nvals(f.a[1]), (f.name, c["E"].shape, a.nvals(f.a[1]))
                assert c["a"].shape == (k,) and c["A"].shape == (9, k, k), f.name
                assert c["W"].shape == (9, k, m) and c["b"].shape == (m,) and c["v"].shape == (m,), f.name
                assert -1 <= f.pad_a < V, f.name
                continue
            b = self.chan(f.b[0])
            assert f.b[1] in b.views, f"{f.name}: {f.b}"
            assert a.h <= b.h, f"{f.name}: a factor is homed on the finer side"
            if f.kind == PAIR:
                assert f.table.shape == (a.nvals(f.a[1]), b.nvals(f.b[1])), f.name
            else:
                assert a.h < b.h, f"{f.name}: count needs a coarser b"
                maxsum = (b.h // a.h) ** 2 * (a.nvals(f.a[1]) - 1)
                assert f.table.shape == (b.nvals(f.b[1]), maxsum + 1), (f.name, f.table.shape)
        for c in self.certs:
            t, d = self.chan(c.tile), self.chan(c.cert)
            assert t.h == d.h and d.D == c.Dmax + 2, c
            assert c.mass in t.views and c.trunk in t.views
            assert c.ports is None or all(p in t.views for p in c.ports), c

    # ----------------------------------------------------------- packing
    def compile(self, home: str, below: bool = False, hard_first: bool = False) -> Packed:
        """hard_first: rows whose table holds +inf first (stable), nhard of
        them; the order kernel.sweep_cap relies on.  Off: today's order."""
        hc = self.chan(home)
        cidx = {c.name: i for i, c in enumerate(self.channels)}
        vidx, views = {}, []
        for c in self.channels:
            for v, arr in c.views.items():
                vidx[c.name, v] = len(views)
                views.append(np.ascontiguousarray(arr, np.int64))
        rows, tabs, src, transp, convs = [], [], [], [], []
        radius = 0
        fi = 0

        def add(kind, b, av, bv, oy, ox, pad, table, tr=False):
            rows.append([kind, b, av, bv, oy, ox, pad, len(tabs)])
            tabs.append(np.ascontiguousarray(table, np.float64).reshape(table.shape[0], -1))
            src.append(fi)
            transp.append(tr)

        for fi, f in enumerate(self.factors):
            if f.a[0] != home:
                if f.kind == CONVPOT:
                    continue                                  # same-level only
                if f.kind == PAIR and f.b[0] == home and self.chan(f.a[0]).h == hc.h:
                    # read from b's side: a sits at p - off
                    add(PAIR, cidx[f.a[0]], vidx[f.b], vidx[f.a], -f.off[0], -f.off[1], f.pad_a, f.table.T, True)
                    radius = max(radius, abs(f.off[0]), abs(f.off[1]))
                elif below and f.kind != UNARY and f.b[0] == home and self.chan(f.a[0]).h < hc.h:
                    # read from the coarse side: the fine cells lie inside the home cell
                    assert f.off == (0, 0), f"{f.name}: a below-pair needs offset (0, 0)"
                    add(BPAIR if f.kind == PAIR else BCOUNT, cidx[f.a[0]], vidx[f.a], vidx[f.b], 0, 0, -1, f.table)
                continue
            if f.kind == UNARY:
                add(UNARY, -1, vidx[f.a], -1, 0, 0, -1, f.table[:, None])
            elif f.kind == CONVPOT:
                rows.append([CONVPOT, cidx[home], vidx[f.a], -1, f.conv["E"].shape[1], f.conv["W"].shape[2],
                             f.pad_a, len(convs)])
                convs.append(f.conv_flat())
                src.append(fi)
                transp.append(False)
                radius = max(radius, 2)
            elif f.kind == PAIR:
                add(PAIR, cidx[f.b[0]], vidx[f.a], vidx[f.b], f.off[0], f.off[1], f.pad_b, f.table)
                if f.b[0] == home:
                    radius = max(radius, abs(f.off[0]), abs(f.off[1]))
                    if f.off != (0, 0):
                        add(PAIR, cidx[home], vidx[f.b], vidx[f.a], -f.off[0], -f.off[1], f.pad_a, f.table.T, True)
            else:
                add(COUNT, cidx[f.b[0]], vidx[f.a], vidx[f.b], 0, 0, -1, f.table)
        cert = np.array([-1, -1, -1, 0, 0, 0], np.int64)
        joins = np.ones((4, 1, 1), np.bool_)
        delta = 0.0
        for c in self.certs:
            if c.tile == home:
                cert = np.array([vidx[home, c.mass], vidx[home, c.trunk], cidx[c.cert], c.Dmax, 1, int(c.tree)], np.int64)
                delta = c.delta
                radius = max(radius, 2)
                joins = np.ones((4, hc.D, hc.D), np.bool_)
                if c.joins is not None:
                    joins = np.asarray(c.joins, np.bool_)
                elif c.ports is not None:
                    P4 = [hc.views[v].astype(bool) for v in c.ports]
                    for d in range(4):
                        joins[d] = P4[d][:, None] & P4[(d + 2) % 4][None, :]
        rows_, cols_ = hc.grid.shape
        yy, xx = np.mgrid[:rows_, :cols_]
        if radius == 0:
            colours, ncol = np.zeros_like(yy), 1
        elif radius == 1:
            colours, ncol = (yy + xx) % 2, 2
        elif radius == 2:
            colours, ncol = (xx + 2 * yy) % 5, 5
        else:
            colours, ncol = np.zeros_like(yy), 1          # sequential: still a valid Gibbs chain
        fac = np.array(rows, np.int64).reshape(-1, 8)
        nhard = 0
        if hard_first and len(rows):
            hard = np.array([r[0] != CONVPOT and bool(np.isposinf(tabs[r[7]]).any()) for r in rows])
            order = np.concatenate([np.flatnonzero(hard), np.flatnonzero(~hard)])
            fac, src, transp, nhard = fac[order], [src[i] for i in order], [transp[i] for i in order], int(hard.sum())
        return Packed(cidx[home], tuple(np.ascontiguousarray(c.grid, np.int32) for c in self.channels),
                      np.array([c.h for c in self.channels], np.int64), tuple(views), fac,
                      tuple(tabs) if tabs else (np.zeros((1, 1)),), np.ascontiguousarray(hc.fixed),
                      np.ascontiguousarray(colours, np.int64), ncol, cert, np.ascontiguousarray(joins),
                      float(delta), radius, np.array(src, np.int64), np.array(transp, np.bool_),
                      tuple(convs) if convs else (np.zeros(1),), nhard)

    # ---------------------------------------------------------- sampling
    def sweep(self, home: str, n: int, seed: int = 0, T: float = 1.0, below: bool = False):
        """n Gibbs sweeps of channel `home` (level 1).  Grids are updated in
        place.  Returns the violation count of the last sweep (sites with no
        finite candidate).  below: also read the factors homed on finer
        channels (the joint's full conditional)."""
        from . import kernel
        P = self.compile(home, below)
        hc = self.chan(home)
        assert P.grids[P.home] is hc.grid or np.shares_memory(P.grids[P.home], hc.grid)
        kernel.seed(seed)
        bad = 0
        for _ in range(n):
            bad = kernel.sweep(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.fixed, P.colours, P.ncol,
                               P.cert, P.joins, P.delta, T, P.convs)
        return bad

    def energy(self, home: str):
        """(total, violations): the factors homed on `home` at the current
        state, each pair once, finite part summed and inf entries counted."""
        from . import kernel
        P = self.compile(home)
        return kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.cert, P.joins, P.delta, P.convs)

    def site_energies(self, home: str, y: int, x: int, below: bool = False):
        """(D,) conditional energies of site (y, x) of `home` over its domain
        (the factor rows only; no certificate), for tests."""
        from . import kernel
        P = self.compile(home, below)
        D = self.chan(home).D
        return kernel.site_energies(y, x, P.home, P.grids, P.hs, P.views, P.fac, P.tabs, D, P.convs)

    def describe(self, home: str) -> str:
        P = self.compile(home)
        names = ["pair", "count", "unary", "bpair", "bcount", "conv"]
        lines = [f"{home}: {P.fac.shape[0]} factor rows, radius {P.radius}, {P.ncol} colours"
                 + (", certificate" if P.cert[4] else "")]
        for r in P.fac:
            if r[0] == CONVPOT:
                lines.append(f"  conv  av={r[2]} k={r[4]} m={r[5]} pad={r[6]} vec{P.convs[r[7]].shape}")
                continue
            lines.append(f"  {names[r[0]]:5s} b={r[1]:2d} av={r[2]} bv={r[3]} off=({r[4]},{r[5]}) pad={r[6]} "
                         f"table{P.tabs[r[7]].shape}")
        return "\n".join(lines)
