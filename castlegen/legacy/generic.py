"""Legacy (G1c, generic/joint sampler era); superseded by castlegen/channels. See notes/history/generic.md.

Joint sampler over exemplar coordinates and promise values, one level at a
time (notes/history/joint_sampler.md).

Level h: a grid R x C = n/h x n/h of cells p (sides free: a patch slot off
the map continues its own cell coherently, so its term vanishes; or periodic;
above and below, two fixed rows of padding cells in the exemplar's edge
continuation, texsyn bounds="edge" with pad >= 2h, whose own patch terms
count; a torus exemplar, bounds=None, has none: all four sides free), each with
    u[p]  exemplar coordinate, flat index y * m + x: the centre of an h x h
          exemplar window (top-left u - h//2; rows clip, columns wrap)
    v[p]  promise value: the surface class of each of the two sub-columns of
          h/2 columns, E (no solid), S (solid, no full column), P (a full
          column, not all), F (all solid); v = 4 c0 + c1, V = 16.
Under the support rule matter reaches the bottom of its sub-column, so a
non-empty upper sub-column needs a full column under it and a full one needs
all: compat[upper, lower] = (u == E or l == F or (u in {S, P} and l == P)) per
sub-column (scale-free; above the map E, below it F).

Cell p's joint conditional at temperature T:
    pi(u, v) ∝ exp(-(lam_c E_c(u) + E_par(u) + cost(u, v)) / T)
               * 1[v compatible with the v above and below]
  E_c    symmetric PCA'd 5x5 patch energy (texsyn.Analysis level h):
         sum over every map cell q of |proj(q) - NE[u_q]|^2, proj(q) =
         sum_s G_s[u_{q + delta_s}] - mean P, G_s = El P_s (slot s of the PCA);
         changing u_p touches the <= 25 cells q whose patch holds p
  E_par  |wrap(u - prolong(u_2h)(p))|^2 / (2 sigma^2), none at the top level
  cost   -log p(v | u): values of the windows within r h of u, smoothed
         toward the level's marginal p_h (cost_table)
Update: u ~ pibar(u) = exp(-(lam_c E_c + E_par) / T) * sum_{v allowed} exp(-cost(u, v) / T),
then v | u.  Exact (knn=None): over all m^2 coordinates.  knn=K: MH with an
independence proposal q(u) = (1 - eps) pibar(u) 1[u in S] / Z_S + eps / m^2,
S = the K nearest exemplar neighbourhoods (knn_table) of each 3x3 neighbour's
coherent coordinate (and of the parent target).  S depends only on the
neighbours, not on u_p, so the move keeps pibar invariant.
"""
from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np

from castlegen.legacy import texsyn

OFF5 = texsyn.OFF5
V = 16
TOP, BOT = 0, 15                                                           # E|E above the map, F|F below


def _gumbel_pick(logits, rng):
    """Gumbel-max per row of (k, S) logits (-inf = forbidden)."""
    assert np.isfinite(logits.max(1)).all(), "a site has no allowed value"
    return np.argmax(logits + rng.gumbel(size=logits.shape), 1)


def support_rule(ts):
    """(2, S, S) allowed pairs [0] left|right, [1] upper/lower: a solid tile
    needs a solid tile below."""
    solid = np.asarray(ts.solid, bool)
    S = len(solid)
    return np.stack([np.ones((S, S), bool), ~solid[:, None] | solid[None, :]])


# ---------------------------------------------------------------- promises
def _classify(cnt, nf, w):
    return np.where(cnt == 0, 0, np.where(nf == w, 3, np.where(nf > 0, 2, 1)))


def surface_class(s):
    """(..., h, h) bool solid block -> (...) value."""
    h = s.shape[-1]
    w = h // 2
    s = s.reshape(s.shape[:-1] + (2, w))                                   # (..., h, 2, w)
    c = _classify(s.sum((-3, -1)), s.all(-3).sum(-1), w)
    return c[..., 0] * 4 + c[..., 1]


def surface_compat():
    """(V, V) bool [upper, lower]."""
    c = np.arange(4)
    u, l = c[:, None], c[None, :]
    ok = (u == 0) | (l == 3) | (((u == 1) | (u == 2)) & (l == 2))
    a, b = np.arange(V) // 4, np.arange(V) % 4
    return ok[a[:, None], a[None, :]] & ok[b[:, None], b[None, :]]


def window_classes(solid, h):
    """(my * m,) value of the h x h window centred on every coordinate of a
    bounded exemplar (solid: (my, m) bool; rows clip, columns wrap).  h = 1:
    the tile's solid bit as E|E or F|F, so compat is the support rule."""
    if h == 1:
        return np.where(np.asarray(solid, bool), BOT, TOP).ravel()
    s = np.asarray(solid, np.int64)
    my, m = s.shape
    w = h // 2
    pad = np.concatenate([np.repeat(s[:1], h, 0), s, np.repeat(s[-1:], h, 0)], 0)
    cs = np.concatenate([np.zeros((1, m), np.int64), np.cumsum(pad, 0)], 0)
    top = np.arange(my) - h // 2 + h
    col = cs[top + h] - cs[top]                                            # (my, m) window rows' solids per column
    out = []
    for j in (0, 1):
        idx = (np.arange(m) - h // 2 + j * w) % m
        sums = []
        for a in (col, (col == h).astype(np.int64)):
            c2 = np.concatenate([np.zeros((my, 1), np.int64), np.cumsum(np.concatenate([a, a], 1), 1)], 1)
            sums.append(c2[:, idx + w] - c2[:, idx])
        out.append(_classify(sums[0], sums[1], w))
    return (out[0] * 4 + out[1]).ravel()


def cost_table(vals, my, m, rad, alpha=1.0):
    """(my * m, V) -log p(v | u): the values of the windows at coordinates
    within rad of u (rows inside the exemplar, columns wrapped), smoothed
    toward the marginal p_h: p = (c_uv + alpha p_h(v)) / (N_u + alpha)."""
    oh = np.eye(V)[vals.reshape(my, m)]                                    # (my, m, V)
    cy = np.concatenate([np.zeros((1, m, V)), np.cumsum(oh, 0)], 0)
    y = np.arange(my)
    col = cy[np.minimum(y + rad + 1, my)] - cy[np.maximum(y - rad, 0)]     # (my, m, V)
    if 2 * rad + 1 >= m:
        cnt = np.repeat(col.sum(1, keepdims=True), m, 1)
    else:
        cx = np.concatenate([np.zeros((my, 1, V)), np.cumsum(np.concatenate([col, col], 1), 1)], 1)
        idx = (np.arange(m) - rad) % m
        cnt = cx[:, idx + 2 * rad + 1] - cx[:, idx]
    cnt = cnt.reshape(-1, V)
    ph = oh.reshape(-1, V).mean(0)
    p = (cnt + alpha * ph) / (cnt.sum(1, keepdims=True) + alpha)
    with np.errstate(divide="ignore"):
        return -np.log(p)


def ex_weight(h, lam_ex=8.0):
    """lam_c derived from the tile-level target (to revisit, notes/history/joint_sampler.md):
    E_ex's weight at scale h, lam_ex h^2 / 25 (castlegen.legacy.exchain; lam_ex = 8
    gives the {4: 5.12, 8: 20.48, 16: 81.92} used so far).  With E_ex's
    features (texsyn.Analysis(..., w_sig=0.5, w_sock=0): one-hot / sqrt 2,
    blurred at h/2 like E_ex's kernels) E_c at level h is E_ex's h-term with
    the fill taken to be the exemplar windows at u, so lam_c E_c and cost
    are both in nats of the target."""
    return lam_ex * h * h / 25.0


def knn_table(NE, K, d_search=8, k_search=48, chunk=8192):
    """(N, K) the K nearest rows of NE to each row (itself first): a k-d tree
    on the first d_search PCA dimensions proposes k_search, re-scored on all."""
    from scipy.spatial import cKDTree
    _, cand = cKDTree(NE[:, :d_search]).query(NE[:, :d_search], k=k_search)
    out = np.empty((len(NE), K), np.int64)
    for a in range(0, len(NE), chunk):
        c = cand[a:a + chunk]
        d2 = ((NE[a:a + chunk, None] - NE[c]) ** 2).sum(-1)
        d2[c == np.arange(a, a + len(c))[:, None]] = -1.0                    # itself first
        out[a:a + chunk] = np.take_along_axis(c, np.argsort(d2, 1)[:, :K], 1)
    return out


# ------------------------------------------------------------------ setup
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache", "generic")
CACHE_VERSION = 1                                                          # bump when a cached table changes


def _digest(*arrays):
    d = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        d.update(str((a.dtype, a.shape)).encode())
        d.update(a.tobytes())
    return d.hexdigest()[:16]


def _cached(key, build):
    """build(), pickled under cache/generic/ by key (a repr-able tuple)."""
    path = os.path.join(CACHE, hashlib.sha1(repr((CACHE_VERSION, key)).encode()).hexdigest()[:16] + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = build()
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return obj


def analysis(ts, E, hs, n_pca=32, torus=False, **feat):
    """texsyn.Analysis(bounds="edge", pad=2 max(hs)) (torus: bounds=None) at
    the sides hs only, without C2, cached on disk; its key also caches
    JointLevel's tables."""
    E = np.asarray(E)
    pad = 0 if torus else 2 * max(hs)
    key = ("analysis", ts.name, _digest(E, ts.np_tables["Eh"], ts.np_tables["Ev"], np.asarray(ts.solid)),
           tuple(sorted(hs)), n_pca, pad, tuple(sorted(feat.items())))
    an = _cached(key, lambda: texsyn.Analysis(ts, E, n_pca=n_pca, bounds=None if torus else "edge", pad=pad, levels=hs, c2=False,
                                              **feat))
    an.key = key
    return an


# ---------------------------------------------------- connectivity promise
# v = O | L << 4.  O: the side set (bit d for side N, E, S, W) where the
# cell's nodes join the nodes across; O = 0: the cell holds no node (dead).
# L within O: the open sides whose neighbour is lower (downhill).  Bit E of a
# cell is bit W of its right neighbour (a seam; likewise S / N; L opposite
# across an open seam), sides off the map are closed.  The orientation is a
# descent certificate: acyclic, with every live cell but one (the root, the
# sink) having an open downhill side, so descending from any live cell ends
# at the root.  Top level: acyclic with at most one live sink (a global
# check; the level is small).  Level h refines level 2h per 2 x 2 group: the
# children are live iff the parent is, the children's O bits on a parent
# side OR to the parent's bit and their L bits there copy the parent's (a
# child descends into any block below its parent, anywhere: all of it
# reaches the root), the internal seams are acyclic, and every live child
# has a downhill side (under the root parent, L = 0: all but at most one).
# The whole orientation is then the lexicographic order of (parent key,
# sibling rank), so acyclic, with one sink; no block need be one component.
# h = 1: a node tile iff O != 0, a connecting socket on every side in O, and
# the tiles across an open seam join.  Then every node of the map is in one
# network.  The h = 1 conditions cost HARD per violation (0 once met), so a
# start that breaks them relaxes toward them; the others are never broken.
# cost(u, v) = cost(u, O): the exemplar says nothing about e.
HARD = 1e3
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPP = (2, 3, 0, 1)


def conn_tables(ts):
    """node (S,); Dh, Dv (S, S): nodes joined across a seam; sock (4, S):
    side d of the tile can join some node."""
    from castlegen.legacy import connmetrics as cm
    node = cm._node(ts)
    Dh = (np.asarray(ts.np_tables["Dh"]) > 0) & node[:, None] & node[None]
    Dv = (np.asarray(ts.np_tables["Dv"]) > 0) & node[:, None] & node[None]
    return node, Dh, Dv, np.stack([Dv.any(0), Dh.any(1), Dv.any(1), Dh.any(0)])


def conn_window_values(ts, E, h):
    """(m m,) side set of the h x h window centred on every coordinate of a
    torus exemplar: bit d iff a node on side d joins the node across it
    (connectivity.abstract of the window's state)."""
    from castlegen.legacy.quantities import connectivity as CQ
    cr = CQ.crossings(ts, E).astype(np.int64)                              # (4, m, m), torus
    m = E.shape[0]
    i = np.arange(m)

    def run(a, axis):                                                      # sum of h entries from i on, wrapped
        c = np.cumsum(np.concatenate([np.zeros_like(a.take([0], axis)), a, a], axis), axis)
        return c.take(i + h, axis) - c.take(i, axis)
    Y, X = np.meshgrid((i - h // 2) % m, (i - h // 2) % m, indexing="ij")  # window top-left per centre
    b = h - 1
    v = ((run(cr[0], 1)[Y, X] > 0).astype(np.int64) | (run(cr[1], 0)[Y, (X + b) % m] > 0) << 1
         | (run(cr[2], 1)[(Y + b) % m, X] > 0) << 2 | (run(cr[3], 0)[Y, X] > 0) << 3)
    return v.ravel()


def conn_cost_h1(tabs, Et):
    """(N, V) h = 1 cost: HARD per violated condition of the tile Et[u]."""
    node, _, _, sock = tabs
    v = np.arange(V)
    bits = (v[None] >> np.arange(4)[:, None]) & 1                          # (4, V)
    n = node[Et][:, None]
    viol = ((v[None] == 0) & n) | ((v[None] != 0) & ~n)
    return HARD * (viol + (bits[None] & ~sock[:, Et].T[:, :, None]).sum(1))


def level_tables(an, solid, h, r, alpha, knn, conn=None):
    """JointLevel's per-level tables: G (25, my m, k) slot projections, c0,
    NE, window values, cost, and the kNN table (None without knn).  conn: a
    tile set, whose connectivity promise replaces the surface class."""
    lv = an.levels[an.L - int(np.log2(h))]
    assert lv is not None and lv["h"] == h, f"level h = {h} not analysed"
    D = lv["El"].shape[-1]
    El = lv["El"].reshape(-1, D)
    P = lv["P"]
    NE = lv["NE"].astype(np.float32)
    if conn is not None:
        assert not an.bounded, "the connectivity promise needs a torus exemplar"
        vals = conn_window_values(conn, an.E, h)
        cost = (conn_cost_h1(conn_tables(conn), an.E.ravel()) if h == 1
                else cost_table(vals, an.my, an.m, int(round(r * h)), alpha))
    elif h == 1:                                                           # v is u's own solid bit
        vals = window_classes(solid[an.E], h)
        cost = np.where(np.arange(V)[None] == vals[:, None], 0.0, np.inf)
    else:
        vals = window_classes(solid[an.E], h)
        cost = cost_table(vals, an.my, an.m, int(round(r * h)), alpha)
    return dict(G=np.stack([El @ P[s * D:(s + 1) * D] for s in range(len(OFF5))]).astype(np.float32),
                c0=(lv["mean"] @ P).astype(np.float32), NE=NE, vals=vals, cost=cost,
                knn=None if knn is None else knn_table(NE, knn))


# ------------------------------------------------------------------ level
class JointLevel:
    """Exact single-cell joint (u, v) heat-bath at level h (module docstring).
    an: texsyn.Analysis(bounds="edge"); solid: (S,) bool per tile; n: map side;
    lam_c: weight of E_c; r: cost window radius in cells (r h tiles);
    sigma: E_par width in tiles (None: no parent term); use_v: False drops
    cost and compat (the plain coordinate MRF); knn: None (exact update) or
    K (MH, module docstring) with uniform weight eps; periodic: wrap the map
    horizontally (else free sides: no E_c term reaches off the map);
    loc: at h = 1, a tile set whose energy E_loc = E_pair / t_loc - sum logz
    (refheights.log_weight's, over the pairs inside the map) joins the
    conditional; fast: sweep() uses the batched colour sweeps (knn only),
    "numba": with generic_nb's kernel for their E_c; conn: a tile set whose
    connectivity promise (above) replaces the surface class (torus exemplar):
    u updates hold v fixed, and every sweep ends with a seam sweep on v."""

    def __init__(self, an, solid, h, n, lam_c=1.0, r=1.0, alpha=1.0, sigma=None, use_v=True, seed=0,
                 knn=None, eps=0.05, periodic=False, loc=None, t_loc=0.9, fast=False, conn=None, lam_d=4.0, rk=4):
        assert n % h == 0
        self.an, self.h, self.R, self.C = an, h, n // h, n // h
        self.torus = not an.bounded                                        # torus exemplar: no padding rows
        self.B = 0 if self.torus else 2
        self.my, self.m = an.my, an.m
        solid = np.asarray(solid, bool)
        build = lambda: level_tables(an, solid, h, r, alpha, knn, conn)
        key = getattr(an, "key", None)
        ckey = None if conn is None else ("conn", conn.name)
        tab = build() if key is None else _cached(("level", key, _digest(solid), h, r, alpha, knn, ckey), build)
        self.G, self.c0, self.NE, self.vals, self.cost, self.knn = (
            tab[k] for k in ("G", "c0", "NE", "vals", "cost", "knn"))
        self.compat = surface_compat()
        self.lam_c, self.sigma, self.use_v = float(lam_c), sigma, use_v
        self.rng = np.random.default_rng(seed)
        self.PX = self.rng.integers(0, self.m, (2 * self.B, self.C))
        self.cy, self.cx = np.divmod(np.arange(self.my * self.m), self.m)
        assert self.torus or an.pad >= 2 * h, "the padding rows need Analysis(pad >= 2 h)"
        self.pad_rows = np.array([0, h, self.my - 1 - h, self.my - 1])       # exemplar rows of the padding
        self.eps, self.periodic = eps, periodic
        self.Et = an.E.ravel()                                              # tile at each coordinate
        self.loc = None
        if loc is not None and h == 1:
            t = loc.np_tables
            self.loc = (np.asarray(t["Eh"], np.float64) / t_loc, np.asarray(t["Ev"], np.float64) / t_loc,
                        np.asarray(t["logz"], np.float64))
        self.fast = fast                                                    # False, True (numpy) or "numba"
        self._off5 = np.ascontiguousarray(np.asarray(OFF5, np.int64))
        assert not fast or knn is not None, "the batched sweeps are MH (knn)"
        assert not (fast and periodic and self.C % 5), "periodic batched sweeps need C % 5 == 0"
        self.conn = None if conn is None else conn_tables(conn)
        assert conn is None or use_v
        if self.conn is not None:
            y, x = np.divmod(np.arange(self.R * self.C), self.C)
            self.mask = ((y == 0) | (x == self.C - 1) << 1 | (y == self.R - 1) << 2
                         | (x == 0) << 3).reshape(self.R, self.C)                # sides off the map
            self.costM = self._marginal_costs()
            assert fast, "the connectivity promise samples (u, v) jointly in the batched sweeps"
        self.lam_d, self.rk = float(lam_d), rk
        self.root = (self.R // 2, self.C // 2)
        self.target = self.vparent = None
        self.U = self.V = None
        self.extra = None

    # ---- coordinates
    # E_c runs over the extended grid: B = 2 padding rows above the map
    # (exemplar rows 0, h: the continuation above its top) and below it (rows
    # my-1-h, my-1), whose own patches reach into the map like any cell's.
    # Their rows are fixed, their columns PX (2B, C) are sampled.  A torus
    # exemplar (Analysis(bounds=None)) has no padding (B = 0): the map's top
    # and bottom are free like its sides.
    B = 2

    def _ext(self, U):
        if not self.B:
            return U
        return np.concatenate([self.pad_rows[:2, None] * self.m + self.PX[:2], U,
                               self.pad_rows[2:, None] * self.m + self.PX[2:]], 0)

    def _wy(self, y):
        """Exemplar row of a row coordinate: clipped (bounded) or wrapped (torus)."""
        return y % self.my if self.torus else np.clip(y, 0, self.my - 1)

    def _dy(self, d):
        """Row offset: as is (bounded) or the shortest one on the torus."""
        return (d + self.my // 2) % self.my - self.my // 2 if self.torus else d

    def _cont(self, u, s):
        """Coordinate(s) coherent with u at slot s: u + h delta_s (rows clipped or wrapped)."""
        dy, dx = OFF5[s]
        uy, ux = np.divmod(u, self.m)
        return self._wy(uy + dy * self.h) * self.m + (ux + dx * self.h) % self.m

    def _off(self, x):
        return not self.periodic and not 0 <= x < self.C

    def _slot(self, X, qy, qx, s):
        """Coordinate in slot s of q's patch on the extended grid X: slots off
        a free side or beyond the padding continue u_q coherently (the
        slot's term cancels; beyond the padding it is the edge continuation)."""
        dy, dx = OFF5[s]
        y, x = qy + dy, qx + dx
        if self._off(x) or not 0 <= y < len(X):
            return self._cont(X[qy, qx], s)
        return X[y, x % self.C]

    def energy_c(self, U=None):
        """E_c, vectorised over the extended grid (energy_c_ref: the loop)."""
        X = self._ext(self.U if U is None else U)
        NR, C = X.shape
        qy, qx = np.divmod(np.arange(NR * C), C)
        uq = X[qy, qx]
        proj = np.broadcast_to(-self.c0, (len(uq), self.c0.size)).astype(np.float64)
        for s, (dy, dx) in enumerate(OFF5):
            y, x = qy + dy, qx + dx
            off = (y < 0) | (y >= NR)
            if not self.periodic:
                off = off | (x < 0) | (x >= C)
            proj = proj + self.G[s, np.where(off, self._cont(uq, s), X[np.clip(y, 0, NR - 1), x % C])]
        return float(((proj - self.NE[uq]) ** 2).sum())

    def energy_c_ref(self, U=None):
        X = self._ext(self.U if U is None else U)
        tot = 0.0
        for qy in range(len(X)):
            for qx in range(self.C):
                proj = -self.c0 + sum(self.G[s, self._slot(X, qy, qx, s)] for s in range(len(OFF5)))
                tot += float(((proj - self.NE[X[qy, qx]]) ** 2).sum())
        return tot

    def cond_c(self, py, px, U=None, cand=None):
        """(my m,) E_c with U[p] = each coordinate (or (len(cand),) for the
        coordinates cand), exact up to a constant shift."""
        return self._cond_ext(py + self.B, px, self._ext(self.U if U is None else U), cand)

    def _cond_ext(self, py, px, X, cand=None):
        cand = np.arange(len(self.NE)) if cand is None else np.asarray(cand)
        hits = {}
        for s, (dy, dx) in enumerate(OFF5):
            qy, qx = py - dy, px - dx
            if 0 <= qy < len(X) and not self._off(qx):
                hits.setdefault((qy, qx % self.C), []).append(s)
        out = np.zeros(len(cand), np.float32)
        for (qy, qx), S in hits.items():
            me = (qy, qx) == (py, px)
            base, var = -self.c0.copy(), []
            for s, (dy, dx) in enumerate(OFF5):
                if s in S:
                    var.append(self.G[s, cand])
                elif me and (self._off(qx + dx) or not 0 <= qy + dy < len(X)):   # continues u_p itself
                    var.append(self.G[s, self._cont(cand, s)])
                else:
                    base = base + self.G[s, self._slot(X, qy, qx, s)]
            proj = base[None] + sum(var)
            tgt = self.NE[cand] if me else self.NE[X[qy, qx]][None]
            out += ((proj - tgt) ** 2).sum(1)
        return out

    def prolong(self, Up):
        """(R, C) targets from the parent grid (R/2, C/2): the centres of the
        parent window's four h x h quarters (parent centre c, top-left c - h)."""
        h = self.h
        out = np.empty((self.R, self.C), np.int64)
        for dy in (0, 1):
            for dx in (0, 1):
                py, px = np.divmod(Up, self.m)
                y = self._wy(py - h + dy * h + h // 2)
                x = (px - h + dx * h + h // 2) % self.m
                out[dy::2, dx::2] = y * self.m + x
        return out

    def e_par(self, py, px):
        if self.target is None or self.sigma is None:
            return 0.0
        ty, tx = divmod(int(self.target[py, px]), self.m)
        dy = self._dy(self.cy - ty)
        dx = (self.cx - tx + self.m // 2) % self.m - self.m // 2
        return (dy ** 2 + dx ** 2) / (2.0 * self.sigma ** 2)

    def e_loc(self, py, px, cand=None):
        """(len(cand),) E_loc terms of cell p's tile set to E[cand] (h = 1)."""
        cand = np.arange(len(self.NE)) if cand is None else np.asarray(cand)
        if self.loc is None:
            return np.zeros(len(cand))
        Eh, Ev, logz = self.loc
        t, T = self.Et[cand], self.Et[self.U]
        e = -logz[t]
        if px > 0:
            e = e + Eh[T[py, px - 1], t]
        if px + 1 < self.C:
            e = e + Eh[t, T[py, px + 1]]
        if py > 0:
            e = e + Ev[T[py - 1, px], t]
        if py + 1 < self.R:
            e = e + Ev[t, T[py + 1, px]]
        return e

    def _joins(self, d, t, n):
        """Tile(s) t joins tile(s) n on its side d."""
        _, Dh, Dv, _ = self.conn
        return (Dv[n, t], Dh[t, n], Dv[t, n], Dh[n, t])[d]

    def e_seam(self, py, px, cand=None):
        """(len(cand),) h = 1: HARD per open seam of p whose tiles would not join."""
        cand = np.arange(len(self.NE)) if cand is None else np.asarray(cand)
        e = np.zeros(len(cand))
        if self.conn is None or self.h != 1:
            return e
        t, T, v = self.Et[cand], self.Et[self.U], int(self.V[py, px]) & 15
        for d, (dy, dx) in enumerate(DIRS):
            if (v >> d) & 1:                                               # open seams lie inside the map
                e = e + HARD * ~self._joins(d, t, T[py + dy, px + dx])
        return e

    # ---- promises
    def _allowed(self, py, px):
        if not self.use_v:
            return np.ones(V, bool)
        if self.conn is not None:                                          # u updates hold v fixed
            return np.arange(V) == self.V[py, px] & 15
        up = self.V[py - 1, px] if py else TOP
        dn = self.V[py + 1, px] if py + 1 < self.R else BOT
        return self.compat[up] & self.compat[:, dn]

    def _cost(self):
        return self.cost if self.use_v else np.zeros_like(self.cost)

    def init(self, U, parent=None, vparent=None, kparent=None):
        """U: (R, C) coordinates; parent: (R/2, C/2) parent coordinates, which
        become E_par's targets.  v is drawn bottom-up given u.  E is compatible
        with anything below, so a value exists whenever the cost is soft; with a
        hard cost (h = 1) a u with no allowed value moves to the nearest
        coordinate that has one (a valid start; the sweeps then sample)."""
        self.U = np.array(U, np.int64)
        self.target = None if parent is None else self.prolong(np.asarray(parent))
        if self.conn is not None:
            self.vparent = None if vparent is None else np.asarray(vparent, np.int64)
            if self.vparent is None:                                       # top: one parent, keys = ranks
                self.RK, self.PK = self.R + self.C, np.zeros((self.R, self.C), np.int64)
                self.pmul = 0
                self.PO = None
            else:
                self.kparent = np.asarray(kparent, np.int64)
                self.RK, self.pmul = self.rk, 2
                self.PK = np.repeat(np.repeat(self.kparent, 2, 0), 2, 1)
                self.PO = np.repeat(np.repeat(self.vparent & 15, 2, 0), 2, 1)
            self.V = self._conn_init()
            return self
        self.V = np.full((self.R, self.C), TOP, np.int64)
        cost = self._cost()
        fin = {}
        for y in range(self.R - 1, -1, -1):
            dn = self.V[y + 1] if y + 1 < self.R else np.full(self.C, BOT)
            ok = self.compat[:, dn].T if self.use_v else np.ones((self.C, V), bool)    # (C, V)
            for x in np.flatnonzero(~np.isfinite(np.where(ok, cost[self.U[y]], np.inf)).any(1)):
                d = int(dn[x])
                if d not in fin:
                    fin[d] = np.flatnonzero(np.isfinite(np.where(ok[x][None], cost, np.inf)).any(1))
                uy, ux = divmod(int(self.U[y, x]), self.m)
                dx = (self.cx[fin[d]] - ux + self.m // 2) % self.m - self.m // 2
                self.U[y, x] = fin[d][np.argmin((self.cy[fin[d]] - uy) ** 2 + dx ** 2)]
            self.V[y] = _gumbel_pick(np.where(ok, -cost[self.U[y]], -np.inf), self.rng)
        return self

    def _marginal_costs(self):
        """(16, N, V) cost per off-map side mask M: a cell's sides off the map
        are closed, so its p(v | u) sums the exemplar's over those bits (the
        window may cross them); v with a bit in M is impossible.  h = 1: the
        tile's own conditions, the same for every M."""
        if self.h == 1:
            return np.broadcast_to(self.cost, (16,) + self.cost.shape)
        with np.errstate(over="ignore"):
            p = np.exp(-self.cost)
        v = np.arange(V)
        out = np.full((16,) + self.cost.shape, np.inf)
        for M in np.unique(self.mask):
            pm = np.zeros_like(p)
            for w in range(V):                                              # w -> w with M's bits cleared
                pm[:, w & ~M] += p[:, w]
            with np.errstate(divide="ignore"):
                out[M] = np.where((v & M) == 0, -np.log(pm), np.inf)
        return out

    def _conn_init(self):
        """A consistent start: top level every in-map seam open, r the Manhattan
        distance to the root; below, every parent side's children open on it,
        every internal seam of a live parent open, r = 4 pot + child index (pot
        0 on a side into a lower parent, else 1; under the root parent 0)."""
        R, C = self.R, self.C
        y, x = np.divmod(np.arange(R * C), C)
        inmap = (y > 0, x < C - 1, y < R - 1, x > 0)
        O = np.zeros(R * C, np.int64)
        if self.vparent is None:
            for d in range(4):
                O |= inmap[d].astype(np.int64) << d
            r = np.abs(y - self.root[0]) + np.abs(x - self.root[1])
            return (O | r << 4).reshape(R, C)
        PO, Kp = self.PO.ravel(), self.kparent
        pot = np.ones(R * C, np.int64)
        for d, (dy, dx) in enumerate(DIRS):
            ext = self._ext_side(y, x, d)
            b = np.where(ext, (PO >> d) & 1, PO != 0) & inmap[d]
            O |= b.astype(np.int64) << d
            gy, gx = np.clip(y // 2 + dy, 0, R // 2 - 1), np.clip(x // 2 + dx, 0, C // 2 - 1)
            pot = np.where(ext & (b == 1) & (Kp[gy, gx] < Kp[y // 2, x // 2]), 0, pot)
        down = np.zeros(R * C, bool)                                        # the parent has a lower side
        for d, (dy, dx) in enumerate(DIRS):
            gy, gx = np.clip(y // 2 + dy, 0, R // 2 - 1), np.clip(x // 2 + dx, 0, C // 2 - 1)
            down |= (((PO >> d) & 1) == 1) & (Kp[gy, gx] < Kp[y // 2, x // 2])
        pot = np.where(down, pot, 0)
        r = np.minimum(4 * pot + (y % 2) * 2 + x % 2, self.RK - 1)
        return (O | r << 4).reshape(R, C)

    @staticmethod
    def _ext_side(y, x, d):
        """Side d of cell (y, x) lies on its parent's side d."""
        return (y % 2 == 0, x % 2 == 1, y % 2 == 1, x % 2 == 0)[d]

    @staticmethod
    def _sib(y, x, d):
        """The sibling on the same parent side d."""
        return (y, x ^ 1) if d in (0, 2) else (y ^ 1, x)

    def keys(self):
        """Distance d = 2 D_parent + r (top: r): a child's range [2 D, 2 D + RK)
        overlaps its neighbouring parents' when RK > 2."""
        return self.PK * self.pmul + (self.V >> 4)

    def _lowbits(self, O, K):
        """(R, C, 4): side d open and the neighbour's key lower."""
        R, C = O.shape
        out = np.zeros((R, C, 4), bool)
        for d, (dy, dx) in enumerate(DIRS):
            ny, nx = np.clip(np.arange(R) + dy, 0, R - 1), np.clip(np.arange(C) + dx, 0, C - 1)
            out[..., d] = (((O >> d) & 1) == 1) & (K[ny][:, nx] < K)
        return out

    def _vtable(self, py, px):
        """(k, 16, RK) the u-independent energy of v = (O, r) at cells (py, px)
        of one colour (5 apart: no shared neighbour), each neighbour's facing
        bit following O: the neighbours' costs, the parent conditions (hard)
        and lam_d per live non-root cell of c and its neighbours without an
        open side into a lower key."""
        R, C, RK, V_ = self.R, self.C, self.RK, self.V
        k = len(py)
        Og, Kg = V_ & 15, self.keys()
        low = self._lowbits(Og, Kg)
        Os = np.arange(16)
        kc = (self.PK[py, px] * self.pmul)[:, None] + np.arange(RK)[None]  # (k, RK)
        Mc = self.mask[py, px]
        B = np.where((Os[None] & Mc[:, None]) == 0, 0.0, np.inf)[:, :, None] + np.zeros((1, 1, RK))
        croot = (py == self.root[0]) & (px == self.root[1])
        c_low = np.zeros((k, 16, RK), bool)
        par = self.vparent is not None
        for d, (dy, dx) in enumerate(DIRS):
            bd = (Os >> d) & 1                                              # (16,)
            inm = ((Mc >> d) & 1) == 0
            ny, nx = np.clip(py + dy, 0, R - 1), np.clip(px + dx, 0, C - 1)
            o = OPP[d]
            kn = Kg[ny, nx]
            c_low |= (bd[None, :, None] == 1) & (inm[:, None, None] & (kn[:, None] < kc)[:, None, :])
            On = (Og[ny, nx] & ~(1 << o))[:, None] | (bd[None] << o)        # (k, 16)
            with np.errstate(invalid="ignore"):
                en = self.costM[self.mask[ny, nx][:, None], self.U[ny, nx][:, None], On]
            if par:
                en = en + np.where((self.PO[ny, nx][:, None] == 0) & (On != 0), np.inf, 0.0)
                ext = self._ext_side(ny, nx, o)
                sy, sx = self._sib(ny, nx, o)
                sb = (Og[sy, sx] >> o) & 1
                en = en + np.where(ext[:, None] & ((bd[None] | sb[:, None]) != ((self.PO[ny, nx] >> o) & 1)[:, None]),
                                   np.inf, 0.0)
            others = low[ny, nx].copy()
            others[:, o] = False
            n_low = others.any(1)[:, None, None] | ((bd[None, :, None] == 1) & (kc[:, None, :] < kn[:, None, None]))
            nroot = (ny == self.root[0]) & (nx == self.root[1])
            n_bad = (On != 0)[:, :, None] & ~n_low & ~nroot[:, None, None]
            B = B + np.where(inm[:, None, None], en[:, :, None] + self.lam_d * n_bad, 0.0)
            if par:
                ext = self._ext_side(py, px, d)
                sy, sx = self._sib(py, px, d)
                sb = (Og[sy, sx] >> d) & 1
                B = B + np.where((ext[:, None] & ((bd[None] | sb[:, None]) != ((self.PO[py, px] >> d) & 1)[:, None]))
                                 [:, :, None], np.inf, 0.0)
        if par:
            B = B + np.where(((self.PO[py, px] == 0)[:, None] & (Os[None] != 0))[:, :, None], np.inf, 0.0)
        return B + self.lam_d * ((Os[None, :, None] != 0) & ~c_low & ~croot[:, None, None])

    def _atable(self, py, px, allc):
        """(k, M, 16) the u-dependent energy of O: cost(u, O); h = 1 also HARD
        per open seam whose tiles do not join."""
        A = self.costM[self.mask[py, px][:, None], allc].astype(np.float64)
        if self.h == 1:
            t, Tg = self.Et[allc], self.Et[self.U]
            Os = np.arange(16)
            for d, (dy, dx) in enumerate(DIRS):
                n = Tg[np.clip(py + dy, 0, self.R - 1), np.clip(px + dx, 0, self.C - 1)][:, None]
                A = A + HARD * (~self._joins(d, t, n))[:, :, None] * ((Os >> d) & 1)[None, None]
        return A

    def _components(self):
        """Components of the live cells under the open seams."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        Og = self.V & 15
        live = (Og != 0).ravel()
        R, C = Og.shape
        idx = np.arange(R * C).reshape(R, C)
        eh, ev = ((Og[:, :-1] >> 1) & 1).astype(bool), ((Og[:-1] >> 2) & 1).astype(bool)
        a = np.concatenate([idx[:, :-1][eh], idx[:-1][ev]])
        b = np.concatenate([idx[:, 1:][eh], idx[1:][ev]])
        _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(R * C, R * C)), directed=False)
        return len(np.unique(lab[live]))

    def conn_violations(self):
        """{seams, parent, descent, h1, comps}: broken hard conditions, live
        non-root cells without a descent (soft), the h = 1 tile conditions, and
        the components of the live cells (1: one network whatever the keys)."""
        Vv, R, C = self.V, self.R, self.C
        Og = Vv & 15
        out = dict(seams=int((((Og[:, :-1] >> 1) & 1) != ((Og[:, 1:] >> 3) & 1)).sum()
                             + (((Og[:-1] >> 2) & 1) != (Og[1:] & 1)).sum()
                             + (Og[0] & 1).sum() + ((Og[-1] >> 2) & 1).sum()
                             + ((Og[:, -1] >> 1) & 1).sum() + ((Og[:, 0] >> 3) & 1).sum()), parent=0)
        if self.vparent is not None:
            y, x = np.mgrid[:R, :C]
            bad = (self.PO == 0) & (Og != 0)
            for d in range(4):
                sy, sx = self._sib(y, x, d)
                bad |= self._ext_side(y, x, d) & ((((Og >> d) | (Og[sy, sx] >> d)) & 1) != ((self.PO >> d) & 1))
            out["parent"] = int(bad.sum())
        root = np.zeros((R, C), bool)
        root[self.root] = True
        out["descent"] = int(((Og != 0) & ~self._lowbits(Og, self.keys()).any(2) & ~root).sum())
        out["h1"] = 0
        if self.h == 1:
            T = self.Et[self.U]
            out["h1"] = int((self.cost[self.U, Og] > 0).sum())
            for d, sl_a, sl_b in ((1, (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                                  (2, (slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
                on = ((Og[sl_a] >> d) & 1).astype(bool)
                out["h1"] += int((on & ~self._joins(d, T[sl_a], T[sl_b])).sum())
        out["comps"] = self._components()
        return out

    def _logpi(self, py, px, T, cand=None):
        """log pibar over all coordinates (or cand), and the (., V) v logits."""
        idx = slice(None) if cand is None else cand
        ep = self.e_par(py, px)
        e = self.lam_c * self.cond_c(py, px, cand=cand) + (ep[idx] if np.ndim(ep) else ep)
        if self.loc is not None:
            e = e + self.e_loc(py, px, cand)
        e = e + self.e_seam(py, px, cand)
        with np.errstate(invalid="ignore", divide="ignore"):
            cost = self._cost() if self.conn is None else self.costM[self.mask[py, px]]
            lv = np.where(self._allowed(py, px)[None], -cost[idx] / T, -np.inf)
            mx = lv.max(1, keepdims=True)
            logz = (mx + np.log(np.exp(lv - np.where(np.isfinite(mx), mx, 0)).sum(1, keepdims=True)))[:, 0]
        return -e / T + logz, lv

    def candidates(self, py, px):
        """S: the K nearest neighbourhoods of each 3x3 neighbour's coherent
        coordinate u_q - h (q - p) (rows clipped, columns wrapped) and of the
        parent target.  Independent of u_p."""
        X, base = self._ext(self.U), []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and not self._off(px + dx):
                    qy, qx = divmod(int(X[py + self.B + dy, (px + dx) % self.C]), self.m)
                    base.append(int(self._wy(qy - dy * self.h)) * self.m + (qx - dx * self.h) % self.m)
        if self.target is not None:
            base.append(int(self.target[py, px]))
        return np.unique(self.knn[base])

    def update(self, py, px, T=1.0):
        if self.knn is not None:
            return self._update_mh(py, px, T)
        lp, lv = self._logpi(py, px, T)
        u = _gumbel_pick((lp - np.nanmax(np.where(np.isfinite(lp), lp, np.nan)))[None], self.rng)[0]
        self.U[py, px] = u
        if self.conn is None:
            self.V[py, px] = _gumbel_pick(lv[u][None], self.rng)[0]

    def _update_mh(self, py, px, T):
        N, rng = len(self.NE), self.rng
        S = self.candidates(py, px)
        u0, uu = int(self.U[py, px]), int(rng.integers(N))
        allc = np.unique(np.concatenate([S, [u0, uu]]))
        lp, lv = self._logpi(py, px, T, allc)
        lpS = np.where(np.isin(allc, S), lp, -np.inf)
        mS = lpS.max()
        with np.errstate(divide="ignore"):
            lq = np.logaddexp(np.log1p(-self.eps) + lpS - mS - np.log(np.exp(lpS - mS).sum())
                              if np.isfinite(mS) else np.full(len(allc), -np.inf), np.log(self.eps / N))
        if rng.random() < self.eps or not np.isfinite(mS):
            j = int(np.searchsorted(allc, uu))
        else:
            j = int(_gumbel_pick((lpS - mS)[None], rng)[0])
        i = int(np.searchsorted(allc, u0))
        if np.log(rng.random()) < lp[j] - lp[i] + lq[i] - lq[j]:
            i = j
        self.U[py, px] = allc[i]
        if self.conn is None:
            self.V[py, px] = _gumbel_pick(lv[i][None], rng)[0]

    def update_pad(self, i, px, T=1.0):
        """Padding cell (row i of PX, column px): its column, exactly."""
        X = self._ext(self.U)
        ey = i if i < self.B else self.R + i
        e = self.lam_c * self._cond_ext(ey, px, X, self.pad_rows[i] * self.m + np.arange(self.m))
        self.PX[i, px] = _gumbel_pick((-(e - e.min()) / T)[None], self.rng)[0]

    def sweep(self, T=1.0):
        if self.fast:
            self.sweep_fast(T)
        else:
            n = (self.R + 2 * self.B) * self.C
            for k in self.rng.permutation(n):
                ey, px = divmod(int(k), self.C)
                if self.B <= ey < self.B + self.R:
                    self.update(ey - self.B, px, T)
                else:
                    self.update_pad(ey if ey < self.B else ey - self.R, px, T)

    # ---- batched colour sweeps: the fast path.  The single-cell methods above
    # (cond_c, e_loc, candidates, _update_mh, update_pad) are the reference;
    # these vectorise them over the cells of one colour (extended-grid cells
    # 5 apart in both directions share no E_c term, no E_loc pair and no
    # seam), and tests/legacy/test_generic.py checks them against it.  Edit the
    # reference first, then mirror the change here.
    def _cond_batch(self, ey, ex, cand, X):
        """(k, M) E_c for cells (ey, ex) of the extended grid X (one colour)
        with u set to each of cand (k, M), up to a per-cell constant."""
        if self.fast == "numba":
            from castlegen.legacy import generic_nb
            return generic_nb.cond_batch(X, ey.astype(np.int64), ex.astype(np.int64), np.asarray(cand, np.int64),
                                         self.G, self.c0, self.NE, self._off5, self.h, self.m, self.my,
                                         bool(self.periodic), bool(self.torus))
        NR, C, G = len(X), self.C, self.G
        ctr = int(np.flatnonzero((np.asarray(OFF5) == 0).all(1))[0])
        out = np.zeros(cand.shape, np.float32)
        for sh, (dy, dx) in enumerate(OFF5):                                # q = p - delta: q's slot sh is p
            qy, qx = ey - dy, ex - dx
            ok = (qy >= 0) & (qy < NR) & (self.periodic | ((qx >= 0) & (qx < C)))
            qy, qx = np.clip(qy, 0, NR - 1), qx % C
            uq = X[qy, qx]
            me = sh == ctr
            base = np.broadcast_to(-self.c0, (len(ey), self.c0.size)).copy()
            offs = []
            for s2, (dy2, dx2) in enumerate(OFF5):
                if s2 == sh:
                    continue
                y2, x2 = qy + dy2, qx + dx2
                off = (y2 < 0) | (y2 >= NR)
                if not self.periodic:
                    off = off | (x2 < 0) | (x2 >= C)
                coord = np.where(off, self._cont(uq, s2), X[np.clip(y2, 0, NR - 1), x2 % C])
                if me:                                                      # off slots continue u_p itself
                    base += np.where(off[:, None], 0.0, G[s2, coord])
                    if off.any():
                        offs.append((s2, off))
                else:
                    base += G[s2, coord]
            g = G[sh, cand]                                                 # (k, M, D)
            if me:
                proj = base[:, None] + g
                for s2, off in offs:
                    proj += off[:, None, None] * G[s2, self._cont(cand, s2)]
                out += ((proj - self.NE[cand]) ** 2).sum(-1)
            else:                                                           # |a + g|^2 - |a|^2, a = base - NE[u_q]
                a = base - self.NE[uq]
                out += ok[:, None] * (2.0 * np.einsum("kmd,kd->km", g, a) + (g ** 2).sum(-1))
        return out

    def _loc_batch(self, py, px, cand):
        Eh, Ev, logz = self.loc
        T, t = self.Et[self.U], self.Et[cand]
        R, C = self.R, self.C
        e = -logz[t]
        for ny, nx, f in ((py, px - 1, lambda n: Eh[n, t]), (py, px + 1, lambda n: Eh[t, n]),
                          (py - 1, px, lambda n: Ev[n, t]), (py + 1, px, lambda n: Ev[t, n])):
            ok = (ny >= 0) & (ny < R) & (nx >= 0) & (nx < C)
            n = T[np.clip(ny, 0, R - 1), np.clip(nx, 0, C - 1)][:, None]
            e = e + np.where(ok[:, None], f(n), 0.0)
        return e

    def _seam_batch(self, py, px, cand):
        T, t, v = self.Et[self.U], self.Et[cand], self.V[py, px]
        e = np.zeros(cand.shape)
        for d, (dy, dx) in enumerate(DIRS):
            on = ((v >> d) & 1).astype(bool)[:, None]
            n = T[np.clip(py + dy, 0, self.R - 1), np.clip(px + dx, 0, self.C - 1)][:, None]
            e = e + HARD * (on & ~self._joins(d, t, n))
        return e

    def _mh_batch(self, py, px, X, T):
        """Batched _update_mh at map cells (py, px) (one colour).  The
        candidate list may repeat coordinates: the proposal is then weighted by
        multiplicity, q(u) = (1 - eps) mult(u) pibar(u) / Z + eps / N."""
        k, N, rng, h, m = len(py), len(self.NE), self.rng, self.h, self.m
        ey, NR = py + self.B, len(X)
        bases, bok = [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    y, x = ey + dy, px + dx
                    ok = (y >= 0) & (y < NR) & (self.periodic | ((x >= 0) & (x < self.C)))
                    qy, qx = np.divmod(X[np.clip(y, 0, NR - 1), x % self.C], m)
                    bases.append(self._wy(qy - dy * h) * m + (qx - dx * h) % m)
                    bok.append(ok)
        if self.target is not None:
            bases.append(self.target[py, px])
            bok.append(np.ones(k, bool))
        K = self.knn.shape[1]
        L = self.knn[np.stack(bases, 1)].reshape(k, -1)                     # (k, M)
        Lok = np.repeat(np.stack(bok, 1), K, 1)
        u0, uu = self.U[py, px], rng.integers(0, N, k)
        allc = np.concatenate([L, u0[:, None], uu[:, None]], 1)             # (k, M + 2)
        e = self.lam_c * self._cond_batch(ey, px, allc, X)
        if self.target is not None and self.sigma is not None:
            ty, tx = np.divmod(self.target[py, px], m)
            dy = self._dy(self.cy[allc] - ty[:, None])
            dx = (self.cx[allc] - tx[:, None] + m // 2) % m - m // 2
            e = e + (dy ** 2 + dx ** 2) / (2.0 * self.sigma ** 2)
        if self.loc is not None:
            e = e + self._loc_batch(py, px, allc)
        if self.extra is not None:                                          # an external term on (cell, u)
            e = e + self.extra(py, px, allc)
        if self.conn is not None:                                          # v = (O, r) summed out jointly
            A = self._atable(py, px, allc)                                  # (k, M + 2, 16)
            Bt = self._vtable(py, px)                                       # (k, 16, RK)
            with np.errstate(invalid="ignore", divide="ignore"):
                mb = Bt.min(2, keepdims=True)
                mb = np.where(np.isfinite(mb), mb, 0.0)
                Bo = mb[..., 0] - T * np.log(np.exp(-(Bt - mb) / T).sum(2))   # (k, 16)
            lv = -(A + Bo[:, None]) / T
            allowed = np.ones((k, V), bool)
        elif self.use_v:
            up = np.where(py > 0, self.V[np.maximum(py - 1, 0), px], TOP)
            dn = np.where(py + 1 < self.R, self.V[np.minimum(py + 1, self.R - 1), px], BOT)
            allowed = self.compat[up] & self.compat[:, dn].T                   # (k, V)
        else:
            allowed = np.ones((k, V), bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.conn is None:
                lv = np.where(allowed[:, None], -self._cost()[allc] / T, -np.inf)   # (k, M + 2, V)
            mx = lv.max(2, keepdims=True)
            logz = (mx + np.log(np.exp(lv - np.where(np.isfinite(mx), mx, 0)).sum(2, keepdims=True)))[..., 0]
            lp = -e / T + logz
            M = L.shape[1]
            lpS = np.where(Lok, lp[:, :M], -np.inf)
            mS = lpS.max(1)
            fin = np.isfinite(mS)
            lZ = np.where(fin, mS + np.log(np.exp(lpS - np.where(fin, mS, 0)[:, None]).sum(1)), -np.inf)

            def lq(col):
                u, l = allc[np.arange(k), col], lp[np.arange(k), col]
                mult = ((L == u[:, None]) & Lok).sum(1)
                return np.logaddexp(np.log1p(-self.eps) + np.log(mult) + l - lZ, np.log(self.eps / N))
            jS = np.argmax(np.where(fin[:, None], lpS, 0.0) + rng.gumbel(size=lpS.shape), 1)
            j = np.where((rng.random(k) < self.eps) | ~fin, M + 1, jS)
            i0 = np.full(k, M)
            acc = np.log(rng.random(k)) < lp[np.arange(k), j] - lp[np.arange(k), i0] + lq(i0) - lq(j)
        col = np.where(acc, j, i0)
        self.U[py, px] = allc[np.arange(k), col]
        lvc = lv[np.arange(k), col]
        vnew = np.argmax(np.where(np.isfinite(lvc), lvc, -np.inf) + rng.gumbel(size=lvc.shape), 1)
        if self.conn is None:
            self.V[py, px] = vnew
            return
        ok = np.isfinite(lvc).any(1)                                        # else keep v
        py, px, O = py[ok], px[ok], vnew[ok]
        lr = -Bt[ok, O] / T
        r = np.argmax(np.where(np.isfinite(lr), lr, -np.inf) + rng.gumbel(size=lr.shape), 1)
        self.V[py, px] = O | r << 4
        for d, (dy, dx) in enumerate(DIRS):                                 # the neighbours' facing bits follow
            inm = ((self.mask[py, px] >> d) & 1) == 0
            ny, nx, o = py[inm] + dy, px[inm] + dx, OPP[d]
            self.V[ny, nx] = (self.V[ny, nx] & ~(1 << o)) | (((O[inm] >> d) & 1) << o)

    def _pad_batch(self, ey, px, X, T):
        """Batched update_pad at padding cells (ey, px) of the extended grid."""
        i = np.where(ey < self.B, ey, ey - self.R)
        cand = self.pad_rows[i][:, None] * self.m + np.arange(self.m)[None]
        e = self.lam_c * self._cond_batch(ey, px, cand, X)
        g = -(e - e.min(1, keepdims=True)) / T
        self.PX[i, px] = np.argmax(g + self.rng.gumbel(size=g.shape), 1)

    def sweep_fast(self, T=1.0):
        NR = self.R + 2 * self.B
        EY, EX = np.divmod(np.arange(NR * self.C), self.C)
        for c in self.rng.permutation(25):
            sel = (EY % 5 == c // 5) & (EX % 5 == c % 5)
            ey, ex = EY[sel], EX[sel]
            X = self._ext(self.U)
            mp = (ey >= self.B) & (ey < self.B + self.R)
            if mp.any():
                self._mh_batch(ey[mp] - self.B, ex[mp], X, T)
            if (~mp).any():
                self._pad_batch(ey[~mp], ex[~mp], X, T)

    def run(self, sweeps, T=1.0, T_hot=None, log=None):
        """Anneal geometrically from T_hot to T over the first half, then at T."""
        hot = sweeps // 2
        for s in range(sweeps):
            t = T * (T_hot / T) ** (1 - s / hot) if T_hot and s < hot else T
            self.sweep(t)
            if log:
                log(s, t, self)
        return self

    def energy_parts(self):
        """{c: lam_c E_c, par: E_par, loc: E_loc, cost: cost(U, V)}."""
        out = dict(c=self.lam_c * self.energy_c(), par=0.0, loc=0.0, cost=0.0)
        if self.target is not None and self.sigma is not None:
            ty, tx = np.divmod(self.target, self.m)
            uy, ux = np.divmod(self.U, self.m)
            dx = (ux - tx + self.m // 2) % self.m - self.m // 2
            out["par"] = float(((self._dy(uy - ty) ** 2 + dx ** 2) / (2.0 * self.sigma ** 2)).sum())
        if self.loc is not None:
            Eh, Ev, logz = self.loc
            T = self.Et[self.U]
            out["loc"] = float(Eh[T[:, :-1], T[:, 1:]].sum() + Ev[T[:-1], T[1:]].sum() - logz[T].sum())
        if self.use_v:
            out["cost"] = float((self.cost[self.U, self.V] if self.conn is None
                                 else self.costM[self.mask, self.U, self.V & 15]).sum())
        if self.conn is not None:
            cv = self.conn_violations()
            out["descent"] = self.lam_d * cv["descent"]
            if self.h == 1:
                out["seam"] = HARD * (cv["h1"] - int((self.cost[self.U, self.V & 15] > 0).sum()))
        return out

    def energy(self):
        """lam_c E_c + E_par + E_loc + cost(U, V); inf if a seam is incompatible."""
        if self.use_v and not self.valid():
            return np.inf
        return sum(self.energy_parts().values())

    def valid(self):
        if self.conn is not None:
            cv = self.conn_violations()
            return not (cv["seams"] or cv["parent"] or cv["descent"] or cv["h1"])
        col = np.concatenate([np.full((1, self.C), TOP), self.V, np.full((1, self.C), BOT)], 0)
        return bool(self.compat[col[:-1], col[1:]].all())

    # ---- output
    def tiles(self, U=None):
        """(n, n) tile map: each cell's exemplar window pasted."""
        U = self.U if U is None else U
        h = self.h
        out = np.empty((self.R * h, self.C * h), self.an.E.dtype)
        for y in range(self.R):
            for x in range(self.C):
                uy, ux = divmod(int(U[y, x]), self.m)
                rows = self._wy(uy - h // 2 + np.arange(h))
                cols = (ux - h // 2 + np.arange(h)) % self.m
                out[y * h:(y + 1) * h, x * h:(x + 1) * h] = self.an.E[rows[:, None], cols[None]]
        return out
