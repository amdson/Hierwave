"""Legacy (G1c, generic/joint sampler era); superseded by castlegen/channels. See notes/history/generic.md.

Port-partition connectivity promise, penalty version (a quick test).

Level h >= 2: each cell holds sigma, a label in {0 closed, 1..m} for each of
its 8 half-side ports p = 2 d + i (side d = N, E, S, W; half i = 0 the
left / top half).  Port (d, i) of a cell at level 2h is side d of one child
at level h.  Labels partition the open ports into parts.  sigma is learned
per level from exemplar windows: the window's walkable components (joins
inside the window) label the ports their tiles cross on; a port crossed by
several components takes the largest; more than m port-touching components
merge the smallest into the largest.  Level 1: the tiles; a side is open iff
the two tiles join.

Rules, each a penalty lam per violation (lam ramps up over the sweeps):
  seam     facing half-ports open on both sides or neither (off the map: closed)
  inherit  a child's side on its parent's side d is open iff the parent's port is
  exit     every child part reaches, over open seams inside the parent block, a
           parent port in E(parent)
  merge    the children touching one parent part's ports are connected inside the block
E(cell) at level h, computed once the level is sampled, from its 2x2 group G at
level h (the cell's parent at 2h): depth 0 for parts on a G port in E(G), then BFS
over G's internal seams; port p of part X is an exit if it lies on a G port in
E(G) or faces a sibling part of lower depth.  Top level: every open port is an
exit (connectivity at larger scales is assumed).
"""
import numpy as np

from castlegen.legacy import generic as GN
from castlegen.legacy.quantities import connectivity as CQ

DIRS = GN.DIRS
OPP = GN.OPP
MLAB = 4                                                                   # parts per cell (labels 1..MLAB)
NN = 4 * MLAB                                                              # nodes of a 2x2 group
WB = (MLAB + 1) ** np.arange(8)


def port_child(q):
    """Child position (a, b) holding parent port q, and the child's side."""
    d, i = divmod(q, 2)
    return ((0, i), (i, 1), (1, i), (i, 0))[d], d


def parent_port(a, b, d):
    """Parent port of side d of the child at (a, b), or -1 if the side is internal."""
    on = (a == 0, b == 1, a == 1, b == 0)[d]
    return 2 * d + (b if d in (0, 2) else a) if on else -1


# ------------------------------------------------------------ learning
def window_labels(ts, E, h, m=MLAB):
    """(N, 8) canonical port labels of the h x h window centred on every
    coordinate of the torus exemplar E (module docstring)."""
    cr = CQ.crossings(ts, E)                                               # (4, H, W)
    H, W = E.shape
    hh = h // 2
    out = np.zeros((H * W, 8), np.int64)
    for cy in range(H):
        for cx in range(W):
            ys, xs = (cy - hh + np.arange(h)) % H, (cx - hh + np.arange(h)) % W
            par = np.arange(h * h)

            def find(a):
                while par[a] != a:
                    par[a] = par[par[a]]
                    a = par[a]
                return a
            for y in range(h):
                for x in range(h):
                    if x + 1 < h and cr[1, ys[y], xs[x]]:
                        par[find(y * h + x)] = find(y * h + x + 1)
                    if y + 1 < h and cr[2, ys[y], xs[x]]:
                        par[find(y * h + x)] = find((y + 1) * h + x)
            comp = np.array([find(a) for a in range(h * h)])
            size = np.bincount(comp, minlength=h * h)
            lab = np.full(8, -1)
            for q in range(8):
                d, i = divmod(q, 2)
                span = np.arange(i * hh, (i + 1) * hh)
                cells = ((0, span), (span, h - 1), (h - 1, span), (span, 0))[d]
                yy, xx = np.broadcast_arrays(*cells)
                on = cr[d, ys[yy], xs[xx]]
                if on.any():
                    cs = comp[yy[on] * h + xx[on]]
                    lab[q] = cs[np.argmax(size[cs])]
            used = [c for c in dict.fromkeys(lab[lab >= 0])]
            while len(used) > m:                                           # merge the smallest into the largest
                used.sort(key=lambda c: size[c])
                small, big = used[0], used[-1]
                lab[lab == small] = big
                size[big] += size[small]
                used = used[1:]
            canon, nxt, res = {}, 1, np.zeros(8, np.int64)
            for q in range(8):
                if lab[q] >= 0:
                    if lab[q] not in canon:
                        canon[lab[q]] = nxt
                        nxt += 1
                    res[q] = canon[lab[q]]
            out[cy * W + cx] = res
    return out


def cost_table(idx, V, my, m, rad, alpha=1.0):
    """generic.cost_table over V values (idx: (N,) value index per coordinate)."""
    oh = np.zeros((my, m, V))
    oh.reshape(-1, V)[np.arange(my * m), idx] = 1.0
    cy = np.concatenate([np.zeros((1, m, V)), np.cumsum(oh, 0)], 0)
    y = np.arange(my)
    col = cy[np.minimum(y + rad + 1, my)] - cy[np.maximum(y - rad, 0)]
    cx = np.concatenate([np.zeros((my, 1, V)), np.cumsum(np.concatenate([col, col], 1), 1)], 1)
    ix = (np.arange(m) - rad) % m
    cnt = (cx[:, ix + 2 * rad + 1] - cx[:, ix]).reshape(-1, V)
    ph = oh.reshape(-1, V).mean(0)
    p = (cnt + alpha * ph) / (cnt.sum(1, keepdims=True) + alpha)
    return -np.log(p)


class PortLevel:
    """sigma at level h >= 2 on top of a JointLevel lvl (u)."""

    def __init__(self, ts, E, lvl, h, m=MLAB, r=0.25, alpha=1.0):
        lab = window_labels(ts, E, h, m)
        codes = lab @ WB
        self.codes, inv = np.unique(codes, return_inverse=True)
        self.LAB = np.zeros((len(self.codes), 8), np.int64)
        self.LAB[inv] = lab
        self.cost = cost_table(inv, len(self.codes), E.shape[0], E.shape[1], max(1, int(round(r * h))), alpha)
        self.lvl, self.h = lvl, h
        self.R, self.C = lvl.R, lvl.C

    def init(self, parent=None, rng=None):
        """parent: the PortLevel above (its S and exits E), or None (top)."""
        self.P = parent
        g = -self.cost[self.lvl.U.ravel()]
        self.S = np.argmax(g + rng.gumbel(size=g.shape), 1).reshape(self.R, self.C)
        self.lvl.extra = lambda py, px, allc: self.cost[allc, self.S[py, px][:, None]]

    def labs(self):
        return self.LAB[self.S]

    # ---- penalties of cell (y, x) for every value, the others fixed
    def cell_pen(self, y, x):
        L = self.LAB                                                       # (V, 8)
        op = L > 0
        pen = np.zeros(len(L))
        Lg = self.labs()
        for d, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            for i in range(2):
                if 0 <= ny < self.R and 0 <= nx < self.C:
                    pen += op[:, 2 * d + i] != (Lg[ny, nx, 2 * OPP[d] + i] > 0)
                else:
                    pen += op[:, 2 * d + i]
        if self.P is None:
            return pen
        gy, gx, a, b = y // 2, x // 2, y % 2, x % 2
        Popen = self.P.labs()[gy, gx] > 0
        for d in range(4):
            q = parent_port(a, b, d)
            if q >= 0:
                pen += (op[:, 2 * d] | op[:, 2 * d + 1]) != Popen[q]
        L4 = np.repeat(Lg[2 * gy:2 * gy + 2, 2 * gx:2 * gx + 2].reshape(1, 4, 8), len(L), 0)
        L4[:, 2 * a + b] = L
        return pen + block_pen(L4, self.P.labs()[gy, gx], self.P.E[gy, gx])

    def sweep(self, lam, T=1.0, rng=None):
        U = self.lvl.U
        for k in rng.permutation(self.R * self.C):
            y, x = divmod(int(k), self.C)
            e = self.cost[U[y, x]] + lam * self.cell_pen(y, x)
            g = -(e - e.min()) / T
            self.S[y, x] = int(np.argmax(g + rng.gumbel(size=g.shape)))

    def violations(self):
        tot = 0.0
        for y in range(self.R):
            for x in range(self.C):
                tot += self.cell_pen(y, x)[self.S[y, x]]
        return tot

    def compute_exits(self):
        """E (R, C, 8) (module docstring)."""
        Lg = self.labs()
        op = Lg > 0
        R, C = self.R, self.C
        E = np.zeros((R, C, 8), bool)
        if self.P is None:
            for d, (dy, dx) in enumerate(DIRS):
                for i in range(2):
                    ny, nx = np.arange(R)[:, None] + dy, np.arange(C)[None] + dx
                    ok = (ny >= 0) & (ny < R) & (nx >= 0) & (nx < C)
                    fac = np.zeros((R, C), bool)
                    fac[ok] = op[np.clip(ny, 0, R - 1), np.clip(nx, 0, C - 1), 2 * OPP[d] + i][ok]
                    E[..., 2 * d + i] = op[..., 2 * d + i] & fac
            self.E = E
            return E
        PE = self.P.E
        for gy in range(R // 2):
            for gx in range(C // 2):
                L4 = Lg[2 * gy:2 * gy + 2, 2 * gx:2 * gx + 2].reshape(4, 8)
                E[2 * gy:2 * gy + 2, 2 * gx:2 * gx + 2] = group_exits(L4, PE[gy, gx]).reshape(2, 2, 8)
        self.E = E
        return E


def _edges(L4):
    """Internal seam edges of a 2x2 group: [(k1, l1, k2, l2)] arrays over the batch."""
    out = []
    for a in range(2):                                                     # horizontal seams
        for i in range(2):
            out.append((2 * a, L4[..., 2 * a, 2 + i], 2 * a + 1, L4[..., 2 * a + 1, 6 + i]))
    for b in range(2):                                                     # vertical seams
        for i in range(2):
            out.append((b, L4[..., b, 4 + i], 2 + b, L4[..., 2 + b, i]))
    return out


def _closure(L4):
    """(..., NN, NN) reachability between nodes MLAB k + l - 1 (k child, l label)."""
    sh = L4.shape[:-2]
    A = np.broadcast_to(np.eye(NN, dtype=bool), sh + (NN, NN)).copy()
    for k1, l1, k2, l2 in _edges(L4):
        on = (l1 > 0) & (l2 > 0)
        n1, n2 = MLAB * k1 + np.maximum(l1, 1) - 1, MLAB * k2 + np.maximum(l2, 1) - 1
        idx = np.nonzero(on)
        A[idx + (n1[idx], n2[idx])] = True
        A[idx + (n2[idx], n1[idx])] = True
    for _ in range(4):
        A = (A.astype(np.uint8) @ A.astype(np.uint8)) > 0
    return A


def _nodes_on(L4, q):
    """(..., NN) nodes on parent port q (the child's open halves there)."""
    (a, b), d = port_child(q)
    k = 2 * a + b
    out = np.zeros(L4.shape[:-2] + (NN,), bool)
    for j in range(2):
        l = L4[..., k, 2 * d + j]
        idx = np.nonzero(l > 0)
        out[idx + (MLAB * k + l[idx] - 1,)] = True
    return out


def block_pen(L4, Plab, PE):
    """(n,) exit + merge penalties of groups L4 (n, 4, 8) under a parent with
    port labels Plab (8,) and exits PE (8,)."""
    A = _closure(L4)
    exists = np.zeros(L4.shape[:1] + (NN,), bool)
    for k in range(4):
        for l in range(1, MLAB + 1):
            exists[:, MLAB * k + l - 1] = (L4[:, k] == l).any(1)
    X = np.zeros_like(exists)
    for q in np.flatnonzero(PE):
        X |= _nodes_on(L4, q)
    reach = (A & X[:, None, :]).any(2)
    pen = (exists & ~reach).sum(1).astype(float)
    for lab in range(1, MLAB + 1):                                         # a part's ports joined pairwise
        Q = np.flatnonzero(Plab == lab)                                    # (a shared port may carry others too)
        if len(Q) < 2:
            continue
        N0 = _nodes_on(L4, Q[0])
        bad = np.zeros(len(L4), bool)
        for q in Q[1:]:
            Nq = _nodes_on(L4, q)
            bad |= N0.any(1) & Nq.any(1) & ~(N0[:, :, None] & Nq[:, None, :] & A).any((1, 2))
        pen += bad
    return pen


def group_exits(L4, PE):
    """(4, 8) exits of a 2x2 group's children given its parent's exits PE (8,)."""
    A = _closure(L4[None])[0]
    Lk = L4
    depth = np.full(NN, 99)
    for q in np.flatnonzero(PE):
        depth[_nodes_on(L4[None], q)[0]] = 0
    for _ in range(NN):                                                    # BFS by relaxation
        for k1, l1, k2, l2 in _edges(L4):
            if l1 > 0 and l2 > 0:
                n1, n2 = MLAB * k1 + l1 - 1, MLAB * k2 + l2 - 1
                depth[n1] = min(depth[n1], depth[n2] + 1)
                depth[n2] = min(depth[n2], depth[n1] + 1)
    E = np.zeros((4, 8), bool)
    for k in range(4):
        a, b = divmod(k, 2)
        for d, (dy, dx) in enumerate(DIRS):
            q = parent_port(a, b, d)
            for i in range(2):
                l = Lk[k, 2 * d + i]
                if l == 0 or depth[MLAB * k + l - 1] >= 99:
                    continue
                if q >= 0:
                    E[k, 2 * d + i] = PE[q]
                else:
                    k2 = 2 * (a + dy) + (b + dx)
                    l2 = Lk[k2, 2 * OPP[d] + i]
                    E[k, 2 * d + i] = l2 > 0 and depth[MLAB * k2 + l2 - 1] < depth[MLAB * k + l - 1]
    return E


# ------------------------------------------------------------ level 1 (tiles)
class TileRules:
    """Penalties of the h = 1 tiles under the level-2 PortLevel P."""

    def __init__(self, ts, lvl, P):
        node, Dh, Dv, _ = GN.conn_tables(ts)
        self.node = np.append(node, False)                                 # index -1: off the map
        self.Dh = np.pad(Dh, ((0, 1), (0, 1)))
        self.Dv = np.pad(Dv, ((0, 1), (0, 1)))
        self.lvl, self.P = lvl, P
        self.Plab, self.PE = P.labs(), P.E
        self.lam = 0.0
        lvl.extra = self.extra

    def _joins(self, W):
        """W (..., 4, 4) tiles -> E joins (..., 4, 3), S joins (..., 3, 4)."""
        nd = self.node[W]
        ej = nd[..., :, :-1] & nd[..., :, 1:] & self.Dh[W[..., :, :-1], W[..., :, 1:]]
        sj = nd[..., :-1, :] & nd[..., 1:, :] & self.Dv[W[..., :-1, :], W[..., 1:, :]]
        return nd, ej, sj

    def win_pen(self, W, gy, gx):
        """(...) penalties of blocks (gy, gx) with 4 x 4 tile windows W (block + ring)."""
        nd, ej, sj = self._joins(W)
        Plab, PE = self.Plab[gy, gx], self.PE[gy, gx]                      # (..., 8)
        ext = np.stack([sj[..., 0, 1], sj[..., 0, 2], ej[..., 1, 2], ej[..., 2, 2],
                        sj[..., 2, 1], sj[..., 2, 2], ej[..., 1, 0], ej[..., 2, 0]], -1)   # (..., 8)
        pen = (ext != (Plab > 0)).sum(-1).astype(float)
        live = nd[..., 1:3, 1:3].reshape(nd.shape[:-2] + (4,))
        A = np.zeros(live.shape + (4,), bool)
        for k in range(4):
            A[..., k, k] = True
        for k1, k2, j in ((0, 1, ej[..., 1, 1]), (2, 3, ej[..., 2, 1]), (0, 2, sj[..., 1, 1]), (1, 3, sj[..., 1, 2])):
            A[..., k1, k2] |= j
            A[..., k2, k1] |= j
        for _ in range(2):
            A = (A.astype(np.uint8) @ A.astype(np.uint8)) > 0
        kq = np.array([2 * port_child(q)[0][0] + port_child(q)[0][1] for q in range(8)])
        X = np.zeros(live.shape, bool)
        for q in range(8):
            X[..., kq[q]] |= ext[..., q] & PE[..., q]
        reach = (A & X[..., None, :]).any(-1)
        pen += (live & ~reach).sum(-1)
        for lab in range(1, MLAB + 1):
            N = np.zeros(live.shape, bool)
            for q in range(8):
                N[..., kq[q]] |= ext[..., q] & (Plab[..., q] == lab)
            pen += (N[..., :, None] & N[..., None, :] & ~A).any((-2, -1))
        return pen

    def _tp(self):
        T = self.lvl.Et[self.lvl.U]
        return np.pad(T, 2, constant_values=-1)

    def extra(self, py, px, allc):
        if self.lam == 0.0:
            return 0.0
        Tp = self._tp()
        cand = self.lvl.Et[allc]                                           # (k, M)
        R2, C2 = self.lvl.R // 2, self.lvl.C // 2
        a, b = py % 2, px % 2
        tot = np.zeros(cand.shape)
        for gy, gx in ((py // 2, px // 2), (py // 2 + np.where(a == 0, -1, 1), px // 2),
                       (py // 2, px // 2 + np.where(b == 0, -1, 1))):
            ok = (gy >= 0) & (gy < R2) & (gx >= 0) & (gx < C2)
            gy, gx = np.clip(gy, 0, R2 - 1), np.clip(gx, 0, C2 - 1)
            y0, x0 = 2 * gy - 1 + 2, 2 * gx - 1 + 2                            # window top-left in Tp
            W = Tp[(y0[:, None] + np.arange(4))[:, :, None], (x0[:, None] + np.arange(4))[:, None, :]]  # (k, 4, 4)
            W = np.repeat(W[:, None], cand.shape[1], 1)                       # (k, M, 4, 4)
            cy, cx = py + 2 - y0, px + 2 - x0
            W[np.arange(len(py)), :, cy, cx] = cand
            tot += ok[:, None] * self.win_pen(W, gy[:, None], gx[:, None])
        return self.lam * tot

    def violations(self):
        Tp = self._tp()
        R2, C2 = self.lvl.R // 2, self.lvl.C // 2
        gy, gx = np.mgrid[:R2, :C2]
        W = Tp[(2 * gy[..., None, None] + 1 + np.arange(4)[:, None]), (2 * gx[..., None, None] + 1 + np.arange(4)[None])]
        return float(self.win_pen(W, gy, gx).sum())
