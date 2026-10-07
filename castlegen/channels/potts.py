"""The Potts test: induced couplings on a three-level hierarchy.

Tiles (h = 1) carry q colours with a cyclic distance d; mid cells (h = BM)
carry a block colour; top cells (h = BM * BT) carry a palette p = {p, p+1}.
The designed energy p* is
    J      sum over tile 4-neighbour pairs of J d(c, c')        (J_h, J_v)
    kappa  sum over tiles of kappa [c != m(parent)]             (kappa, homed on tile)
    lam    sum over mid cells of lam [m not in palette(T)]      (lam, homed on mid)
and nothing else: mid-mid and top-top couplings are induced through the
tiles (at kappa -> inf exactly BM J d(m, m') per shared mid edge).  The
forward model adds learned tables at the coarse levels (mid_h, mid_v, mid_u,
top_h, top_v, top_u) and samples top, mid, tile top-down.  This module
provides
  - the model, its channels, designed and learned factors, stats, render,
  - the forward model on the generic kernel,
  - the oracle: a two-way sampler of p* with collapsed moves; a mid cell is
    drawn with its block's tiles integrated out exactly by a column
    transfer matrix (forward filter, backward sample), a top cell with its
    mid cells integrated out given the tiles.  See notes/potts_test.md."""
import numpy as np
from numba import njit

from .core import Channel, Factor, Model

Q = 4


def dist(c, cp, q=Q):
    """Cyclic distance min(|c - c'|, q - |c - c'|); works on arrays."""
    a = np.abs(np.asarray(c) - np.asarray(cp)) % q
    return np.minimum(a, q - a)


# ------------------------------------------------------------------ model
class Potts:
    def __init__(self, nty, ntx, BM=4, BT=2, q=4, J=1.0, kappa=8.0, lam=3.0):
        self.nty, self.ntx, self.BM, self.BT, self.q = nty, ntx, BM, BT, q
        self.P = q
        self.J, self.kappa, self.lam = float(J), float(kappa), float(lam)
        self.nmy, self.nmx = nty * BT, ntx * BT
        self.H, self.W = self.nmy * BM, self.nmx * BM
        self.PAL = np.zeros((self.P, q), bool)
        for p in range(self.P):
            self.PAL[p, p] = self.PAL[p, (p + 1) % q] = True
        cc = np.arange(q)
        self.Jd = self.J * dist(cc[:, None], cc[None, :], q).astype(np.float64)   # (q, q)

    def channels(self):
        q, P = self.q, self.P
        top = Channel("top", self.BM * self.BT, P).add_view("pal", np.arange(P), P)
        mid = Channel("mid", self.BM, q).add_view("col", np.arange(q), q)
        tile = Channel("tile", 1, q).add_view("col", np.arange(q), q)
        top.grid = np.zeros((self.nty, self.ntx), np.int32)
        mid.grid = np.zeros((self.nmy, self.nmx), np.int32)
        tile.grid = np.zeros((self.H, self.W), np.int32)
        return top, mid, tile

    def designed_factors(self):
        q = self.q
        lamtab = self.lam * (~self.PAL.T).astype(np.float64)                    # (m, p)
        return [Factor.pair(("tile", "col"), ("tile", "col"), (0, 1), self.Jd, name="J_h"),
                Factor.pair(("tile", "col"), ("tile", "col"), (1, 0), self.Jd, name="J_v"),
                Factor.pair(("tile", "col"), ("mid", "col"), (0, 0), self.kappa * (1 - np.eye(q)), name="kappa"),
                Factor.pair(("mid", "col"), ("top", "pal"), (0, 0), lamtab, name="lam")]

    def learned_factors(self, theta):
        M, T = ("mid", "col"), ("top", "pal")
        return [Factor.pair(M, M, (0, 1), theta["mid_h"], name="mid_h"),
                Factor.pair(M, M, (1, 0), theta["mid_v"], name="mid_v"),
                Factor.unary(M, theta["mid_u"], name="mid_u"),
                Factor.pair(T, T, (0, 1), theta["top_h"], name="top_h"),
                Factor.pair(T, T, (1, 0), theta["top_v"], name="top_v"),
                Factor.unary(T, theta["top_u"], name="top_u")]

    def theta0(self):
        q, P = self.q, self.P
        return dict(mid_h=np.zeros((q, q)), mid_v=np.zeros((q, q)), mid_u=np.zeros(q),
                    top_h=np.zeros((P, P)), top_v=np.zeros((P, P)), top_u=np.zeros(P))

    def model(self, top, mid, tile, theta):
        return Model(self.H, self.W, [top, mid, tile], self.designed_factors() + self.learned_factors(theta))

    # ------------------------------------------------------- statistics
    @staticmethod
    def _pairs(g, n):
        h = np.zeros((n, n)); v = np.zeros((n, n))
        np.add.at(h, (g[:, :-1].ravel(), g[:, 1:].ravel()), 1)
        np.add.at(v, (g[:-1, :].ravel(), g[1:, :].ravel()), 1)
        return h / max(h.sum(), 1), v / max(v.sum(), 1)

    def stats(self, top, mid, tile):
        q, P, BM, BT = self.q, self.P, self.BM, self.BT
        tg, mg, fg = top.grid, mid.grid, tile.grid
        mid_h, mid_v = self._pairs(mg, q)
        top_h, top_v = self._pairs(tg, P)
        dh = dist(fg[:, :-1], fg[:, 1:], q).ravel()
        dv = dist(fg[:-1, :], fg[1:, :], q).ravel()
        tile_d = np.bincount(np.concatenate([dh, dv]), minlength=q // 2 + 1) / (dh.size + dv.size)
        mup = np.repeat(np.repeat(mg, BM, 0), BM, 1)
        tup = np.repeat(np.repeat(tg, BT, 0), BT, 1)

        def edge_same(B):
            sh = fg[:, B - 1:-1:B] == fg[:, B::B]
            sv = fg[B - 1:-1:B, :] == fg[B::B, :]
            n = sh.size + sv.size
            return float((sh.sum() + sv.sum()) / n) if n else 0.0

        return dict(mid_h=mid_h, mid_v=mid_v, mid_u=np.bincount(mg.ravel(), minlength=q) / mg.size,
                    top_h=top_h, top_v=top_v, top_u=np.bincount(tg.ravel(), minlength=P) / tg.size,
                    tile_d=tile_d, mismatch=float((fg != mup).mean()),
                    pal_viol=float((~self.PAL[tup, mg]).mean()),
                    edge_same_mid=edge_same(BM), edge_same_top=edge_same(BM * BT))

    def render(self, top, mid, tile):
        """(H, W, 3) uint8: tiles coloured; the first row / column of each mid
        block darkened (thin lines), both sides of each top-block edge darker."""
        base = np.array([(220, 60, 60), (240, 200, 40), (60, 170, 90), (50, 90, 210)], np.float64)
        if self.q > len(base):
            import colorsys
            base = np.array([colorsys.hsv_to_rgb(c / self.q, 0.7, 0.9) for c in range(self.q)]) * 255
        img = base[tile.grid].copy()
        BM, BT = self.BM, self.BM * self.BT
        yy, xx = np.mgrid[:self.H, :self.W]
        mid_edge = (yy % BM == 0) | (xx % BM == 0)
        top_edge = (yy % BT == 0) | (xx % BT == 0) | (yy % BT == BT - 1) | (xx % BT == BT - 1)
        img[mid_edge] *= 0.6
        img[top_edge] *= 0.35 / np.where(mid_edge[top_edge], 0.6, 1.0)[:, None]
        return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------- forward model
class Forward:
    """Top, then mid, then tiles by the generic kernel, with designed +
    learned factors (the model is rebuilt from self.theta at every run)."""

    def __init__(self, P: Potts, theta: dict, seed=0):
        self.P, self.theta = P, theta
        self.top, self.mid, self.tile = P.channels()
        self.model = None
        self.rng = np.random.default_rng(seed)

    def run(self, S_T=30, S_M=30, S_F=20, fresh=True):
        P = self.P
        self.model = P.model(self.top, self.mid, self.tile, self.theta)
        if fresh:
            self.top.grid[:] = 0
            self.mid.grid[:] = 0
            self.tile.grid[:] = 0
        for home, n in (("top", S_T), ("mid", S_M), ("tile", S_F)):
            self.model.sweep(home, n, seed=int(self.rng.integers(1 << 30)))
        return P.stats(self.top, self.mid, self.tile)


# ---------------------------------------------------- oracle: two-way
def _column_tables(BM, q, Jd):
    """Per column state s = sum_r c_r q^r: digits (NS, BM), vertical J d
    within the column (NS,), mismatches against each m (NS, q)."""
    NS = q ** BM
    s = np.arange(NS)
    digs = np.stack([(s // q ** r) % q for r in range(BM)], 1).astype(np.int64)
    vert = np.zeros(NS)
    for r in range(BM - 1):
        vert += Jd[digs[:, r], digs[:, r + 1]]
    mis = np.stack([(digs != m).sum(1) for m in range(q)], 1).astype(np.float64)
    return digs, vert, mis


@njit(cache=True)
def _block_filter(tiles, y0, x0, m, Jd, kappa, digs, vert, mis, do_sample, rng_u):
    """Column transfer matrix over the BM x BM block at (y0, x0) with mid
    colour m.  Tiles outside the block are fixed boundary (none off the
    grid).  Returns log Z; if do_sample, redraws the block's tiles in place
    by backward sampling.  The transition between columns factorises over
    rows, so it is applied one row at a time (NS q per row)."""
    H, W = tiles.shape
    NS, BM = digs.shape
    q = Jd.shape[0]
    Wt = np.exp(-Jd)
    alpha = np.empty((BM, NS))
    u = np.empty(NS)
    v = np.empty(NS)
    logZ = 0.0
    for x in range(BM):
        gx = x0 + x
        emin = np.inf
        for s in range(NS):
            e = vert[s] + kappa * mis[s, m]
            if y0 - 1 >= 0:
                e += Jd[tiles[y0 - 1, gx], digs[s, 0]]
            if y0 + BM < H:
                e += Jd[tiles[y0 + BM, gx], digs[s, BM - 1]]
            if x == 0 and x0 - 1 >= 0:
                for r in range(BM):
                    e += Jd[tiles[y0 + r, x0 - 1], digs[s, r]]
            if x == BM - 1 and x0 + BM < W:
                for r in range(BM):
                    e += Jd[tiles[y0 + r, x0 + BM], digs[s, r]]
            u[s] = e
            if e < emin:
                emin = e
        logZ -= emin
        if x == 0:
            for s in range(NS):
                alpha[0, s] = np.exp(-(u[s] - emin))
        else:
            for s in range(NS):
                v[s] = alpha[x - 1, s]
            stride = 1
            for r in range(BM):
                for s in range(NS):
                    c = digs[s, r]
                    base = s - c * stride
                    acc = 0.0
                    for cp in range(q):
                        acc += v[base + cp * stride] * Wt[cp, c]
                    alpha[x, s] = acc
                for s in range(NS):
                    v[s] = alpha[x, s]
                stride *= q
            for s in range(NS):
                alpha[x, s] = v[s] * np.exp(-(u[s] - emin))
        tot = alpha[x].sum()
        alpha[x] /= tot
        logZ += np.log(tot)
    if do_sample:
        w = np.empty(NS)
        nxt = -1
        for x in range(BM - 1, -1, -1):
            for s in range(NS):
                ws = alpha[x, s]
                if nxt >= 0:
                    for r in range(BM):
                        ws *= Wt[digs[s, r], digs[nxt, r]]
                w[s] = ws
            target = rng_u[x] * w.sum()
            acc = 0.0
            pick = NS - 1
            for s in range(NS):
                acc += w[s]
                if acc >= target:
                    pick = s
                    break
            for r in range(BM):
                tiles[y0 + r, x0 + x] = digs[pick, r]
            nxt = pick
    return logZ


def _softmax(logw):
    p = np.exp(logw - logw.max())
    return p / p.sum()


class Oracle:
    """Two-way sampler of p* with collapsed moves (exact)."""

    def __init__(self, P: Potts, seed=0):
        self.P = P
        self.rng = np.random.default_rng(seed)
        self.top, self.mid, self.tile = P.channels()
        self.top.grid[:] = self.rng.integers(P.P, size=self.top.grid.shape)
        self.mid.grid[:] = self.rng.integers(P.q, size=self.mid.grid.shape)
        self.tile.grid[:] = self.rng.integers(P.q, size=self.tile.grid.shape)
        self.model = Model(P.H, P.W, [self.top, self.mid, self.tile], P.designed_factors())
        self.digs, self.vert, self.mis = _column_tables(P.BM, P.q, P.Jd)
        self.lamtab = P.lam * (~P.PAL).astype(np.float64)                     # (p, m)

    def _filter(self, i, j, m, sample):
        P = self.P
        return _block_filter(self.tile.grid, i * P.BM, j * P.BM, m, P.Jd, P.kappa, self.digs, self.vert,
                             self.mis, sample, self.rng.random(P.BM) if sample else np.zeros(P.BM))

    def mid_logZ(self, i, j):
        return np.array([self._filter(i, j, m, False) for m in range(self.P.q)])

    def mid_probs(self, i, j):
        t = self.top.grid[i // self.P.BT, j // self.P.BT]
        return _softmax(-self.lamtab[t] + self.mid_logZ(i, j))

    def mid_move(self, i, j):
        m = int(self.rng.choice(self.P.q, p=self.mid_probs(i, j)))
        self.mid.grid[i, j] = m
        self._filter(i, j, m, True)

    def _mid_mis(self, i, j):
        """(BT, BT, q) mismatches of each mid cell's block tiles against m."""
        P = self.P
        BM, BT = P.BM, P.BT
        blk = self.tile.grid[i * BM * BT:(i + 1) * BM * BT, j * BM * BT:(j + 1) * BM * BT]
        blk = blk.reshape(BT, BM, BT, BM).transpose(0, 2, 1, 3).reshape(BT, BT, BM * BM)
        return BM * BM - (blk[..., None] == np.arange(P.q)).sum(2)

    def top_probs(self, i, j):
        P = self.P
        e = -P.kappa * self._mid_mis(i, j)                                      # (BT, BT, q)
        lw = -self.lamtab[:, None, None, :] + e[None]                           # (P, BT, BT, q)
        mx = lw.max(3, keepdims=True)
        logw = (np.log(np.exp(lw - mx).sum(3)) + mx[..., 0]).sum((1, 2))
        return _softmax(logw)

    def top_move(self, i, j):
        P = self.P
        t = int(self.rng.choice(P.P, p=self.top_probs(i, j)))
        self.top.grid[i, j] = t
        e = -self.lamtab[t] - P.kappa * self._mid_mis(i, j)
        for a in range(P.BT):
            for b in range(P.BT):
                self.mid.grid[i * P.BT + a, j * P.BT + b] = int(self.rng.choice(P.q, p=_softmax(e[a, b])))

    def tile_sweep(self, n=1):
        self.model.sweep("tile", n, seed=int(self.rng.integers(1 << 30)))

    def sweep(self, n=1, tile_sweeps=2):
        P = self.P
        for _ in range(n):
            for k in self.rng.permutation(P.nty * P.ntx):
                self.top_move(k // P.ntx, k % P.ntx)
            for k in self.rng.permutation(P.nmy * P.nmx):
                self.mid_move(k // P.nmx, k % P.nmx)
            self.tile_sweep(tile_sweeps)

    def moments(self, burn, sweeps):
        self.sweep(burn)
        acc = None
        for _ in range(sweeps):
            self.sweep(1)
            s = self.P.stats(self.top, self.mid, self.tile)
            acc = s if acc is None else {k: acc[k] + s[k] for k in acc}
        return {k: v / sweeps for k, v in acc.items()}
