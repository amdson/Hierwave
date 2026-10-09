"""Legacy (G1b, coarse-to-fine promises era); superseded by castlegen/channels. See notes/history/coarse_to_fine.tex.

The tile level of the generator under the support-envelope language
(notes/history/coarse_to_fine.tex, quantities/envelope.py): sample

    g(x | S_1, pi_K)  ~  tau(x, S_1) exp(-E_p(x)/T) prod_b 1[A_b(x) = pi_b]

with the hard factor applied exactly.  Replaces the old stage chain
tiles -> ground -> repair -> fulfil -> repair(keep_solidity) -> fulfil.

Reduction.  A valid map is a heightmap H(c) >= G (column c solid on rows
>= n - H(c), band = bottom G rows).  For block row r the local height of a
column is t_r = clip(H - K (R-1-r), 0, K), monotone in H, so the min / max of
t_r over an interval J of w = K/I columns is t_r(min_J H) / t_r(max_J H).
The level-K grid therefore constrains each global interval J (the same
column set in every block row) through two H-ranges only:

    min_J H in [a0, a1]    (intersection over block rows of the lo-bin preimages)
    max_J H in [b0, b1]    (same for hi)

i.e. every column of J has H in [a0, b1]; some column has H <= a1 (a "low
witness"); some column has H >= b0 (a "high witness").  Intervals are
independent.  `constraints` computes (a0, a1, b0, b1) and raises on an
unrealisable grid (this check is exact: it passes iff some heightmap
abstracts to P).

Target.  The stationary law of `smooth` is

    pi(x) ~ exp( sum_pairs -E/T + sum_cells logz[x] - tau(x) ) 1[x heightmap, band fixed] 1[A_K(x) = P]

(unaries enter as the log base mass logz, untempered, as in exemplar.gibbs),
tau = w_tex * (#cells whose signature differs from X_tex + #cells whose
solidity differs), band cells excluded.  w_tex = 0 gives p_G(x | A(x) = P).

Moves (each reversible w.r.t. pi; a sweep composes them):
  height   H(c) -> H(c) +- 1: the cell at the column top flips solidity class
           and its material is drawn by heat bath within the new class;
           Metropolis-Hastings acceptance min(1, Z_new / Z_old) where Z_cls
           is the local partition function of the cell over the class
           (proposal and reverse proposal cancel into this ratio).  The
           constraint is an O(1) check on per-interval witness counts.
  swap     two columns of one interval exchange their whole contents.  It
           preserves the interval's multiset of heights, hence its min, max
           and every envelope - so it is always feasible - and moves the
           witness role between columns.  Symmetric proposal, Metropolis.
  material checkerboard heat bath over signatures within the solidity class
           (the band is fixed); keeps every height.
Ergodicity: +-1 moves alone can be frozen (w = 2, min and max pinned: the
two states (a, b) and (b, a) are not +-1 connected); with swaps the height
chain is irreducible on the constrained set (tests/legacy/test_heights.py checks
this by BFS on small instances).  Materials: heat bath within class is
irreducible for fixed heights.

Projection (`project`, initialisation only): per interval, the heights that
minimise the number of non-band cells whose solidity disagrees with X_tex
subject to the constraints, exactly (a 4-state DP over the interval's
columns: "low witness seen", "high witness seen").  Materials: the texture's
tile where its solidity is kept, otherwise ICM on the local energy.
"""
from __future__ import annotations

import time

import numpy as np

from . import exemplar as ex
from .quantities import envelope as EN


# ------------------------------------------------------------- constraints
def constraints(P, K, G, n, lg):
    """-> (a0, a1, b0, b1) arrays over the n / w global intervals (w = K / I):
    min H of the interval in [a0, a1], max H in [b0, b1].  Raises ValueError
    if P is not realisable."""
    P = np.asarray(P, np.int64)
    R = n // K
    assert P.shape == (R, R), (P.shape, n, K)
    I = lg.I
    w = K // I
    assert w * I == K and w >= 1
    Hs = np.arange(n + 1)
    base = K * (R - 1 - np.arange(R))                                    # (R,)
    binsH = lg.bins(np.clip(Hs[None, :] - base[:, None], 0, K), K)       # (R, n+1)
    nJ = n // w
    bx, i = np.arange(nJ) // I, np.arange(nJ) % I
    LO = lg.LO[P][:, bx, i]                                              # (R, nJ)
    HI = lg.HI[P][:, bx, i]
    okH = Hs >= G
    Amask = (binsH[:, None, :] == LO[:, :, None]).all(0) & okH           # (nJ, n+1)
    Bmask = (binsH[:, None, :] == HI[:, :, None]).all(0) & okH
    bad = []
    a0 = np.zeros(nJ, np.int64); a1 = a0.copy(); b0 = a0.copy(); b1 = a0.copy()
    for j in range(nJ):
        A, B = np.flatnonzero(Amask[j]), np.flatnonzero(Bmask[j])
        if not len(A) or not len(B):
            bad.append((j, "empty"))
            continue
        assert A[-1] - A[0] + 1 == len(A) and B[-1] - B[0] + 1 == len(B)   # monotone bins => contiguous
        a0[j], a1[j], b0[j], b1[j] = A[0], A[-1], B[0], B[-1]
        if a0[j] > b1[j] or (w == 1 and max(a0[j], b0[j]) > min(a1[j], b1[j])):
            bad.append((j, "min > max"))
    if bad:
        raise ValueError(f"level-{K} grid is not realisable; bad intervals (index, reason): {bad[:8]}"
                         + (f" ... ({len(bad)} total)" if len(bad) > 8 else ""))
    # clamp the witness ranges into the common range [a0, b1]
    return a0, np.minimum(a1, b1), np.maximum(b0, a0), b1


def heights_of(ts, tiles, G):
    """-> (H, is_heightmap): effective-solid run from the bottom per column,
    and whether every column is air above it."""
    s = EN.effective_solid(ts, tiles, G)
    n = s.shape[0]
    H = np.cumprod(s[::-1], axis=0).sum(0)
    return H.astype(np.int64), bool(s.sum(0).tolist() == H.tolist())


def feasible(H, cons, w):
    a0, a1, b0, b1 = cons
    H = np.asarray(H).reshape(-1, w)
    return bool(((H >= a0[:, None]) & (H <= b1[:, None])).all()
                and (H.min(1) <= a1).all() and (H.max(1) >= b0).all())


# ------------------------------------------------------------------ energy
class _Model:
    """Energy tables and neighbourhoods shared by projection and smoothing."""

    def __init__(self, ts, X_tex, G, T, w_tex, torus):
        self.ts, self.G, self.T, self.torus = ts, G, float(T), torus
        self.Eh = np.asarray(ts.np_tables["Eh"], np.float64)
        self.Ev = np.asarray(ts.np_tables["Ev"], np.float64)
        self.logz = np.asarray(ts.np_tables["logz"], np.float64)
        self.solid = np.asarray(ts.solid, bool)
        S = ts.n_sig
        allowed = np.ones(S, bool)
        allowed[ts.GATE] = False
        self.cls_solid = self.solid & allowed
        self.cls_air = ~self.solid & allowed
        assert self.cls_solid.any() and self.cls_air.any()
        self.WALL = ts.WALL
        self.w_tex = float(w_tex) if X_tex is not None else 0.0
        self.X_tex = None if X_tex is None else np.asarray(X_tex, np.int64)

    def tau(self, n):
        """(n, n, S) texture cost of each signature per cell; 0 on the band."""
        S = self.ts.n_sig
        if self.X_tex is None or self.w_tex == 0.0:
            return np.zeros((n, n, S))
        s = np.arange(S)
        tex = self.X_tex
        t = self.w_tex * ((s[None, None, :] != tex[..., None]).astype(np.float64)
                          + (self.solid[None, None, :] != self.solid[tex][..., None]))
        if self.G:
            t[-self.G:] = 0.0
        return t

    def nbrs(self, g):
        if self.torus:
            return np.roll(g, 1, 1), np.roll(g, -1, 1), np.roll(g, 1, 0), np.roll(g, -1, 0)
        p = np.pad(g, 1, constant_values=self.WALL)
        return p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]

    def site_logits(self, g, tau):
        """(n, n, S) local log-weight of every signature at every cell."""
        l, r, u, d = self.nbrs(g)
        E = self.Eh[l] + self.Eh.T[r] + self.Ev[u] + self.Ev.T[d]
        return -E / self.T + self.logz - tau


def log_weight(ts, tiles, X_tex=None, G=0, T=1.0, w_tex=1.0, torus=True):
    """Total unnormalised log-probability of `tiles` under smooth's target
    (the hard factors excluded)."""
    m = _Model(ts, X_tex, G, T, w_tex, torus)
    x = np.asarray(tiles, np.int64)
    n = x.shape[0]
    if torus:
        eh, ev = ex.pair_energy(ts, x, True)
    else:                                                  # WALL frame, as exemplar.gibbs (frame pairs: a constant)
        eh, ev = ex.pair_energy(ts, np.pad(x, 1, constant_values=ts.WALL), False)
    tau = m.tau(n)
    return float(-(eh.sum() + ev.sum()) / m.T + m.logz[x].sum()
                 - np.take_along_axis(tau, x[..., None], 2).sum())


def _lse(v):
    m = v.max()
    return m + np.log(np.exp(v - m).sum())


# -------------------------------------------------------------- projection
def _interval_dp(cost, lo, hi, a1, b0, jitter):
    """cost: (w, n+1) per column and height.  Minimise the sum subject to
    H in [lo, hi], some H <= a1, some H >= b0.  -> H (w,)."""
    w = cost.shape[0]
    Hs = np.arange(lo, hi + 1)
    c = cost[:, lo:hi + 1] + jitter[:, lo:hi + 1]
    fl = (Hs <= a1).astype(int) + 2 * (Hs >= b0).astype(int)             # flag of each height
    best = np.full((w, 4), np.inf); arg = np.zeros((w, 4), np.int64)
    for f in range(4):
        m = fl == f
        if m.any():
            k = np.argmin(np.where(m[None, :], c, np.inf), 1)
            best[:, f], arg[:, f] = c[np.arange(w), k], Hs[k]
    D = np.full((w + 1, 4), np.inf); D[0, 0] = 0.0
    back = np.zeros((w, 4, 2), np.int64)
    for i in range(w):
        for s in range(4):
            if not np.isfinite(D[i, s]):
                continue
            for f in range(4):
                v = D[i, s] + best[i, f]
                if v < D[i + 1, s | f]:
                    D[i + 1, s | f] = v
                    back[i, s | f] = (s, f)
    assert np.isfinite(D[w, 3]), "interval constraint infeasible"
    H = np.zeros(w, np.int64); s = 3
    for i in range(w - 1, -1, -1):
        ps, f = back[i, s]
        H[i] = arg[i, f]
        s = ps
    return H


def texture_cost(ts, X_tex, G):
    """(n, n+1) per column c and height H: number of non-band cells whose
    solidity under H disagrees with X_tex (inf for H < G)."""
    s = np.asarray(ts.solid)[np.asarray(X_tex)].astype(np.int64)
    n = s.shape[0]
    cs = np.concatenate([np.zeros((1, n), np.int64), np.cumsum(s, 0)], 0)   # cs[y, c]: solids in rows < y
    Hs = np.arange(n + 1)
    top = n - Hs                                                             # first solid row
    cut = n - G
    air_wrong = cs[np.minimum(top, cut)]                                     # solids above the surface
    sol_wrong = (cut - top)[:, None] - (cs[cut][None, :] - cs[np.minimum(top, cut)])
    cost = (air_wrong + np.where((Hs >= G)[:, None], sol_wrong, 0)).astype(np.float64).T
    cost[:, :G] = np.inf
    return cost


def _icm(m, x, mask, tau, iters=2):
    """Greedy (argmax local weight) updates within class on `mask` cells."""
    n = x.shape[0]
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    par = (ys + xs) % 2
    for _ in range(iters):
        for p in (0, 1):
            L = m.site_logits(x, tau)
            cls = np.where(m.solid[x][..., None], m.cls_solid[None, None, :], m.cls_air[None, None, :])
            L = np.where(cls, L, -np.inf)
            upd = mask & (par == p)
            x = np.where(upd, L.argmax(-1), x)
    return x


def project(ts, X_tex, P, K, G, I=2, Q=2, seed=0, T=1.0, ground_tile="stone", torus=True):
    """Exact projection of the texture onto the constraint set (initialisation).
    -> (tiles, stats)."""
    t0 = time.perf_counter()
    lg = EN.lang(I, Q)
    X = np.asarray(X_tex, np.int64)
    n = X.shape[0]
    cons = constraints(P, K, G, n, lg)
    a0, a1, b0, b1 = cons
    w = K // I
    rng = np.random.default_rng([seed, 0x9E1])
    cost = texture_cost(ts, X, G)
    jitter = rng.uniform(0, 1e-3, cost.shape)                               # random tie-breaking
    H = np.zeros(n, np.int64)
    for j in range(n // w):
        cols = slice(j * w, (j + 1) * w)
        H[cols] = _interval_dp(cost[cols], a0[j], b1[j], a1[j], b0[j], jitter[cols])
    rows = np.arange(n)[:, None]
    solid_new = rows >= n - H[None, :]
    m = _Model(ts, None, G, T, 0.0, torus)
    x = X.copy()
    changed = (m.solid[X] != solid_new) | (X == ts.GATE)
    band = np.zeros((n, n), bool)
    if G:
        band[-G:] = True
    changed &= ~band
    init_s = int(np.flatnonzero(m.cls_solid)[np.argmax(m.logz[m.cls_solid])])
    init_a = int(np.flatnonzero(m.cls_air)[np.argmax(m.logz[m.cls_air])])
    x[changed] = np.where(solid_new[changed], init_s, init_a)
    gt = ex.sig_by_name(ts, ground_tile)
    assert ts.solid[gt], "ground tile must be solid"
    x[band] = gt
    x = _icm(m, x, changed, np.zeros((n, n, ts.n_sig)))
    H_tex = np.argmin(cost, 1)
    stats = dict(solidity_flips=int(changed.sum()), cells_changed=int((x != X).sum()),
                 band_repainted=int((X[band] != gt).sum()), texture_cost=int(cost[np.arange(n), H].sum()),
                 unconstrained_cost=int(cost[np.arange(n), H_tex].sum()),
                 height_l1_to_texture=int(np.abs(H - H_tex).sum()), time=time.perf_counter() - t0)
    return x.astype(np.int32), stats


# --------------------------------------------------------------- smoothing
class Chain:
    """The constrained MCMC state (tiles, heights, witness counts)."""

    def __init__(self, ts, tiles, P, K, G, X_tex=None, I=2, Q=2, seed=0, T=1.0, w_tex=1.0, torus=True):
        self.lg = EN.lang(I, Q)
        self.x = np.array(tiles, np.int64)
        self.n = n = self.x.shape[0]
        self.K, self.G, self.w = K, G, K // I
        self.m = _Model(ts, X_tex, G, T, w_tex, torus)
        self.tau = self.m.tau(n)
        self.cons = constraints(P, K, G, n, self.lg)
        H, ok = heights_of(ts, self.x, G)
        if not ok or not feasible(H, self.cons, self.w):
            raise ValueError("initial tiles are not a heightmap meeting the constraints")
        self.H = H
        a0, a1, b0, b1 = self.cons
        J = np.arange(n) // self.w
        self.J = J
        Hw = H.reshape(-1, self.w)
        self.n_low = (Hw <= a1[:, None]).sum(1)
        self.n_high = (Hw >= b0[:, None]).sum(1)
        self.rng = np.random.default_rng([seed, 0x5307])
        self.band = np.zeros((n, n), bool)
        if G:
            self.band[-G:] = True
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.parity = (ys + xs) % 2
        self.stats = dict(height=dict(proposed=0, blocked=0, accepted=0),
                          swap=dict(proposed=0, accepted=0, witness_moves=0),
                          material=dict(changed=0))

    # ---- local weights
    def _cell_logits(self, y, c):
        m, x, n = self.m, self.x, self.n
        if m.torus:
            l, r, u, d = x[y, (c - 1) % n], x[y, (c + 1) % n], x[(y - 1) % n, c], x[(y + 1) % n, c]
        else:
            W = m.WALL
            l = x[y, c - 1] if c > 0 else W
            r = x[y, c + 1] if c < n - 1 else W
            u = x[y - 1, c] if y > 0 else W
            d = x[y + 1, c] if y < n - 1 else W
        E = m.Eh[l] + m.Eh[:, r] + m.Ev[u] + m.Ev[:, d]
        return -E / m.T + m.logz - self.tau[y, c]

    def _cols_logw(self, cols):
        """Log-weight terms touching columns `cols` (vertical pairs inside,
        horizontal pairs on either side, unaries and tau)."""
        m, x, n = self.m, self.x, self.n
        tot = 0.0
        rows = np.arange(n)
        for c in cols:
            col = x[:, c]
            tot += m.logz[col].sum() - self.tau[rows, c, col].sum()
            if m.torus:
                tot -= m.Ev[col, np.roll(col, -1)].sum() / m.T
            else:
                col = np.concatenate([[m.WALL], col, [m.WALL]])
                tot -= m.Ev[col[:-1], col[1:]].sum() / m.T
        wall = np.full(n, m.WALL)
        get = (lambda l: x[:, l % n]) if m.torus else (lambda l: x[:, l] if 0 <= l < n else wall)
        lefts = {(l % n if m.torus else l) for c in cols for l in (c - 1, c)}
        for l in lefts:
            tot -= m.Eh[get(l), get(l + 1)].sum() / m.T
        return tot

    # ---- moves
    def height_move(self, c, up):
        st = self.stats["height"]
        st["proposed"] += 1
        a0, a1, b0, b1 = self.cons
        n, j, h = self.n, self.J[c], int(self.H[c])
        h2 = h + 1 if up else h - 1
        if h2 < a0[j] or h2 > b1[j]:
            st["blocked"] += 1
            return False
        nl = self.n_low[j] - (h <= a1[j]) + (h2 <= a1[j])
        nh = self.n_high[j] - (h >= b0[j]) + (h2 >= b0[j])
        if nl == 0 or nh == 0:
            st["blocked"] += 1
            return False
        y = n - h - 1 if up else n - h
        L = self._cell_logits(y, c)
        m = self.m
        new_cls, old_cls = (m.cls_solid, m.cls_air) if up else (m.cls_air, m.cls_solid)
        Ln = L[new_cls]
        log_acc = _lse(Ln) - _lse(L[old_cls])
        if np.log(self.rng.random()) >= log_acc:
            return False
        g = Ln + self.rng.gumbel(size=Ln.shape)
        self.x[y, c] = np.flatnonzero(new_cls)[np.argmax(g)]
        self.H[c] = h2
        self.n_low[j], self.n_high[j] = nl, nh
        st["accepted"] += 1
        return True

    def swap_move(self, c, c2):
        st = self.stats["swap"]
        x = self.x
        if np.array_equal(x[:, c], x[:, c2]):
            return False
        st["proposed"] += 1
        before = self._cols_logw((c, c2))
        x[:, [c, c2]] = x[:, [c2, c]]
        after = self._cols_logw((c, c2))
        if np.log(self.rng.random()) >= after - before:
            x[:, [c, c2]] = x[:, [c2, c]]
            return False
        a0, a1, b0, b1 = self.cons
        j = self.J[c]
        h, h2 = self.H[c], self.H[c2]
        if (h <= a1[j]) != (h2 <= a1[j]) or (h >= b0[j]) != (h2 >= b0[j]):
            st["witness_moves"] += 1
        self.H[c], self.H[c2] = h2, h
        st["accepted"] += 1
        return True

    def height_pass(self, hmoves=4):
        n = self.n
        for _ in range(hmoves):
            order = self.rng.permutation(n)
            ups = self.rng.random(n) < 0.5
            for c, up in zip(order.tolist(), ups.tolist()):
                self.height_move(c, up)

    def swap_pass(self, swaps=None):
        w = self.w
        if w < 2:
            return
        swaps = w if swaps is None else swaps
        for j in range(self.n // w):
            for _ in range(swaps):
                a, b = self.rng.choice(w, 2, replace=False)
                self.swap_move(j * w + int(a), j * w + int(b))

    def material_pass(self):
        m, x = self.m, self.x
        before = x.copy()
        for p in (0, 1):
            L = m.site_logits(x, self.tau)
            cls = np.where(m.solid[x][..., None], m.cls_solid[None, None, :], m.cls_air[None, None, :])
            L = np.where(cls, L, -np.inf) + self.rng.gumbel(size=L.shape)
            upd = (self.parity == p) & ~self.band
            x = np.where(upd, L.argmax(-1), x)
        self.x = x
        self.stats["material"]["changed"] += int((x != before).sum())

    def sweep(self, hmoves=4, swaps=None, materials=True):
        self.height_pass(hmoves)
        self.swap_pass(swaps)
        if materials:
            self.material_pass()

    def tiles(self):
        return self.x.astype(np.int32)


def _rates(st):
    h, s = st["height"], st["swap"]
    return dict(height_accept=h["accepted"] / max(h["proposed"], 1),
                height_blocked=h["blocked"] / max(h["proposed"], 1),
                swap_accept=s["accepted"] / max(s["proposed"], 1))


def smooth(ts, tiles, P, K, G, X_tex=None, I=2, Q=2, seed=0, T=1.0, w_tex=1.0, sweeps=10, hmoves=4, swaps=None,
           torus=True, callback=None):
    """Constrained MCMC from a valid, P-satisfying `tiles`.  callback(sweep,
    tiles) after every sweep.  -> (tiles, stats)."""
    t0 = time.perf_counter()
    ch = Chain(ts, tiles, P, K, G, X_tex, I, Q, seed, T, w_tex, torus)
    H0 = ch.H.copy()
    for s in range(sweeps):
        ch.sweep(hmoves, swaps)
        if callback is not None:
            callback(s, ch.tiles())
    stats = dict(ch.stats, **_rates(ch.stats), height_l1_moved=int(np.abs(ch.H - H0).sum()),
                 time=time.perf_counter() - t0)
    return ch.tiles(), stats


def sample_tiles(ts, X_tex, P, K, G, I=2, Q=2, seed=0, T=1.0, w_tex=1.0, sweeps=10, ground_tile="stone",
                 hmoves=4, swaps=None, torus=True, callback=None):
    """Project then smooth.  -> (tiles, stats) with stats["project"],
    stats["smooth"], stats["time"]."""
    t0 = time.perf_counter()
    x, sp = project(ts, X_tex, P, K, G, I, Q, seed, T, ground_tile, torus)
    x, ss = smooth(ts, x, P, K, G, X_tex, I, Q, seed, T, w_tex, sweeps, hmoves, swaps, torus, callback)
    X = np.asarray(X_tex)
    return x, dict(project=sp, smooth=ss, cells_changed_total=int((x != X).sum()),
                   time=time.perf_counter() - t0)
