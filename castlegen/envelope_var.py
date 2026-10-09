"""The envelope support promise (quantities/envelope.py) as a hierarchical
variable sampled exactly per child column (notes/coarse_to_fine.tex, the
promise component pi_h of C_h).

Structure.  Under the envelope language every hard constraint is vertical:
the merge of a parent's two stacked children in one child column, and the
seam rule between vertically adjacent blocks.  Side-by-side blocks only
share SOFT pair terms, and a parent's interval part over child column dx
depends only on that child column.  So given the parent level, level h is a
set of CHILD COLUMNS, each a chain over the block rows y = 0 .. R-1 with V
states and hard transitions

    y -> y+1 inside a parent (y even):   pair_mask(parent_part(P[y/2, x/2], x%2), ground)
    y -> y+1 between parents (y odd):    COMPAT
    R-1 -> 0 (the torus wrap seam):      nothing (a chain, not a cycle)

plus unary masks (ground-row validity, clamps, beta F_h) and the beta
seam masks B_h on every transition.  The level energy
(everything divided by T) is

    sum_cells  u[v] + parent[2dy+dx, v, P] + chi(v)
  + sum_vertical seams    pair[1][upper, lower]     (non-wrap seams only)
  + sum_horizontal seams  pair[0][left, right]      (torus in x)

The vertical soft term across the wrap seam is dropped (it would close the
chain into a cycle; the tables are fitted on tori, so this is the one term
of the fitted model that is left out).  The top level has no parent: its
transitions are COMPAT (and B_h) everywhere.

Sampler.  A column conditional given its two neighbour columns is a chain
(the horizontal terms become unaries), drawn exactly by log-space forward
filtering / backward sampling (backward messages, then a forward draw),
vectorised over all columns of one parity.  Columns of the same parity are
conditionally independent, so

  init  (prolong / perturb): even columns drawn given the parent with the
        horizontal terms IGNORED (their neighbours do not exist yet), then
        odd columns drawn given both even neighbours;
  sweep (smooth): even columns given the odd ones, then odd given even -
        an exact block-Gibbs kernel for the level conditional (T > 0).

T = 0 replaces sum-product by min-sum: init is the Viterbi (MAP) chain of
each column, a sweep is exact block coordinate descent.  The same code
serves both.  Randomness is Gumbel-max on hier.noise keyed on (seed, level,
step, parity/phase, row, column, value), so a run is reproducible.  An empty
feasible set raises InfeasibleError, never keeps stale values.

Clamps and beta.  clamps = {(h, by, bx): value}.  Before sampling, `feasible`
computes bottom-up, for every level, a unary set F_h (y, x) and a pair
relation B_h (y, x) on every non-wrap vertical seam [upper, lower]:

  F_K = valid values (ground row), restricted to the clamp if clamped;
  B_K = COMPAT;
  F_2h(parent) = values whose column part dx is realisable, for both dx, by
        an (upper, lower) pair of child column dx (column_pairs) with both
        children in F_h and the pair in B_h;
  B_2h(a over b) = COMPAT(a, b) and, for both child columns, some realising
        pair of a (lower child l) and of b (upper child u) with B_h(l, u)
        on the child seam between them;
  then clamps at 2h, then arc consistency along every column of the level
  (a forward and a backward pass: every surviving value has a feasible
  completion of its whole column at that level).

F_h is a unary mask and B_h a transition mask at every level (beta_h).  This
is exact inside one parent and one seam at a time, not globally (a parent's
pair must serve the seam above and the one below at once; deeper levels are
summarised pairwise), so it is a sound pruning that can in principle still
leave a dead end: an empty set then raises InfeasibleError.  In the tests and
a 1500-pattern stress run of feasible clamp sets (read off valid maps) no
dead end occurred; clamps that are infeasible on their own raise at
`feasible` ("clamps infeasible").  The per-parent F_h alone (without B_h and
arc consistency) did hit dead ends on such patterns.  Memory: B_h is
(n/h - 1) (n/h) V^2 bools per level (10 MB at n = 512, K = 16).

Coupling chi.  The promise side of chi_h is the unary
lam * cost[h][S_h(y, x) flat, v] where S_h is the coordinate grid of the
level (from a CoordPromiseCoupling passed in by hier.run, whose `cost` is
built by `window_costs`).  At init the coordinate level does not exist yet
(promise init runs first), so the prolonged parent coordinates U S_2h are
used; during sweeps the current S_h.  The coordinate side stays in
CoordPromiseCoupling.energy, which reads the promise grid from level.vars.

Hook contract (hier.run): `chain = True`, `init_level(parent, level, ctx,
couplings)`, `sweep_level(level, s, ctx, couplings)`, `sweeps(h)`.
"""
from __future__ import annotations

import numpy as np

from castlegen import hier
from castlegen.promise import Tables
from castlegen.quantities import envelope as EN

INF = np.inf
U_MAX = np.float32(1.0) - np.float32(2.0 ** -24)       # largest float32 below 1


class InfeasibleError(RuntimeError):
    """No consistent assignment exists (clamps / parents admit none)."""


# ------------------------------------------------------------------- chains
def _softmin(x, T, axis=-1):
    """-T log sum exp(-x/T) along axis (min at T = 0); +inf when all +inf."""
    mn = x.min(axis)
    if T <= 0:
        return mn
    fin = np.isfinite(mn)
    safe = np.where(fin, mn, 0.0)
    z = np.exp(-(x - np.expand_dims(safe, axis)) / T)
    with np.errstate(divide="ignore"):
        out = safe - T * np.log(z.sum(axis))
    return np.where(fin, out, INF)


def ffbs(unary, trans, trans_row, T, g=None, what="chain"):
    """Exact draw (T > 0) or MAP (T = 0) of independent chains.

    unary: (N, R, V) energies (+inf = forbidden); trans(y) -> (N or 1, V, V)
    energy of rows y -> y+1 [prev, next]; trans_row(y, prev (N,)) -> (N or 1, V)
    the same rows' slice at the previous values; g: (N, R, V) Gumbel noise
    (T > 0).  Distribution: p(x) ~ exp(-(sum unary + sum trans) / T).
    -> (N, R) int64.  Raises InfeasibleError on an empty feasible set."""
    N, R, V = unary.shape
    m = np.zeros((N, R, V))                        # m[:, y] = cost-to-go of rows > y given x_y
    for y in range(R - 2, -1, -1):
        nxt = unary[:, y + 1] + m[:, y + 1]        # (N, V)
        m[:, y] = _softmin(trans(y) + nxt[:, None, :], T)
    out = np.zeros((N, R), np.int64)
    c = unary[:, 0] + m[:, 0]
    bad = ~np.isfinite(c).any(-1)
    if bad.any():
        raise InfeasibleError(f"{what}: empty feasible set for chain(s) {np.flatnonzero(bad).tolist()}")
    for y in range(R):
        if y > 0:
            c = trans_row(y - 1, out[:, y - 1]) + unary[:, y] + m[:, y]
        if T > 0:
            with np.errstate(invalid="ignore"):
                s = np.where(np.isfinite(c), c / T - g[:, y], INF)
        else:
            s = c
        out[:, y] = np.argmin(s, -1)
        if not np.isfinite(c[np.arange(N), out[:, y]]).all():      # cannot happen for a feasible start
            raise InfeasibleError(f"{what}: dead end at row {y}")
    return out


# --------------------------------------------------------------- the variable
class EnvelopePromise(hier.VarType):
    """Envelope support promise, K <= h <= h_top, sampled per child column.

    tables: promise.Tables with V = lang(I, Q).V (flat when None); T: float
    or callable h -> float; sweeps: column-Gibbs sweeps per level below the
    top; top_sweeps: at h_top; ground: G, the band rows (0 < G < K, or 0 for
    none); clamps: {(h, by, bx): value}; lam: promise-side weight of chi
    (None: the coupling's own lam)."""
    name = "envelope"
    chain = True
    hard = True
    salt = 5

    def __init__(self, K=16, h_top=64, tables=None, T=1.0, sweeps=3, top_sweeps=10, ground=0, I=EN.I_DEFAULT,
                 Q=EN.Q_DEFAULT, clamps=None, lam=None, lang=None):
        self.K, self.h_min, self.h_max = K, K, h_top
        self.lang = EN.lang(I, Q) if lang is None else lang          # any language with Lang's interface
        self.V = self.lang.V
        self.G, self.I, self.Q = ground, I, Q
        assert ground == 0 or 0 < ground < K
        self.tables = tables if tables is not None else Tables.flat(self.V, [K * 2 ** j for j in range(8)])
        assert self.tables.V == self.V, "tables must be fitted on the envelope language"
        self._T, self.n_sweeps, self.top_sweeps = T, sweeps, top_sweeps
        self.clamps = dict(clamps or {})
        self.lam = lam
        self._F, self._Fn = None, None
        self._pm = {}

    # ---- hier.VarType
    def sweeps(self, h):
        return self.top_sweeps if h == self.h_max else self.n_sweeps

    def T(self, h):
        return float(self._T(h) if callable(self._T) else self._T)

    def _seam_base(self, h, R):
        """(R-1, R, V, V) base seam relation of level h: the language's
        seam_compat per row (the seam above the ground row may differ), else COMPAT."""
        lg, V = self.lang, self.V
        if hasattr(lg, "seam_compat"):
            rows = np.stack([lg.seam_compat(h, bool(self.G) and y + 1 == R - 1, self.G) for y in range(R - 1)])
            return np.broadcast_to(rows[:, None], (R - 1, R, V, V)).copy()
        return np.broadcast_to(lg.COMPAT, (R - 1, R, V, V)).copy()

    # ---- beta: feasible sets
    def feasible(self, n):
        """-> (F, S): F {h: (n/h, n/h, V) bool} feasible values per block;
        S {h: (the docstring's B_h) (n/h - 1, n/h, V, V) bool [upper, lower] or None} the
        feasible pairs of every non-wrap vertical seam (None = COMPAT).
        See the module docstring (beta)."""
        if self._Fn == n:
            return self._F, self._S
        lg, V, K = self.lang, self.V, self.K
        for (h, by, bx), v in self.clamps.items():
            assert self.exists(h) and 0 <= by < n // h and 0 <= bx < n // h and 0 <= v < V, f"bad clamp {(h, by, bx)}"
        nparts = lg.P ** (lg.I // 2)
        vals = np.arange(V)
        pv = [lg.parent_part(vals, dx) for dx in (0, 1)]
        F, S, h, prev, Sprev = {}, {}, K, None, None
        while h <= self.h_max:
            R = n // h
            f = np.ones((R, R, V), bool)
            if self.G:
                f[-1] &= lg.valid(vals, True, self.G, h)[None]
            seam = None
            if prev is not None:
                ok, LA, UB = [], [], []
                for dx in (0, 1):
                    Fu, Fl = prev[0::2, dx::2], prev[1::2, dx::2]          # (R, R, V): children of column dx
                    Sin = None if Sprev is None else Sprev[0::2, dx::2]    # seam inside each parent
                    okd = np.zeros((R, R, nparts), bool)
                    la = np.zeros((R, R, nparts, V), bool)                  # lower children usable under part p
                    ub = np.zeros((R, R, nparts, V), bool)                  # upper children usable under part p
                    for p in range(nparts):
                        for gl in ((False, True) if self.G else (False,)):     # last parent row: lower child is ground
                            rows = slice(R - 1, R) if gl else (slice(0, R - 1) if self.G else slice(0, R))
                            u, l = lg.column_pairs(p, False, gl, self.G, h // 2)
                            if len(u) == 0:
                                continue
                            m = Fu[rows][..., u] & Fl[rows][..., l]
                            if Sin is not None:
                                m &= Sin[rows][..., u, l]
                            okd[rows, :, p] = m.any(-1)
                            mf = m.astype(np.float32)
                            la[rows, :, p, :] = mf @ (l[:, None] == vals[None]).astype(np.float32) > 0
                            ub[rows, :, p, :] = mf @ (u[:, None] == vals[None]).astype(np.float32) > 0
                    ok.append(okd)
                    LA.append(la[:, :, pv[dx]])                             # (R, R, V parent, V child)
                    UB.append(ub[:, :, pv[dx]])
                f &= ok[0][:, :, pv[0]] & ok[1][:, :, pv[1]]
                if R > 1:
                    seam = self._seam_base(h, R)
                    for dx in (0, 1):
                        A = LA[dx][:-1].astype(np.float32)                  # upper parents' lower children
                        B = UB[dx][1:].astype(np.float32)                   # lower parents' upper children
                        if Sprev is None:
                            mid = lg.seam_compat(h // 2, False, self.G).astype(np.float32) if hasattr(lg, "seam_compat") \
                                else lg.COMPAT.astype(np.float32)
                        else:
                            mid = Sprev[1:-1:2, dx::2].astype(np.float32)   # child seams between the parents
                        seam &= (A @ mid @ np.swapaxes(B, -1, -2)) > 0
            for (hc, by, bx), v in self.clamps.items():
                if hc == h:
                    keep = f[by, bx, v]
                    f[by, bx] = False
                    f[by, bx, v] = keep
            if R > 1:                                  # arc consistency along every column (exact on a chain)
                if seam is None:
                    seam = self._seam_base(h, R)
                for y in range(R - 2, -1, -1):
                    f[y] &= (seam[y] & f[y + 1][:, None, :]).any(-1)
                for y in range(1, R):
                    f[y] &= (seam[y - 1] & f[y - 1][:, :, None]).any(-2)
                seam &= f[:-1, :, :, None] & f[1:, :, None, :]
            empty = ~f.any(-1)
            if empty.any():
                raise InfeasibleError(f"clamps infeasible: level {h} blocks {np.argwhere(empty).tolist()} have no value")
            F[h], S[h] = f, seam
            prev, Sprev, h = f, seam, 2 * h
        self._F, self._S, self._Fn = F, S, n
        return F, S

    # ---- level energies
    def _pair_masks(self, k, ground):
        """(P^(I/2), V, V) bool pair masks per column part, child side k."""
        key = (k, ground)
        if key not in self._pm:
            lg = self.lang
            self._pm[key] = np.stack([lg.pair_mask(p, False, ground, self.G, k) for p in range(lg.P ** (lg.I // 2))])
        return self._pm[key]

    def _chi(self, level, ctx, couplings, S=None):
        """(R, R, V) promise-side chi energy, or 0."""
        h, out = level.h, 0.0
        for cp in couplings:
            cost = getattr(cp, "cost", None)
            if cost is None or h not in cost:
                continue
            coord = cp.coord.name
            if S is None:
                S = level.vars.get(coord)
            if S is None:
                par = ctx.levels[-1] if ctx.levels else None
                if par is None or par.h != 2 * h or coord not in par.vars:
                    continue
                S = _prolong(par.vars[coord], h, cp.m)
            lam = cp.lam if self.lam is None else self.lam
            out = out + lam * np.asarray(cost[h])[S[..., 0] * cp.m + S[..., 1]]
        return out

    def _setup(self, parent, level, ctx, couplings):
        """Unaries (C, R, V), horizontal table, transition closures."""
        lg, V, h = self.lang, self.V, level.h
        R, C = level.shape
        n = R * h
        u, pair, par = self.tables.level(h)
        E = np.broadcast_to(np.asarray(u, float), (R, C, V)).copy()
        if parent is not None:
            for dy in (0, 1):
                for dx in (0, 1):
                    E[dy::2, dx::2] += np.moveaxis(np.asarray(par[2 * dy + dx], float)[:, parent], 0, -1)
        E += self._chi(level, ctx, couplings)
        Fh, Sh = self.feasible(n)
        Fh, Sh = Fh[h], Sh[h]
        E[~Fh] = INF
        unary = np.ascontiguousarray(np.moveaxis(E, 1, 0))            # (C, R, V)
        p1 = np.asarray(pair[1], float)
        compat = lg.COMPAT
        parts = None
        if parent is not None:
            parts = np.stack([lg.parent_part(parent[:, x // 2], x % 2) for x in range(C)], 1)   # (R/2, C)
        last = R - 1

        def hard(y, cols):                          # (nc or 1, V, V) allowed pairs of rows y -> y+1
            m = compat[None] if Sh is None else Sh[y, cols]
            if parts is not None and y % 2 == 0:
                m = m & self._pair_masks(h, bool(self.G) and y + 1 == last)[parts[y // 2, cols]]
            return m

        def hard_row(y, prev, cols):                # (nc, V) the same at the previous values
            m = compat[prev] if Sh is None else Sh[y, cols, prev]
            if parts is not None and y % 2 == 0:
                m = m & self._pair_masks(h, bool(self.G) and y + 1 == last)[parts[y // 2, cols], prev]
            return m

        trans = lambda y, cols: np.where(hard(y, cols), p1[None], INF)
        trow = lambda y, prev, cols: np.where(hard_row(y, prev, cols), p1[prev], INF)
        return unary, np.asarray(pair[0], float), trans, trow

    def _draw(self, grid, level, ctx, st, cols, horizontal, step, phase):
        """Redraw columns `cols` of grid (R, C) in place, exactly."""
        unary, p0, trans, trow = st
        R, C = grid.shape
        un = unary[cols].copy()
        if horizontal:
            if C == 1:
                un += np.diag(p0)[None, None]
            else:
                left, right = grid[:, (cols - 1) % C].T, grid[:, (cols + 1) % C].T    # (nc, R)
                un += p0[left] + np.moveaxis(p0[:, right], 0, -1)
        T = self.T(level.h)
        g = None
        if T > 0:
            ys = np.arange(R)[None, :, None]
            u = hier.noise(ctx.seed, hier.level_index(level.h), step, phase, ys, cols[:, None, None],
                           np.arange(self.V)[None, None] + (self.salt << 20))
            g = hier.gumbel(np.minimum(u, U_MAX))          # noise can round to exactly 1.0 in float32
        vals = ffbs(un, lambda y: trans(y, cols), lambda y, prev: trow(y, prev, cols), T, g,
                    what=f"envelope level {level.h} columns {cols.tolist()}")
        grid[:, cols] = vals.T

    def init_level(self, parent, level, ctx, couplings=()):
        R, C = level.shape
        st = self._setup(parent, level, ctx, couplings)
        grid = np.zeros((R, C), np.int64)
        even, odd = np.arange(0, C, 2), np.arange(1, C, 2)
        self._draw(grid, level, ctx, st, even, C == 1, hier.INIT_STEP, 2)
        if len(odd):
            self._draw(grid, level, ctx, st, odd, True, hier.INIT_STEP, 3)
        return grid

    def sweep_level(self, level, s, ctx, couplings=()):
        grid = level.vars[self.name]
        h = level.h
        parent = None
        if h < self.h_max:
            parent = next(lv.vars[self.name] for lv in reversed(ctx.levels) if lv.h == 2 * h)
        st = self._setup(parent, level, ctx, couplings)
        C = grid.shape[1]
        for ph, cols in enumerate((np.arange(0, C, 2), np.arange(1, C, 2))):
            if len(cols):
                self._draw(grid, level, ctx, st, cols, True, s, ph)

    # ---- energy / checks
    def level_energy(self, grid, parent, h):
        """Energy of a level grid (no chi, not divided by T; +inf if a hard
        rule or beta mask fails)."""
        lg = self.lang
        u, pair, par = self.tables.level(h)
        g = np.asarray(grid)
        e = float(np.asarray(u)[g].sum()) + float(pair[0][g, np.roll(g, -1, 1)].sum()) + float(pair[1][g[:-1], g[1:]].sum())
        if parent is not None:
            for dy in (0, 1):
                for dx in (0, 1):
                    e += float(par[2 * dy + dx][g[dy::2, dx::2], parent].sum())
        n = g.shape[0] * h
        F, S = self.feasible(n)
        ys, xs = np.arange(g.shape[0])[:, None], np.arange(g.shape[1])[None]
        if not F[h][ys, xs, g].all():
            return INF
        if S[h] is not None and not S[h][ys[:-1], xs, g[:-1], g[1:]].all():
            return INF
        if not lg.consistent(g, parent, self.G, h):
            return INF
        return e

    def satisfaction(self, ts, tiles, levels):
        grids = {lv.h: lv.vars[self.name] for lv in levels if self.name in lv.vars}
        if hasattr(self.lang, "SUB"):                   # average language
            from castlegen.quantities import average as AV
            return AV.satisfaction(ts, tiles, grids, self.K, self.G, self.lang)
        return EN.satisfaction(ts, tiles, grids, self.K, self.G, self.lang)

    # ---- chi: window costs
    def window_costs(self, ts, E, h, viol_w=1.0, viol_cap=4):
        return window_costs(ts, E, h, self.lang, viol_w, viol_cap)


def _prolong(S, h, m):
    """U S_2h: the coordinate prolongation of CoordVar.init_children (no jitter)."""
    out = np.empty((2 * S.shape[0], 2 * S.shape[1], 2), np.int64)
    for dy in (0, 1):
        for dx in (0, 1):
            out[dy::2, dx::2] = S + np.array([(2 * dy - 1) * h // 2, (2 * dx - 1) * h // 2])
    return out % m


def window_costs(ts, E, h, lg=None, viol_w=1.0, viol_cap=4):
    """(m*m, V) float32 graded cost of the h x h exemplar window centred on
    each coordinate u (flat index y*m + x; window top-left u - h//2; torus
    exemplar, no band) honouring each value v:

        sum over intervals |lo bin - LO[v]| + |hi bin - HI[v]|
      + viol_w * min(#columns with a solid cell above their bottom run, viol_cap)

    where the window's per-column local height t is the solid run from the
    window's bottom row (capped at h), binned as in the language."""
    lg = lg or EN.lang()
    s = ts.solid[np.asarray(E)].astype(bool)
    m = s.shape[0]
    I, w = lg.I, h // lg.I
    # run[b, c]: consecutive solid cells ending at row b going up (torus), capped at h
    run = np.zeros((m, m), np.int64)
    cur = np.zeros(m, np.int64)
    for _ in range(2):
        for b in range(m):
            cur = np.where(s[b], np.minimum(cur + 1, h), 0)
            run[b] = cur
    cnt = np.zeros((m, m), np.int64)                       # solid cells in rows b-h+1 .. b
    for i in range(h):
        cnt += np.roll(s, i, 0)
    viol_col = cnt > run                                   # indexed by bottom row b, column c
    c0 = (np.arange(m) - h // 2) % m
    ty, tx = np.meshgrid(c0, c0, indexing="ij")
    by = (ty + h - 1) % m
    lo, hi = [], []
    for i in range(I):
        mn = np.full((m, m), h, np.int64)
        mx = np.zeros((m, m), np.int64)
        for j in range(w):
            t = run[by, (tx + i * w + j) % m]
            mn, mx = np.minimum(mn, t), np.maximum(mx, t)
        lo.append(lg.bins(mn, h))
        hi.append(lg.bins(mx, h))
    lo, hi = np.stack(lo, -1).reshape(-1, I), np.stack(hi, -1).reshape(-1, I)
    nv = np.zeros((m, m), np.int64)
    for j in range(h):
        nv += viol_col[by, (tx + j) % m]
    cost = (np.abs(lo[:, None, :] - lg.LO[None]) + np.abs(hi[:, None, :] - lg.HI[None])).sum(-1)
    cost = cost + viol_w * np.minimum(nv.reshape(-1), viol_cap)[:, None]
    return cost.astype(np.float32)
