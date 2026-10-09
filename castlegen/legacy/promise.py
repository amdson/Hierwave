"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Promise variables: hierarchical variables with an exact bottom-up summary
and hard consistency (notes/history/promises.md sections 3.2, 5, 7).

A promise level is a grid of block values; the level at cell side h has side
n/h and exists for K <= h <= h_top.  Top-down, the coarsest level is sampled
by the variable's own `top` (a small Gibbs with a global check), every finer
level starts from a consistent refinement of its parent (`init_children`) and
is then swept per parent: hier.run sees `group = 2`, so its colours run over
the grid of parents, a candidate is a whole joint update proposed by
`proposals` (the parent's four children, plus whatever seam bits the variable
lets a parent own on its neighbours), and `assign` writes the picked one back.
Candidates that break consistency are never proposed, so every level is
consistent at every step.

Energies come from `Tables` (empirical counts with additive smoothing, fitted
by castlegen.legacy.corpus): per level a unary u(v), a pair term per orientation
(negative pointwise mutual information of the two blocks of a seam) and a
parent term per child position (negative PMI of child and parent).  The
candidate energy is the sum of the terms touching the cells it sets.

Bottom-up, a concrete block has an exact `summarize`d state; `merge` combines
four children's states into the parent's, `abstract` maps a state into the
promise domain, `satisfied(value, state)` is the contract `fulfil` enforces on
a K-block.  CoordPromiseCoupling is the coordinate <-> promise energy of
section 3.4.
"""
from __future__ import annotations

import os

import numpy as np

from castlegen.legacy import hier


# ---------------------------------------------------------------------- tables
class Tables:
    """Energy tables per level h: u[h] (V,), pair[h] (2, V, V) indexed
    [orientation (0: left|right, 1: top/bottom), a, b], parent[h] (4, V, V)
    indexed [child position 2*dy + dx, child, parent]."""

    def __init__(self, V, u, pair, parent, counts=None):
        self.V, self.u, self.pair, self.parent = V, u, pair, parent
        self.counts = counts or {}

    @classmethod
    def flat(cls, V, hs):
        z = {h: np.zeros(V) for h in hs}
        return cls(V, z, {h: np.zeros((2, V, V)) for h in hs}, {h: np.zeros((4, V, V)) for h in hs})

    @classmethod
    def fit(cls, V, grids, alpha=0.5):
        """grids: {h: [value grid (n/h, n/h), ...]} on tori.  Counts with
        additive smoothing alpha; pair and parent terms are negative PMI."""
        u, pair, parent, counts = {}, {}, {}, {}
        for h, gs in grids.items():
            cu = np.zeros(V)
            cp = np.zeros((2, V, V))
            for g in gs:
                np.add.at(cu, g.ravel(), 1)
                np.add.at(cp[0], (g.ravel(), np.roll(g, -1, 1).ravel()), 1)
                np.add.at(cp[1], (g.ravel(), np.roll(g, -1, 0).ravel()), 1)
            pu = (cu + alpha) / (cu.sum() + V * alpha)
            u[h] = -np.log(pu)
            pp = (cp + alpha) / (cp.sum((1, 2), keepdims=True) + V * V * alpha)
            pair[h] = -np.log(pp) + np.log(pp.sum(2, keepdims=True)) + np.log(pp.sum(1, keepdims=True))
            counts[f"u{h}"], counts[f"pair{h}"] = cu, cp
            if 2 * h in grids:
                cc = np.zeros((4, V, V))
                for g, G in zip(gs, grids[2 * h]):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            np.add.at(cc[2 * dy + dx], (g[dy::2, dx::2].ravel(), G.ravel()), 1)
                pc = (cc + alpha) / (cc.sum(1, keepdims=True) + V * alpha)          # p(child | parent, pos)
                parent[h] = -np.log(pc) + np.log(pu)[None, :, None]
                counts[f"parent{h}"] = cc
            else:
                parent[h] = np.zeros((4, V, V))
        return cls(V, u, pair, parent, counts)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrs = {"V": np.array(self.V)}
        for h in self.u:
            arrs[f"u{h}"], arrs[f"pairE{h}"], arrs[f"parentE{h}"] = self.u[h], self.pair[h], self.parent[h]
        for k, v in self.counts.items():
            arrs["count_" + k] = v
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        hs = sorted(int(k[1:]) for k in z.files if k.startswith("u"))
        counts = {k[6:]: z[k] for k in z.files if k.startswith("count_")}
        return cls(int(z["V"]), {h: z[f"u{h}"] for h in hs}, {h: z[f"pairE{h}"] for h in hs},
                   {h: z[f"parentE{h}"] for h in hs}, counts)

    def level(self, h):
        """Tables for level h (the nearest fitted level if h was not fitted)."""
        hs = sorted(self.u)
        k = min(hs, key=lambda x: abs(np.log2(x) - np.log2(h)))
        return self.u[k], self.pair[k], self.parent[k]


# ------------------------------------------------------------------ variable
class PromiseVar(hier.VarType):
    """Base class.  Subclasses define the promise language: `summarize`,
    `merge`, `abstract`, `satisfied`, `merge_promises`, `consistent`,
    `refine`, `init_children`, `top`, `proposals` and `fulfil`."""
    hard = True
    group = 2
    V = 16                                      # domain size

    def __init__(self, K=16, h_top=64, tables=None, T=1.0, sweeps=3):
        self.K, self.h_min, self.h_max = K, K, h_top
        self.tables = tables if tables is not None else Tables.flat(self.V, [K * 2 ** j for j in range(8)])
        self._T, self.n_sweeps = T, sweeps
        self._cur = None

    def sweeps(self, h):
        return 0 if h == self.h_max else self.n_sweeps

    def T(self, h):
        return self._T

    # ---- the per-parent update --------------------------------------------
    def proposals(self, grid, parents, py, px):
        """-> (cells [(y, x)], [values tuple per candidate]): the cells a joint
        update of parent (py, px) may set, and every consistent assignment."""
        raise NotImplementedError

    def candidates(self, level, colour, ctx):
        """(a, b, nC, M) candidate values for the M cells of each parent of
        the colour; the cells are kept in self._cur for energy, couplings and
        assign.  Padding candidates are marked invalid (energy +inf)."""
        grid = level.vars[self.name]
        parents = ctx.levels[-1].vars[self.name]
        i, j = colour
        pys, pxs = np.arange(i, parents.shape[0], 2), np.arange(j, parents.shape[1], 2)
        props = [[self.proposals(grid, parents, py, px) for px in pxs] for py in pys]
        M = max(len(c) for row in props for c, _ in row)
        nC = max(max(len(v), 1) for row in props for _, v in row)
        a, b = len(pys), len(pxs)
        cells = np.zeros((a, b, M, 2), np.int64)
        cmask = np.zeros((a, b, M), bool)
        valid = np.zeros((a, b, nC), bool)
        C = np.zeros((a, b, nC, M), np.int64)
        for ia, row in enumerate(props):
            for ib, (cl, vals) in enumerate(row):
                seen = set()
                for m, c in enumerate(cl):
                    cells[ia, ib, m] = c
                    cmask[ia, ib, m] = c not in seen
                    seen.add(c)
                for m in range(len(cl), M):              # pad with a repeat of the first cell
                    cells[ia, ib, m] = cl[0]
                cur = tuple(int(grid[y, x]) for y, x in cl)
                vals = np.asarray(vals if len(vals) else [cur], np.int64).reshape(-1, len(cl))
                C[ia, ib, :len(vals), :len(cl)] = vals   # never empty: keep the current assignment
                C[ia, ib, :len(vals), len(cl):] = vals[:, :1]
                valid[ia, ib, :len(vals)] = True
                C[ia, ib, len(vals):] = C[ia, ib, 0]
        self._cur = dict(cells=cells, cmask=cmask, valid=valid, parents=parents)
        return C

    def energy(self, C, level, colour, ctx):
        u, pair, par = self.tables.level(level.h)
        grid, parents = level.vars[self.name], self._cur["parents"]
        cells, cmask, valid = self._cur["cells"], self._cur["cmask"], self._cur["valid"]
        g0, g1 = grid.shape
        e = np.full(valid.shape, np.inf)
        for ia, ib in np.ndindex(*valid.shape[:2]):
            cl = [tuple(c) for c, m in zip(cells[ia, ib].tolist(), cmask[ia, ib]) if m]
            idx = [k for k, m in enumerate(cmask[ia, ib]) if m]
            seams = set()
            for y, x in cl:
                seams |= {(y, x, 0), (y, (x - 1) % g1, 0), (y, x, 1), ((y - 1) % g0, x, 1)}
            ks = np.flatnonzero(valid[ia, ib])
            col = {c: C[ia, ib, ks, m] for c, m in zip(cl, idx)}           # (nk,) value of each set cell
            val = lambda y, x: col[y, x] if (y, x) in col else grid[y, x]
            tot = np.zeros(len(ks))                                     # same summation order as a scalar loop
            for y, x in cl:
                v = col[y, x]
                tot = tot + (u[v] + par[2 * (y % 2) + x % 2, v, parents[y // 2, x // 2]])
            for y, x, o in seams:
                ny, nx = (y, (x + 1) % g1) if o == 0 else ((y + 1) % g0, x)
                tot = tot + pair[o, val(y, x), val(ny, nx)]
            e[ia, ib, ks] = tot
        return e

    def assign(self, level, colour, values, ctx):
        grid = level.vars[self.name]
        cells, cmask = self._cur["cells"], self._cur["cmask"]
        for ia, ib in np.ndindex(*values.shape[:2]):
            for m in np.flatnonzero(cmask[ia, ib]):
                y, x = cells[ia, ib, m]
                grid[y, x] = values[ia, ib, m]

    def group_cells(self):
        """(cells (a, b, M, 2), mask (a, b, M)) of the update in progress, for couplings."""
        return self._cur["cells"], self._cur["cmask"]

    # ---- bottom-up ----------------------------------------------------------
    def level_states(self, ts, tiles, h):
        """Exact states of every block of side h, built by merging from K."""
        n = tiles.shape[0]
        g = n // self.K
        S = [[self.summarize(ts, tiles, y * self.K, x * self.K, self.K) for x in range(g)] for y in range(g)]
        k = self.K
        while k < h:
            g //= 2
            S = [[self.merge({(dy, dx): S[2 * y + dy][2 * x + dx] for dy in (0, 1) for dx in (0, 1)})
                  for x in range(g)] for y in range(g)]
            k *= 2
        return S

    def satisfaction(self, ts, tiles, levels):
        """{h: fraction of blocks whose exact state satisfies the promise held
        at that level} for every promise level in `levels` (hier.Level list)."""
        n = tiles.shape[0]
        g = n // self.K
        S = [[self.summarize(ts, tiles, y * self.K, x * self.K, self.K) for x in range(g)] for y in range(g)]
        by_h = {lv.h: lv.vars[self.name] for lv in levels if self.name in lv.vars}
        out, k = {}, self.K
        while True:
            if k in by_h:
                P = by_h[k]
                out[k] = float(np.mean([self.satisfied(int(P[y, x]), S[y][x]) for y in range(g) for x in range(g)]))
            if g == 1 or 2 * k > max(by_h, default=0):
                break
            g //= 2
            S = [[self.merge({(dy, dx): S[2 * y + dy][2 * x + dx] for dy in (0, 1) for dx in (0, 1)})
                  for x in range(g)] for y in range(g)]
            k *= 2
        return out


# ----------------------------------------------------------------- coupling
class CoordPromiseCoupling(hier.Coupling):
    """lam * cost[h][u, promise]: the estimated cost of making the h x h
    exemplar window centred on coordinate u honour the promise (0 when it
    already does).  At h >= K it couples a coordinate cell with the promise of
    the same cell; at h < K with the promise of its ancestor K-block, through
    the K-window the candidate coordinate implies (the cell's window shifted
    back by the cell's offset inside the block).

    cost: {h: (m*m, V) array}, built by the promise type's `window_costs`."""

    def __init__(self, coord, promise, cost, lam=1.0, m=None, bounded=False):
        self.coord, self.promise, self.cost, self.lam = coord, promise, cost, lam
        self.vars = (coord.name, promise.name)
        self.m, self.bounded = m, bounded           # bounded: rows clamp (texsyn.Analysis bounds="edge")

    def energy(self, var, C, lv, colour, ctx):
        h, K, m = lv.h, self.promise.K, self.m
        if var.name == self.promise.name:
            if self.coord.name not in lv.vars or h not in self.cost:
                return 0.0
            cells, cmask = var.group_cells()
            S = lv.vars[self.coord.name][cells[..., 0], cells[..., 1]]          # (a, b, M, 2)
            u = S[..., 0] * m + S[..., 1]
            c = self.cost[h][u[:, :, None, :], C]                               # (a, b, nC, M)
            return self.lam * (c * cmask[:, :, None, :]).sum(-1)
        # coordinate candidates C: (a, b, nC, 2)
        i, j = colour
        if h >= K:
            if self.promise.name not in lv.vars or h not in self.cost:
                return 0.0
            O = lv.vars[self.promise.name][i::2, j::2]
            u = C[..., 0] * m + C[..., 1]
            return self.lam * self.cost[h][u, O[:, :, None]]
        anc = next((l for l in ctx.levels if l.h == K and self.promise.name in l.vars), None)
        if anc is None or K not in self.cost:
            return 0.0
        r = K // h
        ys, xs = hier.colour_coords(lv.shape, colour)
        O = anc.vars[self.promise.name][ys // r, xs // r]                     # (a, b)
        oy, ox = (ys % r) * h, (xs % r) * h
        cy = C[..., 0] - h // 2 - oy[..., None] + K // 2
        cy = np.clip(cy, 0, m - 1) if self.bounded else cy % m
        cx = (C[..., 1] - h // 2 - ox[..., None] + K // 2) % m
        return self.lam * self.cost[K][cy * m + cx, O[:, :, None]]
