"""ActivityAIS: the collapsed target over a child whose structures are
sparse and connected (limbs, roots), by annealing the activity instead of
the couplings.

AISTargets anneals the same-level rows from zero, so its p_0 is independent
sites: for a limb set that is a dense gas of unattached limb cells which
the anneal has to dissolve, and at any affordable K two seeds disagree by
several nats (notes/trees_test.md, stage B).  Here every intermediate
distribution keeps the child's rows at full strength and adds (1 - beta) *
MU to every active value (a limb kind):

    f_beta(z) = exp(-E(z) - (1 - beta) MU n_act(z)),   beta: 0 -> 1

p_0 (beta = 0) is, up to O(e^-MU), the point mass on the inactive state
(sky everywhere free in the window, the forced cells as forced), so
log Z_0 = -E(z_0) exactly computed; the chain grows the structure under the
shipping kernel (all couplings on, including hard ones) as the activity
rises, and the AIS weights are exp(sum_k (beta_{k+1} - beta_k) MU
n_act(z_k)).  Schedule: beta_k = 1 - (1 - k / K)^power, denser near 1
where the structure appears.  The certificate is not part of the model
seen here (the trainer passes the soft model); generation keeps it.

Targets protocol as targets.py: at(model, home, y, x, cand) -> pi over
cand, pi(t) = softmax(-(designed_e(t) - log Z_t)), with region, designed_e,
after_change as in AISTargets.  The child's cells outside the region are
held fixed; cells inside start inactive (value `inactive`, or the forced
value where a hard row to another channel admits one value only: the p_0
draw is the argmin of the unary-like rows)."""
import time
import weakref

import numpy as np
from numba import njit

from . import kernel
from .core import PAIR, UNARY
from .induce import ais_se, logmeanexp


@njit(cache=True)
def _window(y0, y1, x0, x1, fixed):
    nf = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if not fixed[y, x]:
                nf += 1
    ys = np.empty(nf, np.int64)
    xs = np.empty(nf, np.int64)
    k = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if not fixed[y, x]:
                ys[k], xs[k] = y, x
                k += 1
    return ys, xs


@njit(cache=True)
def _z0(ys, xs, home, grids, hs, views, fac0, tabs, act, D):
    """Every free cell at its cheapest value under the unary-like rows with
    the activity at MU: the inactive state, forced cells as forced.
    Written into the grid and returned."""
    g = grids[home]
    e = np.empty(D)
    z0 = np.empty(ys.shape[0], np.int64)
    for i in range(ys.shape[0]):
        kernel._energies(ys[i], xs[i], home, grids, hs, views, fac0, tabs, e)
        best, bt = np.inf, 0
        for t in range(D):
            v = e[t] + act[t]
            if v < best:
                best, bt = v, t
        z0[i] = bt
        g[ys[i], xs[i]] = bt
    return z0


@njit(cache=True)
def _ais_act(ys, xs, z0, home, grids, hs, views, fac, tabs, act, D, betas, logw):
    """AIS over the free cells (ys, xs) from z0 along betas; fills logw (M,)."""
    g = grids[home]
    M = logw.shape[0]
    nf = ys.shape[0]
    e = np.empty(D)
    K = betas.shape[0] - 1
    for m in range(M):
        for i in range(nf):
            g[ys[i], xs[i]] = z0[i]
        lw = 0.0
        for kk in range(K):
            beta = betas[kk]
            for i in range(nf):                                   # one sweep leaving f_beta invariant
                y, x = ys[i], xs[i]
                kernel._energies(y, x, home, grids, hs, views, fac, tabs, e)
                mn = np.inf
                for t in range(D):
                    e[t] += (1.0 - beta) * act[t]
                    if e[t] < mn:
                        mn = e[t]
                acc = 0.0
                for t in range(D):
                    e[t] = np.exp(-(e[t] - mn))
                    acc += e[t]
                u = np.random.random() * acc
                t = 0
                s = e[0]
                while t < D - 1 and s <= u:
                    t += 1
                    s += e[t]
                g[y, x] = t
            nact = 0.0
            for i in range(nf):
                nact += act[g[ys[i], xs[i]]]
            lw += (betas[kk + 1] - beta) * nact
        logw[m] = lw


@njit(cache=True)
def _ais_act_cert(ys, xs, z0, home, grids, hs, views, fac, tabs, act, D, betas, logw, cert, joins, delta):
    """_ais_act with the certificate: the kernel's site_weights at every
    step (all constraints on, only the activity annealed), d redrawn with
    the value as kernel.sweep does.  z0's d: INF on mass 0, 0 on trunks."""
    g = grids[home]
    mass, trunk, g_d = views[cert[0]], views[cert[1]], grids[cert[2]]
    Dmax = cert[3]
    tree = cert[5] == 1
    M = logw.shape[0]
    nf = ys.shape[0]
    e = np.empty(D)
    w = np.empty(D)
    lo = np.empty(D, np.int64)
    hi = np.empty(D, np.int64)
    need = np.empty(4, np.bool_)
    K = betas.shape[0] - 1
    for m in range(M):
        for i in range(nf):
            g[ys[i], xs[i]] = z0[i]
            g_d[ys[i], xs[i]] = 0 if trunk[z0[i]] else Dmax + 1
        lw = 0.0
        for kk in range(K):
            beta = betas[kk]
            for i in range(nf):
                y, x = ys[i], xs[i]
                kernel._energies(y, x, home, grids, hs, views, fac, tabs, e)
                for t in range(D):
                    e[t] += (1.0 - beta) * act[t]
                kernel.site_weights(y, x, e, g, g_d, mass, trunk, joins, Dmax, delta, 1.0, w, lo, hi, need, tree)
                pick = kernel._draw(w, D)
                if pick >= 0:
                    g[y, x] = pick
                    g_d[y, x] = lo[pick] + kernel._tgeom(hi[pick] - lo[pick], delta)
            nact = 0.0
            for i in range(nf):
                nact += act[g[ys[i], xs[i]]]
            lw += (betas[kk + 1] - beta) * nact
        logw[m] = lw


class _Packing:
    def __init__(self, model, child):
        self.ref = weakref.ref(model)
        self.key = self._key(model)
        P = model.compile(child)
        self.cert = P.cert[4] == 1
        assert not (P.fac[:, 0] == 5).any(), "ActivityAIS: no convpot rows"
        self.P = P
        cidx = P.home
        unary_like = (P.fac[:, 0] == UNARY) | ((P.fac[:, 0] == PAIR) & (P.fac[:, 1] != cidx))
        self.fac0 = np.ascontiguousarray(P.fac[unary_like])
        self.D = model.chan(child).D

    @staticmethod
    def _key(model):
        return (tuple(id(c.grid) for c in model.channels), tuple(id(f.table) for f in model.factors))

    def valid(self, model):
        return self.ref() is model and self.key == self._key(model)


def boundary_keep(model, P, child, y0, y1, x0, x1):
    """(rows, cols) bool: window cells on the parent chain of a cell outside
    the window that depends on one inside (the tree certificate: a limb's
    parent is its joined neighbour of smaller d and mass >= its own).  Those
    cells stay fixed at their value, so the emptied window is a valid state
    given the boundary."""
    c = model.chan(child)
    g, g_d = c.grid, model.channels[P.cert[2]].grid
    mass, trunk, joins = P.views[P.cert[0]], P.views[P.cert[1]], P.joins
    rows, cols = g.shape
    inside = np.zeros((rows, cols), bool)
    inside[y0:y1, x0:x1] = True
    keep = np.zeros((rows, cols), bool)
    DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def parent(y, x):
        t, m, dd = g[y, x], mass[g[y, x]], g_d[y, x]
        for d, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                tq = g[ny, nx]
                if mass[tq] >= m and g_d[ny, nx] < dd and joins[d, t, tq]:
                    return ny, nx
        return None

    border = np.zeros((rows, cols), bool)                          # outside cells 4-adjacent to the window
    for dy, dx in DIRS:
        sh = np.zeros((rows, cols), bool)
        ys, xs = slice(max(dy, 0), rows + min(dy, 0)), slice(max(dx, 0), cols + min(dx, 0))
        yq, xq = slice(max(-dy, 0), rows + min(-dy, 0)), slice(max(-dx, 0), cols + min(-dx, 0))
        sh[ys, xs] = inside[yq, xq]
        border |= sh & ~inside
    for y, x in zip(*np.nonzero(border)):
        if mass[g[y, x]] == 0 or trunk[g[y, x]]:
            continue
        p = parent(y, x)
        while p is not None and inside[p] and not keep[p]:
            keep[p] = True
            if trunk[g[p]]:
                break
            p = parent(*p)
    return keep


def betas_power(K, power=2.0):
    u = np.linspace(0.0, 1.0, K + 1)
    return 1.0 - (1.0 - u) ** power


class ActivityAIS:
    """See the module docstring.  act: (D,) activity penalty per child value
    (MU on the active kinds, 0 elsewhere)."""

    def __init__(self, child, region, designed_e, after_change, act, K=100, M=8, power=2.0, seed=0):
        self.child, self.region, self.designed_e, self.after_change = child, region, designed_e, after_change
        self.act = np.ascontiguousarray(act, np.float64)
        self.K, self.M, self.power = int(K), int(M), float(power)
        self.betas = betas_power(self.K, self.power)
        self.rng = np.random.default_rng(seed)
        self._pk = None
        self.n_calls = self.n_ais = self.n_dead = self.n_keep = 0
        self.t_calls = self.t_ais = 0.0
        self.se = []

    def packing(self, model):
        if self._pk is None or not self._pk.valid(model):
            self._pk = _Packing(model, self.child)
        return self._pk

    def log_z(self, model, home, y, x, cand):
        """(log Z_t (n,), standard error (n,)) over cand."""
        t0 = time.perf_counter()
        cand = np.asarray(cand, np.int64)
        n = len(cand)
        lz, se = np.zeros(n), np.zeros(n)
        hg = model.chan(home).grid
        c = model.chan(self.child)
        rows, cols = c.grid.shape
        y0, y1, x0, x1 = self.region(model, home, y, x)
        y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, rows), min(x1, cols)
        old = int(hg[y, x])
        saved = c.grid[y0:y1, x0:x1].copy()
        pk = self.packing(model)
        P = pk.P
        dchan = model.channels[P.cert[2]] if pk.cert else None
        saved_d = dchan.grid[y0:y1, x0:x1].copy() if pk.cert else None
        fixed0 = c.fixed.copy()
        c.fixed[:] = True
        c.fixed[y0:y1, x0:x1] = fixed0[y0:y1, x0:x1]
        if pk.cert:
            keep = boundary_keep(model, P, self.child, y0, y1, x0, x1)
            self.n_keep += int(keep.sum())
            c.fixed |= keep
        try:
            for i, t in enumerate(cand):
                hg[y, x] = t
                self.after_change(model, home, y, x)
                assert np.shares_memory(P.grids[P.home], c.grid)
                logw = np.zeros(self.M)
                kernel.seed(int(self.rng.integers(1 << 30)))
                ta = time.perf_counter()
                ys, xs = _window(y0, y1, x0, x1, c.fixed)
                z0 = _z0(ys, xs, P.home, P.grids, P.hs, P.views, pk.fac0, P.tabs, self.act, pk.D)
                if pk.cert:
                    g_d, trunk = P.grids[P.cert[2]], P.views[P.cert[1]]
                    g_d[ys, xs] = np.where(trunk[z0] == 1, 0, P.cert[3] + 1)
                e0, ninf = kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.cert, P.joins,
                                               P.delta, P.convs)
                if ninf == 0 and pk.cert:
                    _ais_act_cert(ys, xs, z0, P.home, P.grids, P.hs, P.views, P.fac, P.tabs, self.act, pk.D,
                                  self.betas, logw, P.cert, P.joins, P.delta)
                elif ninf == 0:
                    _ais_act(ys, xs, z0, P.home, P.grids, P.hs, P.views, P.fac, P.tabs, self.act, pk.D, self.betas,
                             logw)
                self.t_ais += time.perf_counter() - ta
                self.n_ais += 1
                lz[i] = -e0 + logmeanexp(logw) if ninf == 0 else -np.inf
                se[i] = ais_se(logw) if ninf == 0 else 0.0
                c.grid[y0:y1, x0:x1] = saved
                if pk.cert:
                    dchan.grid[y0:y1, x0:x1] = saved_d
        finally:
            hg[y, x] = old
            c.grid[y0:y1, x0:x1] = saved
            if pk.cert:
                dchan.grid[y0:y1, x0:x1] = saved_d
            c.fixed[:] = fixed0
            self.after_change(model, home, y, x)
        self.n_calls += 1
        self.t_calls += time.perf_counter() - t0
        return lz, se

    def at(self, model, home, y, x, cand):
        cand = np.asarray(cand, np.int64)
        e = np.asarray(self.designed_e(model, home, y, x, cand), np.float64)
        lz, se = self.log_z(model, home, y, x, cand)
        self.se += [float(s) for s in se[np.isfinite(lz)]]
        a = -(e - lz)
        ok = np.isfinite(a)
        if not ok.any():                                          # the fixed boundary admits no window state
            self.n_dead += 1
            return np.full(len(cand), np.nan)
        m = a[ok].max()
        p = np.where(ok, np.exp(np.where(ok, a, m) - m), 0.0)
        return p / p.sum()
