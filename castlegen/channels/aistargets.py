"""AISTargets: p*'s collapsed conditional at one coarse site by AIS on the
one-site window (notes/circles_biome_test.md, "Contracts fixed before the
build"; notes/dsl_updates.md C2 step 2; reference_math section 4).  Generic:
the model enters through callbacks only.  The Targets protocol (duck-typed):

    at(model, home, y, x, cand) -> (len(cand),) float64 summing to 1
        pi(t) = softmax_t(-(E_designed(t) - log Z_t)),  t in cand

log Z_t = log of the partition function of the child channel over the fine
region R(p) = region(model, home, y, x), with z_p = t, the painted channels
repainted (after_change), and every child cell outside R(p) held fixed at
its current value: the child's `fixed` mask is set True outside R(p) for
the call (cells already fixed inside R(p) stay fixed) and restored after.
Only differences of log Z_t across t at one site enter pi, so the free
energy of the child cells outside R(p) cancels; but child cells outside
R(p) coupled to cells inside it by the child's own factors must be held at
their values with those coupling rows kept, which is what the fixed mask
does (the boundary terms are in E).  Where the child cells are independent
given the parents (circles) R(p) can be exactly the written cells plus the
painters' spill; with same-level child factors, R(p) is the written cells
dilated by their reach (or the written cells alone: both are exact
conditionals of p*, conditioning on different fine boundaries).

Z convention: Z_t = sum over the free cells s of R(p) of exp(-E), E = the
child packing's terms that touch at least one free cell (unaries, rows to
other channels, same-level pairs including those to fixed boundary cells,
counts over blocks holding a free cell).  This is the exact window free
energy of reference_math (4), comparable to an exact hook in absolute value.

AIS (as induce.ais_log_z, restricted to R(p)): p_0 = the rows of the child
packing that are per-site unaries given the rest of the state, at full
strength with inf kept: the child's unaries and every pair row reading
another channel (a parent honour like circles' kappa or Potts' kappa reads
the parent / a painted channel, which the child's sweeps never change);
sampled exactly, log Z_0 in closed form.  E_rest = the remaining rows
(same-level pairs, counts), inf -> L, annealed E_beta = E_0 + beta E_rest
over betas_of(K, schedule), one window sweep of the child kernel per step,
M chains.  anneal_names overrides: rows named are annealed (an other-channel
pair may be; a same-level row must be).  For the old circles the default
anneals nothing (tiles independent given dem) and AIS is exact.

hook(model, home, y, x) -> log Z of the current state under the same
convention replaces AIS (e.g. circles' per-tile closed form, Potts'
transfer matrix)."""
import time
import weakref

import numpy as np
from numba import njit

from . import kernel
from .core import COUNT, PAIR, UNARY
from .induce import ais_se, betas_of, logmeanexp


# ------------------------------------------------------------- numba core
@njit(cache=True)
def _erest(y0, y1, x0, x1, home, grids, hs, views, fac, tabs, fixed):
    """Energy of rows `fac` (pairs / counts of home) over the cells of the
    window [y0, y1) x [x0, x1) (clipped), each term once, terms touching no
    free cell skipped."""
    g = grids[home]
    rows, cols = g.shape
    hc = hs[home]
    y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, rows), min(x1, cols)
    tot = 0.0
    for y in range(y0, y1):
        for x in range(x0, x1):
            t = g[y, x]
            pf = not fixed[y, x]
            for f in range(fac.shape[0]):
                av = views[fac[f, 2]]
                tab = tabs[fac[f, 7]]
                b = fac[f, 1]
                hb = hs[b]
                gb = grids[b]
                if fac[f, 0] == 0:
                    if b == home and (fac[f, 4] < 0 or (fac[f, 4] == 0 and fac[f, 5] < 0)):
                        continue
                    qy = (y * hc) // hb + fac[f, 4]
                    qx = (x * hc) // hb + fac[f, 5]
                    if 0 <= qy < gb.shape[0] and 0 <= qx < gb.shape[1]:
                        vb = views[fac[f, 3]][gb[qy, qx]]
                        qf = b == home and not fixed[qy, qx]
                    elif fac[f, 6] >= 0:
                        vb = fac[f, 6]
                        qf = False
                    else:
                        continue
                    if pf or qf:
                        tot += tab[av[t], vb]
                else:                                           # count: once per block, at its first window cell
                    r = hb // hc
                    by, bx = (y // r) * r, (x // r) * r
                    if y != max(by, y0) or x != max(bx, x0):
                        continue
                    s = 0
                    anyf = False
                    for yy in range(by, by + r):
                        for xx in range(bx, bx + r):
                            s += av[g[yy, xx]]
                            anyf = anyf or not fixed[yy, xx]
                    if anyf:
                        tot += tab[views[fac[f, 3]][gb[(y * hc) // hb, (x * hc) // hb]], s]
    return tot


@njit(cache=True)
def _ais(y0, y1, x0, x1, wy0, wy1, wx0, wx1, home, grids, hs, views, fac0, fac1, tabs, fixed, colours, ncol,
         betas, D, logw):
    """AIS over the free cells of [y0, y1) x [x0, x1) of `home`; fills logw
    (M,) and returns log Z_0 (-inf if some free cell has no finite p_0
    value).  E_rest is summed over the window w* (R dilated by the reach)."""
    g = grids[home]
    M = logw.shape[0]
    nf = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if not fixed[y, x]:
                nf += 1
    ys = np.empty(nf, np.int64)
    xs = np.empty(nf, np.int64)
    cdf = np.empty((nf, D))
    e = np.empty(D)
    e1 = np.empty(D)
    logz0 = 0.0
    k = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if fixed[y, x]:
                continue
            kernel._energies(y, x, home, grids, hs, views, fac0, tabs, e)
            mn = e.min()
            if mn == np.inf:
                logw[:] = 0.0
                return -np.inf
            acc = 0.0
            for t in range(D):
                acc += np.exp(-(e[t] - mn))
                cdf[k, t] = acc
            logz0 += np.log(acc) - mn
            ys[k], xs[k] = y, x
            k += 1
    K = betas.shape[0] - 1
    for m in range(M):
        for k in range(nf):
            u = np.random.random() * cdf[k, D - 1]
            t = 0
            while t < D - 1 and cdf[k, t] <= u:
                t += 1
            g[ys[k], xs[k]] = t
        lw = 0.0
        if fac1.shape[0] > 0:
            for kk in range(K):
                b = betas[kk]
                for col in range(ncol):
                    for k in range(nf):
                        y, x = ys[k], xs[k]
                        if colours[y, x] != col:
                            continue
                        kernel._energies(y, x, home, grids, hs, views, fac0, tabs, e)
                        if b > 0.0:
                            kernel._energies(y, x, home, grids, hs, views, fac1, tabs, e1)
                            for t in range(D):
                                e[t] += b * e1[t]
                        pick = kernel._draw(e, D)
                        if pick >= 0:
                            g[y, x] = pick
                lw -= (betas[kk + 1] - b) * _erest(wy0, wy1, wx0, wx1, home, grids, hs, views, fac1, tabs, fixed)
        logw[m] = lw
    return logz0


# ------------------------------------------------------------ the targets
class _Packing:
    """The child packing split into p_0 rows (fac0) and annealed rows (fac1,
    inf -> L), cached per model while its grids and tables are the same."""

    def __init__(self, model, child, anneal_names, L):
        self.ref = weakref.ref(model)
        self.key = self._key(model)
        P = model.compile(child)
        assert P.cert[4] == 0, "AISTargets: certificates are not annealed"
        assert not (P.fac[:, 0] == 5).any(), "AISTargets: convpot rows are not annealed"
        cidx = P.home
        unary_like = (P.fac[:, 0] == UNARY) | ((P.fac[:, 0] == PAIR) & (P.fac[:, 1] != cidx))
        names = [model.factors[s].name for s in P.src]
        if anneal_names is None:
            ann = ~unary_like
        else:
            ann = np.array([n in anneal_names for n in names], bool)
            assert not (P.fac[ann, 0] == UNARY).any(), "a unary is part of p_0"
            bad = [n for n, a, u in zip(names, ann, unary_like) if not a and not u]
            assert not bad, f"{bad}: same-level rows of {child} must be annealed"
        assert np.isin(P.fac[ann, 0], (PAIR, COUNT)).all()
        self.anneal_names = sorted({n for n, a in zip(names, ann) if a})
        tabs = list(P.tabs)
        for f in np.flatnonzero(ann):
            ti = P.fac[f, 7]
            tabs[ti] = np.ascontiguousarray(np.where(np.isinf(tabs[ti]), L, tabs[ti]))
        self.P, self.tabs = P, tuple(tabs)
        self.fac0 = np.ascontiguousarray(P.fac[~ann])
        self.fac1 = np.ascontiguousarray(P.fac[ann])
        self.D = model.chan(child).D

    @staticmethod
    def _key(model):
        return (tuple(id(c.grid) for c in model.channels), tuple(id(f.table) for f in model.factors))

    def valid(self, model):
        return self.ref() is model and self.key == self._key(model)


class AISTargets:
    """See the module docstring.  Counters: n_calls / t_calls (at, log_z),
    n_ais / t_ais (AIS runs, one per candidate), n_hook."""

    def __init__(self, child, region, designed_e, after_change, K=32, M=16, L=30.0, schedule="linear",
                 anneal_names=None, hook=None, seed=0):
        self.child, self.region, self.designed_e, self.after_change = child, region, designed_e, after_change
        self.K, self.M, self.L, self.schedule = K, M, float(L), schedule
        self.anneal_names, self.hook = anneal_names, hook
        self.betas = betas_of(K, schedule)
        self.rng = np.random.default_rng(seed)
        self._pk = None
        self.n_calls = self.n_ais = self.n_hook = 0
        self.t_calls = self.t_ais = 0.0

    def packing(self, model):
        if self._pk is None or not self._pk.valid(model):
            self._pk = _Packing(model, self.child, self.anneal_names, self.L)
        return self._pk

    def log_z(self, model, home, y, x, cand):
        """(log Z_t (n,), AIS standard error (n,)) over cand; se 0 with the hook."""
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
        fixed0 = c.fixed.copy()
        c.fixed[:] = True
        c.fixed[y0:y1, x0:x1] = fixed0[y0:y1, x0:x1]
        pk = None if self.hook is not None else self.packing(model)
        try:
            for i, t in enumerate(cand):
                hg[y, x] = t
                self.after_change(model, home, y, x)
                if self.hook is not None:
                    lz[i] = self.hook(model, home, y, x)
                    self.n_hook += 1
                else:
                    ta = time.perf_counter()
                    P, r = pk.P, pk.P.radius
                    assert np.shares_memory(P.grids[P.home], c.grid) and np.shares_memory(P.fixed, c.fixed)
                    logw = np.zeros(self.M)
                    kernel.seed(int(self.rng.integers(1 << 30)))
                    lz0 = _ais(y0, y1, x0, x1, y0 - r, y1 + r, x0 - r, x1 + r, P.home, P.grids, P.hs, P.views,
                               pk.fac0, pk.fac1, pk.tabs, c.fixed, P.colours, P.ncol, self.betas, pk.D, logw)
                    lz[i] = lz0 + logmeanexp(logw) if np.isfinite(lz0) else -np.inf
                    se[i] = ais_se(logw) if np.isfinite(lz0) else 0.0
                    self.n_ais += 1
                    self.t_ais += time.perf_counter() - ta
                c.grid[y0:y1, x0:x1] = saved
        finally:
            hg[y, x] = old
            c.grid[y0:y1, x0:x1] = saved
            c.fixed[:] = fixed0
            self.after_change(model, home, y, x)
        self.n_calls += 1
        self.t_calls += time.perf_counter() - t0
        return lz, se

    def at(self, model, home, y, x, cand):
        e = np.asarray(self.designed_e(model, home, y, x, cand), np.float64)
        lz, _ = self.log_z(model, home, y, x, cand)
        a = -(e - lz)
        a[np.isnan(a)] = -np.inf
        assert np.isfinite(a).any(), "AISTargets: no candidate has finite weight"
        p = np.exp(a - a.max())
        return p / p.sum()
