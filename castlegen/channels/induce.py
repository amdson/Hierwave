"""Induced potentials: the free energy a fine level induces on a coarse
channel, estimated on small clamped windows and fitted by an additive
potential (notes/induce_test.md).  Generic: nothing here knows a channel set.

ais_log_z    annealed importance sampling of log Z over channel `home` of a
             Model: p_0 = the unary factors of `home` (log Z_0 closed form),
             E_rest = every other factor of the home's packing, annealed
             linearly (or geometrically) with inf -> L; one generic-kernel
             sweep per beta step, M independent chains
ais_se       the estimator's standard error (delta method on mean(w))
windows      F(item) = -log Z_fine for a list of window items through a model
             hook: hook.window_fine_model(item) -> (Model, home) for `ais`,
             hook.window_free_energy_exact(item) for `exact`
fit_tables   ridge fit of F(z) - F(r) ~ sum u(z_p) + sum g_h + sum g_v over a
             window's cells and internal adjacencies, in the reference gauge
             (g(r, .) = g(., r) = 0)
materialise  write the tables into a theta dict (zero-mean per table)

Z convention: Z = sum over the free (non-fixed) cells of `home` of
exp(-E), E = Model.energy(home) at that state (fixed cells' own terms are
constants inside E).  Factors homed on finer channels that read `home` are
not part of E (the window model should not hold finer free channels)."""
import numpy as np

from . import kernel
from .core import UNARY


# ------------------------------------------------------------------ helpers
def logmeanexp(a):
    a = np.asarray(a, float)
    m = a.max()
    return float(m + np.log(np.mean(np.exp(a - m))))


def ais_se(logw):
    """Standard error of log mean(w) (delta method: sd(w) / (sqrt(M) mean(w)))."""
    lw = np.asarray(logw, float)
    w = np.exp(lw - lw.max())
    M = len(w)
    if M < 2:
        return float("nan")
    return float(w.std(ddof=1) / (np.sqrt(M) * w.mean()))


def anneal_names_of(model, home):
    """Names of every factor in the packing of `home` that is not a unary on it."""
    P = model.compile(home)
    out = []
    for f in sorted(set(P.src[P.fac[:, 0] != 2].tolist())):
        n = model.factors[f].name
        if n not in out:
            out.append(n)
    return out


def betas_of(K, schedule="linear", beta0=1e-3):
    if schedule == "linear":
        return np.linspace(0.0, 1.0, K + 1)
    if schedule == "geometric":
        return np.concatenate([[0.0], np.geomspace(beta0, 1.0, K)])
    raise ValueError(schedule)


# ---------------------------------------------------------------------- AIS
class _Anneal:
    """The packing of `home` split into p_0 rows (unaries of home) and anneal
    rows (tables with inf -> L, scaled by beta)."""

    def __init__(self, model, home, anneal_names=None, L=30.0):
        self.model, self.home = model, home
        P = model.compile(home)
        assert P.cert[4] == 0, "ais_log_z: certificates are not annealed"
        names = anneal_names_of(model, home) if anneal_names is None else list(anneal_names)
        hc = model.chan(home)
        assert np.shares_memory(P.grids[P.home], hc.grid), "home grid must be contiguous int32"
        self.P, self.hc = P, hc
        unary = P.fac[:, 0] == 2
        self.is_anneal = np.zeros(P.fac.shape[0], bool)
        for f in range(P.fac.shape[0]):
            nm = model.factors[P.src[f]].name
            if unary[f]:
                assert nm not in names, f"{nm}: a unary on {home} is part of p_0"
                continue
            assert P.fac[f, 0] in (0, 1), f"row {f} ({nm}): below rows are not supported"
            assert nm in names, f"{nm}: a non-unary factor of {home} must be annealed"
            self.is_anneal[f] = True
        self.full = tuple(np.where(np.isinf(t), L, t) if a else t for t, a in zip(P.tabs, self.is_anneal))
        self.rest = tuple(t if a else np.zeros_like(t) for t, a in zip(self.full, self.is_anneal))
        # per-site unary energies (position free: one vector for every site)
        D = hc.D
        u = np.zeros(D)
        for f in np.flatnonzero(unary):
            av = P.views[P.fac[f, 2]]
            u += P.tabs[P.fac[f, 7]][av, 0]
        self.u = u
        free = ~hc.fixed
        self.free = free
        fin = np.isfinite(u)
        assert fin.any(), "every value has an infinite unary"
        umin = u[fin].min()
        p = np.exp(-(u - umin))
        self.logZ0 = free.sum() * (np.log(p.sum()) - umin) - float(u[hc.grid[~free]].sum())
        self.p0 = p / p.sum()

    def tabs(self, beta):
        return tuple(t * beta if a else t for t, a in zip(self.full, self.is_anneal))

    def e_rest(self):
        P = self.P
        e, nv = kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, self.rest, P.cert, P.joins, P.delta)
        assert nv == 0
        return e

    def sweep(self, tabs):
        P = self.P
        kernel.sweep(P.home, P.grids, P.hs, P.views, P.fac, tabs, P.fixed, P.colours, P.ncol, P.cert, P.joins,
                     P.delta, 1.0)


def ais_log_z(model, home, anneal_names=None, K=32, M=16, seed=0, L=30.0, schedule="linear"):
    """(log Z, log w (M,)): AIS from p_0 (the unaries of `home`, sampled
    exactly) to the full conditional of `home`, E_beta = E_unary + beta E_rest,
    K beta steps with one kernel sweep each.  The home grid's free cells are
    overwritten (left at the last chain's final state); fixed cells kept."""
    A = _Anneal(model, home, anneal_names, L)
    betas = betas_of(K, schedule)
    tabs = [A.tabs(b) for b in betas[:-1]]
    rng = np.random.default_rng(seed)
    g = A.hc.grid
    fy, fx = np.nonzero(A.free)
    cdf = np.cumsum(A.p0)
    logw = np.zeros(M)
    for m in range(M):
        g[fy, fx] = np.minimum(np.searchsorted(cdf, rng.random(len(fy)) * cdf[-1], side="right"),
                               len(cdf) - 1).astype(np.int32)
        kernel.seed(int(rng.integers(1 << 30)))
        lw = 0.0
        for k in range(K):
            A.sweep(tabs[k])
            lw -= (betas[k + 1] - betas[k]) * A.e_rest()
        logw[m] = lw
    return A.logZ0 + logmeanexp(logw), logw


# ------------------------------------------------------------------ windows
def windows(hook, items, method="ais", K=32, M=16, seed=0, L=30.0, schedule="linear", anneal_names=None):
    """(F (N,), se (N,)): F(item) = -log Z of the fine level of each window
    item.  method 'exact': hook.window_free_energy_exact(item), se 0;
    'ais': ais_log_z on hook.window_fine_model(item) -> (Model, home)."""
    N = len(items)
    F, se = np.zeros(N), np.zeros(N)
    rng = np.random.default_rng(seed)
    for n, it in enumerate(items):
        if method == "exact":
            F[n] = hook.window_free_energy_exact(it)
            continue
        m, home = hook.window_fine_model(it)
        lz, lw = ais_log_z(m, home, anneal_names, K, M, int(rng.integers(1 << 30)), L, schedule)
        F[n], se[n] = -lz, ais_se(lw)
    return F, se


# ---------------------------------------------------------------------- fit
def window_features(wv, D, ref=0):
    """(N, D + 2 (D-1)^2) counts: unary counts, then horizontal and vertical
    internal-adjacency pair counts over the non-reference values (the
    reference gauge g(r, .) = g(., r) = 0)."""
    wv = np.asarray(wv, np.int64)
    N = wv.shape[0]
    nr = [v for v in range(D) if v != ref]
    idx = -np.ones(D, np.int64)
    idx[nr] = np.arange(D - 1)
    d1 = D - 1
    X = np.zeros((N, D + 2 * d1 * d1))
    rows = np.arange(N)
    for y in range(wv.shape[1]):
        for x in range(wv.shape[2]):
            np.add.at(X, (rows, wv[:, y, x]), 1)
    for o, (a, b) in ((D, (wv[:, :, :-1], wv[:, :, 1:])), (D + d1 * d1, (wv[:, :-1, :], wv[:, 1:, :]))):
        ia, ib = idx[a], idx[b]
        ok = (ia >= 0) & (ib >= 0)
        r = np.broadcast_to(rows[:, None, None], a.shape)
        np.add.at(X, (r[ok], o + ia[ok] * d1 + ib[ok]), 1)
    return X


def fit_tables(wv, targets, D, ridge=1e-3, ref=0):
    """Ridge fit of targets ~ sum_cells u + sum_h g_h + sum_v g_v.  Returns
    (u (D,), g_h (D, D), g_v (D, D), residual RMS, target RMS); pair tables
    in the reference gauge (row / column `ref` zero), then each table
    zero-meaned (a constant per table: no change to any conditional)."""
    X = window_features(wv, D, ref)
    y = np.asarray(targets, float)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    th = np.linalg.solve(A, X.T @ y)
    res = y - X @ th
    d1 = D - 1
    nr = [v for v in range(D) if v != ref]
    u = th[:D].copy()
    gh, gv = np.zeros((D, D)), np.zeros((D, D))
    gh[np.ix_(nr, nr)] = th[D:D + d1 * d1].reshape(d1, d1)
    gv[np.ix_(nr, nr)] = th[D + d1 * d1:].reshape(d1, d1)
    return (u - u.mean(), gh - gh.mean(), gv - gv.mean(), float(np.sqrt(np.mean(res ** 2))),
            float(np.sqrt(np.mean(y ** 2))))


def materialise(theta, u, g_h, g_v, names):
    """theta[names] = (u, g_h, g_v), each zero-meaned (a new dict)."""
    out = dict(theta)
    for n, t in zip(names, (u, g_h, g_v)):
        t = np.array(t, float)
        out[n] = t - t.mean()
    return out
