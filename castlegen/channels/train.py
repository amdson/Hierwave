"""Generic pieces for fitting learned tables by moment matching / self-play.

counts      sufficient statistics of named factors at the current state
site_probs  the kernel's one-site conditionals of a channel, softmax(-e / T)
rb_gap      Rao-Blackwellised expected change of the counts under one-site
            moves of a channel with given per-site probabilities
step        theta += eta * gap / scale (- prior pull), zero-mean per table

The bootstrap of notes/dsl_updates.md C2 (q -> p*, Rao-Blackwellised
pseudo-likelihood over the sampler's candidate set):
fit               contexts from q, local p* targets, K p*-steps, refit theta
tabular_feats     candidate features of learned tables homed on a channel
designed_energies the designed-only energy row at a site (fixed offset)
pack / unpack / gauge / double_centre   flat theta <-> tables
consistency       pair closure error of one-site odds (fit diagnostic)
autocorr_clamped  integrated autocorrelation time of a channel's energy with
                  its parents clamped (C1 criterion)

Sign convention: E = sum theta[n] * N[n], p ~ exp(-E); gap = (forward
counts) - (target counts), so an over-produced feature gains energy.

Pads: a pair term with an off-grid partner exists when the pad is >= 0 and
reads the entry (va, pad_b) (or (pad_a, vb) for a same-level pair seen from
b's side, a off the grid), exactly as the kernel does; counts and rb_gap
count these entries too, so E = sum theta * N is the energy whose one-site
conditionals the kernel samples."""
import numpy as np
from numba import njit

import time

from .core import COUNT, PAIR, UNARY
from .kernel import _energies
from . import kernel as _kernel


# ----------------------------------------------------------------- counts
def _pair_counts(f, A, B, va, vb, shape):
    N = np.zeros(shape)
    ra, ca = A.grid.shape
    rb, cb = B.grid.shape
    yy, xx = np.mgrid[:ra, :ca]
    qy = (yy * A.h) // B.h + f.off[0]
    qx = (xx * A.h) // B.h + f.off[1]
    on = (qy >= 0) & (qy < rb) & (qx >= 0) & (qx < cb)
    a = va[A.grid]
    np.add.at(N, (a[on], vb[B.grid[qy[on], qx[on]]]), 1)
    if f.pad_b >= 0:
        np.add.at(N[:, f.pad_b], a[~on], 1)
    if f.pad_a >= 0 and A.h == B.h:                                 # b's cells whose a partner is off the grid
        yy, xx = np.mgrid[:rb, :cb]
        py, px = yy - f.off[0], xx - f.off[1]
        off = (py < 0) | (py >= ra) | (px < 0) | (px >= ca)
        np.add.at(N[f.pad_a], vb[B.grid[off]], 1)
    return N


def counts(model, names):
    """{name: N} with N shaped like the factor's table: pairs once each (a
    at p, b at q, plus pad entries), counts once per block, unaries per cell."""
    out = {}
    for f in model.factors:
        if f.name not in names:
            continue
        A = model.chan(f.a[0])
        va = A.views[f.a[1]]
        if f.kind == UNARY:
            out[f.name] = np.bincount(va[A.grid].ravel(), minlength=f.table.shape[0]).astype(np.float64)
            continue
        B = model.chan(f.b[0])
        vb = B.views[f.b[1]]
        if f.kind == PAIR:
            out[f.name] = _pair_counts(f, A, B, va, vb, f.table.shape)
        else:
            r = B.h // A.h
            rb, cb = B.grid.shape
            s = va[A.grid].reshape(rb, r, cb, r).sum(axis=(1, 3))
            N = np.zeros(f.table.shape)
            np.add.at(N, (vb[B.grid].ravel(), s.ravel()), 1)
            out[f.name] = N
    missing = set(names) - set(out)
    assert not missing, f"no factor named {missing}"
    return out


# ------------------------------------------------------------- site probs
@njit(cache=True)
def _all_energies(home, grids, hs, views, fac, tabs, D, convs):
    g = grids[home]
    rows, cols = g.shape
    out = np.empty((rows, cols, D))
    e = np.empty(D)
    for y in range(rows):
        for x in range(cols):
            _energies(y, x, home, grids, hs, views, fac, tabs, e, convs)
            out[y, x] = e
    return out


def site_probs(model, home, T=1.0, below=True):
    """(rows, cols, D): softmax(-e / T) of the kernel's candidate energies
    at every site of `home`; fixed cells (and sites with no finite
    candidate) get a one-hot on their current value."""
    P = model.compile(home, below)
    hc = model.chan(home)
    e = _all_energies(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, hc.D, P.convs) / T
    m = e.min(axis=2, keepdims=True)
    dead = ~np.isfinite(m[..., 0])
    w = np.exp(-(e - np.where(np.isfinite(m), m, 0.0)))
    w /= np.maximum(w.sum(axis=2, keepdims=True), 1e-300)
    onehot = hc.fixed | dead
    w[onehot] = 0.0
    yy, xx = np.nonzero(onehot)
    w[yy, xx, hc.grid[yy, xx]] = 1.0
    return w


# ----------------------------------------------------------------- rb gap
@njit(cache=True)
def _site_rb(y, x, cur, pr, home, grids, hs, views, fac, sel, trans, base, ncol, out):
    """At site (y, x) with current value cur: for every selected row f,
    out[entry(t)] += pr[t] and out[entry(cur)] -= pr[t]; entry in the
    factor's own orientation, flat index base[i] + row * ncol[i] + col."""
    g_home = grids[home]
    D = pr.shape[0]
    hc = hs[home]
    for i in range(sel.shape[0]):
        f = sel[i]
        kind = fac[f, 0]
        av = views[fac[f, 2]]
        o, nc = base[i], ncol[i]
        if kind == 2:
            for t in range(D):
                out[o + av[t]] += pr[t]
                out[o + av[cur]] -= pr[t]
            continue
        b = fac[f, 1]
        hb = hs[b]
        g = grids[b]
        if kind == 0:
            qy = (y * hc) // hb + fac[f, 4]
            qx = (x * hc) // hb + fac[f, 5]
            if 0 <= qy < g.shape[0] and 0 <= qx < g.shape[1]:
                vb = views[fac[f, 3]][g[qy, qx]]
            elif fac[f, 6] >= 0:
                vb = fac[f, 6]
            else:
                continue
            for t in range(D):
                if trans[i]:
                    out[o + vb * nc + av[t]] += pr[t]
                    out[o + vb * nc + av[cur]] -= pr[t]
                else:
                    out[o + av[t] * nc + vb] += pr[t]
                    out[o + av[cur] * nc + vb] -= pr[t]
        elif kind == 1:                                 # count homed here: home is the fine side
            r = hb // hc
            qy = (y * hc) // hb
            qx = (x * hc) // hb
            vb = views[fac[f, 3]][g[qy, qx]]
            s = 0
            for yy in range(qy * r, qy * r + r):
                for xx in range(qx * r, qx * r + r):
                    if yy != y or xx != x:
                        s += av[g_home[yy, xx]]
            for t in range(D):
                out[o + vb * nc + s + av[t]] += pr[t]
                out[o + vb * nc + s + av[cur]] -= pr[t]
        else:                                           # 3 / 4: home is the coarse side
            r = hc // hb
            vh = views[fac[f, 3]]
            if kind == 3:
                for yy in range(y * r, y * r + r):
                    for xx in range(x * r, x * r + r):
                        u = av[g[yy, xx]]
                        for t in range(D):
                            out[o + u * nc + vh[t]] += pr[t]
                            out[o + u * nc + vh[cur]] -= pr[t]
            else:
                s = 0
                for yy in range(y * r, y * r + r):
                    for xx in range(x * r, x * r + r):
                        s += av[g[yy, xx]]
                for t in range(D):
                    out[o + vh[t] * nc + s] += pr[t]
                    out[o + vh[cur] * nc + s] -= pr[t]


@njit(cache=True)
def _rb(home, grids, hs, views, fac, sel, trans, base, ncol, probs, out):
    """_site_rb at every site of home with probs[y, x]."""
    g_home = grids[home]
    rows, cols = g_home.shape
    for y in range(rows):
        for x in range(cols):
            _site_rb(y, x, g_home[y, x], probs[y, x], home, grids, hs, views, fac, sel, trans, base, ncol, out)


@njit(cache=True)
def _site_feats(y, x, cand, D, home, grids, hs, views, fac, sel, trans, base, ncol, out):
    """out[i] = N(z with z_p = cand[i]) - N(z) over the selected rows."""
    cur = grids[home][y, x]
    pr = np.zeros(D)
    for i in range(cand.shape[0]):
        pr[:] = 0.0
        pr[cand[i]] = 1.0
        _site_rb(y, x, cur, pr, home, grids, hs, views, fac, sel, trans, base, ncol, out[i])


def _rb_setup(model, home, names):
    """The below=True packing of `home` and the rows of the named factors:
    (P, sel, trans, base, ncol, shapes, offs, n)."""
    P = model.compile(home, below=True)
    idx = {f.name: i for i, f in enumerate(model.factors)}
    missing = set(names) - set(idx)
    assert not missing, f"no factor named {missing}"
    want = {idx[n]: n for n in names}
    shapes = {n: model.factors[idx[n]].table.shape for n in names}
    offs, o = {}, 0
    for n in names:
        offs[n] = o
        o += int(np.prod(shapes[n]))
    sel = [f for f in range(P.fac.shape[0]) if P.src[f] in want]
    nm = [want[P.src[f]] for f in sel]
    base = np.array([offs[n] for n in nm], np.int64)
    ncol = np.array([shapes[n][1] if len(shapes[n]) > 1 else 1 for n in nm], np.int64)
    return P, np.array(sel, np.int64), P.transposed[sel].copy(), base, ncol, shapes, offs, o


def rb_gap(model, home, names, probs):
    """{name: dN}: sum_p sum_t probs[p, t] (N(z with z_p = t) - N(z)), each
    site of `home` moved alone from the current state.  Read off the
    below=True packed rows of `home` (home-side, reflected and below rows),
    one entry change per row per site; factors that do not touch `home`
    give zeros.  Fixed cells contribute nothing if probs is one-hot there."""
    P, sel, trans, base, ncol, shapes, offs, o = _rb_setup(model, home, names)
    probs = np.ascontiguousarray(probs, np.float64)
    assert probs.shape == model.chan(home).grid.shape + (model.chan(home).D,), probs.shape
    out = np.zeros(o)
    _rb(P.home, P.grids, P.hs, P.views, P.fac, sel, trans, base, ncol, probs, out)
    return {n: out[offs[n]:offs[n] + int(np.prod(shapes[n]))].reshape(shapes[n]) for n in names}


# ------------------------------------------------------------------- step
def step(theta, gap, prior=None, eta=1.0, lam_prior=0.0, scale=None):
    """theta[k] + eta * gap[k] / scale[k] - lam_prior * (theta[k] - prior[k]),
    then zero-mean per table.  gap = forward counts - target counts (per
    site); keys missing from gap are only pulled and re-centred."""
    new = {}
    for k, th in theta.items():
        t = np.array(th, np.float64)
        if k in gap:
            s = 1.0 if scale is None else (scale[k] if isinstance(scale, dict) else scale)
            t = t + eta * np.asarray(gap[k]) / s
        if prior is not None and lam_prior:
            t = t - lam_prior * (np.asarray(th) - np.asarray(prior[k]))
        new[k] = t - t.mean()
    return new


# ------------------------------------------------------ theta <-> tables
def pack(theta, names):
    """Flat vector of the tables theta[n], n in names order."""
    return np.concatenate([np.ravel(np.asarray(theta[n], np.float64)) for n in names])


def unpack(flat, shapes):
    """{name: table} from a flat vector; shapes: {name: shape} in pack order."""
    out, o = {}, 0
    for n, s in shapes.items():
        k = int(np.prod(s))
        out[n] = np.asarray(flat[o:o + k], np.float64).reshape(s)
        o += k
    assert o == len(flat), (o, len(flat))
    return out


def gauge(theta):
    """Zero-mean per table (the constant level of a table is not identifiable)."""
    return {k: np.asarray(v, np.float64) - np.mean(v) for k, v in theta.items()}


def double_centre(A):
    """A pair table minus its row and column means plus the grand mean (a
    unary: minus its mean): the part not confounded with unaries."""
    A = np.asarray(A, np.float64)
    if A.ndim == 1:
        return A - A.mean()
    return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()


def _same_grids(P, model):
    return all(g is c.grid for g, c in zip(P.grids, model.channels))


class tabular_feats:
    """feats callback for tabular learned factors (pair / unary / count tables
    whose rows reach `home`, as in Potts and the old circles):
    X[i] = N(z with z_p = cand[i]) - N(z), N = counts(model, names) flattened
    in `names` order, so the learned energy change is X @ theta with theta =
    pack(tables, names).  Read off the below=True packing of `home`, one
    numba pass per site; the packing is cached per (model, home) while the
    model's grids are the same arrays."""

    def __init__(self, names):
        self.names = list(names)
        self._cache = {}

    def _setup(self, model, home):
        c = self._cache.get(home)
        if c is None or c[0] is not model or not _same_grids(c[1][0], model):
            c = (model, _rb_setup(model, home, self.names))
            self._cache[home] = c
        return c[1]

    def shapes(self, model, home):
        return self._setup(model, home)[5]

    def __call__(self, model, home, y, x, cand):
        P, sel, trans, base, ncol, shapes, offs, n = self._setup(model, home)
        cand = np.asarray(cand, np.int64)
        out = np.zeros((len(cand), n))
        _site_feats(y, x, cand, model.chan(home).D, P.home, P.grids, P.hs, P.views, P.fac, sel, trans, base,
                    ncol, out)
        return out


def designed_energies(model_designed, home):
    """designed_e callback: the factor rows of `home` in a designed-only model
    (no below rows: the finer levels' terms are what F integrates), at
    (y, x), restricted to cand.  model_designed must share its Channel
    objects with the contexts' models (it reads the same grids)."""
    cache = {}

    def f(model, home_, y, x, cand):
        assert home_ == home, (home_, home)
        P = cache.get("P")
        if P is None or not _same_grids(P, model_designed):
            P = cache["P"] = model_designed.compile(home)
        e = _kernel.site_energies(y, x, P.home, P.grids, P.hs, P.views, P.fac, P.tabs,
                                  model_designed.chan(home).D, P.convs)
        return e[np.asarray(cand, np.int64)]
    return f


# --------------------------------------------------------------- fitting
def _site_order(model, home):
    """Active (not fixed) sites of `home` in the kernel's order: colour classes
    of Model.compile(home).colours in order, raster within a class."""
    P = model.compile(home)
    fixed = model.chan(home).fixed
    out = []
    for c in range(P.ncol):
        yy, xx = np.nonzero((P.colours == c) & ~fixed)
        out += list(zip(yy.tolist(), xx.tolist()))
    return out, P


def _pair_offsets(P):
    """Same-level pair offsets of the kernel (rows of kind PAIR reading home);
    the 4-neighbourhood when there are none."""
    offs = {(int(r[4]), int(r[5])) for r in P.fac if r[0] == PAIR and r[1] == P.home and (r[4] or r[5])}
    return sorted(offs) or [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _site_record(model, home, y, x, targets, feats, designed_e):
    """(cand, X, e, pi) at (y, x), or None with no admissible value."""
    D = model.chan(home).D
    e = np.asarray(designed_e(model, home, y, x, np.arange(D)), np.float64)
    cand = np.flatnonzero(np.isfinite(e))
    if len(cand) == 0:
        return None
    pi = np.asarray(targets.at(model, home, y, x, cand), np.float64)
    return cand, np.asarray(feats(model, home, y, x, cand), np.float64), e[cand], pi


def _probe(model, home, p, q, targets, feats, designed_e, repaint):
    """Records at p for every value of q and at q for every value of p, the
    rest of the state as it is; the state is restored."""
    g = model.chan(home).grid
    D = model.chan(home).D
    out = []
    for a, b in ((p, q), (q, p)):
        side = []
        old = g[b]
        for v in range(D):
            g[b] = v
            if repaint is not None:
                repaint(model, home, *b)
            side.append(_site_record(model, home, a[0], a[1], targets, feats, designed_e))
        g[b] = old
        if repaint is not None:
            repaint(model, home, *b)
        out.append(side)
    return out


def _logp_full(rec, D, theta=None):
    """log of the record's distribution over the full domain (-inf outside
    cand): the target pi (theta None) or the fit softmax(-(e + X theta))."""
    lp = np.full(D, -np.inf)
    if rec is None:
        return lp
    cand, X, e, pi = rec
    if theta is None:
        with np.errstate(divide="ignore"):
            lp[cand] = np.log(pi)
    else:
        a = -(e + X @ theta)
        lp[cand] = a - (np.log(np.exp(a - a.max()).sum()) + a.max())
    return lp


def consistency(probe, D, theta=None):
    """Max over value pairs t != t' of the closure error of the one-site odds
    around the square (z_p, z_q) = (t, t') -> (t', t') -> (t', t) -> (t, t)
    -> (t, t'):

        C = [lp_{q=t'}(t') - lp_{q=t'}(t)] + [lq_{p=t'}(t) - lq_{p=t'}(t')]
          + [lp_{q=t}(t)   - lp_{q=t}(t')] + [lq_{p=t}(t')  - lq_{p=t}(t)]

    lp_{q=v} = log of the conditional at p with z_q = v (rest of the state
    fixed), lq likewise.  C = 0 for every square iff the two conditionals
    come from one joint over (z_p, z_q) given the rest.  Squares with any
    zero probability (inadmissible values, one-hot targets) are skipped;
    nan if none is left.  probe from _probe; theta None scores the targets,
    else the fitted conditional softmax(-(e + X theta))."""
    P_side, Q_side = probe
    lp = np.array([_logp_full(r, D, theta) for r in P_side])        # [v of q, t at p]
    lq = np.array([_logp_full(r, D, theta) for r in Q_side])        # [v of p, t at q]
    t, tp = np.meshgrid(np.arange(D), np.arange(D), indexing="ij")
    with np.errstate(invalid="ignore"):
        C = (lp[tp, tp] - lp[tp, t] + lq[tp, t] - lq[tp, tp] + lp[t, t] - lp[t, tp] + lq[t, tp] - lq[t, t])
    ok = np.isfinite(C) & (t != tp)
    return float(np.abs(C[ok]).max()) if ok.any() else float("nan")


def _compress(r):
    """A record with X as COO triples (row, col, val) plus its shape."""
    cand, X, e, pi = r
    i, j = np.nonzero(X)
    return cand, (i, j, X[i, j], X.shape), e, pi


def _stack(recs):
    """Pad compressed records to X (N * Dm, nf) (scipy CSR when importable,
    else dense), e, pi, mask (N, Dm)."""
    N = len(recs)
    Dm = max(len(r[0]) for r in recs)
    nf = recs[0][1][3][1]
    E = np.zeros((N, Dm)); PI = np.zeros((N, Dm)); M = np.zeros((N, Dm), bool)
    R, C, V = [], [], []
    for n, (cand, (i, j, v, _), e, pi) in enumerate(recs):
        k = len(cand)
        E[n, :k], PI[n, :k], M[n, :k] = e, pi, True
        R.append(n * Dm + i); C.append(j); V.append(v)
    R, C, V = np.concatenate(R), np.concatenate(C), np.concatenate(V)
    try:
        from scipy.sparse import csr_matrix
        X = csr_matrix((V, (R, C)), shape=(N * Dm, nf))
    except ImportError:
        X = np.zeros((N * Dm, nf))
        np.add.at(X, (R, C), V)
    return X, E, PI, M


def kl_loss(theta, X, E, PI, M, l2=0.0):
    """(loss, grad): mean over records of KL(pi || softmax(-(e + X theta)))
    over each record's candidates (mask M), + l2 |theta|^2.  X is
    (N, Dm, nf) or (N * Dm, nf), dense or sparse.  grad = mean of
    E_pi[X] - E_q[X] + 2 l2 theta."""
    N, Dm = E.shape
    if X.ndim == 3:
        X = X.reshape(N * Dm, -1)
    a = np.where(M, -(E + np.asarray(X @ theta).reshape(N, Dm)), -np.inf)
    mx = a.max(1, keepdims=True)
    logq = a - (np.log(np.exp(a - mx).sum(1, keepdims=True)) + mx)
    q = np.where(M, np.exp(logq), 0.0)
    pos = PI > 0
    kl = np.where(pos, PI * (np.log(np.where(pos, PI, 1.0)) - np.where(M, logq, 0.0)), 0.0).sum() / N
    g = np.asarray(X.T @ (PI - q).ravel()).ravel() / N
    return kl + l2 * theta @ theta, g + 2 * l2 * theta


def _minimise(X, E, PI, M, l2, nf):
    """argmin of kl_loss from theta = 0: scipy L-BFGS if importable, else Adam
    stopped when the gradient norm is < 1e-7 or after 20000 steps."""
    th = np.zeros(nf)
    if E.shape[0] == 0:
        return th
    try:
        from scipy.optimize import minimize
        r = minimize(kl_loss, th, args=(X, E, PI, M, l2), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=5000, gtol=1e-9, ftol=1e-15))
        return r.x
    except ImportError:
        m = np.zeros(nf); v = np.zeros(nf)
        for t in range(1, 20001):
            _, g = kl_loss(th, X, E, PI, M, l2)
            if np.linalg.norm(g) < 1e-7:
                break
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            th -= 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
        return th


def fit(contexts, home, targets, feats, designed_e, iters, n_contexts, K, l2=1e-4, after_change=None,
        holdout=0.1, rng=0, repaint=None, n_pairs=8, theta0=None, verbose=False):
    """The bootstrap of dsl_updates.md C2 / the fit pseudocode of
    dsl_interface.md.  Returns (theta, report).

    contexts(theta, n, rng): iterable of n models in fresh states at the flat
        parameters theta (None at the first iteration unless theta0 is
        given: the learned potential is zero) -- the forward chain, or
        samples of p* that ignore theta.  Each model is used while it is the
        current item (the next may overwrite shared grids).
    targets.at(model, home, y, x, cand) -> pi over cand (targets.py).
    feats(model, home, y, x, cand) -> (len(cand), nfeat) feature deltas; the
        learned conditional is softmax(-(e + X theta)) (linear psi).
    designed_e(model, home, y, x, cand) -> designed energies (a fixed offset).
    after_change(model, home, y, x): called after a K-step writes z_p
        (repaint painted channels; for a collapsed move, redraw the fine
        block given the new value).
    repaint(model, home, y, x): deterministic repaint used by the
        consistency probes when they change a value and when they restore
        it (None: nothing to repaint).  Never after_change, which may draw.

    Per iteration, per context: for k = 0..K, every active site in the
    kernel's colour order: cand = admissible values (finite designed
    energy), pi = targets.at, record (X, e, pi) (sites with one candidate
    carry no information and are not recorded); if k < K draw z_p ~ pi,
    write it, after_change.  Then n_pairs consistency probes at the final
    state (random active p, q = p + one of the kernel's same-level pair
    offsets).  Then theta = argmin over all records so far (aggregated, refit
    from zero) of mean KL(pi || softmax(-(e + X theta))) + l2 |theta|^2;
    records are assigned to the held-out set with probability `holdout` and
    never fitted.  K = 0 is S1, K > 0 is CD-K / S3.

    report: per iteration lists 'n_records', 'kl_train', 'kl_held',
    'viol_targets', 'viol_fit' (max of `consistency` over this iteration's
    probes; the fit's at the refitted theta), 'seconds'; plus 'seconds' total
    and 'nfeat'."""
    rng = np.random.default_rng(rng)
    theta = None if theta0 is None else np.asarray(theta0, np.float64)
    recs, held = [], []
    rep = dict(n_records=[], kl_train=[], kl_held=[], viol_targets=[], viol_fit=[], seconds=[])
    T0 = time.time()
    for it in range(iters):
        t0 = time.time()
        probes = []
        D = None
        for model in contexts(theta, n_contexts, rng):
            hc = model.chan(home)
            D = hc.D
            order, P = _site_order(model, home)
            for k in range(K + 1):
                for y, x in order:
                    r = _site_record(model, home, y, x, targets, feats, designed_e)
                    if r is None:
                        continue
                    cand, X, e, pi = r
                    if len(cand) > 1:
                        (held if rng.random() < holdout else recs).append(_compress(r))
                    if k < K:
                        hc.grid[y, x] = cand[rng.choice(len(cand), p=pi / pi.sum())]
                        if after_change is not None:
                            after_change(model, home, y, x)
            offs = _pair_offsets(P)
            rows, cols = hc.grid.shape
            for _ in range(n_pairs if order else 0):
                for _try in range(20):
                    p = order[rng.integers(len(order))]
                    o = offs[rng.integers(len(offs))]
                    q = (p[0] + o[0], p[1] + o[1])
                    if 0 <= q[0] < rows and 0 <= q[1] < cols and not hc.fixed[q]:
                        probes.append(_probe(model, home, p, q, targets, feats, designed_e, repaint))
                        break
        nf = recs[0][1][3][1] if recs else (held[0][1][3][1] if held else 0)
        Xs = _stack(recs) if recs else None
        theta = _minimise(*Xs, l2, nf) if recs else np.zeros(nf)
        rep["n_records"].append(len(recs) + len(held))
        rep["kl_train"].append(float(kl_loss(theta, *Xs)[0]) if recs else float("nan"))
        rep["kl_held"].append(float(kl_loss(theta, *_stack(held))[0]) if held else float("nan"))
        vt = [consistency(pr, D) for pr in probes]
        vf = [consistency(pr, D, theta) for pr in probes]
        rep["viol_targets"].append(float(np.nanmax(vt)) if vt and not np.all(np.isnan(vt)) else float("nan"))
        rep["viol_fit"].append(float(np.nanmax(vf)) if vf and not np.all(np.isnan(vf)) else float("nan"))
        rep["seconds"].append(time.time() - t0)
        if verbose:
            print(f"  fit it {it}: {rep['n_records'][-1]} records  KL train {rep['kl_train'][-1]:.5f} "
                  f"held {rep['kl_held'][-1]:.5f}  viol targets {rep['viol_targets'][-1]:.2e} "
                  f"fit {rep['viol_fit'][-1]:.2e}  {rep['seconds'][-1]:.1f}s", flush=True)
    rep["seconds_total"] = time.time() - T0
    rep["nfeat"] = int(len(theta))
    return theta, rep


# ------------------------------------------------- autocorrelation (C1)
def iat(a):
    """Integrated autocorrelation time of a series, in steps, by Geyer's
    initial monotone sequence: Gamma_m = rho(2m) + rho(2m + 1), summed up to
    the first non-positive Gamma_m and made non-increasing; tau = -1 +
    2 sum_m Gamma_m (1 for an independent series).  1.0 for a constant."""
    a = np.asarray(a, np.float64) - np.mean(a)
    n = len(a)
    if n < 4 or a.var() == 0:
        return 1.0
    f = np.fft.rfft(a, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / n
    rho = ac / ac[0]
    m = n // 2
    G = rho[0:2 * m:2] + rho[1:2 * m:2]
    neg = np.flatnonzero(G <= 0)
    G = G[:neg[0]] if len(neg) else G
    G = np.minimum.accumulate(G)
    return float(max(-1.0 + 2.0 * G.sum(), 1.0 / n))


def autocorr_clamped(model, home, parent_names, sweeps, burn, seed=0):
    """tau of the energy of `home` (Model.energy: the factors homed on it,
    finite part) along `sweeps` sweeps of its kernel after `burn`, with the
    named parent channels clamped: only `home` is swept, so every other
    channel (the parents in particular) stays at its current state; asserted
    at the end.  The C1 criterion: the fine kernel mixes with the coarse
    values fixed.  Estimator: iat (Geyer's initial monotone sequence)."""
    before = {n: model.chan(n).grid.copy() for n in parent_names}
    P = model.compile(home)
    _kernel.seed(seed)
    sw = lambda: _kernel.sweep(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.fixed, P.colours, P.ncol,
                               P.cert, P.joins, P.delta, 1.0, P.convs)
    for _ in range(burn):
        sw()
    es = np.empty(sweeps)
    for i in range(sweeps):
        sw()
        es[i] = _kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.cert, P.joins, P.delta,
                                     P.convs)[0]
    assert all(np.array_equal(model.chan(n).grid, g) for n, g in before.items())
    return iat(es)
