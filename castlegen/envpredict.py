"""Linear energy predictor for the envelope promise (quantities/envelope.py),
trained by exact column likelihood.

The sampler (envelope_var.EnvelopePromise) reads per-level tables u (V,),
pair (2, V, V) and parent (4, V, V).  A table indexes a whole block value, so
values never seen whole in the corpus get an uninformed energy.  Here every
term is a linear function of interval features, shared across interval
positions, so the energy of an unseen value is built from the parts it shares
with seen ones:

  u(v)              one-hot of each interval's (lo, hi) pair, summed over
                    intervals; plus adjacency features of the block's own
                    neighbouring intervals (weights shared with pair[0])
  pair[0](a, b)     adjacency features of a's last and b's first interval:
                    |d lo|, |d hi|, [one interval full and the other empty]
  pair[1](u, l)     per interval, one-hot of (what u presents downward:
                    none / some / all columns with matter at its bottom row,
                    what l presents upward: none / some / all columns full),
                    summed over intervals
  parent(pos, c, P) per child interval, the child's envelope lifted to the
                    parent's scale vs the parent interval it lies in: one-hot
                    of (dy, lo matches, hi matches) and |d lo|, |d hi| by dy

with one weight vector per level h.  `to_tables` expands the weights into a
promise.Tables, so the sampler is unchanged.

Training objective: for every child column of every corpus map at every
level, the exact log-probability of the observed column under the column
conditional the sampler draws from in its sweeps (unary u + parent term +
horizontal pair terms with the observed neighbour columns, vertical pair
terms, hard merge / seam / ground masks), computed by the forward algorithm
(a column is a chain).  This is block pseudo-likelihood with columns as
blocks; minimised with L-BFGS, L2 penalty `l2`.  `column_nll` evaluates the
same objective for any Tables (e.g. count tables) on held-out maps.
"""
from __future__ import annotations

import numpy as np

from castlegen.promise import Tables
from castlegen.quantities import envelope as EN

BIG = 1e9


# ------------------------------------------------------------------ features
def _adjacency(lg, a_lo, a_hi, b_lo, b_hi):
    """(..., 3) features of two side-by-side intervals."""
    F = lg.F
    full_empty = ((a_lo == F) & (b_hi == 0)) | ((b_lo == F) & (a_hi == 0))
    return np.stack([np.abs(a_lo - b_lo), np.abs(a_hi - b_hi), full_empty], -1).astype(float)


def features(lg):
    """Feature tensors for one level: dict of
       U   (V, dU)        unary (pair one-hots),
       A_in (V, 3)        adjacency inside the block,
       A_x  (V, V, 3)     adjacency across a vertical seam [left, right],
       Pv  (V, V, 9)      vertical pair [upper, lower],
       Par (4, V, V, 12)  parent [pos, child, parent]."""
    if hasattr(lg, "SUB"):
        return features_avg(lg)
    V, I, P, F = lg.V, lg.I, lg.P, lg.F
    LO, HI = lg.LO, lg.HI
    pidx = np.stack([(np.arange(V) // P ** i) % P for i in range(I)], 1)
    U = np.zeros((V, P))
    for i in range(I):
        U[np.arange(V), pidx[:, i]] += 1
    A_in = sum(_adjacency(lg, LO[:, i], HI[:, i], LO[:, i + 1], HI[:, i + 1]) for i in range(I - 1))
    A_x = _adjacency(lg, LO[:, -1][:, None], HI[:, -1][:, None], LO[:, 0][None, :], HI[:, 0][None, :])
    down = np.where(LO > 0, 2, np.where(HI > 0, 1, 0))          # upper: matter at its bottom row in none/some/all columns
    up = np.where(LO == F, 2, np.where(HI == F, 1, 0))           # lower: full in none/some/all columns
    Pv = np.zeros((V, V, 9))
    for i in range(I):
        idx = 3 * down[:, i][:, None] + up[:, i][None, :]
        np.add.at(Pv, (np.arange(V)[:, None].repeat(V, 1), np.arange(V)[None, :].repeat(V, 0), idx), 1)
    Par = np.zeros((4, V, V, 12))
    h2 = I // 2
    for dy in (0, 1):
        lift = lg.up_upper if dy == 0 else lg.up
        clo, chi = lift(LO), lift(HI)                            # (V, I) child intervals at parent scale
        for dx in (0, 1):
            pos = 2 * dy + dx
            for i in range(I):
                j = dx * h2 + i // 2                             # parent interval covering child interval i
                plo, phi = LO[:, j][None, :], HI[:, j][None, :]
                mlo, mhi = clo[:, i][:, None] == plo, chi[:, i][:, None] == phi
                oh = 4 * dy + 2 * mlo + mhi
                for k in range(8):
                    Par[pos, :, :, k] += oh == k
                Par[pos, :, :, 8 + 2 * dy] += np.abs(clo[:, i][:, None] - plo)
                Par[pos, :, :, 9 + 2 * dy] += np.abs(chi[:, i][:, None] - phi)
    return dict(U=U, A_in=A_in, A_x=A_x, Pv=Pv, Par=Par)


def features_avg(lg):
    """Features for the average language (quantities/average.py), whose value
    is two sub-column heights (b0, b1) in 0..w, same keys as `features`:
       U     one-hot of each b_j, summed over j
       A     |d b|, (d b)^2 / w, [one full and the other empty] of two
             side-by-side sub-columns (inside the block, and across a seam)
       Pv    per sub-column, one-hot of (b_upper, b_lower), summed
       Par   per dy, one-hot of clip(s - 2p, -4, 4) and |s - 2p| / w, where s
             is the child's sub-column sum over the parent sub-column p it
             lies in (a column pair u, l merges to p = (s_u + s_l) // 4)."""
    V, P, w, S = lg.V, lg.P, lg.w, lg.SUB
    U = np.zeros((V, P))
    for j in range(S.shape[1]):
        U[np.arange(V), S[:, j]] += 1

    def adj(a, b):
        d = (a - b).astype(float)
        return np.stack([np.abs(d), d * d / w, ((a == w) & (b == 0)) | ((a == 0) & (b == w))], -1).astype(float)
    A_in = adj(S[:, 0], S[:, 1])
    A_x = adj(S[:, 1][:, None], S[:, 0][None, :])
    Pv = np.zeros((V, V, P * P))
    vi, vj = np.meshgrid(np.arange(V), np.arange(V), indexing="ij")
    for j in range(S.shape[1]):
        np.add.at(Pv, (vi, vj, P * S[vi, j] + S[vj, j]), 1)
    Par = np.zeros((4, V, V, 20))
    s = S.sum(1)
    for dy in (0, 1):
        for dx in (0, 1):
            r = s[:, None] - 2 * S[None, :, dx]                      # (child, parent)
            oh = np.clip(r, -4, 4) + 4
            for k in range(9):
                Par[2 * dy + dx, :, :, 10 * dy + k] = oh == k
            Par[2 * dy + dx, :, :, 10 * dy + 9] = np.abs(r) / w
    return dict(U=U, A_in=A_in, A_x=A_x, Pv=Pv, Par=Par)


def _slices(f):
    dims = [("U", f["U"].shape[-1]), ("A", f["A_in"].shape[-1]), ("Pv", f["Pv"].shape[-1]), ("Par", f["Par"].shape[-1])]
    out, s = {}, 0
    for k, d in dims:
        out[k] = slice(s, s + d)
        s += d
    return out, s


def level_tables(w, f, xp=np):
    """(u, pair0, pair1, parent) of one level from its weights."""
    sl, _ = _slices(f)
    wa = w[sl["A"]]
    u = f["U"] @ w[sl["U"]] + f["A_in"] @ wa
    p0 = f["A_x"] @ wa
    p1 = f["Pv"] @ w[sl["Pv"]]
    par = f["Par"] @ w[sl["Par"]]
    return u, p0, p1, par


def to_tables(W, lg=None):
    """promise.Tables from {h: weights}."""
    lg = lg or EN.lang()
    f = features(lg)
    u, pair, parent = {}, {}, {}
    for h, w in W.items():
        a, p0, p1, par = level_tables(np.asarray(w, float), f)
        u[h], pair[h], parent[h] = a, np.stack([p0, p1]), par
    return Tables(lg.V, u, pair, parent)


# ---------------------------------------------------------------- column data
def column_data(grids, h, lg, G):
    """Every child column of every map at level h as training chains:
    X (N, R) observed values, Lv / Rv (N, R) neighbour columns, pos (N, R)
    parent-term position (or None at the top), Pv (N, R) parent values,
    mask_u (N, R, V), mask_t (N, R-1, V, V)."""
    X, L, Rt, POS, PAR, MU, MT = [], [], [], [], [], [], []
    has_parent = (2 * h) in grids
    for m, g in enumerate(grids[h]):
        R, C = g.shape
        par = grids[2 * h][m] if has_parent else None
        mu = np.ones((R, lg.V), bool)
        if G:
            mu[-1] = lg.valid(np.arange(lg.V), True, G, h)
        for x in range(C):
            X.append(g[:, x])
            L.append(g[:, (x - 1) % C])
            Rt.append(g[:, (x + 1) % C])
            MU.append(mu)
            if hasattr(lg, "seam_compat"):             # level-dependent seam relation (average language)
                mt = np.stack([lg.seam_compat(h, bool(G) and y + 1 == R - 1, G) for y in range(R - 1)]) \
                    if R > 1 else np.zeros((0, lg.V, lg.V), bool)
            else:
                mt = np.repeat(lg.COMPAT[None], R - 1, 0).copy()
            if has_parent:
                prow = par[np.arange(R) // 2, x // 2]
                POS.append(2 * (np.arange(R) % 2) + x % 2)
                PAR.append(prow)
                for y in range(0, R - 1, 2):
                    part = int(lg.parent_part(prow[y], x % 2))
                    gl = bool(G) and y + 1 == R - 1
                    mt[y] &= lg.pair_mask(part, False, gl, G, h)
            MT.append(mt)
    d = dict(X=np.array(X), L=np.array(L), R=np.array(Rt), mask_u=np.array(MU), mask_t=np.array(MT))
    if has_parent:
        d["pos"], d["par"] = np.array(POS), np.array(PAR)
    assert d["mask_u"][np.arange(len(X))[:, None], np.arange(d["X"].shape[1])[None], d["X"]].all()
    Xa = d["X"]
    assert d["mask_t"][np.arange(len(X))[:, None], np.arange(Xa.shape[1] - 1)[None], Xa[:, :-1], Xa[:, 1:]].all()
    return d


def _nll_fn(xp, logsumexp):
    """Mean negative log-likelihood of the observed columns given the level
    tables (u, p0, p1, par) - numpy or jax."""
    def nll(u, p0, p1, par, d):
        X, Lv, Rv = d["X"], d["L"], d["R"]
        N, R = X.shape
        E = u[None, None, :] + p0[Lv] + p0.T[Rv]                           # (N, R, V)
        if "pos" in d:
            parT = xp.moveaxis(par, 2, 1)                                  # (4, Vparent, Vchild)
            E = E + parT[d["pos"], d["par"]]
        E = xp.where(d["mask_u"], E, BIG)
        T = xp.where(d["mask_t"], p1[None, None], BIG)                     # (N, R-1, V, V)
        a = -E[:, 0]
        for y in range(R - 1):
            a = logsumexp(a[:, :, None] - T[:, y], axis=1) - E[:, y + 1]
        logZ = logsumexp(a, axis=1)
        n = xp.arange(N)
        obs = E[n[:, None], xp.arange(R)[None], X].sum(1)
        if R > 1:
            obs = obs + p1[X[:, :-1], X[:, 1:]].sum(1)
        return (obs + logZ).mean()
    return nll


def column_nll(tables, grids, h, lg=None, G=8):
    """Held-out objective for any Tables at level h (nats per column)."""
    from scipy.special import logsumexp
    lg = lg or EN.lang()
    u, pair, par = tables.level(h)
    d = column_data(grids, h, lg, G)
    return float(_nll_fn(np, logsumexp)(np.asarray(u, float), np.asarray(pair[0], float),
                                         np.asarray(pair[1], float), np.asarray(par, float), d))


# --------------------------------------------------------------------- fitting
def fit(grids, lg=None, G=8, l2=1e-2, maxiter=300, log=print):
    """{h: weights} fitted per level on mined grids {h: [grid per map]}."""
    import jax
    with jax.enable_x64(True):                      # scoped: the rest of the repo's jax code runs in 32 bit
        return _fit(grids, lg, G, l2, maxiter, log)


def _fit(grids, lg, G, l2, maxiter, log):
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    from scipy.optimize import minimize
    lg = lg or EN.lang()
    f = {k: jnp.asarray(v) for k, v in features(lg).items()}
    _, D = _slices(features(lg))
    nll = _nll_fn(jnp, logsumexp)
    W = {}
    for h in sorted(grids):
        d = {k: jnp.asarray(v) for k, v in column_data(grids, h, lg, G).items()}

        def obj(w):
            return nll(*level_tables(w, f, jnp), d) + l2 * jnp.sum(w ** 2)
        vg = jax.jit(jax.value_and_grad(obj))
        res = minimize(lambda w: tuple(np.asarray(t, float) for t in vg(jnp.asarray(w))), np.zeros(D),
                       jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
        W[h] = res.x
        log(f"envpredict: h={h} columns={d['X'].shape[0]} nll/column={res.fun:.3f} ({res.nit} it)")
    return W


def tables(ts, maps, K=16, ground=8, h_top=64, l2=1e-2, lg=None, log=print):
    """Tables from the linear predictor fitted on corpus maps."""
    from castlegen import refheights as RH
    lg = lg or EN.lang()
    grids = RH.mine(ts, maps, K, ground, h_top, lg)
    return to_tables(fit(grids, lg, ground, l2, log=log), lg)
