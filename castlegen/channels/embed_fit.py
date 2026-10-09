"""Learned value embeddings for the bootstrap (notes/circles_biome_stage4c.md;
dsl_updates.md C2 "Feature set": learned token embeddings vs derived).

Model at a site with candidate c and neighbour values nb_j (j over the
neighbour slots, -1 off the grid: the term vanishes, like the kernel's pad -1):

    psi(c; ctx) = u[c] + e(c) . h(ctx) (+ X_stamp(c) . theta, mixed)
    bilinear:   h = sum_j B_j e(nb_j),  B_j = A[d_j] (forward slot: the table
                g_d[c, nb] = e(c)^T A_d e(nb)) or A[d_j]^T (reflected slot:
                g_d[nb, c]), so the materialised tables are g_d = E A_d E^T
    mlp:        h = W2 relu(W1 [e(nb_1) .. e(nb_n)] + b1) + b2 (not a joint
                potential; no tables)

fix0: e(0) = 0, u[0] = 0 (value 0 = absent: the gauge materialise uses, u[0]
and g_d[0, .] exactly 0).  Off-grid neighbours read a zero row.

Data(recs)           padded records (cand, e, pi, nb[, X]) -> CAND, M, E, PI, NB, XS
init / pack          parameter dicts <-> flat vectors
loss                 (mean KL(pi || softmax(-(e + psi))) + l2 |params but u|^2, grad)
fit_embed            Adam from random restarts (+ L-BFGS polish), best by loss
tables               materialised g_d (n_off, D, D) and u of a bilinear model
tables_energy        psi from tables (any pairwise model, e.g. tabular)
kl                   KL of a psi matrix against the data's targets
bootstrap            train.fit's loop (contexts, targets, K steps, aggregated
                     records, holdout) for a non-linear-in-theta fitter"""
import time

import numpy as np

from . import train


# ------------------------------------------------------------------ data
class Data:
    """recs: list of (cand, e, pi, nb) or (cand, e, pi, nb, X), over a domain
    of D values (default max candidate + 1).  Held on the full domain: MF, EF,
    PIF (N, D) (mask, designed energy, target); NB (N, nd) with -1 off the
    grid, NBi the same with -1 -> D (a zero row), OH[j] its one-hot (sparse);
    CAND / M (N, Dm) padded candidates and XS (N, Dm, nf) for stamp features."""

    def __init__(self, recs, D=None):
        from scipy.sparse import csr_matrix
        N = len(recs)
        Dm = max(len(r[0]) for r in recs)
        D = D or int(max(r[0].max() for r in recs)) + 1
        nd = len(recs[0][3])
        self.N, self.D = N, D
        self._H = None
        self.CAND = np.zeros((N, Dm), np.int64)
        self.M = np.zeros((N, Dm), bool)
        self.MF = np.zeros((N, D), bool)
        self.EF = np.zeros((N, D))
        self.PIF = np.zeros((N, D))
        self.NB = np.asarray([r[3] for r in recs], np.int64).reshape(N, nd)
        self.XS = None
        if len(recs[0]) > 4 and recs[0][4] is not None:
            self.XS = np.zeros((N, Dm, recs[0][4].shape[1]))
        for n, r in enumerate(recs):
            c = np.asarray(r[0], np.int64)
            k = len(c)
            self.CAND[n, :k], self.M[n, :k] = c, True
            self.MF[n, c], self.EF[n, c], self.PIF[n, c] = True, r[1], r[2]
            if self.XS is not None:
                self.XS[n, :k] = r[4]
        self.NBi = np.where(self.NB < 0, D, self.NB)
        self.OH = [csr_matrix((np.ones(N), (np.arange(N), self.NBi[:, j])), shape=(N, D + 1)) for j in range(nd)]
        self.NEG = np.where(self.MF, 0.0, -np.inf)
        self.R, self.Cc = np.nonzero(self.M)
        self.Cf = self.CAND[self.R, self.Cc]
        self.XF = None if self.XS is None else np.ascontiguousarray(self.XS[self.R, self.Cc])   # (n_entries, nf)


def _entropy_term(d):
    P = d.PIF
    return float(np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0).sum() / d.N)


def kl(psi, d):
    """(KL mean over records, dKL/dpsi) for learned energies psi (N, D):
    KL = H + mean_n [sum_t pi (e + psi) + log Z_n], H = mean sum pi log pi."""
    a = d.NEG - d.EF - psi
    mx = a.max(1, keepdims=True)
    w = np.exp(a - mx)
    Z = w.sum(1, keepdims=True)
    if d._H is None or d._Hsrc is not d.PIF:
        d._H, d._Hsrc = _entropy_term(d), d.PIF
    v = d._H + (np.einsum("nt,nt->", d.PIF, d.EF + psi) + (np.log(Z) + mx).sum()) / d.N
    return v, (d.PIF - w / Z) / d.N


# ------------------------------------------------------------ parameters
def init(D, k, n_off, rng, scale=0.3, nf=0, mlp=0, nd=0, fix0=True):
    p = dict(E=rng.normal(0, scale, (D, k)), u=np.zeros(D))
    if mlp:
        p.update(W1=rng.normal(0, 1 / np.sqrt(nd * k), (mlp, nd * k)), b1=np.zeros(mlp),
                 W2=rng.normal(0, 1 / np.sqrt(mlp), (k, mlp)), b2=np.zeros(k))
    else:
        p["A"] = rng.normal(0, scale, (n_off, k, k))
    if nf:
        p["theta"] = np.zeros(nf)
    if fix0:
        p["E"][0] = 0.0
    return p


def pack(p):
    return np.concatenate([p[k].ravel() for k in sorted(p)])


def unpack(x, like):
    out, o = {}, 0
    for k in sorted(like):
        n = like[k].size
        out[k] = x[o:o + n].reshape(like[k].shape)
        o += n
    return out


def n_params(p, fix0=True):
    n = sum(v.size for v in p.values())
    return n - (p["E"].shape[1] + 1 if fix0 else 0)


# -------------------------------------------------------------- forward
def _slots(dirs):
    return [(int(d), bool(f)) for d, f in dirs]


def psi(p, d, dirs, grad_psi=None):
    """psi (N, D) over the full domain; with grad_psi = dL/dpsi also returns
    dL/dparams (dict)."""
    E = p["E"]
    D, k = E.shape
    EE = np.vstack([E, np.zeros((1, k))])
    EN = EE[d.NBi]                                                    # (N, nd, k)
    if "A" in p:
        Bs = [p["A"][o] if f else p["A"][o].T for o, f in _slots(dirs)]
        h = sum(EN[:, j] @ B.T for j, B in enumerate(Bs))
    else:
        z = EN.reshape(d.N, -1)
        a1 = z @ p["W1"].T + p["b1"]
        r = np.maximum(a1, 0.0)
        h = r @ p["W2"].T + p["b2"]
    out = h @ E.T + p["u"][None, :]
    if "theta" in p:
        out[d.R, d.Cf] += d.XF @ p["theta"]
    if grad_psi is None:
        return out
    G = grad_psi
    g = {kk: np.zeros_like(v) for kk, v in p.items()}
    g["u"] = G.sum(0)
    Gh = G @ E
    gEE = np.zeros_like(EE)
    gEE[:D] = G.T @ h
    if "A" in p:
        for j, ((o, f), B) in enumerate(zip(_slots(dirs), Bs)):
            gEE += d.OH[j].T @ (Gh @ B)
            gB = Gh.T @ EN[:, j]
            g["A"][o] += gB if f else gB.T
    else:
        g["W2"] = Gh.T @ r
        g["b2"] = Gh.sum(0)
        ga = (Gh @ p["W2"]) * (a1 > 0)
        g["W1"] = ga.T @ z
        g["b1"] = ga.sum(0)
        gz = (ga @ p["W1"]).reshape(EN.shape)
        for j in range(EN.shape[1]):
            gEE += d.OH[j].T @ gz[:, j]
    g["E"] = gEE[:D]
    if "theta" in p:
        g["theta"] = d.XF.T @ G[d.R, d.Cf]
    return out, g


def loss(p, d, dirs, l2=1e-4, fix0=True):
    P = psi(p, d, dirs)
    v, G = kl(P, d)
    _, g = psi(p, d, dirs, G)
    for kk in p:
        if kk == "u":                     # no ridge on the unary: in the u[0] = 0 gauge it is the
            continue                      # object's whole free energy (~25 nats here)
        v += l2 * float((p[kk] ** 2).sum())
        g[kk] = g[kk] + 2 * l2 * p[kk]
    if fix0:
        g["E"][0] = 0.0
        g["u"][0] = 0.0
    return v, g


# ------------------------------------------------------------------ fit
def _adam(f, x, steps, lr):
    m = np.zeros_like(x); v = np.zeros_like(x)
    for t in range(1, steps + 1):
        _, g = f(x)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        a = lr * 0.5 * (1 + np.cos(np.pi * t / steps)) + 1e-4               # cosine decay
        x = x - a * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
    return x


def fit_embed(d, D, k, dirs, n_off, l2=1e-4, restarts=3, steps=2000, lr=0.03, polish=2000, seed=0, init_p=None,
              nf=0, mlp=0, fix0=True, fixed=(), theta0=None, scale=None):
    """argmin of loss by Adam from `restarts` random inits (plus init_p when
    given) then L-BFGS (`polish` iterations, 0: none); best by final loss.
    fixed: parameter names held at their init (gradient zeroed); theta0: the
    stamp part's init (mixed); scale: multiplies the random E inits.
    Returns (params, info: losses per start, seconds)."""
    rng = np.random.default_rng(seed)
    nd = d.NB.shape[1]
    starts = ([init_p] if init_p is not None else []) + \
        [init(D, k, n_off, rng, nf=nf, mlp=mlp, nd=nd, fix0=fix0) for _ in range(restarts)]
    for p in starts[len(starts) - restarts:]:
        if theta0 is not None:
            p["theta"] = np.array(theta0, float)
        if scale is not None:
            p["E"] *= scale
    t0 = time.time()
    best, losses = None, []
    for p0 in starts:
        p0 = {kk: np.array(v, float) for kk, v in p0.items()}

        def f(x, like=p0):
            v, g = loss(unpack(x, like), d, dirs, l2, fix0)
            for kk in fixed:
                g[kk] = np.zeros_like(g[kk])
            return v, pack(g)
        x = _adam(f, pack(p0), steps, lr) if steps else pack(p0)
        if polish:
            from scipy.optimize import minimize
            x = minimize(f, x, jac=True, method="L-BFGS-B", options=dict(maxiter=polish, gtol=1e-10, ftol=1e-15)).x
        v = f(x)[0]
        losses.append(float(v))
        if best is None or v < best[0]:
            best = (v, unpack(x, p0))
    return best[1], dict(losses=losses, seconds=time.time() - t0)


# ----------------------------------------------------------- materialise
def tables(p):
    """(G (n_off, D, D), u (D,)) of a bilinear model: G[d] = E A_d E^T."""
    E = p["E"]
    return np.einsum("ik,dkl,jl->dij", E, p["A"], E), p["u"].copy()


def tables_energy(G, u, d, dirs):
    """psi (N, D) of a pairwise model with tables G[off] (t at p, t' at p +
    off) and unary u; neighbour -1 contributes nothing."""
    D = len(u)
    out = np.tile(np.asarray(u, float), (d.N, 1))
    for j, (o, f) in enumerate(_slots(dirs)):
        T = G[o] if f else G[o].T
        out += np.hstack([T, np.zeros((D, 1))])[:, d.NBi[:, j]].T
    return out


# ------------------------------------------------------------- bootstrap
def bootstrap(contexts, home, targets, ctx, designed_e, fitter, iters, n_contexts, K=0, after_change=None,
              holdout=0.1, rng=0, verbose=False):
    """train.fit's loop with records (cand, e, pi, *ctx(model, home, y, x,
    cand)) and fitter(Data, prev) -> (params, kl_train); probes omitted.
    Returns (params, report with n_records, kl_train, kl_held, seconds per
    iteration; held: the held-out Data)."""
    rng = np.random.default_rng(rng)
    params, recs, held = None, [], []
    rep = dict(n_records=[], kl_train=[], kl_held=[], seconds=[])
    T0 = time.time()
    for it in range(iters):
        t0 = time.time()
        for model in contexts(params, n_contexts, rng):
            hc = model.chan(home)
            order, _ = train._site_order(model, home)
            for k in range(K + 1):
                for y, x in order:
                    e = np.asarray(designed_e(model, home, y, x, np.arange(hc.D)), float)
                    cand = np.flatnonzero(np.isfinite(e))
                    if len(cand) == 0:
                        continue
                    pi = np.asarray(targets.at(model, home, y, x, cand), float)
                    if len(cand) > 1:
                        r = (cand, e[cand], pi) + tuple(ctx(model, home, y, x, cand))
                        (held if rng.random() < holdout else recs).append(r)
                    if k < K:
                        hc.grid[y, x] = cand[rng.choice(len(cand), p=pi / pi.sum())]
                        if after_change is not None:
                            after_change(model, home, y, x)
        D = model.chan(home).D
        params, ktr, kh = fitter(Data(recs, D), params, Data(held, D) if held else None)
        rep["n_records"].append(len(recs) + len(held))
        rep["kl_train"].append(float(ktr))
        rep["kl_held"].append(float(kh))
        rep["seconds"].append(time.time() - t0)
        if verbose:
            print(f"  fit it {it}: {rep['n_records'][-1]} records  KL train {ktr:.5f} held {kh:.5f}  "
                  f"{rep['seconds'][-1]:.1f}s", flush=True)
    rep["seconds_total"] = time.time() - T0
    rep["held"] = Data(held, D) if held else None
    return params, rep
