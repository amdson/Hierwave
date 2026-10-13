"""Gradient check for the cpass actor estimator (spec test 5).

A tiny two-level periodic problem whose every complete draw outcome can be
enumerated; exact J(theta) = E_q[S + sum_l lambda_l H_l + tau log q] and its
gradient by enumeration (forced scoring of each outcome, differentiated with
jax); and the cpass estimator's gradient averaged over independent batches.

The cpass actor loss is the mean over worlds of sum_l (sum of level-l draw
terms) / nval[l]: a positive scale per parameter block (parameters are per
level); `estimator_grad` multiplies level l by nval[l] so that it estimates
grad J itself."""
from __future__ import annotations

import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import INF, Chan, Problem, Region, count, pair, unary

from . import _cpass_core as core


# ------------------------------------------------------------------ problem
def tiny_problem(D1=3):
    """T = 1, H = 2, W = 8: 16 binary fine cells t, 4 coarse cells c (D1 values).
    Hard: support (dirt above air), no two horizontally adjacent dirt cells
    (crosses blocks, so the one-step lookahead can be trapped), count
    t -> c bands, no two adjacent c = 2.  Soft: random-ish unaries and pairs."""
    rng = np.random.default_rng(7)
    chans = [Chan("c", 1, D1), Chan("t", 0, 2)]
    supp = np.array([[0.0, 0.0], [INF, 0.0]])          # t(y)=1 above t(y+1)=0
    horiz = np.array([[0.0, 0.3], [-0.2, INF]])
    cnt = np.full((D1, 5), INF)
    cnt[0, 0:2] = [0.0, 0.4]
    cnt[1, 1:4] = [0.2, -0.3, 0.1]
    if D1 > 2:
        cnt[2, 2:5] = [0.5, -0.2, 0.0]
    cc = rng.normal(scale=0.5, size=(D1, D1))
    if D1 > 2:
        cc[2, 2] = INF
    factors = [
        unary(("c", "id"), rng.normal(scale=0.5, size=D1), name="cu"),
        pair(("c", "id"), ("c", "id"), (0, 1), cc, name="cc"),
        count(("t", "id"), ("c", "id"), cnt, name="count"),
        pair(("t", "id"), ("t", "id"), (1, 0), supp, name="support"),
        pair(("t", "id"), ("t", "id"), (0, 1), horiz, name="horiz"),
        unary(("t", "id"), [0.0, -0.4], name="tu"),
    ]
    return Problem("gradcheck_tiny", T=1, H=2, chans=chans, factors=factors, W_train=8, W_eval=8)


def all_states(problem, region):
    """Every assignment of the sampled channels: {name: (N, h, w)} int32."""
    shapes = [(c.name, problem.shape(c.name, region), c.D) for c in problem.sampled()]
    sizes = [int(np.prod(s)) for _, s, _ in shapes]
    Ds = sum(([D] * n for (_, _, D), n in zip(shapes, sizes)), [])
    grid = np.array(list(itertools.product(*[range(D) for D in Ds])), np.int32)
    out, o = {}, 0
    for (name, s, _), n in zip(shapes, sizes):
        out[name] = grid[:, o:o + n].reshape((-1,) + s)
        o += n
    return out


def reachable_states(lay):
    """Every complete outcome the pass can draw (q > 0), by expanding the draw
    tree step by step: a site's options are its least-violating set A (closed
    violations + lookahead), computed exactly as coloured_pass does from the
    state before the step.  {name: (N, h, w)} int32."""
    p = lay.p
    st = {c.name: np.full((1,) + lay.hw[c.level], -1, np.int32) for c in p.chans}
    for l in lay.levels:
        for k in range(core.NCOL):
            Y, X, valid = (np.asarray(a) for a in lay.sites(l, k))
            for c in lay.chans[l]:
                def masks(state, c=c, Y=Y, X=X):
                    _, nv, look = lay.cond(state, c.name, jnp.asarray(Y), jnp.asarray(X))
                    v = nv + look
                    return v == jnp.min(v, axis=1, keepdims=True)
                m = np.asarray(jax.vmap(masks)({n: jnp.asarray(a) for n, a in st.items()}))   # (N, n, D)
                for s in np.flatnonzero(valid):
                    rows, vals = np.nonzero(m[:, s])
                    st = {n: a[rows] for n, a in st.items()}
                    m = m[rows]
                    st[c.name][:, Y[s], X[s]] = vals
    return {c.name: st[c.name] for c in p.sampled()}


# ------------------------------------------------------------------- exact
class Exact:
    """Reachable outcomes and their reference energies, for one layout."""

    def __init__(self, var, lam, chunk=8192, cache=None, reference=True, ref_check=300, brute=False):
        self.var, self.lay = var, var.layout(Region(var.problem.W_train))
        p, region = var.problem, Region(var.problem.W_train)
        lay = self.lay
        self.lam = jnp.asarray(lam, jnp.float32)

        def lq(params, forced, tau):
            return core.coloured_pass(lay, params, 0, {}, tau, self.lam, forced=forced)[2]

        self._lq = jax.jit(jax.vmap(lq, in_axes=(None, 0, None)))
        names = [c.name for c in p.sampled()]
        if cache is not None and os.path.exists(cache):
            z = np.load(cache)
            self.X = {k: z["X_" + k] for k in names}
            self.S, self.H = z["S"], z["H"]
        else:
            if brute:
                X = all_states(p, region)
                N = len(next(iter(X.values())))
                keep = []
                for s in range(0, N, chunk):
                    v = np.asarray(self._lq(var.params, {k: jnp.asarray(a[s:s + chunk]) for k, a in X.items()},
                                            1.0))
                    keep.append(np.isfinite(v))
                keep = np.concatenate(keep)
                # reachability is fixed by the masks (tau / params never zero a probability)
                self.X = {k: a[keep] for k, a in X.items()}
            else:
                self.X = reachable_states(lay)
            n = len(next(iter(self.X.values())))
            self.S = np.zeros(n)
            self.H = np.zeros((n, p.T + 1))
            if reference:
                todo = range(n)
            else:
                # the pass's own closed energies (sum of draw costs - tau log q): fast; the
                # reference energy is then checked on `ref_check` random outcomes
                self.S, self.H = self.pass_energy(var.params)
                todo = np.random.default_rng(0).choice(n, min(ref_check, n), replace=False)
            for i in todo:
                e = en.energy(p, {k: a[i] for k, a in self.X.items()}, region)
                Hi = [e.viol[l] for l in range(p.T + 1)]
                if not reference:
                    assert abs(self.S[i] - e.soft) < 1e-3 and (self.H[i] == Hi).all(), (i, self.S[i], e.soft)
                self.S[i], self.H[i] = e.soft, Hi
            if cache is not None:
                np.savez_compressed(cache, S=self.S, H=self.H, **{"X_" + k: self.X[k] for k in names})
        self.n = len(self.S)
        self.E = self.S + self.H @ np.asarray(lam)

    def pass_energy(self, params, chunk=8192):
        """(S, H) of every reachable outcome from the forced pass's records: closed
        soft energy and violations of each draw, summed (H per drawing level)."""
        lay, p = self.lay, self.var.problem
        lam0 = jnp.zeros(p.T + 1, jnp.float32)

        def one(forced):
            _, tot, lq, rec = core.coloured_pass(lay, params, 0, {}, 1.0, lam0, forced=forced, record=True)
            S = jnp.sum(tot) - lq
            H = []
            for l in range(p.T + 1):
                h = 0.0
                for c in lay.chans[l]:
                    r = rec[core.key_of(l, c.name)]
                    nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
                    h = h + jnp.sum(jnp.where(r["valid"], nva, 0))
                H.append(h)
            return S, jnp.stack(H)

        f = jax.jit(jax.vmap(one))
        S, H = [], []
        n = len(next(iter(self.X.values())))
        for s in range(0, n, chunk):
            a, b = f({k: jnp.asarray(v[s:s + chunk]) for k, v in self.X.items()})
            S.append(np.asarray(a, np.float64))
            H.append(np.asarray(b, np.float64))
        return np.concatenate(S), np.concatenate(H)

    def logq(self, params, tau):
        return self._lq(params, {k: jnp.asarray(a) for k, a in self.X.items()}, tau)

    def J_and_grad(self, params, tau):
        """J = sum_x q(x) (E(x) + tau log q(x)); grad = sum_x grad q(x) (E + tau log q)."""
        X = {k: jnp.asarray(a) for k, a in self.X.items()}
        lq, vjp = jax.vjp(lambda pr: self._lq(pr, X, tau), params)
        lq = np.asarray(lq, np.float64)
        q = np.exp(lq)
        J = float(np.sum(q * (self.E + tau * lq)))
        g, = vjp(jnp.asarray(q * (self.E + tau * lq), jnp.float32))
        return J, g, {"sum_q": float(q.sum()), "p_viol": float(q @ (self.H.sum(1) > 0)),
                      "E_S": float(q @ self.S), "E_H": (q @ self.H).tolist(), "E_logq": float(q @ lq)}


# --------------------------------------------------------------- estimator
def estimator_grad(var, seeds, tau, lam, gscale=1.0, part="all", truncate=None):
    """The cpass actor-loss gradient for one batch of worlds (seeds), rescaled
    by nval[l] per level so that its expectation should be grad J.  part: "all", "branch" (only the coupled
    replay term, scaled by the number of top draws) or "rest" (top draws
    without it).  truncate=n: a deliberately biased negative control, returns
    summed over the next n steps only (unbranched draws).
    Returns (grad pytree, batch info)."""
    lay = var.layout(Region(var.problem.W_train))
    if not hasattr(lay, "actor_loss"):
        var._build_train(lay)
    B = len(seeds)
    region = Region(var.problem.W_train)
    desl = [var.designed(int(s), region) for s in seeds]
    des = {k: jnp.stack([d[k] for d in desl]) for k in desl[0]} if desl[0] else {}
    seeds = jnp.asarray(seeds, jnp.int32)
    tau_, lam_ = jnp.float32(tau), jnp.asarray(lam, jnp.float32)
    extra = jnp.concatenate([jnp.array([tau]), lam_]).astype(jnp.float32)
    st, tot, recs = lay.jit_rec(var.params, seeds, des, tau_, lam_, extra)
    G, after = var._returns(lay, tot)
    br = var._branches(lay, recs, after, seeds, des, tau_, lam_)
    if truncate is not None:
        cs = np.concatenate([np.zeros((B, 1)), np.cumsum(np.asarray(tot, np.float64), -1)], -1)
        ns = cs.shape[1] - 1
        idx = np.arange(ns)
        aft = cs[:, np.minimum(idx + 1 + truncate, ns)] - cs[:, idx + 1]
        G, _ = var._returns(lay, jnp.asarray(np.diff(cs, axis=-1)))   # for the key layout
        for l in lay.levels:
            for ci, c in enumerate(lay.chans[l]):
                steps = np.array([lay.step(l, k, ci) for k in range(core.NCOL)])
                G[core.key_of(l, c.name)] = jnp.asarray(aft[:, steps], jnp.float32)
    T = var.problem.T
    ntop = lay.nval[T]
    scale = {l: float(lay.nval[l]) for l in lay.levels}      # loss = mean_worlds sum_l (draw terms) / nval[l]
    if part == "branch":         # only the coupled-branch term: top draws' own terms switched off
        recs = {k: ({**r, "valid": jnp.zeros_like(r["valid"])} if k in lay.topkeys else r) for k, r in recs.items()}
        scale[T] *= ntop         # the branched draw is one of ntop, chosen uniformly
    elif part == "rest":         # only the unbranched top draws: the branch term switched off
        br = {**br, "chan": jnp.full_like(br["chan"], -1)}
        scale[T] *= ntop / (ntop - 1)
    if not hasattr(lay, "jit_agrad"):
        lay.jit_agrad = jax.jit(jax.grad(lay.actor_loss))
    g = lay.jit_agrad(var.params, var.cparams, recs, G, tau_, lam_, br, jnp.float32(gscale))
    g = {k: jax.tree_util.tree_map(lambda x: x * scale[int(k[1:].split("_")[0])], v) for k, v in g.items()}
    return g, {"cost": np.asarray(tot).sum(-1), "enum": np.asarray(br["enum"])}


def flat(g, keys=None):
    keys = sorted(g) if keys is None else keys
    return np.concatenate([np.ravel(np.asarray(x, np.float64)) for k in keys
                           for x in jax.tree_util.tree_leaves(g[k])])


def randomize(params, key, scale=1.0, last_only=False):
    """Random nonzero parameters (the init zeroes the output layer, so q would
    ignore the network)."""
    out = {}
    for k, p in sorted(params.items()):
        out[k] = {}
        for n, x in sorted(p.items()):
            key, sk = jax.random.split(key)
            if last_only and n not in ("W3", "b3"):
                out[k][n] = x
            elif n in ("W3", "b3"):
                out[k][n] = scale * jax.random.normal(sk, x.shape)
            else:
                out[k][n] = x if n.startswith("W") else 0.3 * jax.random.normal(sk, x.shape)
    return out


def zero_critic(cparams):
    return {k: {n: (jnp.zeros_like(x) if n in ("W3", "b3") else x) for n, x in p.items()}
            for k, p in cparams.items()}


def fit_critic(var, tau, lam, steps, B, seed0, lr=3e-3):
    """Fit the critic alone (actor fixed) with cpass's critic loss."""
    from .cpass import adam_init, adam_step
    lay = var.layout(Region(var.problem.W_train))
    if not hasattr(lay, "actor_loss"):
        var._build_train(lay)
    tau_, lam_ = jnp.float32(tau), jnp.asarray(lam, jnp.float32)
    extra = jnp.concatenate([jnp.array([tau]), lam_]).astype(jnp.float32)
    cg = jax.jit(jax.value_and_grad(lay.critic_loss))
    opt = adam_init(var.cparams)
    losses = []
    for i in range(steps):
        seeds = jnp.asarray(np.arange(seed0 + i * B, seed0 + (i + 1) * B), jnp.int32)
        st, tot, recs = lay.jit_rec(var.params, seeds, {}, tau_, lam_, extra)
        G, after = var._returns(lay, tot)
        br = var._branches(lay, recs, after, seeds, {}, tau_, lam_)
        l, g = cg(var.cparams, recs, G, br, jnp.float32(1.0))
        var.cparams, opt = adam_step(var.cparams, g, opt, lr)
        losses.append(float(l))
    return losses


def block_slices(g, keys=None):
    keys = sorted(g) if keys is None else keys
    out, o = {}, 0
    for k in keys:
        n = sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(g[k]))
        out[k] = slice(o, o + n)
        o += n
    return out


def ztest(est, exact, B):
    """est: (M, P) batch-mean gradients (B worlds each); exact: (P,).

    proj_z: z of the mean projection on the exact gradient's direction minus
            |grad J| (the most sensitive single test of a bias along grad J);
    rand_z: the same along a fixed random direction;
    chi2:   mean of z^2 over parameters with nonzero variance (1 if unbiased);
    maxz:   max |z| over parameters;
    rel:    |mean - exact| / |exact|;
    snr_world: |grad J|^2 / E|g_world - grad J|^2 (one world's gradient);
    cos:    cosine between the mean and the exact gradient."""
    M = est.shape[0]
    m = est.mean(0)
    sd = est.std(0, ddof=1)
    ok = sd > 1e-12 * (np.abs(m).max() + 1e-30)
    z = (m[ok] - exact[ok]) / (sd[ok] / np.sqrt(M))
    nrm = np.linalg.norm(exact)
    u = exact / max(nrm, 1e-30)
    d = est @ u
    r = np.random.default_rng(0).normal(size=exact.shape)
    r /= np.linalg.norm(r)
    dr = est @ r
    var_world = B * np.sum(sd ** 2)
    # Hotelling T^2 on k = 16 fixed random directions plus grad J's: calibrated even
    # though coordinates are strongly correlated (an MLP's gradient has low rank)
    # restricted to coordinates that varied in the sample: a parameter read only in
    # contexts too rare to have occurred has zero sample variance but a nonzero
    # exact gradient, which is not a bias but would make T^2 explode
    R = np.random.default_rng(1).normal(size=(exact.size, 16))
    R = np.concatenate([u[:, None], R / np.linalg.norm(R, axis=0)], 1) * ok[:, None]
    P = est @ R
    dev = P.mean(0) - exact @ R
    C = np.cov(P, rowvar=False) / M
    # whitened, keeping eigen-directions with variance >= 1e-6 of the largest: an
    # MLP's gradient is nearly low-rank, and along directions with ~1e-8 relative
    # variance float32 round-off (~1e-4 absolute) and unseen rare contexts dominate
    w, V = np.linalg.eigh(C)
    keep = w >= 1e-6 * w.max()
    k = int(keep.sum())
    T2 = float(np.sum((V[:, keep].T @ dev) ** 2 / w[keep]))
    F = T2 * (M - k) / (k * (M - 1))
    from scipy.stats import f as fdist
    pval = float(fdist.sf(F, k, M - k))
    return {"hotelling_p": pval, "hotelling_F": F, "hotelling_k": k,
            "unseen_frac": float(np.linalg.norm(exact[~ok]) / max(nrm, 1e-30)),"proj_z": float((d.mean() - nrm) / (d.std(ddof=1) / np.sqrt(M))),
            "proj_ratio": float(d.mean() / max(nrm, 1e-30)),
            "rand_z": float((dr.mean() - r @ exact) / (dr.std(ddof=1) / np.sqrt(M))),
            "chi2": float(np.mean(z ** 2)), "maxz": float(np.abs(z).max()), "npar": int(ok.sum()),
            "rel": float(np.linalg.norm(m - exact) / max(nrm, 1e-30)),
            "rel_noise": float(np.sqrt(np.sum(sd ** 2) / M) / max(nrm, 1e-30)),
            "cos": float(m @ exact / max(np.linalg.norm(m) * nrm, 1e-30)),
            "snr_world": float(nrm ** 2 / max(var_world, 1e-30)), "norm": float(nrm), "M": M}
