"""Exact path KL for crevise on a tiny enumerable problem (spec test 5 for the
path-KL estimator, tests/variants/test_crevise.py).

With the erase coins fixed (mask_seed), every trajectory the sampler can
produce -- pass-1 values, then the redrawn values of the erased cells, level
by level -- is enumerated by expanding the draw tree step by step exactly as
revise_pass does (least-violating sets from the state before each draw).
Forced scoring of every trajectory gives log Q(traj) and the summed per-draw
costs C(traj); then

    J(theta) = sum_traj Q(traj) C(traj)        (the path-KL integrand, + log Z)
    grad J   = sum_traj grad Q(traj) C(traj)   (C's tau log Q term has zero expected gradient)

and the estimator's mean over fresh worlds drawn with the same mask_seed must
equal grad J.  Grouping trajectories by their final state gives q(x_0) and
hence the exact reverse KL of x_0, so the bound gap J - sum_x q (log q + E) is
exact too (log Z cancels)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench import noise
from castlegen.bench.problem import INF, Chan, Problem, Region, pair, unary

from . import _cpass_core as core
from ._crevise_core import gkey, mask_tag, revise_pass


def tiny_flat_problem():
    """T = 0, H = 2, W = 8: 16 binary cells with gradcheck's fine factors only
    (hard support and no-adjacent-dirt, soft unary and horizontal pair)."""
    supp = np.array([[0.0, 0.0], [INF, 0.0]])
    horiz = np.array([[0.0, 0.3], [-0.2, INF]])
    factors = [pair(("t", "id"), ("t", "id"), (1, 0), supp, name="support"),
               pair(("t", "id"), ("t", "id"), (0, 1), horiz, name="horiz"),
               unary(("t", "id"), [0.0, -0.4], name="tu")]
    return Problem("crevise_tiny_flat", T=0, H=2, chans=[Chan("t", 0, 2)], factors=factors, W_train=8, W_eval=8)


def masks_of(sch, rho, mask_seed):
    """{group key (m >= 1): (16, n) bool} erase masks of the fixed coins (jnp, as the pass)."""
    lay = sch.lay
    out = {}
    for key, (l, ci, cname, m, kind) in sch.groups.items():
        if m == 0:
            continue
        h, w = lay.hw[l]
        Ys, Xs, Vs = [], [], []
        for k in range(core.NCOL):
            Y, X, valid = lay.sites(l, k)
            Ys.append(Y); Xs.append(X); Vs.append(valid)
        Y, X, valid = jnp.stack(Ys), jnp.stack(Xs), jnp.stack(Vs)
        Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
        u = noise.uniform(jnp, mask_seed, mask_tag(cname, m), l, Y, Xn)
        out[key] = np.asarray((u < rho) & valid)
    return out


def enumerate_trajectories(sch, rho, mask_seed):
    """Every trajectory with q > 0 under the fixed masks: {group key: (N, h, w)}
    int32 forced values (sites a group does not draw hold the current state)
    plus the final states {name: (N, h, w)}.  Recorded groups travel in the
    same dict as the channel states, so every branching expands them too."""
    lay = sch.lay
    p = lay.p
    masks = masks_of(sch, rho, mask_seed)
    names = [c.name for c in p.chans]
    st = {c.name: np.full((1,) + lay.hw[c.level], -1, np.int32) for c in p.chans}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue
        for m in range(sch.R[l] + 1):
            if m > 0:
                for c in chans:
                    mk = masks[gkey(l, c.name, m)]
                    for k in range(core.NCOL):
                        Y, X, valid = (np.asarray(a) for a in lay.sites(l, k))
                        for s_ in np.flatnonzero(mk[k]):
                            st[c.name][:, Y[s_], X[s_]] = -1
            for k in range(core.NCOL):
                Y, X, valid = (np.asarray(a) for a in lay.sites(l, k))
                for c in chans:
                    draw = valid if m == 0 else masks[gkey(l, c.name, m)][k]
                    sites = np.flatnonzero(draw)
                    if sites.size == 0:
                        continue

                    def amask(state, c=c, Y=Y, X=X):
                        _, nv, look = lay.cond(state, c.name, jnp.asarray(Y), jnp.asarray(X))
                        v = nv + look
                        return v == jnp.min(v, axis=1, keepdims=True)
                    A = np.asarray(jax.vmap(amask)({n: jnp.asarray(st[n]) for n in names}))   # (N, n, D)
                    for s_ in sites:
                        rows, vals = np.nonzero(A[:, s_])
                        st = {n: a[rows] for n, a in st.items()}
                        A = A[rows]
                        st[c.name][:, Y[s_], X[s_]] = vals
            for c in chans:
                st["rec:" + gkey(l, c.name, m)] = st[c.name].copy()
    forced = {k[4:]: v for k, v in st.items() if k.startswith("rec:")}
    return forced, {c.name: st[c.name] for c in p.sampled()}, masks


class ExactPath:
    """Enumerated trajectories, their log Q(traj) and costs, exact J and grad."""

    def __init__(self, var, tau, lam, mask_seed, forced, final):
        self.var = var
        lay = var.layout(Region(var.problem.W_train))
        self.lay, self.sch = lay, var.sched(lay)
        rho, share, guide = float(var.cfg["rho"]), bool(var.cfg["share"]), bool(var.cfg.get("guide", False))
        self.tau, self.lam = float(tau), jnp.asarray(lam, jnp.float32)
        self.forced, self.final = forced, final
        self.N = len(next(iter(forced.values())))
        sch = self.sch
        logD = {k: float(np.log(var.problem.chan[g[2]].D)) for k, g in sch.groups.items() if g[3] > 0}

        def one(params, fz, tau):
            _, recs = revise_pass(sch, params, 0, {}, tau, self.lam, rho, share, forced=fz, mask_seed=mask_seed, guide=guide)
            lq = sum(jnp.sum(jnp.where(r["valid"], jnp.take_along_axis(r["logq"], r["a"][..., None], -1)[..., 0], 0.0))
                     for r in recs.values())
            C = sum(jnp.sum(r["cost"]) for r in recs.values())
            ld = sum(logD[k] * jnp.sum(r["valid"]) for k, r in recs.items() if k in logD)
            return lq, C, ld

        self._f = jax.jit(jax.vmap(one, in_axes=(None, 0, None)))

    def score(self, params, tau=None, chunk=4096):
        tau = self.tau if tau is None else tau
        lq, C, nm = [], [], []
        for s in range(0, self.N, chunk):
            fz = {k: jnp.asarray(a[s:s + chunk]) for k, a in self.forced.items()}
            a, b, c = self._f(params, fz, tau)
            lq.append(np.asarray(a, np.float64)); C.append(np.asarray(b, np.float64)); nm.append(np.asarray(c))
        return np.concatenate(lq), np.concatenate(C), np.concatenate(nm)

    def J_and_grad(self, params, tau=None):
        tau = self.tau if tau is None else tau
        lq, C, nm = self.score(params, tau)
        Q = np.exp(lq)
        J = float(np.sum(Q * C))
        X = {k: jnp.asarray(a) for k, a in self.forced.items()}
        g_acc = None
        for s in range(0, self.N, 4096):
            fz = {k: v[s:s + 4096] for k, v in X.items()}
            _, vjp = jax.vjp(lambda pr: self._f(pr, fz, tau)[0], params)
            g, = vjp(jnp.asarray(Q[s:s + 4096] * C[s:s + 4096], jnp.float32))
            g_acc = g if g_acc is None else jax.tree_util.tree_map(jnp.add, g_acc, g)
        return J, g_acc, {"sum_q": float(Q.sum()), "lq": lq, "C": C, "nmask": nm}

    def reverse_kl_parts(self, params, tau=None):
        """(path J, exact reverse-KL integrand of x_0, gap) at tau; both without log Z.
        E(x_0) = S + lam . H from the reference energy of each distinct final state."""
        tau = self.tau if tau is None else tau
        lq, C, nm = self.score(params, tau)
        Q = np.exp(lq)
        p, region = self.var.problem, Region(self.var.problem.W_train)
        keys = np.concatenate([self.final[c.name].reshape(self.N, -1) for c in p.sampled()], 1)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        qx = np.bincount(inv, weights=Q, minlength=len(uniq))
        E = np.zeros(len(uniq))
        lam = np.asarray(self.lam, np.float64)
        for i in range(len(uniq)):
            j = int(np.flatnonzero(inv == i)[0])
            x = {c.name: self.final[c.name][j] for c in p.sampled()}
            e = en.energy(p, x, region)
            E[i] = e.soft + lam @ np.array([e.viol[l] for l in range(p.T + 1)])
        J = float(np.sum(Q * C))
        rkl = float(np.sum(qx * (E + tau * np.log(np.maximum(qx, 1e-300)))))
        # cost identity per trajectory: C = E(x_0) + tau (log Q + sum_masked log D)
        ident = C - (E[inv] + tau * lq + tau * nm)
        return {"J_path": J, "rkl_x0": rkl, "gap": J - rkl, "masked_logD": float(np.sum(Q * nm)), "n_traj": int(self.N), "n_final": int(len(uniq)),
                "sum_q": float(Q.sum()), "ident_max": float(np.abs(ident).max()),
                "E_mean": float(np.sum(Q * E[inv])), "H_logq_traj": float(-np.sum(Q * lq)),
                "H_x0": float(-np.sum(qx * np.log(np.maximum(qx, 1e-300))))}
