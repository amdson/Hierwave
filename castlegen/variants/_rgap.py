"""rgap (ideas #19): exact bound gap of the path-KL revising sampler with a designed fixed refill r
versus the uniform refill, by trajectory enumeration (copy of _crevise_gradcheck.enumerate_trajectories
that also accumulates, per trajectory, -log r_des(old | x_notM) of every erased cell).

r_des(a | x_notM) = softmax_a(-s(a) - lam_r nv(a)) from lay.cond on the state with the whole step's
erased set unknown (the same closed s, nv the draw uses), full support over the D values.
Path KL with r_des: C' = C - tau*sum_M log D + tau*sum_M (-log r_des(old)); gap' = J' - reverse KL(x_0)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._crevise_core import gkey
from ._crevise_gradcheck import ExactPath, masks_of
from . import _cpass_core as core

LAMS = (5.0, 10.0, 20.0)


def enumerate_with_logr(sch, rho, mask_seed, lam_rs=LAMS):
    """As enumerate_trajectories, plus nlogr {lam_r: (N,)} = sum over erased cells of -log r_des(old)."""
    lay, p = sch.lay, sch.lay.p
    masks = masks_of(sch, rho, mask_seed)
    names = [c.name for c in p.chans]
    st = {c.name: np.full((1,) + lay.hw[c.level], -1, np.int32) for c in p.chans}
    for j, _ in enumerate(lam_rs):
        st[f"_nlr{j}"] = np.zeros(1)
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue
        for m in range(sch.R[l] + 1):
            if m > 0:
                old = {c.name: st[c.name].copy() for c in chans}
                for c in chans:
                    mk = masks[gkey(l, c.name, m)]
                    for k in range(core.NCOL):
                        Y, X, valid = (np.asarray(a) for a in lay.sites(l, k))
                        for s_ in np.flatnonzero(mk[k]):
                            st[c.name][:, Y[s_], X[s_]] = -1
                # designed r for every erased cell, given the state with the whole erased set unknown
                for c in chans:
                    mk = masks[gkey(l, c.name, m)]
                    for k in range(core.NCOL):
                        Y, X, valid = (np.asarray(a) for a in lay.sites(l, k))
                        sites = np.flatnonzero(mk[k])
                        if sites.size == 0:
                            continue
                        def cf(state, c=c, Y=Y, X=X):
                            s, nv, _ = lay.cond(state, c.name, jnp.asarray(Y), jnp.asarray(X))
                            return s, nv
                        S, NV = jax.vmap(cf)({n: jnp.asarray(st[n]) for n in names})   # (N, n, D)
                        S, NV = np.asarray(S, np.float64), np.asarray(NV, np.float64)
                        for j, lr in enumerate(lam_rs):
                            lg = -(S + lr * NV)
                            lg = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
                            for s_ in sites:
                                o = old[c.name][:, Y[s_], X[s_]]
                                st[f"_nlr{j}"] = st[f"_nlr{j}"] - lg[np.arange(len(o)), s_, o]
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
                    A = np.asarray(jax.vmap(amask)({n: jnp.asarray(st[n]) for n in names}))
                    for s_ in sites:
                        rows, vals = np.nonzero(A[:, s_])
                        st = {n: a[rows] for n, a in st.items()}
                        A = A[rows]
                        st[c.name][:, Y[s_], X[s_]] = vals
            for c in chans:
                st["rec:" + gkey(l, c.name, m)] = st[c.name].copy()
    forced = {k[4:]: v for k, v in st.items() if k.startswith("rec:")}
    nlogr = {lr: st[f"_nlr{j}"] for j, lr in enumerate(lam_rs)}
    return forced, {c.name: st[c.name] for c in p.sampled()}, masks, nlogr


class RGap(ExactPath):
    def __init__(self, *a, nlogr, **kw):
        super().__init__(*a, **kw)
        self.nlogr = nlogr

    def parts(self, params, tau=1.0):
        """{r name: dict(gap, J, charge)} plus rkl; charge = E_Q[-tau log r(old)] (uniform: |M| log D)."""
        base = self.reverse_kl_parts(params, tau)
        lq, C, nm = self.score(params, tau)
        Q = np.exp(lq)
        out = {"rkl": base["rkl_x0"], "H_x0": base["H_x0"], "E_M": float(np.sum(Q * self._nM())),
               "uniform": {"J": base["J_path"], "gap": base["gap"], "charge": base["masked_logD"]}}
        for lr, v in self.nlogr.items():
            ch = float(np.sum(Q * tau * v))
            J = base["J_path"] - tau * base["masked_logD"] + ch
            out[f"des{lr:g}"] = {"J": J, "gap": J - base["rkl_x0"], "charge": ch}
        return out

    def _nM(self):
        return sum(m.sum() for m in self.masks.values())

    def grad_with(self, params, Cfun, tau=1.0):
        """Exact grad of J = sum Q C' where C' = C - tau*nm + tau*nlogr[Cfun] (Cfun None = uniform)."""
        lq, C, nm = self.score(params, tau)
        Q = np.exp(lq)
        Cp = C if Cfun is None else C - tau * nm + tau * self.nlogr[Cfun]
        g_acc = None
        for s in range(0, self.N, 4096):
            fz = {k: jnp.asarray(a[s:s + 4096]) for k, a in self.forced.items()}
            _, vjp = jax.vjp(lambda pr: self._f(pr, fz, tau)[0], params)
            g, = vjp(jnp.asarray(Q[s:s + 4096] * Cp[s:s + 4096], jnp.float32))
            g_acc = g if g_acc is None else jax.tree_util.tree_map(jnp.add, g_acc, g)
        return float(np.sum(Q * Cp)), g_acc
