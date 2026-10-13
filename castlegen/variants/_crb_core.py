"""crb training functions: cfast's fused update (K = 0) with Rao-Blackwellised
downstream returns.

cfast's return for draw i is G_i = sum_{j in D_i} l_j(a_j), the realised
costs l_j(a) = s_j(a) + lam n_j(a) + tau log q_j(a) of the later draws in its
dependency cone D_i.  crb replaces each l_j(a_j) by its expectation under
draw j's own conditional,

    lbar_j = sum_{a in A_j} q_j(a | h_j) l_j(a),

which the pass already has (soft, nv, mask and logq are recorded for every
draw).  States are unchanged (draws are still sampled); only the return
changes.  Unbiased by the tower property: for j > i (a later step),
grad log q_i(a_i) is a function of h_j (everything fixed before step j), and
a_j ~ q_j(. | h_j) with fresh noise, so
    E[grad log q_i(a_i) l_j(a_j)] = E[grad log q_i(a_i) E[l_j(a_j) | h_j]]
                                  = E[grad log q_i(a_i) lbar_j].
D_i never contains i itself or draws of i's own step, so the conditioning is
always strictly on a later step.  The direct all-action term of the actor
loss (sum_a q_i(a) (l_i(a) + Q_i(a))) is untouched, and returns carry no
gradient.

rb=False gives cfast's estimator exactly (same code path), for paired
variance comparisons."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import _cpass_core as core
from ._cfast_cone import Cone
from ._cfast_core import train_pass
from .cpass import adam_step, clip_by_level


def expected_cost(r, tau, lam_l):
    """lbar (B, 16, n): sum_a q(a) [soft(a) + lam nv(a) + tau log q(a)] over
    the draw's least-violating set, 0 at invalid sites."""
    m = r["mask"]
    lq = jnp.where(m, r["logq"], 0.0)
    ell = jnp.where(m, r["soft"] + lam_l * r["nv"] + tau * lq, 0.0)
    q = jnp.where(m, jnp.exp(lq), 0.0)
    return jnp.where(r["valid"], jnp.sum(q * ell, -1), 0.0)


def uniform_cost(r, tau, lam_l):
    """NEGATIVE CONTROL ONLY (biased): the expectation of l(a) under the uniform
    distribution over A instead of q."""
    m = r["mask"]
    lq = jnp.where(m, r["logq"], 0.0)
    ell = jnp.where(m, r["soft"] + lam_l * r["nv"] + tau * lq, 0.0)
    return jnp.where(r["valid"], jnp.sum(ell, -1) / jnp.maximum(jnp.sum(m, -1), 1), 0.0)


def build_train(var, lay, rb=True, neg=None):
    """neg="uniform": a deliberately biased estimator (tests only)."""
    cfg = var.cfg
    assert cfg["K"] == 0, "crb: coupled replays (K > 0) not supported"
    assert cfg["cone"], "crb: cone returns only"
    T = var.problem.T
    levels = list(range(T + 1))
    keys = [core.key_of(l, c.name) for l in lay.levels for c in lay.chans[l]]
    level_of = {core.key_of(l, c.name): l for l in levels for c in lay.chans[l]}
    nval = {}
    for l in levels:
        cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(core.NCOL))
        nval[l] = cnt * len(lay.chans[l])
    cone = Cone(lay, core.ACTOR_WIN)
    idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
    sfrac = {}
    for l in levels:
        for ci, c in enumerate(lay.chans[l]):
            sfrac[core.key_of(l, c.name)] = jnp.asarray(
                [lay.step(l, k, ci) / lay.nsteps for k in range(core.NCOL)], jnp.float32)
    critic_h1 = cfg["critic"] == "h1"
    sub = cfg["sub"]

    def cfeat(r, key, tau, lam, params):
        f = r["feat"]
        if critic_h1:
            p = params[key]
            f = jax.lax.stop_gradient(jnp.tanh(f @ p["W1"] + p["b1"]))
        ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
        ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
        sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
        return jnp.concatenate([f, ex, sf], -1)

    def actor_loss(params, cparams, recs, G, tau, lam, gs, p0, wgt):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = core.key_of(l, c.name)
                r = recs[key]
                b = core.mlp(params[key], r["feat"])
                logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, -1)
                q = jnp.exp(logq)
                lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                Q = jax.lax.stop_gradient(core.mlp(cparams[key], cfeat(r, key, tau, lam, p0))) * gs[key]
                t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                t2 = la * jax.lax.stop_gradient(G[key] - Qa)
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], t1 + t2, 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def critic_loss(cparams, recs, G, tau, lam, gs, p0, wgt):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = core.key_of(l, c.name)
                r = recs[key]
                Q = core.mlp(cparams[key], cfeat(r, key, tau, lam, p0))
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def batch(params, seeds, des, tau, lam, rkey):
        """Pass, (RB) cone returns, subsampled records."""
        st, recs = jax.vmap(lambda s, d: train_pass(lay, params, s, d, tau, lam))(seeds, des)
        viol = jnp.zeros(T + 1)
        for key, r in recs.items():
            nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
            viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
        viol = viol / seeds.shape[0]
        tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())      # realised (S + lam H + tau log q)
        if neg == "uniform":
            costs = {k: uniform_cost(r, tau, lam[level_of[k]]) for k, r in recs.items()}
        elif rb:
            costs = {k: expected_cost(r, tau, lam[level_of[k]]) for k, r in recs.items()}
        else:
            costs = {k: r["cost"] for k, r in recs.items()}
        costs = jax.lax.stop_gradient(costs)
        ii = cone.integral(costs)                                            # (B, size)
        Bn = seeds.shape[0]
        G, wgt, rs = {}, {}, {}
        for key, r in recs.items():
            n = r["a"].shape[-1]
            if not sub or n <= sub:
                sel = jnp.broadcast_to(jnp.arange(n), (Bn, core.NCOL, n))
            else:
                rkey, k1 = jax.random.split(rkey)
                sel = jnp.argsort(jax.random.uniform(k1, (Bn, core.NCOL, n)), -1)[..., :sub]
                wgt[key] = n / sub
            rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                       for f, x in r.items()}
            ix = idx[key][jnp.arange(core.NCOL)[None, :, None], sel]           # (B, 16, m, S, 4)
            v = jax.vmap(lambda a, i: a[i])(ii, ix)
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        return st, rs, G, wgt, viol, tot

    def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey):
        """cfast's fused update signature (lr = 0 gives the baseline draws)."""
        st, recs, G, wgt, viol, tot = batch(params, seeds, des, tau, lam, rkey)
        p0 = params
        loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, p0, wgt)
        g = clip_by_level(g, levels, clip)
        params, opt = adam_step(params, g, opt, lr)
        closs, gc = jax.value_and_grad(critic_loss)(cparams, recs, G, tau, lam, gs, p0, wgt)
        gc = clip_by_level(gc, levels, clip)
        cparams, copt = adam_step(cparams, gc, copt, lrc)
        gstd = {k: jnp.std(v) for k, v in G.items()}
        return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd

    def agrad(params, cparams, gs, seeds, des, tau, lam, rkey):
        """Actor gradient of one batch (no update, no clipping): tests, variance."""
        st, recs, G, wgt, viol, tot = batch(params, seeds, des, tau, lam, rkey)
        g = jax.grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, params, wgt)
        return g, tot

    def agrad_pair(params, cparams, gs, seeds, des, tau, lam, rkey):
        """Per-world actor gradients of cfast (realised) and crb (expected)
        returns on the same worlds and subsample: (g_cfast, g_crb), each with a
        leading world axis."""
        st, recs = jax.vmap(lambda s, d: train_pass(lay, params, s, d, tau, lam))(seeds, des)
        outs = []
        for use_rb in (False, True):
            if use_rb:
                costs = {k: expected_cost(r, tau, lam[level_of[k]]) for k, r in recs.items()}
            else:
                costs = {k: r["cost"] for k, r in recs.items()}
            ii = cone.integral(costs)
            outs.append(ii)
        Bn = seeds.shape[0]
        Gs = ({}, {})
        wgt, rs = {}, {}
        for key, r in recs.items():
            n = r["a"].shape[-1]
            if not sub or n <= sub:
                sel = jnp.broadcast_to(jnp.arange(n), (Bn, core.NCOL, n))
            else:
                rkey, k1 = jax.random.split(rkey)
                sel = jnp.argsort(jax.random.uniform(k1, (Bn, core.NCOL, n)), -1)[..., :sub]
                wgt[key] = n / sub
            rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                       for f, x in r.items()}
            ix = idx[key][jnp.arange(core.NCOL)[None, :, None], sel]
            for o, ii in enumerate(outs):
                v = jax.vmap(lambda a, i: a[i])(ii, ix)
                Gs[o][key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)

        def one(rw, G):                         # per-world gradient (world axis 1 for actor_loss)
            r1 = jax.tree_util.tree_map(lambda x: x[None], rw)
            G1 = jax.tree_util.tree_map(lambda x: x[None], G)
            return jax.grad(actor_loss)(params, cparams, r1, G1, tau, lam, gs, params, wgt)

        return tuple(jax.vmap(one)(rs, G) for G in Gs)

    return {"fused": jax.jit(fused), "agrad": jax.jit(agrad), "agrad_pair": jax.jit(agrad_pair),
            "batch": jax.jit(batch),
            "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s))),
            "nval": nval, "cone": cone, "level_of": level_of}
