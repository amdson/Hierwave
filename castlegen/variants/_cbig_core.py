"""cbig's training pass and fused update: cfast's (same draws, same cone
returns, same subsampled loss), with the subsample chosen BEFORE the pass so
the pass records features and logits only for the m chosen sites per (step,
world).  Per-site costs and the chosen action's violation count are still
recorded at every site (cone returns, penalty updates).  The choice is
uniform and independent of the draws, weighted n / m, so the loss is the same
unbiased estimate as cfast's; only memory changes (cfast records every site,
about 27 MB per ground world, which runs out of memory at B = 1024 on 24 GB).

Also builds `grad`: the actor gradient of one batch (for SNR measurements),
with the n / m weights."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from . import _cpass_core as core
from ._cfast_cone import Cone
from ._cpass_core import ACTOR_WIN, NCOL, key_of, mlp
from .cpass import adam_step, clip_by_level


def train_pass_sel(lay, params, seed, designed, tau, lam, sel):
    """One world.  sel[key]: (16, m) site indices to record.  Returns (state,
    recs, costs, nva): recs[key] fields (16, m, ...), costs / nva (16, n)."""
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs, costs, nvas = {}, {}, {}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue

        def body(state, k, l=l, chans=chans):
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
            outs = []
            for ci, c in enumerate(chans):
                key = key_of(l, c.name)
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = lay.features(state, l, Y, X, ACTOR_WIN)
                b = mlp(params[key], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                idx = jnp.arange(Y.shape[0])
                la = logq[idx, a]
                cost = jnp.where(valid, soft[idx, a] + lam[l] * nv[idx, a] + tau * la, 0.0)
                nva = jnp.where(valid, nv[idx, a], 0.0)
                s = sel[key][k]
                rec = {"feat": feat[s], "soft": soft[s], "nv": nv[s], "mask": mask[s], "a": a[s],
                       "valid": valid[s]}
                outs.append((rec, cost, nva))
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, outs

        state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            key = key_of(l, c.name)
            recs[key], costs[key], nvas[key] = ys[ci]
    return state, recs, costs, nvas


def build_train(var, lay):
    """The fused update and helpers for a CBig instance on layout lay."""
    cfg = var.cfg
    T = var.problem.T
    levels = list(range(T + 1))
    level_of = {key_of(l, c.name): l for l in levels for c in lay.chans[l]}
    keys = list(level_of)
    nval = {}
    for l in levels:
        cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(NCOL))
        nval[l] = cnt * len(lay.chans[l])
    cone = Cone(lay, ACTOR_WIN)
    idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
    sfrac = {}
    for l in levels:
        for ci, c in enumerate(lay.chans[l]):
            sfrac[key_of(l, c.name)] = jnp.asarray(
                [lay.step(l, k, ci) / lay.nsteps for k in range(NCOL)], jnp.float32)
    sub = cfg["sub"]
    nsite = {k: lay.nsite(level_of[k]) for k in keys}
    wgt = {k: (nsite[k] / sub if nsite[k] > sub else 1.0) for k in keys}

    def choose(rkey, Bn):
        sel = {}
        for key in keys:
            n = nsite[key]
            if n <= sub:
                sel[key] = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (Bn, NCOL, n))
            else:
                rkey, k1 = jax.random.split(rkey)
                sel[key] = jnp.argsort(jax.random.uniform(k1, (Bn, NCOL, n)), -1)[..., :sub].astype(jnp.int32)
        return sel

    def cfeat(r, key, tau, lam, params):
        p = params[key]
        f = jax.lax.stop_gradient(jnp.tanh(r["feat"] @ p["W1"] + p["b1"]))
        ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
        ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
        sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
        return jnp.concatenate([f, ex, sf], -1)

    def actor_loss(params, cparams, recs, G, tau, lam, gs, p0):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = key_of(l, c.name)
                r = recs[key]
                b = mlp(params[key], r["feat"])
                logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, -1)
                q = jnp.exp(logq)
                lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                Q = jax.lax.stop_gradient(mlp(cparams[key], cfeat(r, key, tau, lam, p0))) * gs[key]
                t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                t2 = la * jax.lax.stop_gradient(G[key] - Qa)
                lev = lev + wgt[key] * jnp.sum(jnp.where(r["valid"], t1 + t2, 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def critic_loss(cparams, recs, G, tau, lam, gs, p0):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = key_of(l, c.name)
                r = recs[key]
                Q = mlp(cparams[key], cfeat(r, key, tau, lam, p0))
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                lev = lev + wgt[key] * jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def batch(params, seeds, des, tau, lam, rkey):
        Bn = seeds.shape[0]
        sel = choose(rkey, Bn)
        st, recs, costs, nvas = jax.vmap(lambda s, d, se: train_pass_sel(lay, params, s, d, tau, lam, se))(
            seeds, des, sel)
        viol = jnp.zeros(T + 1)
        for key, nv in nvas.items():
            viol = viol.at[level_of[key]].add(jnp.sum(nv))
        tot = sum(jnp.sum(c, (1, 2)) for c in costs.values())
        ii = cone.integral(costs)                                          # (B, size)
        G = {}
        for key in keys:
            ix = idx[key][jnp.arange(NCOL)[None, :, None], sel[key]]       # (B, 16, m, S, 4)
            v = jax.vmap(lambda a, i: a[i])(ii, ix)
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        return st, recs, G, viol / Bn, tot

    def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey):
        st, recs, G, viol, tot = batch(params, seeds, des, tau, lam, rkey)
        p0 = params
        loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, p0)
        g = clip_by_level(g, levels, clip)
        params, opt = adam_step(params, g, opt, lr)
        closs, gc = jax.value_and_grad(critic_loss)(cparams, recs, G, tau, lam, gs, p0)
        gc = clip_by_level(gc, levels, clip)
        cparams, copt = adam_step(cparams, gc, copt, lrc)
        gstd = {k: jnp.std(v) for k, v in G.items()}
        return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd

    def grad(params, cparams, gs, seeds, des, tau, lam, rkey):
        """Unclipped actor gradient of one batch (per-level nval and 1/B
        scaling as in the loss) and the cone-return std per key."""
        _, recs, G, _, _ = batch(params, seeds, des, tau, lam, rkey)
        return jax.grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, params)

    return {"fused": jax.jit(fused), "grad": jax.jit(grad),
            "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s))),
            "nval": nval, "cone": cone}
