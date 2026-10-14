"""cstart training functions (ideas #14): cfast's fused update (K = 0) on
worlds with a per-world start depth L.

Each world b carries L_b in 1..T+1 and a context state ctx (every sampled
channel at levels >= 1).  In the pass, every draw at a level l >= L_b is
replaced by the context's value (the same replacement as
_creplay_clamp.clamped_pass); draws at levels < L_b are sampled as usual from
the same keyed noise.  L_b = T + 1 is a full-depth world (cfast's world).

In the loss, clamped draws are removed: they are marked invalid (so neither
the actor's t1 + t2 nor the critic's regression sees them) and their costs are
zeroed before the cone returns (they precede every unclamped draw of the same
world, coarse-to-fine, so they are in no unclamped draw's cone anyway).  The
recorded action at a clamped site is the model's own argmax (finite log q);
the state carries the context value.  Draws below L get their normal cone
return and critic.

With every L_b = T + 1 the update is cfast's: the replacement and masks are
`where`s on all-false conditions (tested bit for bit)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from ._cpass_core import NCOL, key_of, mlp


def train_pass_ctx(lay, params, seed, designed, tau, lam, ctx, L):
    """One world.  ctx: {chan name: (h, w) int32} for sampled channels at
    levels >= 1 (values used only where level >= L); L: int32 scalar.
    Returns (state, recs) like _cfast_core.train_pass, recs[key] also has
    'clamped' (16, n) bool; for clamped draws valid is False and cost 0."""
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs = {}
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
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = lay.features(state, l, Y, X, lay.actor_win)
                b = mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                if c.name in ctx:
                    on = l >= L
                    cval = ctx[c.name][jnp.minimum(Y, h - 1), jnp.mod(X, w)]
                    a_st = jnp.where(on, cval, a)
                else:
                    on = jnp.bool_(False)
                    a_st = a
                clamped = jnp.broadcast_to(on, valid.shape)
                idx = jnp.arange(Y.shape[0])
                la = logq[idx, a]
                cost = jnp.where(valid, soft[idx, a] + lam[l] * nv[idx, a] + tau * la, 0.0)
                cost = jnp.where(clamped, 0.0, cost)
                outs.append({"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a,
                             "valid": valid & ~clamped, "logq": logq, "cost": cost,
                             "clamped": clamped})
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a_st, mode="drop")}
            return state, outs

        state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            recs[key_of(l, c.name)] = ys[ci]
    return state, recs


def build_train(var, lay):
    """cfast's functions plus 'fused_ctx' (the clamped-world update)."""
    from .cfast import CFast
    cfg = var.cfg
    assert cfg["K"] == 0 and cfg["cone"], "cstart: cone returns, K = 0 only"
    base = CFast._build_train(var, lay)
    T = var.problem.T
    levels = list(range(T + 1))
    level_of = {key_of(l, c.name): l for l in levels for c in lay.chans[l]}
    cone = base["cone"]
    idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
    step = base["step"]
    sub = cfg["sub"]

    def batch(params, seeds, des, tau, lam, ctx, Ls, rkey):
        st, recs = jax.vmap(lambda s, d, c, L: train_pass_ctx(lay, params, s, d, tau, lam, c, L))(
            seeds, des, ctx, Ls)
        viol = jnp.zeros(T + 1)
        for key, r in recs.items():
            nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
            viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
        # per level: worlds that sampled it (all B when nothing is clamped)
        nw = jnp.sum(jnp.arange(T + 1)[None, :] < Ls[:, None], 0).astype(jnp.float32)
        viol = jnp.where(nw == seeds.shape[0], viol / seeds.shape[0], viol / jnp.maximum(nw, 1.0))
        tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
        costs = {k: r["cost"] for k, r in recs.items()}
        ii = cone.integral(costs)
        Bn = seeds.shape[0]
        G, wgt, rs = {}, {}, {}
        for key, r in recs.items():
            n = r["a"].shape[-1]
            if not sub or n <= sub:
                sel = jnp.broadcast_to(jnp.arange(n), (Bn, NCOL, n))
            else:
                rkey, k1 = jax.random.split(rkey)
                sel = jnp.argsort(jax.random.uniform(k1, (Bn, NCOL, n)), -1)[..., :sub]
                wgt[key] = n / sub
            rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                       for f, x in r.items() if f != "clamped"}
            ix = idx[key][jnp.arange(NCOL)[None, :, None], sel]
            v = jax.vmap(lambda a, i: a[i])(ii, ix)
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        return st, rs, G, wgt, viol, tot

    def fused_ctx(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey, ctx, Ls):
        st, recs, G, wgt, viol, tot = batch(params, seeds, des, tau, lam, ctx, Ls, rkey)
        params, cparams, opt, copt, loss, closs = step(params, cparams, opt, copt, recs, G, tau, lam,
                                                       {}, gs, lr, lrc, clip, wgt)
        gstd = {k: jnp.std(v) for k, v in G.items()}
        return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd

    return {**base, "fused_ctx": jax.jit(fused_ctx), "batch": jax.jit(batch)}

