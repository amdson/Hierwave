"""crecur core (constitution amendment 2026-10-16 (b)): K-sample recursive cost for the coarse levels.

split_pass  = _cfast_core.train_pass with the Gumbel noise of levels >= l0 keyed by a coarse seed and of
              levels < l0 by a fine seed.  The coarse levels are drawn first and read only coarse noise and
              the designed channels, so worlds with the same coarse seed carry the SAME coarse state (the
              clamped coarse state) and differ only in their fine completions: K fine completions of one
              coarse draw are K split passes with one coarse seed and K fine seeds.

Recursive cost.  For a world with coarse state c and fine completions t_1..t_K (drawn from the sampler's
own fine levels given c), log w_k = -sum of the fine draws' costs (soft + lam_l nv + log q, tau = 1)
= -(E_fine(c, t_k) + log q_fine(t_k | c)), and
        R_K(c) = -log (1/K) sum_k w_k        (-log Z_K; E[R_1] = F(c) + KL(q_fine || p*_fine | c)).
Per coarse draw i the return is the coarse levels' cone return (cfast's cone, coarse costs only) plus
R_K restricted to the fine draws in i's cone ("cone-local": the fine costs outside cone(i) do not depend
on a_i in any replica; see report for why the whole-world version is degenerate on ground).  The coarse
actor's gradient: cfast's all-action term on the draw's own cost (soft + lam nv + tau log q, sg log q)
plus log q(a_i) * sg(G_i - b_i), b_i = leave-one-out mean of G over the other worlds of the batch at the
same (channel, colour, site) (independent worlds: a zero-mean control variate).  Only levels >= l0 get
this gradient.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from ._cfast_core import _cols, _encode, _geom, fast_features
from ._cpass_core import NCOL, key_of, mlp
from .cpass import adam_step, clip_by_level


def split_pass(lay, params, cseed, fseed, designed, tau, lam, l0, record=True, clamp=None, cflag=None, Lc=None):
    """train_pass with noise seed cseed at levels >= l0 and fseed below.  Same returns as train_pass.
    clamp/cflag/Lc (ideas #14 contexts): when cflag, the draws of levels >= Lc take the values clamp[name]
    (e.g. a probe lift's coarsest level) instead of the drawn ones; their records are kept (masked in the loss)."""
    assert not lay.resid
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs = {}
    Ep = None
    for l in lay.levels:
        chans = lay.chans[l]
        Es = _encode(lay, l, state)
        if not chans:
            Ep = Es
            continue
        g = {k: jnp.asarray(v) for k, v in _geom(lay, l).items()}
        h, w = lay.hw[l]
        seed = cseed if l >= l0 else fseed
        gum = [noise.gumbel(jnp, seed, noise.tag_of(c.name), l, g["Y"], g["Xn"], c.D) for c in chans]
        xs = {**g, "k": jnp.arange(NCOL), "gum": gum}
        Ep_ = Ep

        def body(carry, x, l=l, chans=chans, Ep_=Ep_, h=h, w=w):
            state, Es = carry
            Y, X, valid = x["Y"], x["X"], x["valid"]
            outs = []
            for ci, c in enumerate(chans):
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = fast_features(lay, l, Es, Ep_, x)
                b = mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                a = jnp.argmax(logits + x["gum"][ci], axis=1).astype(jnp.int32)
                if clamp is not None and l >= Lc:
                    a = jnp.where(cflag, clamp[c.name][jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)], a)
                idx = jnp.arange(Y.shape[0])
                la = logq[idx, a]
                cost = jnp.where(valid, soft[idx, a] + lam[l] * nv[idx, a] + tau * la, 0.0)
                if record:
                    outs.append({"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a,
                                 "valid": valid, "logq": logq, "cost": cost})
                else:
                    outs.append(cost)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
                rows = jnp.concatenate([lay.emb[c.name][a], jnp.ones((a.shape[0], 1), jnp.float32)], -1)
                o, kk = _cols(lay, l)[c.name]
                Es = Es.at[x["sidx"][:, None], jnp.arange(o, o + kk + 1)[None, :]].set(rows, mode="drop")
            return (state, Es), outs

        (state, Es), ys = jax.lax.scan(body, (state, Es), xs)
        Ep = Es
        for ci, c in enumerate(chans):
            recs[key_of(l, c.name)] = ys[ci]
    return state, recs


def foot_sums(lay, fine_costs, L):
    """{fine key: (N, 16, n) per-draw costs} -> (N, 16, n_L): for every level-L draw site, the summed cost of the
    fine draws (levels < l0) inside its footprint (its subtree)."""
    N = next(iter(fine_costs.values())).shape[0]
    hL, wL = lay.hw[L]
    tot = jnp.zeros((N, hL, wL))
    for key, c in fine_costs.items():
        l = int(key[1:key.index("_")])
        h, w = lay.hw[l]
        g = _geom(lay, l)
        img = jnp.zeros((N, h * w + 1)).at[:, jnp.asarray(g["sidx"]).reshape(-1)].add(c.reshape(N, -1))
        f = 1 << (L - l)
        tot = tot + img[:, :h * w].reshape(N, hL, f, wL, f).sum((2, 4))
    gL = _geom(lay, L)
    flat = jnp.concatenate([tot.reshape(N, -1), jnp.zeros((N, 1))], 1)
    return flat[:, jnp.asarray(gL["sidx"])]                                          # (N, 16, n_L)


def build(lay, cone, l0, K, tau=1.0, mode="foot", Lc=None):
    """Jitted recursive step and pieces.  mode:
      "foot"  (default): G_i = coarse cone return + fine cone return of replica 0 - fine footprint cost of replica 0
              + R_K(footprint): the K-sample subtree term replaces only the fine draws in i's footprint (subtree);
              K = 1 is exactly cfast's cone return (unbiased joint reverse-KL gradient).  On ground its return std is
              ~100-200 nats (the fine cone), so the K-sample change (a few nats) is lost in the noise.
      "footonly": G_i = coarse cone return + R_K(footprint): the per-coarse-cell recursive cost; fine draws outside
              the footprint that read a_i are dropped (boundary truncation, a bias even at K = 1).
      "cone": R_K over the fine draws in i's whole dependency cone.
      "world": R_K of the whole world's fine cost, the same for every coarse draw of a world."""
    T = lay.T
    levels = list(range(T + 1))
    ckeys = [key_of(l, c.name) for l in levels if l >= l0 for c in lay.chans[l]]
    fkeys = [key_of(l, c.name) for l in levels if l < l0 for c in lay.chans[l]]
    level_of = {key_of(l, c.name): l for l in levels for c in lay.chans[l]}
    clevels = sorted({level_of[k] for k in ckeys})
    nval = {}
    for l in levels:
        cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(NCOL))
        nval[l] = cnt * len(lay.chans[l])
    cidx = {k: jnp.asarray(cone.idx[k]) for k in ckeys}
    tau = float(tau)

    def coarse_returns(costs):
        ii = cone.integral(costs)
        G = {}
        for key, ix in cidx.items():
            v = ii[:, ix]
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        return G

    def rollout(params, cseeds, fseeds, des, lam, clamp=None, cflag=None):
        """cseeds (B,), fseeds (B, K), des {name: (B, ...)}.  Returns recs (replica 0, all levels, recorded),
        G (coarse keys: coarse cone return + R_K), diagnostics."""
        Bn = cseeds.shape[0]
        if clamp is None:
            clamp = {}
            cflag = jnp.zeros(Bn, bool)
        cl = clamp if clamp else None
        st, recs = jax.vmap(lambda s, f, d, c, fl: split_pass(lay, params, s, f, d, tau, lam, l0, clamp=cl and c, cflag=fl,
                                                              Lc=Lc))(cseeds, fseeds[:, 0], des, clamp, cflag)
        if cl:          # clamped draws: no loss, no return (masked like cstart's clamped draws)
            recs = {k: ({**r, "valid": r["valid"] & ~cflag[:, None, None]} if level_of[k] >= Lc else r)
                    for k, r in recs.items()}
        costs0 = {k: r["cost"] for k, r in recs.items()}
        zero = lambda k, c: jnp.zeros_like(c)
        Gc = coarse_returns({k: (c if level_of[k] >= l0 else zero(k, c)) for k, c in costs0.items()})
        fine0 = {k: (c if level_of[k] < l0 else zero(k, c)) for k, c in costs0.items()}
        if K > 1:
            rep = lambda x: jnp.repeat(x, K - 1, axis=0)
            cs = rep(cseeds)
            fs = fseeds[:, 1:].reshape(-1)
            ds = {k: rep(v) for k, v in des.items()}
            _, costs = jax.vmap(lambda s, f, d, c, fl: split_pass(lay, params, s, f, d, tau, lam, l0, record=False,
                                                                   clamp=cl and c, cflag=fl, Lc=Lc))(
                cs, fs, ds, {k: rep(v) for k, v in clamp.items()}, rep(cflag))
            fineK = {k: jnp.concatenate([fine0[k][:, None], (c if level_of[k] < l0 else jnp.zeros_like(c))
                                         .reshape((Bn, K - 1) + c.shape[1:])], 1).reshape((Bn * K,) + c.shape[1:])
                     for k, c in costs.items()}
        else:
            fineK = fine0
        world_f = sum(jnp.sum(c, (1, 2)) for c in fineK.values()).reshape(Bn, K)          # (B, K)
        diag = {"world_logw_std": jnp.mean(jnp.std(-world_f, 1)),
                "world_ess": jnp.mean(_ess(-world_f)),
                "world_R": jnp.mean(-jax.nn.logsumexp(-world_f, 1) + jnp.log(K)),
                "world_fine_mean": jnp.mean(world_f)}
        G = {}
        if mode in ("foot", "footonly"):
            Gf = coarse_returns(fineK)
            fkeys_ = [k for k in fineK if level_of[k] < l0]
            for key in ckeys:
                L = level_of[key]
                ft = foot_sums(lay, {k: fineK[k] for k in fkeys_}, L)
                ft = ft.reshape((Bn, K) + ft.shape[1:])
                g0 = Gf[key].reshape((Bn, K) + Gf[key].shape[1:])[:, 0]
                Rf = -jax.nn.logsumexp(-ft, 1) + jnp.log(K)
                G[key] = Gc[key] + (g0 - ft[:, 0] if mode == "foot" else 0.0) + Rf
                vm = recs[key]["valid"]
                nv_ = jnp.maximum(jnp.sum(vm), 1)
                diag[f"ess_{key}"] = jnp.sum(jnp.where(vm, _ess(-ft, 1), 0.0)) / nv_
                diag[f"footstd_{key}"] = jnp.sum(jnp.where(vm, jnp.std(ft, 1), 0.0)) / nv_
                diag[f"foot1_{key}"] = jnp.sum(jnp.where(vm, ft[:, 0], 0.0)) / nv_
                diag[f"Rfoot_{key}"] = jnp.sum(jnp.where(vm, Rf, 0.0)) / nv_
        elif mode == "cone":
            Gf = coarse_returns(fineK)
            for key in ckeys:
                g = Gf[key].reshape((Bn, K) + Gf[key].shape[1:])                         # (B, K, 16, n)
                R = -jax.nn.logsumexp(-g, 1) + jnp.log(K)
                G[key] = Gc[key] + R
                lw = -g
                diag[f"ess_{key}"] = jnp.sum(jnp.where(recs[key]["valid"], _ess(lw, 1), 0.0)) / jnp.maximum(
                    jnp.sum(recs[key]["valid"]), 1)
                diag[f"R_{key}"] = jnp.sum(jnp.where(recs[key]["valid"], R, 0.0)) / jnp.maximum(jnp.sum(recs[key]["valid"]), 1)
                diag[f"R1_{key}"] = jnp.sum(jnp.where(recs[key]["valid"], g[:, 0], 0.0)) / jnp.maximum(jnp.sum(recs[key]["valid"]), 1)
        else:
            Rw = -jax.nn.logsumexp(-world_f, 1) + jnp.log(K)                                  # (B,)
            for key in ckeys:
                G[key] = Gc[key] + Rw[:, None, None]
        viol = jnp.zeros(T + 1)
        for key, r in recs.items():
            nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
            viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
        return st, recs, G, viol / Bn, diag

    def loss(pc, params, recs, G, lam, baseline=True):
        Bn = recs[ckeys[0]]["a"].shape[0]
        total = 0.0
        for l in clevels:
            lev = 0.0
            for c in lay.chans[l]:
                key = key_of(l, c.name)
                r = recs[key]
                b = mlp(pc[key], r["feat"])
                logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, -1)
                q = jnp.exp(logq)
                lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                t1 = jnp.sum(jnp.where(r["mask"], q * ell, 0.0), -1)
                la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                g = G[key]
                if baseline and Bn > 1:
                    bl = (jnp.sum(g, 0, keepdims=True) - g) / (Bn - 1)
                else:
                    bl = 0.0
                t2 = la * jax.lax.stop_gradient(g - bl)
                lev = lev + jnp.sum(jnp.where(r["valid"], t1 + t2, 0.0))
            total = total + lev / nval[l]
        return total / Bn

    def step(params, opt, cseeds, fseeds, des, lam, lr, clip, clamp=None, cflag=None):
        st, recs, G, viol, diag = rollout(params, cseeds, fseeds, des, lam, clamp, cflag)
        pc = {k: params[k] for k in ckeys}
        lval, g = jax.value_and_grad(loss)(pc, params, recs, G, lam)
        g = clip_by_level(g, clevels, clip)
        pc, opt = adam_step(pc, g, opt, lr)
        params = {**params, **pc}
        gstd = {k: jnp.std(G[k]) for k in ckeys}
        return params, opt, lval, viol, diag, gstd

    grad_fn = jax.jit(lambda params, cseeds, fseeds, des, lam: _grad_only(rollout, loss, ckeys, params, cseeds,
                                                                             fseeds, des, lam))
    return {"step": jax.jit(step), "rollout": jax.jit(rollout), "grad": grad_fn, "ckeys": ckeys, "fkeys": fkeys,
            "nval": nval, "clevels": clevels}


def _grad_only(rollout, loss, ckeys, params, cseeds, fseeds, des, lam):
    st, recs, G, viol, diag = rollout(params, cseeds, fseeds, des, lam)
    pc = {k: params[k] for k in ckeys}
    return jax.grad(loss)(pc, params, recs, G, lam), G, st


def _ess(lw, axis=-1):
    """Normalised effective sample size (1/K .. 1) of log weights along axis."""
    K = lw.shape[axis]
    a = jax.nn.logsumexp(lw, axis)
    b = jax.nn.logsumexp(2 * lw, axis)
    return jnp.exp(2 * a - b) / K
