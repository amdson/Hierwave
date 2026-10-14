"""chot training functions: cfast's fused update (K = 0) on worlds drawn with
epsilon-hot exploration, importance-weighted back to the training objective.

Behaviour.  Each draw i is independently "hot" with probability eps_l (its
level's rate; a hash of (seed, "hot:" + channel, level, Y, X), so replays
redraw the same coins).  A hot draw samples from the whole-logit tempered
conditional q_b(a) ∝ q(a)^(1/tau_b) over the least-violating set A (q = the
target policy at the current tau, logits -s/tau - b); a cold draw samples q.
Both use the same per-value Gumbel noise.  The coin is an auxiliary variable
independent of everything else, so marginally draw i follows the mixture
    b_i(a | h_i) = (1 - eps) q_i(a | h_i) + eps q_b,i(a | h_i),
and every draw carries the ratio rho_i = q_i(a_i | h_i) / b_i(a_i | h_i), which is
bounded by 1 / (1 - eps), equals 1 exactly at levels with eps = 0, and has
chi2(q || b) <= eps chi2(q || q_b) (convexity: lower variance than weighting
only the hot draws by q / q_b given the coin).

Weights (unbiased).  For any term f of the actor loss that depends on a set of
draws, let C be the ancestral closure of that set (every draw any of them
reads, transitively).  For any S ⊇ C,  E_b[f prod_{m in S} rho_m] = E_q[f]:
sum out the draws outside C in reverse step order; none of C reads them, so
each sums its own factor b_m rho_m = q_m to 1 (outside S: b_m to 1).  chot
takes S = every draw of every hot level in the world, a superset of every
term's closure, so each world carries one weight W = prod rho over its hot
levels and multiplies all of its actor-loss terms (all-action term,
score-function term, critic terms).  The critic is a control variate, so it
may be trained (unweighted) on behaviour draws.  The cost l_i keeps tau log q_i
(the target's), returns are cfast's cone returns of those costs.

Variance: Var W = prod (1 + chi2_m) - 1 over the hot levels' draws, so eps
times the number of hot draws per world must stay small; the per-update ESS
(sum W)^2 / sum W^2 is logged."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from . import _cpass_core as core
from ._cfast_cone import Cone
from ._cln_core import hidden1
from ._cpass_core import NCOL, key_of, mlp
from .cpass import adam_step, clip_by_level


def train_pass_hot(lay, params, seed, designed, tau, lam, eps, tau_b, override=None, neg=False):
    """cfast's train_pass with eps-hot draws.  eps: (T + 1,) per-level hot rate.
    Records also "logrho" = log q(a) - log b(a) (0 where eps = 0 or invalid)
    and "hot".  neg=True is a NEGATIVE CONTROL (tests only): logrho forced to 0."""
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
                logqb = jax.nn.log_softmax(jnp.where(mask, logq / tau_b, -jnp.inf), axis=1)
                g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                u = noise.uniform(jnp, seed, noise.tag_of("hot:" + c.name), l, Y, Xn)
                hot = u < eps[l]
                a = jnp.argmax(jnp.where(hot[:, None], logqb, logits) + g, axis=1).astype(jnp.int32)
                st = lay.step(l, 0, ci) + k * len(chans)
                if override is not None:
                    os_, oj, oval, on = override
                    hit = on & (st == os_) & (jnp.arange(Y.shape[0]) == oj)
                    a = jnp.where(hit, oval, a)
                idx = jnp.arange(Y.shape[0])
                la = logq[idx, a]
                lb = jnp.logaddexp(jnp.log1p(-eps[l]) + la, jnp.log(eps[l]) + logqb[idx, a])
                logrho = jnp.where(valid & (eps[l] > 0) & (not neg), la - lb, 0.0)
                cost = jnp.where(valid, soft[idx, a] + lam[l] * nv[idx, a] + tau * la, 0.0)
                outs.append({"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a,
                             "valid": valid, "logq": logq, "cost": cost, "logrho": logrho,
                             "hot": hot & valid})
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, outs

        state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            recs[key_of(l, c.name)] = ys[ci]
    return state, recs


def build_train(var, lay, neg=False):
    cfg = var.cfg
    assert cfg["K"] == 0 and cfg["cone"], "chot: cone returns, K = 0 only"
    T = var.problem.T
    tau_b = float(cfg["tau_b"])
    levels = list(range(T + 1))
    keys = [key_of(l, c.name) for l in lay.levels for c in lay.chans[l]]
    level_of = {key_of(l, c.name): l for l in levels for c in lay.chans[l]}
    nval = {}
    for l in levels:
        cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(NCOL))
        nval[l] = cnt * len(lay.chans[l])
    cone = Cone(lay, lay.actor_win)
    idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
    sfrac = {}
    for l in levels:
        for ci, c in enumerate(lay.chans[l]):
            sfrac[key_of(l, c.name)] = jnp.asarray(
                [lay.step(l, k, ci) / lay.nsteps for k in range(NCOL)], jnp.float32)
    critic_h1 = cfg["critic"] == "h1"
    sub = cfg["sub"]

    def cfeat(r, key, tau, lam, params):
        f = r["feat"]
        if critic_h1:
            f = jax.lax.stop_gradient(hidden1(params[key], f))
        ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
        ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
        sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
        return jnp.concatenate([f, ex, sf], -1)

    def actor_loss(params, cparams, recs, G, tau, lam, gs, p0, wgt, W):
        """cfast's actor loss with each world's terms multiplied by W (B,)."""
        total = 0.0
        Wb = jax.lax.stop_gradient(W)[:, None, None]
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
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], Wb * (t1 + t2), 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def critic_loss(cparams, recs, G, tau, lam, gs, p0, wgt):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = key_of(l, c.name)
                r = recs[key]
                Q = mlp(cparams[key], cfeat(r, key, tau, lam, p0))
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def batch(params, seeds, des, tau, lam, eps, rkey):
        st, recs = jax.vmap(lambda s, d: train_pass_hot(lay, params, s, d, tau, lam, eps, tau_b,
                                                        neg=neg))(seeds, des)
        logW = sum(jnp.sum(r["logrho"], (1, 2)) for r in recs.values())          # (B,)
        W = jnp.exp(logW)
        violw = jnp.zeros((seeds.shape[0], T + 1))
        diag = {}
        for key, r in recs.items():
            l = level_of[key]
            nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
            violw = violw.at[:, l].add(jnp.sum(nva * r["valid"], (1, 2)))
            m = r["mask"]
            Hq = -jnp.sum(jnp.where(m, jnp.exp(r["logq"]) * r["logq"], 0.0), -1)
            nf = r["valid"] & (jnp.sum(m, -1) > 1)
            diag[key] = jnp.stack([jnp.sum(jnp.where(nf, Hq, 0.0)), jnp.sum(nf),
                                   jnp.sum(r["hot"]), jnp.sum(r["valid"])])
        viol = jnp.mean(W[:, None] * violw, 0)                     # importance-weighted (unbiased for q)
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
                       for f, x in r.items() if f not in ("logrho", "hot")}
            ix = idx[key][jnp.arange(NCOL)[None, :, None], sel]
            v = jax.vmap(lambda a, i: a[i])(ii, ix)
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        return st, rs, G, wgt, viol, tot, W, diag

    def fused_hot(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey, eps):
        st, recs, G, wgt, viol, tot, W, diag = batch(params, seeds, des, tau, lam, eps, rkey)
        p0 = params
        loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, p0, wgt, W)
        g = clip_by_level(g, levels, clip)
        params, opt = adam_step(params, g, opt, lr)
        closs, gc = jax.value_and_grad(critic_loss)(cparams, recs, G, tau, lam, gs, p0, wgt)
        gc = clip_by_level(gc, levels, clip)
        cparams, copt = adam_step(cparams, gc, copt, lrc)
        gstd = {k: jnp.std(v) for k, v in G.items()}
        return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd, W, diag

    eps0 = jnp.zeros(T + 1, jnp.float32)

    def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey):
        """cfast's signature, no hot draws (the baseline measurement)."""
        return fused_hot(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey, eps0)[:10]

    def agrad(params, cparams, gs, seeds, des, tau, lam, rkey, eps):
        """Actor gradient of one batch (no update, no clipping): tests."""
        st, recs, G, wgt, viol, tot, W, diag = batch(params, seeds, des, tau, lam, eps, rkey)
        g = jax.grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, params, wgt, W)
        return g, tot, W

    return {"fused": jax.jit(fused), "fused_hot": jax.jit(fused_hot), "agrad": jax.jit(agrad),
            "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s))),
            "nval": nval, "cone": cone, "level_of": level_of, "topkeys": []}
