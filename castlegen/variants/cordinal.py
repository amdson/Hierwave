"""cordinal: cpass with ordinal input encodings and smooth, mask-relative
output heads for large channels (ideas.md #6; core in _cordinal_core.py).

A pure head / encoding A/B against cpass: the estimator, critic (per-value
outputs, cpass's encoder (a) applied to its inputs too), replays, penalties,
schedules and generation are cpass's.  Switches (CFG, or CORDINAL_CFG as JSON
in the environment for ablations):
  enc     "thermo" (a) or "poly" (cpass's u, u^2)
  smooth  True (b) or False (per-value logits for every channel)
  rel     True (c) or False (basis over the view's full range)
  nbasis, rscale  basis size (cos(pi k u), k = 1..nbasis), residual scale."""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cordinal_core as core
from .cpass import CFG as CPASS_CFG
from .cpass import CPass, adam_init, adam_step, clip_by_level

CFG = {**CPASS_CFG, "enc": "thermo", "smooth": True, "rel": True, "nbasis": 8, "rscale": 0.1}


@register
class COrdinal(CPass):
    name = "cordinal"

    def __init__(self, problem, seed, workdir, **cfg):
        env = json.loads(os.environ.get("CORDINAL_CFG", "{}"))
        cfg = {**CFG, **env, **cfg}
        super().__init__(problem, seed, workdir, **cfg)
        T = problem.T
        lay = self.layout(Region(problem.W_train))
        self.specs = {}
        for l in range(T + 1):
            for c in lay.chans[l]:
                self.specs[core.key_of(l, c.name)] = core.head_spec(
                    c, K=self.cfg["nbasis"], rscale=self.cfg["rscale"], rel=self.cfg["rel"], smooth=self.cfg["smooth"])
        # re-initialise with this layout's feature sizes and head output sizes
        key = jax.random.PRNGKey(seed)
        n_extra = 1 + (T + 1) + 1
        for l in range(T + 1):
            for c in lay.chans[l]:
                key, k1, k2 = jax.random.split(key, 3)
                kk = core.key_of(l, c.name)
                sp = self.specs[kk]
                self.params[kk] = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN),
                                                sp.dout if sp is not None else c.D, self.cfg["width"])
                self.cparams[kk] = core.mlp_init(k2, lay.feat_dim(l, core.CRITIC_WIN, n_extra), c.D,
                                                 self.cfg["width"])
        self.opt, self.copt = adam_init(self.params), adam_init(self.cparams)

    # ------------------------------------------------------------- layouts
    def layout(self, region):
        key = (region.W, region.periodic, region.x0)
        if key not in self.lays:
            lay = core.OLayout(self.problem, region.W, region.periodic, region.x0, enc=self.cfg["enc"])
            lay.jit_gen = jax.jit(lambda params, seed, des, lam: core.coloured_pass(
                lay, params, self.specs, seed, des, 1.0, lam)[0])
            lay.jit_repair = jax.jit(lambda st, seed: core.repair(lay, st, seed))
            lay.jit_score = jax.jit(lambda params, des, forced, lam: core.coloured_pass(
                lay, params, self.specs, 0, des, 1.0, lam, forced=forced)[2])
            self.lays[key] = lay
        return self.lays[key]

    # -------------------------------------------------------------- training
    def _build_train(self, lay):
        """cpass._build_train with b from the cordinal head."""
        T, K = self.problem.T, self.cfg["K"]
        levels = list(range(T + 1))
        topkeys = [core.key_of(T, c.name) for c in lay.chans[T]]
        specs = self.specs
        nval = {}
        for l in levels:
            cnt = 0
            for k in range(core.NCOL):
                _, _, v = lay.sites(l, k)
                cnt += int(np.asarray(v).sum())
            nval[l] = cnt * len(lay.chans[l])

        def gen_rec(params, seed, des, tau, lam, extra):
            st, tot, _, rec = core.coloured_pass(lay, params, specs, seed, des, tau, lam, record=True,
                                                 critic_extra=extra)
            return st, tot, rec

        def gen_tot(params, seed, des, tau, lam, ov):
            return core.coloured_pass(lay, params, specs, seed, des, tau, lam, override=ov)[1]

        lay.jit_rec = jax.jit(jax.vmap(gen_rec, in_axes=(None, 0, 0, None, None, None)))
        lay.jit_tot = jax.jit(jax.vmap(gen_tot, in_axes=(None, 0, 0, None, None, 0)))

        def actor_loss(params, cparams, recs, G, tau, lam, br, gs):
            total = 0.0
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    b = core.head(specs[key], params[key], r["feat"], r["mask"])   # (B, 16, n, D)
                    logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, -1)
                    q = jnp.exp(logq)
                    lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                    ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                    Q = jax.lax.stop_gradient(core.mlp(cparams[key], r["cfeat"])) * gs
                    t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                    la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    t2 = la * jax.lax.stop_gradient(G[key][..., None] - Qa)
                    keep = r["valid"]
                    if key in topkeys:
                        keep = keep & ~br["sel"][key]
                        bl = branch_loss(key, r, b, logq, q, ell, Q, br)
                        lev = lev + jnp.sum(bl)
                    lev = lev + jnp.sum(jnp.where(keep, t1 + t2, 0.0))
                total = total + lev / nval[l]
            return total / recs[topkeys[0]]["a"].shape[0]

        def branch_loss(key, r, b, logq, q, ell, Q, br):
            Bn = r["a"].shape[0]
            ix = jnp.arange(Bn)
            k, j = br["k"], br["j"]
            on = br["chan"] == topkeys.index(key)
            qq, el, QQ, mk = q[ix, k, j], ell[ix, k, j], Q[ix, k, j], r["mask"][ix, k, j]
            lq = jnp.where(mk, logq[ix, k, j], 0.0)
            acts = jnp.where(on[:, None], jnp.clip(br["acts"], 0, el.shape[-1] - 1), 0)
            Gk, enum = br["Gk"], br["enum"]
            Gfull = jnp.zeros_like(el).at[ix[:, None], acts].set(Gk)
            le = jnp.sum(jnp.where(mk, qq * jax.lax.stop_gradient(el + Gfull), 0.0), -1)
            t1 = jnp.sum(jnp.where(mk, qq * jax.lax.stop_gradient(el + QQ), 0.0), -1)
            rk = Gk - jnp.take_along_axis(QQ, acts, -1)
            beta = (jnp.sum(rk, -1, keepdims=True) - rk) / (K - 1)
            ls = t1 + jnp.mean(jnp.take_along_axis(lq, acts, -1) * jax.lax.stop_gradient(rk - beta), -1)
            return jnp.where(on, jnp.where(enum, le, ls), 0.0)

        def critic_loss(cparams, recs, G, br, gs):
            total = 0.0
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    Q = core.mlp(cparams[key], r["cfeat"])
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    err = (Qa - G[key][..., None] / gs) ** 2
                    lev = lev + jnp.sum(jnp.where(r["valid"], err, 0.0))
                    if key in topkeys:
                        ix = jnp.arange(Q.shape[0])
                        on = br["chan"] == topkeys.index(key)
                        acts = jnp.where(on[:, None], jnp.clip(br["acts"], 0, Q.shape[-1] - 1), 0)
                        Qk = jnp.take_along_axis(Q[ix, br["k"], br["j"]], acts, -1)
                        e2 = jnp.sum((Qk - br["Gk"] / gs) ** 2, -1)
                        lev = lev + jnp.sum(jnp.where(on, e2, 0.0))
                total = total + lev / nval[l]
            return total / recs[topkeys[0]]["a"].shape[0]

        def train_step(params, cparams, opt, copt, recs, G, tau, lam, br, gs, lr, lrc, clip):
            loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, br, gs)
            g = clip_by_level(g, levels, clip)
            params, opt = adam_step(params, g, opt, lr)
            closs, gc = jax.value_and_grad(critic_loss)(cparams, recs, G, br, gs)
            gc = clip_by_level(gc, levels, clip)
            cparams, copt = adam_step(cparams, gc, copt, lrc)
            return params, cparams, opt, copt, loss, closs

        lay.jit_step = jax.jit(train_step)
        lay.topkeys, lay.nval = topkeys, nval
        lay.actor_loss, lay.critic_loss = actor_loss, critic_loss


# ----------------------------------------------------------------- ablations
@register
class COrdinalEncOnly(COrdinal):
    """(a) only: thermometer inputs, cpass's per-value head."""
    name = "cordinal_a"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{"smooth": False, **cfg})


@register
class COrdinalAbs(COrdinal):
    """(a) + (b) without (c): basis over each view's full range."""
    name = "cordinal_ab"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{"rel": False, **cfg})
