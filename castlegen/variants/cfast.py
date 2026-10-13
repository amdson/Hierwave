"""cfast: cpass with dependency-cone returns and a cheaper update.

Generation is cpass's (same pass, features, noise, repair); only training
changes, and it stays an unbiased estimator of the same objective:

- Returns.  G_i sums the costs of a conservative superset of draw i's
  read-graph descendants (the dependency cone, _cfast_cone.py) instead of
  every later draw.  Draws outside the cone cannot change with a_i, so their
  costs only add variance.  Computed on device from per-site costs by box
  sums over integral images.
- Critic.  Reads the actor's first hidden layer (pre-update parameters, no
  gradient) plus (tau, lambda, step fraction); its target is the cone return
  scaled per (level, channel).  No critic features are recorded during the
  pass (cpass's 8 x 8 + 7 x 7 critic window cost more than the pass itself).
- Subsampled loss (sub = m).  The pass draws every site; the loss reads m
  uniformly chosen sites per (step, world) where a step has more, weighted
  n / m: an unbiased estimate of the full sum at a fraction of the
  forward/backward cost (fine levels only; cone returns of only those sites).
- Replays.  K coupled replays of one coarsest draw per world, scored by the
  same cone; K = 0 (default) turns them off: the coarsest draws then use the
  score-function term like every other draw.  (K > 0 ignores sub.)
- One fused jitted update (pass, returns, actor and critic steps; K = 0),
  which at lr = 0 also gives the untrained tau = 1 draws for the baseline;
  repair batched; layouts and compiled functions shared by all instances in
  the process (the second training seed pays no trace / compile); tau
  anneals over the first 20 % of the time left after setup.
- lr 3e-3 (cpass 1e-3).
"""
from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cpass_core as core
from ._cfast_cone import Cone
from ._cfast_core import train_pass
from .cpass import CPass, adam_init, adam_step, clip_by_level

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cfast"))

_LAYS = {}
CFG = dict(B=8, K=0, width=64, lr=3e-3, lr_critic=1e-3, lam0=10.0, lam_max=20.0, eta=0.01,
           alpha=0.05, tau0=2.0, tau1=1.0, anneal=0.2, clip=10.0, W_train=None,
           cone=True, critic="h1", sub=64)


@register
class CFast(CPass):
    name = "cfast"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CFG, **cfg})
        T = problem.T
        lay = self.layout(Region(self.cfg["W_train"] or problem.W_train))
        n_extra = 1 + (T + 1) + 1
        key = jax.random.PRNGKey(seed + 7919)
        for l in range(T + 1):
            for c in lay.chans[l]:
                key, k2 = jax.random.split(key)
                din = self.cfg["width"] if self.cfg["critic"] == "h1" else lay.feat_dim(l, core.ACTOR_WIN)
                self.cparams[core.key_of(l, c.name)] = core.mlp_init(k2, din + n_extra, c.D, self.cfg["width"])
        self.copt = adam_init(self.cparams)
        self.gs = None

    # ------------------------------------------------------------- layouts
    def layout(self, region):
        """Layouts (and their compiled functions) are shared by every instance
        on the same problem and region: a second training seed in the same
        process reuses the traced and compiled functions."""
        key = (self.problem.name, self.problem.version, region.W, region.periodic, region.x0)
        if key not in _LAYS:
            self.lays = {}
            _LAYS[key] = CPass.layout(self, region)
        return _LAYS[key]

    def train_fns(self, lay):
        key = (self.cfg["K"], self.cfg["cone"], self.cfg["critic"], self.cfg["sub"])
        if not hasattr(lay, "cfast_fns"):
            lay.cfast_fns = {}
        if key not in lay.cfast_fns:
            lay.cfast_fns[key] = self._build_train(lay)
        return lay.cfast_fns[key]

    # -------------------------------------------------------------- training
    def _build_train(self, lay):
        T, K = self.problem.T, self.cfg["K"]
        levels = list(range(T + 1))
        keys = [core.key_of(l, c.name) for l in lay.levels for c in lay.chans[l]]
        topkeys = [core.key_of(T, c.name) for c in lay.chans[T]]
        level_of = {core.key_of(l, c.name): l for l in levels for c in lay.chans[l]}
        nval = {}
        for l in levels:
            cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(core.NCOL))
            nval[l] = cnt * len(lay.chans[l])
        cone = Cone(lay, core.ACTOR_WIN)
        idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
        idx_top = jnp.stack([idx[k] for k in topkeys])                  # (ntop, 16, n, S, 4)
        sfrac = {}
        for l in levels:
            for ci, c in enumerate(lay.chans[l]):
                sfrac[core.key_of(l, c.name)] = jnp.asarray(
                    [lay.step(l, k, ci) / lay.nsteps for k in range(core.NCOL)], jnp.float32)
        use_cone = self.cfg["cone"]

        def returns(costs):
            if use_cone:
                return cone.returns(costs, idx)
            # all later draws (cpass)
            tot = jnp.zeros((next(iter(costs.values())).shape[0], lay.nsteps))
            for key, c in costs.items():
                l = level_of[key]
                ci = [x.name for x in lay.chans[l]].index(key[key.index("_") + 1:])
                st = np.array([lay.step(l, k, ci) for k in range(core.NCOL)])
                tot = tot.at[:, st].add(c.sum(-1))
            after = jnp.sum(tot, -1, keepdims=True) - jnp.cumsum(tot, -1)
            G = {}
            for key, c in costs.items():
                l = level_of[key]
                ci = [x.name for x in lay.chans[l]].index(key[key.index("_") + 1:])
                st = np.array([lay.step(l, k, ci) for k in range(core.NCOL)])
                G[key] = jnp.broadcast_to(after[:, st][..., None], c.shape)
            return G

        def rec_fn(params, seeds, des, tau, lam):
            st, recs = jax.vmap(lambda s, d: train_pass(lay, params, s, d, tau, lam))(seeds, des)
            G = returns({k: r["cost"] for k, r in recs.items()})
            viol = jnp.zeros(T + 1)
            for key, r in recs.items():
                nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
                viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
            tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
            top = {k: {f: recs[k][f] for f in ("valid", "mask", "logq", "a")} for k in topkeys}
            return st, recs, G, viol / seeds.shape[0], tot, top


        def replay_fn(params, seeds, des, tau, lam, ov, chan, kk, jj):
            costs = jax.vmap(lambda s, d, o: train_pass(lay, params, s, d, tau, lam, override=o,
                                                        record=False)[1])(seeds, des, ov)
            if use_cone:
                ii = cone.integral(costs)                                # (BK, size)
                ix = idx_top[chan, kk, jj]                               # (BK, S, 4)
                v = jnp.take_along_axis(ii[:, None, :], ix.reshape(ix.shape[0], 1, -1), -1)
                v = v.reshape(ix.shape)
                return jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
            G = returns(costs)
            Gt = jnp.stack([G[k] for k in topkeys])                       # (ntop, BK, 16, n)
            return Gt[chan, jnp.arange(seeds.shape[0]), kk, jj]


        critic_h1 = self.cfg["critic"] == "h1"
        sub = self.cfg["sub"]

        def cfeat(r, key, tau, lam, params):
            f = r["feat"]
            if critic_h1:                     # the actor's first hidden layer, fixed for this update
                p = params[key]
                f = jax.lax.stop_gradient(jnp.tanh(f @ p["W1"] + p["b1"]))
            ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
            ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
            sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
            return jnp.concatenate([f, ex, sf], -1)

        def actor_loss(params, cparams, recs, G, tau, lam, br, gs, p0, wgt):
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
                    keep = r["valid"]
                    if K > 0 and key in topkeys:
                        keep = keep & ~br["sel"][key]
                        lev = lev + jnp.sum(branch_loss(key, r, logq, q, ell, Q, br))
                    lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(keep, t1 + t2, 0.0))
                total = total + lev / nval[l]
            return total / recs[keys[0]]["a"].shape[0]

        def branch_loss(key, r, logq, q, ell, Q, br):
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

        def critic_loss(cparams, recs, G, tau, lam, br, gs, p0, wgt):
            total = 0.0
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    Q = core.mlp(cparams[key], cfeat(r, key, tau, lam, p0))
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0))
                    if K > 0 and key in topkeys:
                        ix = jnp.arange(Q.shape[0])
                        on = br["chan"] == topkeys.index(key)
                        acts = jnp.where(on[:, None], jnp.clip(br["acts"], 0, Q.shape[-1] - 1), 0)
                        Qk = jnp.take_along_axis(Q[ix, br["k"], br["j"]], acts, -1)
                        lev = lev + jnp.sum(jnp.where(on, jnp.sum((Qk - br["Gk"] / gs[key]) ** 2, -1), 0.0))
                total = total + lev / nval[l]
            return total / recs[keys[0]]["a"].shape[0]

        def train_step(params, cparams, opt, copt, recs, G, tau, lam, br, gs, lr, lrc, clip, wgt=None):
            wgt = wgt or {}
            p0 = params
            loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, br, gs, p0, wgt)
            g = clip_by_level(g, levels, clip)
            params, opt = adam_step(params, g, opt, lr)
            closs, gc = jax.value_and_grad(critic_loss)(cparams, recs, G, tau, lam, br, gs, p0, wgt)
            gc = clip_by_level(gc, levels, clip)
            cparams, copt = adam_step(cparams, gc, copt, lrc)
            return params, cparams, opt, copt, loss, closs

        def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey):
            """K = 0: pass, cone returns and both optimiser steps in one call
            (lr = 0 gives the untrained tau = 1 draws for the baseline).  With
            sub = m, the loss reads m uniformly chosen sites per (step, world)
            where a step has more, weighted n / m (unbiased)."""
            if not sub:
                st, recs, G, viol, tot, _ = rec_fn(params, seeds, des, tau, lam)
                wgt = {}
            else:
                st, recs = jax.vmap(lambda s, d: train_pass(lay, params, s, d, tau, lam))(seeds, des)
                viol = jnp.zeros(T + 1)
                for key, r in recs.items():
                    nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
                    viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
                viol = viol / seeds.shape[0]
                tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
                costs = {k: r["cost"] for k, r in recs.items()}
                ii = cone.integral(costs) if use_cone else None                 # (B, size)
                Gall = None if use_cone else returns(costs)
                Bn = seeds.shape[0]
                G, wgt, rs = {}, {}, {}
                for key, r in recs.items():
                    n = r["a"].shape[-1]
                    if n <= sub:
                        sel = jnp.broadcast_to(jnp.arange(n), (Bn, core.NCOL, n))
                    else:
                        rkey, k1 = jax.random.split(rkey)
                        sel = jnp.argsort(jax.random.uniform(k1, (Bn, core.NCOL, n)), -1)[..., :sub]
                        wgt[key] = n / sub
                    rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                               for f, x in r.items()}
                    if use_cone:                # cone returns of the chosen draws only
                        ix = idx[key][jnp.arange(core.NCOL)[None, :, None], sel]      # (B, 16, m, S, 4)
                        v = jax.vmap(lambda a, i: a[i])(ii, ix)
                        G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
                    else:
                        G[key] = jnp.take_along_axis(Gall[key], sel, 2)
                recs = rs
            params, cparams, opt, copt, loss, closs = train_step(params, cparams, opt, copt, recs, G, tau, lam,
                                                                 {}, gs, lr, lrc, clip, wgt)
            gstd = {k: jnp.std(v) for k, v in G.items()}
            return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd

        fns = {"rec": jax.jit(rec_fn), "step": jax.jit(train_step), "fused": jax.jit(fused),
               "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s))),
               # diagnostics: the actor gradient of one batch (variance measurements)
               "actor_grad": jax.jit(lambda params, cparams, recs, G, tau, lam, gs: jax.grad(actor_loss)(
                   params, cparams, recs, G, tau, lam, {}, gs, params, {})),
               "topkeys": topkeys, "nval": nval, "cone": cone}
        if K > 0:
            fns["replay"] = jax.jit(replay_fn)
        return fns

    def _branches(self, lay, top, G, seeds, des, tau, lam):
        """One coarsest draw per world; K coupled replays scored by its cone."""
        K, B = self.cfg["K"], len(seeds)
        tk = self.fn["topkeys"]
        top = jax.device_get(top)
        sel = {k: np.zeros(top[k]["a"].shape, bool) for k in tk}
        chan, kk, jj = np.zeros(B, int), np.zeros(B, int), np.zeros(B, int)
        acts, enum = np.zeros((B, K), int), np.zeros(B, bool)
        for b in range(B):
            cand = [(ci, k, j) for ci, key in enumerate(tk) for k, j in zip(*np.nonzero(top[key]["valid"][b]))]
            ci, k, j = cand[self.rng.integers(len(cand))]
            chan[b], kk[b], jj[b] = ci, k, j
            sel[tk[ci]][b, k, j] = True
            r = top[tk[ci]]
            mask = r["mask"][b, k, j]
            A = np.nonzero(mask)[0]
            if len(A) <= K:
                enum[b] = True
                acts[b] = np.concatenate([A, np.full(K - len(A), A[0])])
            else:
                lq = np.asarray(r["logq"][b, k, j], float)
                p = np.exp(lq - lq.max()) * mask
                p /= p.sum()
                acts[b, 0] = int(r["a"][b, k, j])
                acts[b, 1:] = self.rng.choice(len(p), K - 1, p=p)
        T = self.problem.T
        steps = np.array([lay.step(T, kk[b], chan[b]) for b in range(B)])
        rep = lambda x: jnp.repeat(x, K, axis=0)
        ov = (jnp.asarray(np.repeat(steps, K)), jnp.asarray(np.repeat(jj, K)),
              jnp.asarray(acts.reshape(-1), jnp.int32), jnp.ones(B * K, bool))
        Gk = self.fn["replay"](self.params, rep(seeds), {k: rep(v) for k, v in des.items()}, tau, lam, ov,
                            jnp.asarray(np.repeat(chan, K)), jnp.asarray(np.repeat(kk, K)),
                            jnp.asarray(np.repeat(jj, K)))
        Gk = np.asarray(Gk).reshape(B, K)
        Gt = np.stack([np.asarray(G[k]) for k in tk])                   # (ntop, B, 16, n)
        g0 = Gt[chan, np.arange(B), kk, jj]
        assert np.allclose(Gk[~enum, 0], g0[~enum], rtol=1e-3, atol=1e-1), (Gk[~enum, 0], g0[~enum])
        return {"sel": {k: jnp.asarray(v) for k, v in sel.items()}, "chan": jnp.asarray(chan),
                "k": jnp.asarray(kk), "j": jnp.asarray(jj), "acts": jnp.asarray(acts),
                "Gk": jnp.asarray(Gk, jnp.float32), "enum": jnp.asarray(enum)}

    def _batch(self, seeds, region):
        desl = [self.designed(int(s), region) for s in seeds]
        return {k: jnp.stack([d[k] for d in desl]) for k in desl[0]}

    def _measure_baseline(self, lay, log):
        """Untrained raw (the pass at tau = 1) and repaired violation fractions -> eps_l."""
        T, nb = self.problem.T, self.cfg["B"]          # same batch shape as training: no extra compile
        region = Region(lay.W)
        seeds = np.arange(500_000 + 1000 * self.seed, 500_000 + 1000 * self.seed + nb)
        des = self._batch(seeds, region)
        lam = jnp.asarray(self.lam, jnp.float32)
        js = jnp.asarray(seeds, jnp.int32)
        if self.cfg["K"] == 0:                         # the fused update at lr = 0 (its outputs are discarded)
            ones = {k: jnp.float32(1.0) for k in self.params}
            out = self.fn["fused"](self.params, self.cparams, self.opt, self.copt, ones, js, des,
                                   jnp.float32(1.0), lam, 0.0, 0.0, self.cfg["clip"], jax.random.PRNGKey(0))
            st, gstd = out[0], out[-1]
        else:
            out = self.fn["rec"](self.params, js, des, jnp.float32(1.0), lam)
            st, gstd = out[0], {k: jnp.std(v) for k, v in out[2].items()}
        # critic output scale per (level, channel), fixed from the untrained draws
        self.gs = {k: jnp.float32(max(float(v), 1.0)) for k, v in gstd.items()}
        fin = self.fn["repair_b"](st, js)
        st, fin = jax.device_get(st), jax.device_get(fin)
        raw, ff_, n = np.zeros(T + 1), np.zeros(T + 1), np.zeros(T + 1)
        for b in range(nb):
            d = {k: np.asarray(v[b]) for k, v in des.items()}
            for kind, x, acc in (("raw", st, raw), ("final", fin, ff_)):
                xs = {c.name: np.asarray(x[c.name][b]) for c in self.problem.sampled()}
                E = en.energy(self.problem, {**d, **xs}, region)
                for l in range(T + 1):
                    acc[l] += E.viol[l]
                    if kind == "raw":
                        n[l] += E.nhard[l]
        self.nhard = n / nb
        rf, ff = raw / np.maximum(n, 1), ff_ / np.maximum(n, 1)
        rho = np.where(rf > 0, ff / np.maximum(rf, 1e-12), 1.0)
        self.eps = np.clip(1e-4 / np.maximum(rho, 1e-12), 1e-4, 1e-3)
        log({"baseline_raw": rf.tolist(), "baseline_final": ff.tolist(), "rho": rho.tolist(),
             "eps": self.eps.tolist()})
        self.baseline = {"raw": rf.tolist(), "final": ff.tolist(), "eps": self.eps.tolist()}

    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        self.fn = self.train_fns(lay)
        self._measure_baseline(lay, log)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True})
        B = cfg["B"]
        tm = {"rec": 0.0, "replay": 0.0, "step": 0.0}
        while True:
            frac = (time.time() - t0) / max(budget_s, 1e-9)
            if frac >= 1.0:
                break
            # tau anneals over the first `anneal` of the time left after setup (compiles)
            ftr = (time.time() - t0 - t_setup) / max(budget_s - t_setup, 1e-9)
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - ftr / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            des = self._batch(seeds, region)
            jseeds = jnp.asarray(seeds, jnp.int32)
            ta = time.time()
            br = None
            if cfg["K"] == 0:
                out = self.fn["fused"](self.params, self.cparams, self.opt, self.copt, self.gs, jseeds, des,
                                       tau, lam, cfg["lr"], cfg["lr_critic"], cfg["clip"],
                                       jax.random.PRNGKey(self.updates + 7919 * self.seed))
                _, self.params, self.cparams, self.opt, self.copt, loss, closs, viol, tot, _ = out
                viol = np.asarray(viol)
                tb = tc = time.time()
            else:
                st, recs, G, viol, tot, top = self.fn["rec"](self.params, jseeds, des, tau, lam)
                tb = time.time()
                br = self._branches(lay, top, G, jseeds, des, tau, lam)
                tc = time.time()
                self.params, self.cparams, self.opt, self.copt, loss, closs = self.fn["step"](
                    self.params, self.cparams, self.opt, self.copt, recs, G, tau, lam, br,
                    self.gs, cfg["lr"], cfg["lr_critic"], cfg["clip"])
                viol = np.asarray(viol)
            td = time.time()
            tm["rec"] += tb - ta
            tm["replay"] += tc - tb
            tm["step"] += td - tc
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            log({"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                 "critic_loss": float(closs), "cost_per_world": float(np.asarray(tot).mean()),
                 "raw_viol": hf.tolist(), "lam": self.lam.tolist(),
                 "enum_frac": float(np.asarray(br["enum"]).mean()) if br else None,
                 "time_rec": tm["rec"], "time_replay": tm["replay"], "time_step": tm["step"]})
        self.timing = {**tm, "setup": t_setup, "total": time.time() - t0}

    def info(self):
        return {**super().info(), "timing": getattr(self, "timing", None)}


def _variant(name, **over):
    """Register a cfast configuration under its own name (for comparisons)."""
    cls = type(f"CFast_{name}", (CFast,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("cfast_k4", K=4)                 # with cpass's coupled replays
_variant("cfast_full", cone=False)        # all later draws as returns (cpass's returns), cfast otherwise
_variant("cfast_s64", lr=1e-3)            # development runs: lr 1e-3 (cpass's)
_variant("cfast_s64_lr3")                 # development runs: the default
