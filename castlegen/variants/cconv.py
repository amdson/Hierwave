"""cconv: cfast with a convolutional actor (ideas.md #5; net in _cconv_core.py).

Unchanged from cfast: the schedule, conditional, noise, logits form
(-s / tau - b on the least-violating set), repair, the objective and its
estimator (cone returns, subsampled score-function loss with a critic control
variate, K = 0), penalties and tau schedule.

Changed:
- Actor b_theta: per-level conv trunk (3 x 3 stem + r - 1 residual pre-LN
  SiLU blocks, radius r, width cw) over input planes of the level and its
  parent, recomputed densely at every step of the pass; per-channel linear
  head plus candidate-scoring head (site features, value embedding, s(t),
  nv(t), lookahead(t)).
- Loss: the trunk re-evaluated on (2r + 1)^2 patches around the subsampled
  sites, the state at each step rebuilt from the final state and the static
  schedule (nothing per site is recorded beyond cfast's draw records).
- Critic: tanh MLP on the stop-gradient trunk features at the site (pre-update
  parameters) plus (tau, lambda, step fraction).
- Cone: built with the trunk's window (radius r around the site, parent cells
  under it), so cone returns stay a superset of the descendants.
- Margin for open windows: the sum of every step's reach (_cconv_core.margin).
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

from . import _cconv_core as cc
from . import _cpass_core as core
from ._cfast_cone import Cone
from .cfast import CFast
from .cpass import CPass, adam_init, adam_step, clip_by_level


_LAYS = {}
CFG = dict(B=64, K=0, width=64, lr=1e-3, lr_critic=1e-3, lam0=10.0, lam_max=20.0, eta=0.01,
           alpha=0.05, tau0=2.0, tau1=1.0, anneal=0.2, clip=10.0, W_train=None,
           cone=True, critic="trunk", sub=64, r=4, cw=32, hc=32)


@register
class CConv(CFast):
    name = "cconv"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CFG, **cfg})
        T = problem.T
        assert self.cfg["K"] == 0 and self.cfg["cone"] and self.cfg["sub"]
        lay = self.layout(Region(self.cfg["W_train"] or problem.W_train))
        key = jax.random.PRNGKey(seed + 104729)
        k1, key = jax.random.split(key)
        self.params = lay.net.init(k1)
        n_extra = 1 + (T + 1) + 1
        self.cparams = {}
        for l in range(T + 1):
            for c in lay.chans[l]:
                key, k2 = jax.random.split(key)
                self.cparams[core.key_of(l, c.name)] = core.mlp_init(k2, self.cfg["cw"] + n_extra, c.D,
                                                                     self.cfg["width"])
        self.opt, self.copt = adam_init(self.params), adam_init(self.cparams)

    # ------------------------------------------------------------- layouts
    def layout(self, region):
        cfg = self.cfg
        key = (self.problem.name, self.problem.version, region.W, region.periodic, region.x0,
               cfg["r"], cfg["cw"], cfg["hc"])
        if key not in _LAYS:
            lay = core.Layout(self.problem, region.W, region.periodic, region.x0)
            net = cc.ConvNet(lay, cfg["r"], cfg["cw"], cfg["hc"])
            lay.net = net
            lay.jit_gen = jax.jit(lambda params, seed, des, lam: cc.conv_pass(
                lay, net, params, seed, des, 1.0, lam, record=False)[0])
            lay.jit_repair = jax.jit(lambda st, seed: core.repair(lay, st, seed))
            lay.jit_score = jax.jit(lambda params, des, forced, lam: cc.conv_pass(
                lay, net, params, 0, des, 1.0, lam, forced=forced)[1])
            _LAYS[key] = lay
        return _LAYS[key]

    def margin(self):
        lay = self.layout(Region(self.cfg["W_train"] or self.problem.W_train))
        return cc.margin(self.problem, lay.chans, self.cfg["r"])

    def train_fns(self, lay):
        if not hasattr(lay, "cconv_fns"):
            lay.cconv_fns = self._build_train(lay)
        return lay.cconv_fns

    # -------------------------------------------------------------- training
    def _build_train(self, lay):
        T = self.problem.T
        net = lay.net
        levels = list(range(T + 1))
        keys = [core.key_of(l, c.name) for l in lay.levels for c in lay.chans[l]]
        level_of = {core.key_of(l, c.name): l for l in levels for c in lay.chans[l]}
        emb_of = {core.key_of(l, c.name): lay.emb[c.name] for l in levels for c in lay.chans[l]}
        nval = {}
        for l in levels:
            cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(core.NCOL))
            nval[l] = cnt * len(lay.chans[l])
        cone = Cone(lay, cc.conv_window(self.cfg["r"]))
        idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
        steps = {}
        for l in levels:
            steps.update(net.set_step(l))
        sfrac, sYX, sstep = {}, {}, {}
        for l in levels:
            Ys, Xs = zip(*[lay.sites(l, k)[:2] for k in range(core.NCOL)])
            for ci, c in enumerate(lay.chans[l]):
                key = core.key_of(l, c.name)
                st = np.array([lay.step(l, k, ci) for k in range(core.NCOL)])
                sfrac[key] = jnp.asarray(st / lay.nsteps, jnp.float32)
                sstep[key] = jnp.asarray(st, jnp.int32)
                sYX[key] = (jnp.stack(Ys), jnp.stack(Xs))                    # (16, n)
        sub = self.cfg["sub"]

        def feats(params, st, key, sel):
            """Trunk features (B, 16, m, C) at the chosen sites, state as before each step."""
            l = level_of[key]
            Y = jnp.take_along_axis(jnp.broadcast_to(sYX[key][0], (sel.shape[0],) + sYX[key][0].shape), sel, 2)
            X = jnp.take_along_axis(jnp.broadcast_to(sYX[key][1], (sel.shape[0],) + sYX[key][1].shape), sel, 2)
            s = jnp.broadcast_to(sstep[key][None, :, None], sel.shape)

            def one(fin, y, x, s_):
                return net.patch_feat(params, fin, l, y.ravel(), x.ravel(), s_.ravel(), steps)
            f = jax.vmap(one)(st, Y, X, s)
            return f.reshape(sel.shape + (f.shape[-1],))

        def cfeat(f, key, tau, lam):
            ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
            ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
            sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
            return jnp.concatenate([jax.lax.stop_gradient(f), ex, sf], -1)

        def actor_loss(params, cparams, recs, G, tau, lam, gs, wgt, st, sels):
            total, fs = 0.0, {}
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    f = feats(params, st, key, sels[key])
                    fs[key] = jax.lax.stop_gradient(f)
                    b = net.head(params[key], f, emb_of[key], r["soft"], r["nv"], r["look"])
                    logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, -1)
                    q = jnp.exp(logq)
                    lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                    ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                    Q = jax.lax.stop_gradient(core.mlp(cparams[key], cfeat(f, key, tau, lam))) * gs[key]
                    t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                    la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    t2 = la * jax.lax.stop_gradient(G[key] - Qa)
                    lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], t1 + t2, 0.0))
                total = total + lev / nval[l]
            return total / recs[keys[0]]["a"].shape[0], fs

        def critic_loss(cparams, recs, G, tau, lam, gs, wgt, fs):
            total = 0.0
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    Q = core.mlp(cparams[key], cfeat(fs[key], key, tau, lam))
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0))
                total = total + lev / nval[l]
            return total / recs[keys[0]]["a"].shape[0]

        def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey):
            """Pass, cone returns of m sampled sites per (step, world) (weight
            n / m), actor and critic steps (lr = 0: untrained tau = 1 draws)."""
            st, recs = jax.vmap(lambda s, d: cc.conv_pass(lay, net, params, s, d, tau, lam))(seeds, des)
            viol = jnp.zeros(T + 1)
            for key, r in recs.items():
                nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
                viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
            viol = viol / seeds.shape[0]
            tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
            ii = cone.integral({k: r["cost"] for k, r in recs.items()})
            Bn = seeds.shape[0]
            G, wgt, rs, sels = {}, {}, {}, {}
            for key, r in recs.items():
                n = r["a"].shape[-1]
                if n <= sub:
                    sel = jnp.broadcast_to(jnp.arange(n), (Bn, core.NCOL, n))
                else:
                    rkey, k1 = jax.random.split(rkey)
                    sel = jnp.argsort(jax.random.uniform(k1, (Bn, core.NCOL, n)), -1)[..., :sub]
                    wgt[key] = n / sub
                sels[key] = sel
                rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                           for f, x in r.items()}
                ix = idx[key][jnp.arange(core.NCOL)[None, :, None], sel]
                v = jax.vmap(lambda a, i: a[i])(ii, ix)
                G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
            (loss, fs), g = jax.value_and_grad(actor_loss, has_aux=True)(
                params, cparams, rs, G, tau, lam, gs, wgt, st, sels)
            g = clip_by_level(g, levels, clip)
            params, opt = adam_step(params, g, opt, lr)
            closs, gc = jax.value_and_grad(critic_loss)(cparams, rs, G, tau, lam, gs, wgt, fs)
            gc = clip_by_level(gc, levels, clip)
            cparams, copt = adam_step(cparams, gc, copt, lrc)
            gstd = {k: jnp.std(v) for k, v in G.items()}
            return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd

        return {"fused": jax.jit(fused),
                "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s))),
                "pass_b": jax.jit(jax.vmap(lambda p, s, d, tau, lam: cc.conv_pass(lay, net, p, s, d, tau, lam),
                                           in_axes=(None, 0, 0, None, None))),
                "feats": feats, "topkeys": [core.key_of(T, c.name) for c in lay.chans[T]],
                "nval": nval, "cone": cone}

    def _measure_baseline(self, lay, log):
        """cfast's, with the critic scales keyed by channel only."""
        T, nb = self.problem.T, self.cfg["B"]
        region = Region(lay.W)
        seeds = np.arange(500_000 + 1000 * self.seed, 500_000 + 1000 * self.seed + nb)
        des = self._batch(seeds, region)
        lam = jnp.asarray(self.lam, jnp.float32)
        js = jnp.asarray(seeds, jnp.int32)
        ones = {k: jnp.float32(1.0) for k in self.cparams}
        out = self.fn["fused"](self.params, self.cparams, self.opt, self.copt, ones, js, des,
                               jnp.float32(1.0), lam, 0.0, 0.0, self.cfg["clip"], jax.random.PRNGKey(0))
        st, gstd = out[0], out[-1]
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

    # ------------------------------------------------------------ generation
    def generate(self, seed, region):
        if region.periodic:
            lay = self.layout(region)
            des = self.designed(seed, region)
            raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
            fin = lay.jit_repair(raw, seed)
            crop = lambda st, c: np.asarray(st[c.name])
        else:
            m = self.margin()
            ext = Region(region.W + 2 * m, periodic=False, x0=region.x0 - m)
            lay = self.layout(ext)
            des = self.designed(seed, ext)
            raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
            fin = lay.jit_repair(raw, seed)
            crop = lambda st, c: np.asarray(st[c.name])[:, (m >> c.level):(m >> c.level) + (region.W >> c.level)]
        out = {}
        for kind, st in (("raw", raw), ("final", fin)):
            out[kind] = {c.name: crop(st, c) for c in self.problem.sampled()}
        return out

    def info(self):
        return {**super().info(), "margin": self.margin()}


@register
class CConvB16(CConv):
    """cconv at B = 16: B = 64 needs 7+ GB on ground (OOM on the shared GPU)."""
    name = "cconv_b16"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{"B": 16, **cfg})
