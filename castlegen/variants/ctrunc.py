"""ctrunc: cpass's generator and objective, trained from truncated-depth,
bootstrapped returns (ideas.md #2; the user's approved exception to
constitution constraint 3: the gradient is biased).

Generation is cpass's: the coloured pass at tau = 1, then one repair round
(inherited unchanged).  Only the training signal differs.

Training (one update) is a chain, coarse to fine.  Level T is drawn for B_T
worlds; level l for the first B_l of the worlds that drew level l + 1
(B_l non-increasing toward the fine level: coarse levels get many cheap,
on-policy worlds).  Every draw is scored by the current model; start states
are the model's own coarse draws.

Return of a draw at level l, colour k, channel c (truncated at depth d):
    G = box_R( later same-level costs + U_l )  at the draw's site,
    U_l = per-level-l-cell downstream cost: the realised costs of levels
          l-1 .. l-d+1 (summed into their level-l ancestor) where the world
          continued, plus v_{l-d}'s estimate below that (v of the level where
          the world or the depth stops).
box_R sums a (2R + 1)^2 box of level-l cells around the site: the return is
local as well as truncated.  Actor loss and critic as in cpass (all-action
term with the critic Q, plus the score-function correction with G - Q(a)); no
coupled replays.

Value v_m (per level-(m + 1) cell, _ctrunc_core) is regressed on the same
d-step target: the realised level-m cost of the cell's children plus U_m,
i.e. bootstrapped from v_{m-d}.  d = 1 is pure one-level bootstrapping; d = T + 1
uses realised costs all the way down for worlds that reach level 0."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cpass_core as core
from . import _ctrunc_core as tc
from .cpass import CFG as CPASS_CFG
from .cpass import CPass, adam_init, adam_step

CFG = {**CPASS_CFG, "depth": 1, "R": 3, "cells": 2048, "Bmax": 64, "lr_value": 3e-3, "lr_critic": 3e-3}


def clip_tree(g, c):
    n = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g)))
    s = jnp.minimum(1.0, c / (n + 1e-12))
    return jax.tree_util.tree_map(lambda x: x * s, g)


@register
class CTrunc(CPass):
    name = "ctrunc"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CFG, **cfg})
        T = problem.T
        lay = self.layout(Region(problem.W_train))
        key = jax.random.PRNGKey(10_000 + seed)
        self.vparams = {}
        for m in range(T):
            key, k1 = jax.random.split(key)
            self.vparams[tc.vkey(m)] = core.mlp_init(k1, lay.feat_dim(m + 1, tc.VALUE_WIN, T + 2), 1,
                                                     self.cfg["width"])
        self.vopt = adam_init(self.vparams)
        c = self.cfg
        # worlds per level: about c["cells"] level-l cells per update, at most Bmax
        # (non-increasing toward the fine level, as the chain needs)
        self.B = {l: int(np.clip(c["cells"] // int(np.prod(lay.hw[l])), 1, c["Bmax"])) for l in range(T + 1)}
        for l in range(1, T + 1):
            self.B[l] = max(self.B[l], self.B[l - 1])
        self.gs = None                     # per-level return scale (critic), fixed after the first batch
        self.vs = None                     # per-level value scale, fixed after the first batch
        self.vcheck = []

    # ------------------------------------------------------------ build
    def _build_trunc(self, lay):
        T, R = self.problem.T, self.cfg["R"]
        levels = list(range(T + 1))
        nval = {}
        for l in levels:
            _, _, V = tc.site_index(lay, l)
            nval[l] = int(V.sum()) * len(lay.chans[l])
        lay.nval = nval

        def fwd(l):
            def f(params, vparams, seed, st, tau, lam, extra, vs):
                st, recs = tc.level_pass(lay, params, seed, st, tau, lam, l, extra)
                C = tc.cost_grid(lay, l, recs, tau, lam[l])
                Vm = (tc.value_map(lay, vparams[tc.vkey(l - 1)], st, l - 1, extra, vs[l - 1]) if l > 0
                      else jnp.zeros(lay.hw[l]))
                return st, recs, C, Vm
            return jax.jit(jax.vmap(f, in_axes=(None, None, 0, 0, None, None, None, None)))

        def ret(l):
            def f(C, U):
                A = jnp.sum(C, 1, keepdims=True) - jnp.cumsum(C, 1) + U[:, None]   # strictly later + below
                G = jax.vmap(lambda a: tc.gather_steps(lay, l, a))(tc.box(A, R))
                y = tc.down(jnp.sum(C, 1) + U) if l < T else None
                return G, y
            return jax.jit(f)

        def loss_l(l):
            keys = [core.key_of(l, c.name) for c in lay.chans[l]]

            def actor_loss(pa, pc, recs, G, tau, lam, gs):
                tot = 0.0
                for key in keys:
                    r = recs[key]
                    b = core.mlp(pa[key], r["feat"])
                    logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, -1)
                    q = jnp.exp(logq)
                    lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                    ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                    Q = jax.lax.stop_gradient(core.mlp(pc[key], r["cfeat"])) * gs
                    t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                    la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    t2 = la * jax.lax.stop_gradient(G[key] - Qa)
                    tot = tot + jnp.sum(jnp.where(r["valid"], t1 + t2, 0.0))
                return tot / nval[l] / r["a"].shape[0]

            def critic_loss(pc, recs, G, gs):
                tot = 0.0
                for key in keys:
                    r = recs[key]
                    Q = core.mlp(pc[key], r["cfeat"])
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    tot = tot + jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs) ** 2, 0.0))
                return tot / nval[l] / r["a"].shape[0]

            def value_loss(pv, st_in, y, extra, vs):
                pred = jax.vmap(lambda s: tc.value_map(lay, pv, s, l, extra, 1.0))(st_in)
                return jnp.mean((pred - y / vs) ** 2)

            def f(pa, pc, pv, recs, G, y, st_in, tau, lam, extra, gs, vs):
                la, ga = jax.value_and_grad(actor_loss)(pa, pc, recs, G, tau, lam, gs)
                lc, gc = jax.value_and_grad(critic_loss)(pc, recs, G, gs)
                ga, gc = clip_tree(ga, self.cfg["clip"]), clip_tree(gc, self.cfg["clip"])
                if pv is None:
                    return ga, gc, None, la, lc, 0.0
                lv, gv = jax.value_and_grad(value_loss)(pv, st_in, y, extra, vs)
                return ga, gc, clip_tree(gv, self.cfg["clip"]), la, lc, lv
            return jax.jit(f), keys

        lay.fwd = {l: fwd(l) for l in levels}
        lay.ret = {l: ret(l) for l in levels}
        lay.loss = {l: loss_l(l) for l in levels}

        def apply(params, cparams, vparams, opt, copt, vopt, g, gc, gv, lr, lrc, lrv):
            params, opt = adam_step(params, g, opt, lr)
            cparams, copt = adam_step(cparams, gc, copt, lrc)
            if vparams:
                vparams, vopt = adam_step(vparams, gv, vopt, lrv)
            return params, cparams, vparams, opt, copt, vopt
        lay.apply = jax.jit(apply)

    # ------------------------------------------------------------ chain
    def _chain(self, lay, seeds, tau, lam, extra, vs):
        """Forward chain coarse to fine; returns per-level records and maps."""
        T, region = self.problem.T, Region(self.problem.W_train)
        BT = self.B[T]
        desl = [self.designed(int(s), region) for s in seeds[:BT]]
        st = {}
        for c in self.problem.chans:
            if c.designed is not None:
                st[c.name] = jnp.stack([d[c.name] for d in desl])
            else:
                st[c.name] = jnp.full((BT,) + lay.hw[c.level], -1, jnp.int32)
        out = {}
        for l in range(T, -1, -1):
            Bl = self.B[l]
            st_in = {k: v[:Bl] for k, v in st.items()}
            if not lay.chans[l]:
                out[l] = None
                st = st_in
                continue
            st, recs, C, Vm = lay.fwd[l](self.params, self.vparams, jnp.asarray(seeds[:Bl], jnp.int32), st_in,
                                         tau, lam, extra, vs)
            out[l] = {"st_in": st_in, "st_out": st, "recs": recs, "C": C, "Vm": Vm}
        return out

    def _downstream(self, lay, out, depth):
        """U_l (B_l, h_l, w_l): truncated downstream cost below each level-l cell."""
        T = self.problem.T
        U = {}
        for l in range(T + 1):
            if l == 0:
                U[0] = [jnp.zeros((self.B[0],) + lay.hw[0])] * depth
                continue
            Vm = out[l]["Vm"]
            Ul = [Vm]
            for j in range(1, depth):
                nb = self.B[l - 1]
                below = tc.down(jnp.sum(out[l - 1]["C"], 1) + U[l - 1][j - 1])
                Ul.append(jnp.concatenate([below, Vm[nb:]], 0))
            U[l] = Ul
        return {l: U[l][depth - 1] for l in U}

    # ------------------------------------------------------------ train
    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(self.problem.W_train)
        lay = self.layout(region)
        self._measure_baseline(lay, log)
        self._build_trunc(lay)
        depth = int(min(cfg["depth"], T + 1))
        cells = self.problem.H * self.problem.W_train
        while True:
            frac = (time.time() - t0) / max(budget_s, 1e-9)
            if frac >= 1.0:
                break
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - frac / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            extra = jnp.concatenate([jnp.array([self.tau]), lam]).astype(jnp.float32)
            nb = self.B[T]
            seeds = np.arange(self.train_seed, self.train_seed + nb)
            self.train_seed += nb
            vs = jnp.asarray(self.vs if self.vs is not None else np.ones(T + 1), jnp.float32)
            out = self._chain(lay, seeds, tau, lam, extra, vs)
            U = self._downstream(lay, out, depth)
            rets = {l: lay.ret[l](out[l]["C"], U[l]) for l in range(T + 1) if out[l] is not None}
            if self.gs is None:
                # calibration (first batch only): value scale and output offset from the
                # realised per-fine-cell cost below each level, then the critic's from the
                # returns that the calibrated values give; the chain is redrawn after it
                n0 = self.B[0]
                pc = np.array([float(jnp.sum(out[l]["C"][:n0])) / n0 / cells if out[l] is not None else 0.0
                               for l in range(T + 1)])
                self.vs = np.ones(T + 1)
                for m_ in range(T):
                    cm = pc[:m_ + 1].sum() * 4 ** (m_ + 1)
                    self.vs[m_] = max(1.0, abs(cm))
                    vp = dict(self.vparams[tc.vkey(m_)])
                    vp["b3"] = jnp.full_like(vp["b3"], cm / self.vs[m_])
                    self.vparams[tc.vkey(m_)] = vp
                vs = jnp.asarray(self.vs, jnp.float32)
                out = self._chain(lay, seeds, tau, lam, extra, vs)
                U = self._downstream(lay, out, depth)
                rets = {l: lay.ret[l](out[l]["C"], U[l]) for l in range(T + 1) if out[l] is not None}
                self.gs = np.ones(T + 1)
                for l in rets:
                    g = np.concatenate([np.asarray(x).ravel() for x in rets[l][0].values()])
                    self.gs[l] = max(1.0, float(np.std(g)), abs(float(np.mean(g))))
                    for key in lay.loss[l][1]:
                        cp = dict(self.cparams[key])
                        cp["b3"] = jnp.full_like(cp["b3"], float(np.mean(g)) / self.gs[l])
                        self.cparams[key] = cp
                self.opt, self.copt, self.vopt = adam_init(self.params), adam_init(self.cparams), adam_init(
                    self.vparams)
            ga, gc, gv = {}, {}, {}
            losses = np.zeros((T + 1, 3))
            for l, (G, y) in rets.items():
                f, keys = lay.loss[l]
                pa = {k: self.params[k] for k in keys}
                pc = {k: self.cparams[k] for k in keys}
                pv = self.vparams.get(tc.vkey(l)) if l < T else None
                a, c, v, la, lc, lv = f(pa, pc, pv, out[l]["recs"], G, y, out[l]["st_in"], tau, lam, extra,
                                        jnp.float32(self.gs[l]), jnp.float32(self.vs[l]))
                ga.update(a)
                gc.update(c)
                if v is not None:
                    gv[tc.vkey(l)] = v
                losses[l] = [float(la), float(lc), float(lv)]
            for k, v in self.vparams.items():
                gv.setdefault(k, jax.tree_util.tree_map(jnp.zeros_like, v))
            self.params, self.cparams, self.vparams, self.opt, self.copt, self.vopt = lay.apply(
                self.params, self.cparams, self.vparams, self.opt, self.copt, self.vopt, ga, gc, gv,
                cfg["lr"], cfg["lr_critic"], cfg["lr_value"])
            # penalties: raw violated fraction per level (each level's own batch)
            viol = np.zeros(T + 1)
            for l in range(T + 1):
                if out[l] is None:
                    continue
                for c in lay.chans[l]:
                    r = out[l]["recs"][core.key_of(l, c.name)]
                    nv_a = np.take_along_axis(np.asarray(r["nv"]), np.asarray(r["a"])[..., None], -1)[..., 0]
                    viol[l] += float((nv_a * np.asarray(r["valid"])).sum()) / self.B[l]
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            # value check on the worlds that reached level 0: V's prediction (made before
            # this update) against the realised cost of every level below, per world
            n0 = self.B[0]
            lev_tot = {l: np.asarray(jnp.sum(out[l]["C"][:n0], (1, 2, 3))) if out[l] is not None else np.zeros(n0)
                       for l in range(T + 1)}
            vchk = {}
            for l in range(1, T + 1):
                if out[l] is None:
                    continue
                pred = np.asarray(jnp.sum(out[l]["Vm"][:n0], (1, 2)))
                real = sum(lev_tot[m] for m in range(l))
                vchk[str(l)] = [pred.tolist(), real.tolist()]
            log({"t": time.time() - t0, "update": self.updates, "tau": self.tau,
                 "loss": float(losses[:, 0].sum()), "critic_loss": float(losses[:, 1].sum()),
                 "value_loss": losses[:, 2].tolist(),
                 "cost_per_world": float(sum(lev_tot[l].mean() for l in lev_tot)),
                 "raw_viol": hf.tolist(), "lam": self.lam.tolist(), "vcheck": vchk})
            self.vcheck.append(vchk)
        self.cells = cells

    def info(self):
        d = super().info()
        d.update({"B": self.B, "gs": None if self.gs is None else self.gs.tolist(),
                  "vs": None if self.vs is None else self.vs.tolist()})
        return d


@register
class CTruncFull(CTrunc):
    """Full-depth control: realised costs all the way down where a world reaches
    level 0 (still local, still no replays)."""
    name = "ctrunc_full"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{"depth": 99, **cfg})


@register
class CTrunc2(CTrunc):
    name = "ctrunc_d2"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{"depth": 2, **cfg})
