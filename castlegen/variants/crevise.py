"""crevise (ideas.md #19; constitution Amendment 2026-10-14 (b)): cfast's
coloured pass followed by R fixed revising steps per level, trained by the
path KL.  Mechanics and the per-draw cost decomposition: _crevise_core.py.

Generation: per level the pass, then R revising steps (each: erase every
(channel, site) independently with probability rho by a coordinate-keyed
coin, redraw the erased cells colour by colour with lookahead), then cpass's
one repair round.  Finite support: (1 + R) sweeps per level (`margin`).

Training (K = 0 only): cfast's fused update on the path-KL costs: pass, cone
returns over the pass + revising step list (cmulti's MCone), subsampled loss
(sub = m sites per step and world), all-action + score-function terms with
the h1 critic (one critic per (level, channel, step) group) as control
variate, lambda control on the violations of the FINAL state (the telescoped
dv), tau 2 -> 1 over the first 20 %.  One actor per (level, channel) shared by
the pass and its revising steps (share=True; the known flags in the window
tell the steps apart), or one per group (share=False).  R = 0 is cfast: same
draws, same costs, same update (tests/variants/test_crevise.py).

score() returns None: log q(x_0) of a revised draw is intractable (the bench
then reports F and the probe metrics as missing).  info() reports the bound
F_ub = mean per fine cell of S(x_0) + log Q(traj) + |M| log D at tau = 1,
lam = 0 on fresh worlds (an upper bound on the bench's free energy) and its
parts.  Objective stated in every log entry: "path_kl".
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cpass_core as core
from ._cln_core import hidden1
from ._crevise_core import RCone, Sched, check_colouring, margin, revise_pass, steps_per_level
from .cfast import CFast
from .cpass import adam_init, adam_step, clip_by_level

CFG = dict(R=2, rho=0.5, share=True, guide=True)


@register
class CRevise(CFast):
    name = "crevise"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CFG, **cfg})
        assert self.cfg["K"] == 0, "crevise: coupled replays (K > 0) are not supported"
        check_colouring(problem)
        self.R = steps_per_level(problem, self.cfg["R"])
        lay = self.layout(Region(self.cfg["W_train"] or problem.W_train))
        sch = self.sched(lay)
        n_extra = 1 + (problem.T + 1) + 1
        key = jax.random.PRNGKey(seed + 15485863)
        for k, (l, ci, cn, m, kind) in sch.groups.items():
            if m == 0:
                continue
            D = problem.chan[cn].D
            key, k1, k2 = jax.random.split(key, 3)
            if not self.cfg["share"] or self.cfg["guide"]:
                din = lay.feat_dim(l, lay.actor_win) + (sch.field_dim(l) if self.cfg["guide"] else 0)
                self.params[k] = core.mlp_init(k1, din, D, self.cfg["width"])
            self.cparams[k] = core.mlp_init(k2, self.cfg["width"] + n_extra, D, self.cfg["width"])
        self.opt, self.copt = adam_init(self.params), adam_init(self.cparams)

    def sched(self, lay):
        key = tuple(sorted(self.R.items()))
        if not hasattr(lay, "crevise_sched"):
            lay.crevise_sched = {}
        if key not in lay.crevise_sched:
            lay.crevise_sched[key] = Sched(lay, self.cfg["R"])
        return lay.crevise_sched[key]

    def train_fns(self, lay):
        key = ("crevise", tuple(sorted(self.R.items())), float(self.cfg["rho"]), bool(self.cfg["share"]),
               bool(self.cfg["guide"]), self.cfg["sub"], self.cfg.get("net", "tanh"))
        if not hasattr(lay, "cfast_fns"):
            lay.cfast_fns = {}
        if key not in lay.cfast_fns:
            lay.cfast_fns[key] = self._build_train(lay)
        return lay.cfast_fns[key]

    # -------------------------------------------------------------- training
    def _build_train(self, lay):
        T = self.problem.T
        levels = list(range(T + 1))
        sch = self.sched(lay)
        dkeys = sch.draw_keys()
        level_of = {k: g[0] for k, g in sch.groups.items()}
        logD = {k: float(np.log(self.problem.chan[g[2]].D)) if g[3] > 0 else 0.0 for k, g in sch.groups.items()}
        rho, share, guide = float(self.cfg["rho"]), bool(self.cfg["share"]), bool(self.cfg["guide"])
        nval = {}
        for l in levels:
            cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(core.NCOL))
            nval[l] = max(cnt * len(lay.chans[l]) * (1.0 + self.R[l] * rho), 1.0)   # expected draws per level
        cone = RCone(sch, lay.actor_win, guide)
        idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
        sfrac = {k: jnp.asarray([sch.step(k, kk) / sch.nsteps for kk in range(core.NCOL)], jnp.float32)
                 for k in dkeys}
        sub = self.cfg["sub"]

        def cfeat(r, key, tau, lam, params):
            f = jax.lax.stop_gradient(hidden1(params[sch.actor_key(key, share, guide)], r["feat"]))
            ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
            ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
            sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
            return jnp.concatenate([f, ex, sf], -1)

        def actor_loss(params, cparams, recs, G, tau, lam, gs, p0, wgt):
            total = 0.0
            for key in dkeys:
                l = level_of[key]
                r = recs[key]
                b = core.mlp(params[sch.actor_key(key, share, guide)], r["feat"])
                logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, -1)
                q = jnp.exp(logq)
                lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                # revising steps: the cost of each candidate in the hybrid state, relative to the old value
                # (soft_c, nv_c; the old value's terms and log D are constant over the candidates)
                ell = r["soft_c"] + lam[l] * r["nv_c"] + tau * (lq_sg + logD[key])
                Q = jax.lax.stop_gradient(core.mlp(cparams[key], cfeat(r, key, tau, lam, p0))) * gs.get(key, 1.0)
                t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                t2 = la * jax.lax.stop_gradient(G[key] - Qa)
                total = total + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], t1 + t2, 0.0)) / nval[l]
            return total / recs[dkeys[0]]["a"].shape[0]

        def critic_loss(cparams, recs, G, tau, lam, gs, p0, wgt):
            total = 0.0
            for key in dkeys:
                r = recs[key]
                Q = core.mlp(cparams[key], cfeat(r, key, tau, lam, p0))
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                total = total + wgt.get(key, 1.0) * jnp.sum(
                    jnp.where(r["valid"], (Qa - G[key] / gs.get(key, 1.0)) ** 2, 0.0)) / nval[level_of[key]]
            return total / recs[dkeys[0]]["a"].shape[0]

        def passes(params, seeds, des, tau, lam):
            st, recs = jax.vmap(lambda s, d: revise_pass(sch, params, s, d, tau, lam, rho, share, guide=guide))(seeds, des)
            viol = jnp.zeros(T + 1)
            for key in dkeys:
                viol = viol.at[level_of[key]].add(jnp.sum(recs[key]["dv"]))
            viol = viol / seeds.shape[0]                               # H_l(x_0) per world (telescoped)
            tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
            return st, recs, viol, tot

        def rec_fn(params, seeds, des, tau, lam):
            st, recs, viol, tot = passes(params, seeds, des, tau, lam)
            G = cone.returns({k: r["cost"] for k, r in recs.items()})
            return st, recs, G, viol, tot

        def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rk):
            """cfast's fused update (same signature and outputs) on the path-KL costs."""
            st, recs, viol, tot = passes(params, seeds, des, tau, lam)
            ii = cone.integral({k: r["cost"] for k, r in recs.items()})
            Bn = seeds.shape[0]
            G, wgt, rs = {}, {}, {}
            for key, r in recs.items():
                n = r["valid"].shape[-1]
                if not sub or n <= sub:
                    sel = jnp.broadcast_to(jnp.arange(n), (Bn, core.NCOL, n))
                else:
                    rk, k1 = jax.random.split(rk)
                    sel = jnp.argsort(jax.random.uniform(k1, (Bn, core.NCOL, n)), -1)[..., :sub]
                    wgt[key] = n / sub
                rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                           for f, x in r.items()}
                ix = idx[key][jnp.arange(core.NCOL)[None, :, None], sel]
                v = jax.vmap(lambda a, i: a[i])(ii, ix)
                G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
            p0 = params
            loss, g = jax.value_and_grad(actor_loss)(params, cparams, rs, G, tau, lam, gs, p0, wgt)
            g = clip_by_level(g, levels, clip)
            params, opt = adam_step(params, g, opt, lr)
            closs, gc = jax.value_and_grad(critic_loss)(cparams, rs, G, tau, lam, gs, p0, wgt)
            gc = clip_by_level(gc, levels, clip)
            cparams, copt = adam_step(cparams, gc, copt, lrc)
            gstd = {k: jnp.std(v) for k, v in G.items()}
            return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd

        def actor_grad(params, cparams, recs, G, tau, lam, gs):
            return jax.grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, params, {})

        def evaluate(params, seeds, des, lam):
            """tau = 1: final states, per-world path-KL integrand (lam = 0: F_ub * cells),
            log Q(traj), the masked-cell log D term, violations H_l(x_0)."""
            st, recs = jax.vmap(lambda s, d: revise_pass(sch, params, s, d, 1.0, lam, rho, share, guide=guide))(seeds, des)
            tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
            lq = sum(jnp.sum(jnp.where(recs[k]["valid"], jnp.take_along_axis(
                recs[k]["logq"], recs[k]["a"][..., None], -1)[..., 0], 0.0), (1, 2)) for k in dkeys)
            ld = sum(logD[k] * jnp.sum(recs[k]["valid"], (1, 2)) for k in dkeys)
            viol = jnp.stack([sum((jnp.sum(recs[k]["dv"], (1, 2)) for k in dkeys if level_of[k] == l),
                                  jnp.zeros(seeds.shape[0]))
                              for l in levels], -1)
            return st, tot, lq, ld, viol

        return {"fused": jax.jit(fused), "rec": jax.jit(rec_fn), "actor_grad": jax.jit(actor_grad),
                "evaluate": jax.jit(evaluate), "cone": cone, "sched": sch, "nval": nval, "level_of": level_of,
                "topkeys": [], "logD": logD,
                "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s)))}

    def train(self, budget_s, log):
        wrapped = lambda d: log({**d, "objective": "path_kl", "R": self.cfg["R"], "rho": self.cfg["rho"],
                                 "guide": self.cfg["guide"]} if "update" in d else d)
        super().train(budget_s, wrapped)

    # ------------------------------------------------------------ generation
    def _gen_fn(self, lay):
        key = ("crevise", tuple(sorted(self.R.items())), float(self.cfg["rho"]), bool(self.cfg["share"]),
               bool(self.cfg["guide"]))
        if not hasattr(lay, "crevise_gen"):
            lay.crevise_gen = {}
        if key not in lay.crevise_gen:
            sch = Sched(lay, self.cfg["R"])
            rho, share, guide = float(self.cfg["rho"]), bool(self.cfg["share"]), bool(self.cfg["guide"])

            def gen(params, seed, des, lam):
                st = revise_pass(sch, params, seed, des, 1.0, lam, rho, share, record=False, guide=guide)[0]
                return st, core.repair(lay, st, seed)
            lay.crevise_gen[key] = jax.jit(gen)
        return lay.crevise_gen[key]

    def generate(self, seed, region):
        if region.periodic:
            lay = self.layout(region)
            des = self.designed(seed, region)
            crop = lambda st, c: np.asarray(st[c.name])
        else:
            m = margin(self.problem, self.cfg["R"], self.cfg["guide"])
            ext = Region(region.W + 2 * m, periodic=False, x0=region.x0 - m)
            lay = self.layout(ext)
            des = self.designed(seed, ext)
            crop = lambda st, c: np.asarray(st[c.name])[:, (m >> c.level):(m >> c.level) + (region.W >> c.level)]
        raw, fin = self._gen_fn(lay)(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
        return {kind: {c.name: crop(st, c) for c in self.problem.sampled()} for kind, st in (("raw", raw), ("final", fin))}

    def score(self, x, region):
        if all(r == 0 for r in self.R.values()):
            return super().score(x, region)          # no revisions: log q(x) is cpass's
        return None

    def score_sites(self, x, region):
        if all(r == 0 for r in self.R.values()):
            return super().score_sites(x, region)
        return None

    def bound_report(self, n=16):
        """F_ub (tau = 1, lam = 0): mean per fine cell of S(x_0) + log Q(traj) + |M| log D,
        an upper bound on the bench's free energy, and its parts."""
        if not hasattr(self, "fn"):
            return None
        region = Region(self.cfg["W_train"] or self.problem.W_train)
        cells = self.problem.H * region.W
        seeds = np.arange(3_000_000 + 1000 * self.seed, 3_000_000 + 1000 * self.seed + n)
        des = self._batch(seeds, region)
        lam0 = jnp.zeros(self.problem.T + 1, jnp.float32)
        st, tot, lq, ld, viol = self.fn["evaluate"](self.params, jnp.asarray(seeds, jnp.int32), des, lam0)
        st = jax.device_get(st)
        S = []
        for b in range(n):
            d = {k: np.asarray(v[b]) for k, v in des.items()}
            xs = {c.name: np.asarray(st[c.name][b]) for c in self.problem.sampled()}
            S.append(en.energy(self.problem, {**d, **xs}, region).soft)
        tot, lq, ld = np.asarray(tot), np.asarray(lq), np.asarray(ld)
        return {"F_ub": float(np.mean(tot)) / cells, "soft": float(np.mean(S)) / cells,
                "logq_traj": float(np.mean(lq)) / cells, "masked_logD": float(np.mean(ld)) / cells,
                "telescope_err": float(np.mean(tot - (np.asarray(S) + lq + ld))),
                "viol_final": np.asarray(viol).mean(0).tolist()}

    def info(self):
        return {**super().info(), "objective": "path_kl", "R": self.R, "rho": self.cfg["rho"],
                "margin": margin(self.problem, self.cfg["R"], self.cfg["guide"]), "guide": self.cfg["guide"],
                "bound": self.bound_report()}


def _variant(name, **over):
    cls = type(f"CRevise_{name}", (CRevise,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("crevise_r0", R=0)                     # = cfast through crevise's code (sanity)
_variant("crevise_noguide", guide=False)        # fixed-rate revisions without the violation-field features (first run)
_variant("crevise_r1", R=1)                     # one revising step per level
_variant("crevise_sep_noguide", share=False, guide=False)   # one actor per (level, channel, step), no field
