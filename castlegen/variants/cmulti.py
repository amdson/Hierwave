"""cmulti: cfast with P coloured passes per level (ideas.md #3).

Generation: at every level (coarse to fine) pass 1 is cfast's coloured pass;
passes 2..P sweep the 16 colours again and redraw every site given all its
current neighbours, each (level, channel, pass) with its own actor; then
cpass's one repair round.  Fixed work: P sweeps per level.

Training objective (needs the user's approval: constitution item 4).  log q
of the final draw x^P is intractable once sites are redrawn, so cmulti
minimises the auxiliary-variable upper bound

    J = E_q[ S(x^P) + sum_l lam_l c_l(x^P) + tau (log q(x^1..x^P) - log r(x^{<P} | x^P)) ]
      = (stated objective) + tau E_q KL( q(x^{<P} | x^P) || r(x^{<P} | x^P) )

with a learned, per-site factorised backward kernel r (_cmulti_core).  Both
the actors and r descend J (r by maximum likelihood on the recorded earlier
values, which is J's exact r-gradient).  Estimator as cfast: per-draw costs
that telescope to J's integrand, dependency-cone returns, h1 critic,
subsampled loss (unbiased for J).  With P = 1 it is cfast exactly.

score() returns None: log q(x) of a final draw is not available.  info()
reports J's integrand at tau = 1, lam = 0 per fine cell on fresh worlds
(F_ub, an upper bound on the bench's free energy) and its parts."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cpass_core as core
from ._cmulti_core import MCone, Sched, check_colouring, gkey, margin, multi_pass, rkey
from .cfast import CFast
from .cpass import adam_init, adam_step, clip_by_level

CFG = dict(P=2, lr_r=3e-3)


@register
class CMulti(CFast):
    name = "cmulti"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CFG, **cfg})
        check_colouring(problem)
        P = self.cfg["P"]
        lay = self.layout(Region(self.cfg["W_train"] or problem.W_train))
        sch = self.sched(lay)
        n_extra = 1 + (problem.T + 1) + 1
        key = jax.random.PRNGKey(seed + 104729)
        self.rparams = {}
        for k, (l, ci, cn, p, kind) in sch.groups.items():
            if p == 1:
                continue
            D = problem.chan[cn].D
            key, k1, k2 = jax.random.split(key, 3)
            if kind == "draw":
                self.params[k] = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), D, self.cfg["width"])
                self.cparams[k] = core.mlp_init(k2, self.cfg["width"] + n_extra, D, self.cfg["width"])
            else:
                self.rparams[k] = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), D, self.cfg["width"])
        self.opt, self.copt, self.ropt = adam_init(self.params), adam_init(self.cparams), adam_init(self.rparams)
        assert P >= 1

    def sched(self, lay):
        P = self.cfg["P"]
        if not hasattr(lay, "cmulti_sched"):
            lay.cmulti_sched = {}
        if P not in lay.cmulti_sched:
            lay.cmulti_sched[P] = Sched(lay, P)
        return lay.cmulti_sched[P]

    def train_fns(self, lay):
        key = ("cmulti", self.cfg["P"], self.cfg["sub"])
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
        dkeys, rkeys = sch.draw_keys(), sch.r_keys()
        level_of = {k: g[0] for k, g in sch.groups.items()}
        nval = {}
        for l in levels:
            cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(core.NCOL))
            nval[l] = max(cnt * len(lay.chans[l]), 1)
        cone = MCone(sch, core.ACTOR_WIN)
        idx = {k: jnp.asarray(v) for k, v in cone.idx.items() if k in dkeys}
        sfrac = {k: jnp.asarray([sch.step(k, kk) / sch.nsteps for kk in range(core.NCOL)], jnp.float32)
                 for k in dkeys}
        sub = self.cfg["sub"]

        def cfeat(r, key, tau, lam, params):
            p = params[key]
            f = jax.lax.stop_gradient(jnp.tanh(r["feat"] @ p["W1"] + p["b1"]))
            ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
            ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
            sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
            return jnp.concatenate([f, ex, sf], -1)

        def actor_loss(params, cparams, recs, G, tau, lam, gs, p0, wgt):
            total = 0.0
            for key in dkeys:
                l = level_of[key]
                r = recs[key]
                b = core.mlp(params[key], r["feat"])
                logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, -1)
                q = jnp.exp(logq)
                lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg       # the old value's terms are constant in a
                Q = jax.lax.stop_gradient(core.mlp(cparams[key], cfeat(r, key, tau, lam, p0))) * gs[key]
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
                    jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0)) / nval[level_of[key]]
            return total / recs[dkeys[0]]["a"].shape[0]

        def r_loss(rparams, recs, wgt):
            total = 0.0
            for key in rkeys:
                r = recs[key]
                lr = jax.nn.log_softmax(core.mlp(rparams[key], r["feat"]), -1)
                lo = jnp.take_along_axis(lr, r["o"][..., None], -1)[..., 0]
                total = total - wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], lo, 0.0)) / nval[level_of[key]]
            return total / recs[dkeys[0]]["a"].shape[0]

        def fused(params, cparams, rparams, opt, copt, ropt, gs, seeds, des, tau, lam, lr, lrc, lrr, clip, rk):
            st, recs = jax.vmap(lambda s, d: multi_pass(sch, params, rparams, s, d, tau, lam))(seeds, des)
            viol = jnp.zeros(T + 1)
            for key in dkeys:
                viol = viol.at[level_of[key]].add(jnp.sum(recs[key]["dv"]))
            viol = viol / seeds.shape[0]
            tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
            rterm = sum(jnp.sum(recs[k]["cost"], (1, 2)) for k in rkeys) if rkeys else jnp.zeros(seeds.shape[0])
            ii = cone.integral({k: r["cost"] for k, r in recs.items()})
            Bn = seeds.shape[0]
            G, wgt, rs = {}, {}, {}
            for key, r in recs.items():
                n = r["valid"].shape[-1]
                if n <= sub:
                    sel = jnp.broadcast_to(jnp.arange(n), (Bn, core.NCOL, n))
                else:
                    rk, k1 = jax.random.split(rk)
                    sel = jnp.argsort(jax.random.uniform(k1, (Bn, core.NCOL, n)), -1)[..., :sub]
                    wgt[key] = n / sub
                rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                           for f, x in r.items()}
                if key in idx:
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
            if rkeys:
                rloss, gr = jax.value_and_grad(r_loss)(rparams, rs, wgt)
                gr = clip_by_level(gr, levels, clip)
                rparams, ropt = adam_step(rparams, gr, ropt, lrr)
            else:
                rloss = jnp.float32(0.0)
            gstd = {k: jnp.std(v) for k, v in G.items()}
            return st, params, cparams, rparams, opt, copt, ropt, loss, closs, rloss, viol, tot, rterm, gstd

        def evaluate(params, rparams, seeds, des, lam):
            """tau = 1: final states, per-world J integrand (lam = 0 gives F_ub
            * cells), r term, log q(trajectory), changed fraction per pass."""
            st, recs = jax.vmap(lambda s, d: multi_pass(sch, params, rparams, s, d, 1.0, lam))(seeds, des)
            tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
            rterm = sum(jnp.sum(recs[k]["cost"], (1, 2)) for k in rkeys) if rkeys else jnp.zeros(seeds.shape[0])
            lq = sum(jnp.sum(jnp.where(recs[k]["valid"], jnp.take_along_axis(
                recs[k]["logq"], recs[k]["a"][..., None], -1)[..., 0], 0.0), (1, 2)) for k in dkeys)
            return st, tot, rterm, lq

        return {"fused": jax.jit(fused), "evaluate": jax.jit(evaluate), "cone": cone, "sched": sch,
                "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s)))}

    def _call_fused(self, gs, js, des, tau, lam, lr, lrc, lrr, rk):
        out = self.fn["fused"](self.params, self.cparams, self.rparams, self.opt, self.copt, self.ropt, gs, js, des,
                               tau, lam, lr, lrc, lrr, self.cfg["clip"], rk)
        return out

    def _measure_baseline(self, lay, log):
        T, nb = self.problem.T, self.cfg["B"]
        region = Region(lay.W)
        seeds = np.arange(500_000 + 1000 * self.seed, 500_000 + 1000 * self.seed + nb)
        des = self._batch(seeds, region)
        lam = jnp.asarray(self.lam, jnp.float32)
        js = jnp.asarray(seeds, jnp.int32)
        ones = {k: jnp.float32(1.0) for k in self.params}
        out = self._call_fused(ones, js, des, jnp.float32(1.0), lam, 0.0, 0.0, 0.0, jax.random.PRNGKey(0))
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
        tstep = 0.0
        while True:
            if time.time() - t0 >= budget_s:
                break
            ftr = (time.time() - t0 - t_setup) / max(budget_s - t_setup, 1e-9)
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - ftr / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            des = self._batch(seeds, region)
            ta = time.time()
            out = self._call_fused(self.gs, jnp.asarray(seeds, jnp.int32), des, tau, lam, cfg["lr"],
                                   cfg["lr_critic"], cfg["lr_r"], jax.random.PRNGKey(self.updates + 7919 * self.seed))
            (_, self.params, self.cparams, self.rparams, self.opt, self.copt, self.ropt,
             loss, closs, rloss, viol, tot, rterm, _) = out
            viol = np.asarray(viol)
            tstep += time.time() - ta
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            log({"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                 "critic_loss": float(closs), "r_loss": float(rloss),
                 "cost_per_world": float(np.asarray(tot).mean()), "r_term_per_world": float(np.asarray(rterm).mean()),
                 "raw_viol": hf.tolist(), "lam": self.lam.tolist(), "enum_frac": None,
                 "time_rec": tstep, "time_replay": 0.0, "time_step": 0.0})
        self.timing = {"step": tstep, "setup": t_setup, "total": time.time() - t0}

    # ------------------------------------------------------------ generation
    def _gen_fn(self, lay):
        P = self.cfg["P"]
        if not hasattr(lay, "cmulti_gen"):
            lay.cmulti_gen = {}
        if P not in lay.cmulti_gen:
            sch = Sched(lay, P)

            def gen(params, rparams, seed, des, lam):
                st = multi_pass(sch, params, rparams, seed, des, 1.0, lam, record=False)[0]
                return st, core.repair(lay, st, seed)
            lay.cmulti_gen[P] = jax.jit(gen)
        return lay.cmulti_gen[P]

    def generate(self, seed, region):
        if region.periodic:
            lay = self.layout(region)
            des = self.designed(seed, region)
            crop = lambda st, c: np.asarray(st[c.name])
        else:
            m = margin(self.problem, self.cfg["P"])
            ext = Region(region.W + 2 * m, periodic=False, x0=region.x0 - m)
            lay = self.layout(ext)
            des = self.designed(seed, ext)
            crop = lambda st, c: np.asarray(st[c.name])[:, (m >> c.level):(m >> c.level) + (region.W >> c.level)]
        raw, fin = self._gen_fn(lay)(self.params, self.rparams, seed, des, jnp.asarray(self.lam, jnp.float32))
        return {kind: {c.name: crop(st, c) for c in self.problem.sampled()} for kind, st in (("raw", raw), ("final", fin))}

    def score(self, x, region):
        return None

    def bound_report(self, n=16):
        """F_ub (tau = 1, lam = 0): mean per fine cell of S(x^P) + log q(z) - log r,
        an upper bound on the bench's free energy; and its parts."""
        if not hasattr(self, "fn"):
            return None
        region = Region(self.cfg["W_train"] or self.problem.W_train)
        cells = self.problem.H * region.W
        seeds = np.arange(3_000_000 + 1000 * self.seed, 3_000_000 + 1000 * self.seed + n)
        des = self._batch(seeds, region)
        lam0 = jnp.zeros(self.problem.T + 1, jnp.float32)
        st, tot, rterm, lq = self.fn["evaluate"](self.params, self.rparams, jnp.asarray(seeds, jnp.int32), des, lam0)
        st = jax.device_get(st)
        S = []
        for b in range(n):
            d = {k: np.asarray(v[b]) for k, v in des.items()}
            xs = {c.name: np.asarray(st[c.name][b]) for c in self.problem.sampled()}
            S.append(en.energy(self.problem, {**d, **xs}, region).soft)
        return {"F_ub": float(np.mean(np.asarray(tot))) / cells, "soft": float(np.mean(S)) / cells,
                "logq_traj": float(np.mean(np.asarray(lq))) / cells,
                "minus_log_r": float(np.mean(np.asarray(rterm))) / cells,
                "telescope_err": float(np.mean(np.asarray(tot) - (np.asarray(S) + np.asarray(lq) + np.asarray(rterm))))}

    def info(self):
        return {**super().info(), "bound": self.bound_report()}


def _variant(name, **over):
    cls = type(f"CMulti_{name}", (CMulti,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("cmulti_p1", P=1)      # = cfast through cmulti's code (sanity)
_variant("cmulti_p3", P=3)
