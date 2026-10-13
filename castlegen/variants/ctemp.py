"""ctemp: cpass with temperature-conditioned training (ideas.md #1).

Changes against cpass (everything else, estimator included, is cpass's):
- tau_cond: the actor reads log tau as an extra input, so one network holds
  q(. | tau) for a range of temperatures; generation and scoring use tau = 1
  (log tau = 0).
- tau schedule "coarse" (default): as "mix" below, but only the levels above
  the finest are heated; the finest level always trains at tau = 1, on the
  wider spread of contexts the hot coarse levels give it.
- tau schedule "mix": for the first mix_frac of the budget each update draws
  its tau, 1 with probability p_one, else log-uniform on [1, tau_max]; the
  rest of the budget trains at tau = 1 only (the objective at the end of
  training is the fixed one).  "anneal" is cpass's 2 -> 1 ramp.
- lam_tau1_only: the penalty controller steps only on tau = 1 batches
  (hot batches violate more by design and would drive lambda up).
- stat logging: the problem's statistics on the training batch every
  stat_every updates (st_* keys), for diagnosing when entropy is lost.

Config comes from CFG below; CTEMP_CFG (JSON in the environment) overrides
it for experiments only."""
from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cpass_core as core
from . import _ctemp_core as tcore
from .cpass import CFG as CPASS_CFG
from .cpass import CPass, adam_init, adam_step, clip_by_level

CFG = {**CPASS_CFG, "tau_mode": "coarse", "tau_max": 3.0, "mix_frac": 0.5, "p_one": 0.25, "tau_cond": True,
       "lam_tau1_only": True, "stat_every": 10}


@register
class CTemp(CPass):
    name = "ctemp"

    def __init__(self, problem, seed, workdir, **cfg):
        env = json.loads(os.environ.get("CTEMP_CFG", "{}"))
        cfg = {**CFG, **env, **cfg}
        super().__init__(problem, seed, workdir, **cfg)
        T = problem.T
        self.n_aex = 1 if self.cfg["tau_cond"] else 0
        lay = self.layout(Region(problem.W_train))
        key = jax.random.PRNGKey(seed + 7919)
        for l in range(T + 1):                              # actor: + log tau_l; critic: tau, lambda per level
            for c in lay.chans[l]:
                key, k1, k2 = jax.random.split(key, 3)
                if self.n_aex:
                    din = lay.feat_dim(l, core.ACTOR_WIN, self.n_aex)
                    self.params[core.key_of(l, c.name)] = core.mlp_init(k1, din, c.D, self.cfg["width"])
                din = lay.feat_dim(l, core.CRITIC_WIN, 2 * (T + 1) + 1)
                self.cparams[core.key_of(l, c.name)] = core.mlp_init(k2, din, c.D, self.cfg["width"])
        self.opt, self.copt = adam_init(self.params), adam_init(self.cparams)
        self.tauv = np.full(T + 1, self.tau)

    def aex(self, tau):
        """Actor inputs per level, (T + 1, 1): log tau_l."""
        if not self.cfg["tau_cond"]:
            return None
        return jnp.log(jnp.asarray(tau, jnp.float32))[:, None]

    def ones(self):
        return jnp.ones(self.problem.T + 1, jnp.float32)

    # ------------------------------------------------------------- layouts
    def layout(self, region):
        key = (region.W, region.periodic, region.x0)
        if key not in self.lays:
            lay = core.Layout(self.problem, region.W, region.periodic, region.x0)
            lay.jit_gen = jax.jit(lambda params, seed, des, lam: tcore.coloured_pass(
                lay, params, seed, des, self.ones(), lam, actor_extra=self.aex(self.ones()))[0])
            lay.jit_repair = jax.jit(lambda st, seed: core.repair(lay, st, seed))
            lay.jit_score = jax.jit(lambda params, des, forced, lam: tcore.coloured_pass(
                lay, params, 0, des, self.ones(), lam, forced=forced, actor_extra=self.aex(self.ones()))[2])
            self.lays[key] = lay
        return self.lays[key]

    # -------------------------------------------------------------- training
    def _build_train(self, lay):
        """cpass._build_train with the pass taking the actor's extra input."""
        T, K = self.problem.T, self.cfg["K"]
        levels = list(range(T + 1))
        topkeys = [core.key_of(T, c.name) for c in lay.chans[T]]
        nval = {}
        for l in levels:
            cnt = 0
            for k in range(core.NCOL):
                _, _, v = lay.sites(l, k)
                cnt += int(np.asarray(v).sum())
            nval[l] = cnt * len(lay.chans[l])

        def gen_rec(params, seed, des, tau, lam, extra):
            st, tot, _, rec = tcore.coloured_pass(lay, params, seed, des, tau, lam, record=True,
                                                  critic_extra=extra, actor_extra=self.aex(tau))
            return st, tot, rec

        def gen_tot(params, seed, des, tau, lam, ov):
            return tcore.coloured_pass(lay, params, seed, des, tau, lam, override=ov, actor_extra=self.aex(tau))[1]

        lay.jit_rec = jax.jit(jax.vmap(gen_rec, in_axes=(None, 0, 0, None, None, None)))
        lay.jit_tot = jax.jit(jax.vmap(gen_tot, in_axes=(None, 0, 0, None, None, 0)))

        def actor_loss(params, cparams, recs, G, tau, lam, br, gs):
            total = 0.0
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    b = core.mlp(params[key], r["feat"])
                    logits = jnp.where(r["mask"], -r["soft"] / tau[l] - b, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, -1)
                    q = jnp.exp(logq)
                    lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                    ell = r["soft"] + lam[l] * r["nv"] + tau[l] * lq_sg
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
            ix = jnp.arange(r["a"].shape[0])
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

    def _tau_at(self, frac):
        """Per-level temperatures (T + 1,) for this update."""
        cfg, T = self.cfg, self.problem.T
        if cfg["tau_mode"] == "anneal":
            return np.full(T + 1, cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - frac / cfg["anneal"]))
        if frac >= cfg["mix_frac"] or self.rng.random() < cfg["p_one"]:
            return np.ones(T + 1)
        t = float(np.exp(self.rng.uniform(0.0, np.log(cfg["tau_max"]))))
        if cfg["tau_mode"] == "mix":                        # every level at one sampled tau
            return np.full(T + 1, t)
        assert cfg["tau_mode"] == "coarse"                  # the finest level stays at tau = 1
        return np.where(np.arange(T + 1) > 0, t, 1.0)

    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(self.problem.W_train)
        lay = self.layout(region)
        self._measure_baseline(lay, log)
        self._build_train(lay)
        B = cfg["B"]
        cells = self.problem.H * self.problem.W_train
        while True:
            frac = (time.time() - t0) / max(budget_s, 1e-9)
            if frac >= 1.0:
                break
            self.tauv = self._tau_at(frac)
            self.tau = float(self.tauv.max())
            tau, lam = jnp.asarray(self.tauv, jnp.float32), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            desl = [self.designed(int(s), region) for s in seeds]
            des = {k: jnp.stack([d[k] for d in desl]) for k in desl[0]}
            extra = jnp.concatenate([tau, lam]).astype(jnp.float32)
            st, tot, recs = lay.jit_rec(self.params, jnp.asarray(seeds, jnp.int32), des, tau, lam, extra)
            G, after = self._returns(lay, tot)
            if self.gscale is None:
                self.gscale = float(max(np.std(np.asarray(after)), 1.0))
            br = self._branches(lay, recs, after, jnp.asarray(seeds, jnp.int32), des, tau, lam)
            self.params, self.cparams, self.opt, self.copt, loss, closs = lay.jit_step(
                self.params, self.cparams, self.opt, self.copt, recs, G, tau, lam, br,
                jnp.float32(self.gscale), cfg["lr"], cfg["lr_critic"], cfg["clip"])
            viol = np.zeros(T + 1)
            for l in range(T + 1):
                for c in lay.chans[l]:
                    r = recs[core.key_of(l, c.name)]
                    nv_a = np.take_along_axis(np.asarray(r["nv"]), np.asarray(r["a"])[..., None], -1)[..., 0]
                    viol[l] += float((nv_a * np.asarray(r["valid"])).sum()) / B
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            if np.all(self.tauv == 1.0) or not cfg["lam_tau1_only"]:
                self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
                step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
                self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            cpw = float(np.asarray(tot).sum(-1).mean())
            rec = {"t": time.time() - t0, "update": self.updates, "tau": self.tau, "tau_vec": self.tauv.tolist(), "loss": float(loss),
                   "critic_loss": float(closs), "cost_per_world": cpw, "F_cell": cpw / cells,
                   "raw_viol": hf.tolist(), "lam": self.lam.tolist(),
                   "enum_frac": float(np.asarray(br["enum"]).mean())}
            if self.problem.stats and self.updates % cfg["stat_every"] == 0:
                acc = {}
                for b in range(B):
                    s = self.problem.stats({k: np.asarray(v[b]) for k, v in st.items()})
                    for k, v in s.items():
                        acc.setdefault(k, []).append(v)
                rec.update({"st_" + k: float(np.mean(v)) for k, v in acc.items()})
            log(rec)
