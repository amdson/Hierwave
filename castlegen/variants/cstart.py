"""cstart: cfast with mixed start depths (ideas #14, user-approved).

Each training world picks a start depth L.  A fraction `frac` of each batch
(stratified: round(frac * B) worlds) are clamped worlds with L drawn
uniformly from `Ls` (default 1..T); the rest are full-depth worlds
(L = T + 1, exactly cfast's world).  In a clamped world every draw at a level
>= L takes its value from a context state instead of being sampled; levels
< L sample as usual and get their normal cone returns and critic.  Clamped
draws are excluded from the loss (actor and critic) and their costs from
every return (_cstart_core).

Context source `src`:
  "probe"  a probe state of the world's seed (problem.probes: valid,
           target-like states with every coarse channel lifted exactly; on
           ground they follow the designed fill line with 0-3 rows of
           jitter), one of the probes chosen uniformly per world.  Generic:
           any problem that declares probes; a problem without probes (or
           with T = 0, e.g. dimers) gets no clamped worlds and trains as cfast.

The fine levels' gradient is then an average over contexts from the mixture
(own draws in full-depth worlds, probe contexts in clamped worlds) instead of
own draws alone.  It is not the unbiased gradient of J as written
(constraint 3), but each fine conditional has the same minimiser (the
target's conditional given its parents); the user approved this in chat
(ideas #14).  Generation is cfast's, unchanged.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cstart_core
from .cfast import CFast


@register
class CStart(CFast):
    name = "cstart"

    def __init__(self, problem, seed, workdir, **cfg):
        cfg = {"frac": 0.5, "Ls": None, "src": "probe", "n_probes": 4, **cfg}
        super().__init__(problem, seed, workdir, **cfg)
        T = problem.T
        self.Ls = list(self.cfg["Ls"] or range(1, T + 1))
        self.can_clamp = T >= 1 and problem.probes is not None and len(self.Ls) > 0
        self.crng = np.random.default_rng(1_000_003 + seed)
        self.ctx_chans = [c for c in problem.sampled() if c.level >= 1]

    def train_fns(self, lay):
        key = ("cstart", self.cfg["critic"], self.cfg["sub"], self.cfg.get("net", "tanh"))
        if not hasattr(lay, "cstart_fns"):
            lay.cstart_fns = {}
        if key not in lay.cstart_fns:
            lay.cstart_fns[key] = _cstart_core.build_train(self, lay)
        return lay.cstart_fns[key]

    def _build_train(self, lay):
        return _cstart_core.build_train(self, lay)

    # -------------------------------------------------------------- contexts
    def contexts(self, seeds, region, Ls_w):
        """Context arrays {chan: (B, h, w)} and per-world L (B,) int32."""
        T = self.problem.T
        ctx = {c.name: np.zeros((len(seeds),) + self.layout(region).hw[c.level], np.int64)
               for c in self.ctx_chans}
        for b, s in enumerate(seeds):
            if Ls_w[b] <= T:
                k = int(self.crng.integers(self.cfg["n_probes"]))
                try:                           # ground: probe k only depends on k (cheaper)
                    pr = self.problem.probes(int(s), region, n=k + 1)[k]
                except TypeError:
                    prs = self.problem.probes(int(s), region)
                    pr = prs[k % len(prs)]
                for c in self.ctx_chans:
                    ctx[c.name][b] = pr[c.name]
        return ({k: jnp.asarray(v, jnp.int32) for k, v in ctx.items()},
                jnp.asarray(Ls_w, jnp.int32))

    def draw_L(self, B):
        T = self.problem.T
        Ls_w = np.full(B, T + 1, np.int64)
        if self.can_clamp and self.cfg["frac"] > 0:
            nc = int(round(self.cfg["frac"] * B))
            who = self.crng.permutation(B)[:nc]
            Ls_w[who] = self.crng.choice(self.Ls, size=nc)
        return Ls_w

    # -------------------------------------------------------------- training
    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        self.fn = self.train_fns(lay)
        self._measure_baseline(lay, log)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True, "frac": cfg["frac"], "Ls": self.Ls,
             "can_clamp": self.can_clamp})
        B = cfg["B"]
        tm = {"rec": 0.0, "ctx": 0.0}
        nclamp = np.zeros(T + 2)
        while True:
            if (time.time() - t0) / max(budget_s, 1e-9) >= 1.0:
                break
            ftr = (time.time() - t0 - t_setup) / max(budget_s - t_setup, 1e-9)
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - ftr / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            des = self._batch(seeds, region)
            tc = time.time()
            Ls_w = self.draw_L(B)
            ctx, Lj = self.contexts(seeds, region, Ls_w)
            for L in Ls_w:
                nclamp[L] += 1
            ta = time.time()
            out = self.fn["fused_ctx"](self.params, self.cparams, self.opt, self.copt, self.gs,
                                       jnp.asarray(seeds, jnp.int32), des, tau, lam, cfg["lr"],
                                       cfg["lr_critic"], cfg["clip"],
                                       jax.random.PRNGKey(self.updates + 7919 * self.seed), ctx, Lj)
            _, self.params, self.cparams, self.opt, self.copt, loss, closs, viol, tot, _ = out
            viol = np.asarray(viol)
            tm["ctx"] += ta - tc
            tm["rec"] += time.time() - ta
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            log({"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                 "critic_loss": float(closs), "cost_per_world": float(np.asarray(tot).mean()),
                 "raw_viol": hf.tolist(), "lam": self.lam.tolist(), "L_worlds": Ls_w.tolist(),
                 "time_rec": tm["rec"], "time_ctx": tm["ctx"]})
        self.timing = {**tm, "setup": t_setup, "total": time.time() - t0,
                       "L_counts": nclamp.tolist()}


def _variant(name, **over):
    cls = type(f"CStart_{name}", (CStart,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("cstart_off", frac=0.0)          # control: cfast through cstart's code path
_variant("cstart_L34", Ls=(3, 4))          # clamped worlds start at L = 3 or 4 only
_variant("cstart_f9L4", frac=0.9, Ls=(4,))  # diagnostic upper bound: 90 % of worlds clamped at L = 4
