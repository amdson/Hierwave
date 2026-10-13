"""creplay: cfast plus coupled replays for many coarse draws per world.

cfast's dependency-cone returns fix credit at the fine levels, but at the
coarse levels the cones cover nearly the whole world and the score-function
term's SNR is about 0.02 per world.  creplay branches many coarse draws per
world instead of one (cpass) or none (cfast): every draw of the top level and
R draws of the next (configurable), K actions each (all of A when |A| <= K),
replayed under the world's stored noise and scored by the draw's cone return.
The estimator stays unbiased (see _creplay_core.py; gradcheck in
tests/variants/test_creplay.py).  Generation is cpass's / cfast's, unchanged.

Config: Kr (actions per branched draw), blev (number of top levels branched),
R (per branched level from the top: None = all draws, int = random subset per
world), chunk (replays per lax.map chunk, memory)."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _creplay_core as rcore
from .cfast import CFast

CFG = dict(K=0, Kr=4, blev=2, R=(None, 64), chunk=1024)


@register
class CReplay(CFast):
    name = "creplay"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CFG, **cfg})

    def train_fns(self, lay):
        c = self.cfg
        key = ("creplay", c["Kr"], c["blev"], tuple(c["R"]), c["chunk"], c["critic"], c["sub"])
        if not hasattr(lay, "cfast_fns"):
            lay.cfast_fns = {}
        if key not in lay.cfast_fns:
            lay.cfast_fns[key] = rcore.build(self, lay)
        return lay.cfast_fns[key]

    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        self.fn = self.train_fns(lay)
        self._measure_baseline(lay, log)          # cfast's: the fused update at lr = 0 (same compile)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True,
             "branch": {str(l): {"R": b["R"], "Nc": b["Nc"]} for l, b in self.fn["btab"].items()}})
        B = cfg["B"]
        while True:
            frac = (time.time() - t0) / max(budget_s, 1e-9)
            if frac >= 1.0:
                break
            ftr = (time.time() - t0 - t_setup) / max(budget_s - t_setup, 1e-9)
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - ftr / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            des = self._batch(seeds, region)
            out = self.fn["fused"](self.params, self.cparams, self.opt, self.copt, self.gs,
                                   jnp.asarray(seeds, jnp.int32), des, tau, lam, cfg["lr"], cfg["lr_critic"],
                                   cfg["clip"], jax.random.PRNGKey(self.updates + 7919 * self.seed))
            _, self.params, self.cparams, self.opt, self.copt, loss, closs, viol, tot, diag, _ = out
            viol = np.asarray(viol)
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            d = jax.device_get(diag)
            ent = {"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                   "critic_loss": float(closs), "cost_per_world": float(np.asarray(tot).mean()),
                   "raw_viol": hf.tolist(), "lam": self.lam.tolist()}
            if self.updates % 10 == 1:
                ent["branch"] = {l: {k: float(v) for k, v in x.items()} for l, x in d.items()}
            if d:
                ent["enum_frac"] = float(np.mean([x["enum_frac"] for x in d.values()]))
            log(ent)
        self.timing = {"setup": t_setup, "total": time.time() - t0}


def _variant(name, **over):
    cls = type(f"CReplay_{name}", (CReplay,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)
