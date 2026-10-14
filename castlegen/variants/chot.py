"""chot: cfast with epsilon-hot exploration, importance-weighted back to the
training objective (ideas.md #12; user-approved 2026-10-14).

Training worlds are drawn with each draw at a "hot" level independently hot
with probability eps (coins keyed by the coordinate noise); a hot draw samples
the whole-logit tempered conditional q^(1/tau_b) over A (csat: tempering only
the designed term cannot move a saturated bias).  Every draw at a hot level
carries the bounded mixture ratio q / ((1 - eps) q + eps q_b) <= 1 / (1 - eps),
and each world's actor-loss terms are multiplied by the product of its
ratios: a superset of every term's ancestral closure, so the gradient is an
unbiased estimate of cfast's (constraint 3; argument in _chot_core.py).
Penalty updates use importance-weighted violation counts.  Generation, pass,
cone returns, critic, subsampling, optimiser and schedules are cfast's.

cfg: eps (hot rate; None: n_hot / number of draws at the hot levels, i.e. a
fixed expected number of hot draws per world, which bounds Var W whatever
the world size), n_hot, tau_b (behaviour temperature), hot ("top": level T
only, "all": every level, or an int k: the top k levels)."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _chot_core
from .cfast import CFast


@register
class CHot(CFast):
    name = "chot"

    def __init__(self, problem, seed, workdir, **cfg):
        cfg = {"eps": None, "n_hot": 1.28, "tau_b": 3.0, "hot": "top", **cfg}
        super().__init__(problem, seed, workdir, **cfg)
        T, h = problem.T, self.cfg["hot"]
        k = 1 if h == "top" else (T + 1 if h == "all" else int(h))
        hot = [l > T - k for l in range(T + 1)]
        eps = self.cfg["eps"]
        if eps is None:              # ground, top level: 64 draws -> eps = 0.02
            lay = self.layout(Region(self.cfg["W_train"] or problem.W_train))
            nd = sum(int(np.asarray(lay.sites(l, kk)[2]).sum()) * len(lay.chans[l])
                     for l in range(T + 1) if hot[l] for kk in range(16))
            eps = min(self.cfg["n_hot"] / max(nd, 1), 0.5)
        self.eps_vec = np.array([eps if hot[l] else 0.0 for l in range(T + 1)], np.float32)

    def train_fns(self, lay):
        key = ("chot", self.cfg["tau_b"], self.cfg["critic"], self.cfg["sub"], self.cfg.get("net", "tanh"))
        if not hasattr(lay, "chot_fns"):
            lay.chot_fns = {}
        if key not in lay.chot_fns:
            lay.chot_fns[key] = _chot_core.build_train(self, lay)
        return lay.chot_fns[key]

    def _build_train(self, lay):
        return _chot_core.build_train(self, lay)

    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        self.fn = self.train_fns(lay)
        self._measure_baseline(lay, log)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True, "eps": self.eps_vec.tolist(), "tau_b": cfg["tau_b"]})
        B = cfg["B"]
        eps = jnp.asarray(self.eps_vec)
        keys = [k for k in self.params]
        lev = self.fn["level_of"]
        while True:
            if (time.time() - t0) / max(budget_s, 1e-9) >= 1.0:
                break
            ftr = (time.time() - t0 - t_setup) / max(budget_s - t_setup, 1e-9)
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - ftr / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            des = self._batch(seeds, region)
            out = self.fn["fused_hot"](self.params, self.cparams, self.opt, self.copt, self.gs,
                                       jnp.asarray(seeds, jnp.int32), des, tau, lam, cfg["lr"],
                                       cfg["lr_critic"], cfg["clip"],
                                       jax.random.PRNGKey(self.updates + 7919 * self.seed), eps)
            (_, self.params, self.cparams, self.opt, self.copt, loss, closs, viol, tot, _, W, diag) = out
            viol, W, tot = np.asarray(viol), np.asarray(W, np.float64), np.asarray(tot, np.float64)
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            rec = {"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                   "critic_loss": float(closs), "cost_per_world": float(tot.mean()),
                   "cost_per_world_w": float((W * tot).mean()), "raw_viol": hf.tolist(),
                   "lam": self.lam.tolist(), "W_mean": float(W.mean()), "W_max": float(W.max()),
                   "ess": float(W.sum() ** 2 / max((W ** 2).sum(), 1e-300))}
            if self.updates % 10 == 1:
                d = jax.device_get(diag)
                H, hot = {}, {}
                for l in range(T + 1):
                    ks = [k for k in keys if lev[k] == l]
                    s = np.sum([d[k] for k in ks], 0)
                    H[l] = float(s[0] / max(s[1], 1))
                    hot[l] = float(s[2] / max(s[3], 1))
                    for k in ks:
                        rec["H_" + k] = float(d[k][0] / max(d[k][1], 1))
                rec["H_level"] = [H[l] for l in range(T + 1)]
                rec["hot_frac"] = [hot[l] for l in range(T + 1)]
            log(rec)
        self.timing = {"setup": t_setup, "total": time.time() - t0}


def _variant(name, **over):
    cls = type(f"CHot_{name}", (CHot,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("chot_off", eps=0.0)                  # control: cfast through chot's code path
_variant("chot_all", hot="all")               # hot draws at every level (same expected count per world)
_variant("chot_t2", hot=2)                    # top two levels
