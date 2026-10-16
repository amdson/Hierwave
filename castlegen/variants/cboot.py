"""cboot (2026-10-16 evening, user request): clongfit's kernel route (ML-only fit on own draws refined by
p*-invariant kernels, no penalty steps) started from a q pre-fitted on a few exact example worlds.

The examples are the problem's own probe worlds (problem.probes(seed, region): exact samples on columns and
dimers, target-like lifts on ground), so nothing here is problem-specific.  Pre-fit = boot_steps ML steps on
boot_n examples (minibatch ml_B) before the kernel-route loop; the loop itself (population of own draws, refine,
weighted ML) is clongfit's, unchanged.  The examples are not added to the population.

Knobs: boot_n (32 example worlds), boot_steps (200 ML steps), boot_lr (ml_lr).
cboot_stage = cboot + cstage's fixed stage as generation (Amendment 2026-10-16), the clongfit_stage pattern.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.variant import VARIANTS, register

from . import clongfit  # noqa: F401  (registers clongfit)
from .cstage import CKPopHardPenStage

BOOT = dict(boot_n=32, boot_steps=200, boot_lr=None)


class CBoot(VARIANTS["clongfit"]):
    name = "cboot"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**BOOT, **cfg})
        self._booted = False
        self.boot_log = []

    def _examples(self, n, region):
        """n example worlds from the problem's probe generator (+ designed channels)."""
        p, out = self.problem, []
        s = 70_000_000 + 1_000_000 * self.seed
        while len(out) < n:
            for x in p.probes(s, region):
                out.append({**{k: np.asarray(v) for k, v in x.items()}, **p.designed_values(s, region)})
                if len(out) >= n:
                    break
            s += 1
        return {ch.name: jnp.asarray(np.stack([x[ch.name] for x in out]), jnp.int32) for ch in p.chans}

    def _build(self, region):
        super()._build(region)
        if self._booted:
            return
        self._booted = True
        from castlegen.kernels.logq import ml_opt_init
        c, p = self.cfg, self.problem
        t0 = time.time()
        ex = self._examples(int(c["boot_n"]), region)
        s, nv, _ = self.E(self._to_flat(ex))
        nv = np.asarray(nv)
        opt = ml_opt_init(self.params)
        on = jnp.ones(p.T + 1, jnp.float32)
        lr = jnp.float32(c["boot_lr"] if c["boot_lr"] is not None else c["ml_lr"])
        B = min(int(c["ml_B"]), int(c["boot_n"]))
        cells = p.H * region.W
        for u in range(int(c["boot_steps"])):
            idx = self.rng.choice(int(c["boot_n"]), B, replace=False)
            self.params, opt, info = self.mlstep(self.params, opt, {k: x[idx] for k, x in ex.items()},
                                                 jnp.ones(B), lr, on)
            if u % 20 == 0 or u == int(c["boot_steps"]) - 1:
                self.boot_log.append({"step": u, "loss_per_cell": float(info["loss"]) / cells, "t": time.time() - t0})
        self.boot_info = {"n": int(c["boot_n"]), "steps": int(c["boot_steps"]), "example_viol_mean": float(nv.mean()),
                          "example_stats": p.stats({k: np.asarray(v[0]) for k, v in ex.items()}),
                          "t_boot": time.time() - t0, "log": self.boot_log}
        print(f"[cboot] pre-fit on {c['boot_n']} examples (viol {nv.mean():.3f}), {c['boot_steps']} steps: "
              f"loss/cell {self.boot_log[0]['loss_per_cell']:.4f} -> {self.boot_log[-1]['loss_per_cell']:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    def info(self):
        d = super().info()
        d = dict(d) if isinstance(d, dict) else {"base": d}
        d["boot"] = getattr(self, "boot_info", None)
        return d


register(CBoot)

_BORROW = ("STAGE_R", "STAGE_F", "REACH_ROUND", "REACH_REFILL", "BATCH", "_stage_RF", "_reach", "_pass")


class CBootStage(CBoot):
    name = "cboot_stage"
    staged = True   # amendment 2026-10-16

    def generate(self, seed, region):
        return CKPopHardPenStage.generate(self, seed, region)

    def info(self):
        d = super().info()
        R, F = self._stage_RF()
        d["stage"] = {"R": R, "F": F}
        return d


for _k in _BORROW:
    setattr(CBootStage, _k, CKPopHardPenStage.__dict__[_k])
register(CBootStage)
