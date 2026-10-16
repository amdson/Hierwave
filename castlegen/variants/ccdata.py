"""ccdata (overnight 2026-10-16, agent cdata): cfitpen_ml2 with more data / more weight for the coarse levels in the ML steps.

Engineering freedoms only (population size, step counts, step sizes); objective, kernels, penalty term, lambda
control and generation unchanged.  Zero-change config is cfitpen_ml2 exactly (the wrapped ML step is the original).

Knobs (defaults = cfitpen_ml2):
  pop_mult     population size (nctx) multiplier (nctx 32 -> 32*pop_mult)
  w3, w2       per-level ML step weight for levels >= 3 / >= 2 (w3 wins at levels >= 3).  The ML update is Adam, which is
               invariant to a gradient scale, so a loss/gradient weight would be a no-op; the weight is therefore applied as
               the size of the parameter step: p <- p_old + w * (p_adam - p_old) for those levels.
  extra_coarse extra ML steps per iteration that update only levels >= 3 (same population, fresh minibatches)
  init_pkl     {"params","lam"} pickle to continue from (None: ckernel's usual init)
"""
from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.variant import register

from .cfitpen import CFitPen
from .ckernel import CK


class CCData(CFitPen):
    name = "ccdata"

    def __init__(self, problem, seed, workdir, **cfg):
        cfg = {"upi": 16, "pop_mult": 1, "w3": 1.0, "w2": 1.0, "extra_coarse": 0, "init_pkl": None, **cfg}
        cfg["nctx"] = int(cfg.get("nctx", CK["nctx"]) * cfg["pop_mult"])
        self._upi0 = int(cfg["upi"])
        cfg["upi"] = self._upi0 + int(cfg["extra_coarse"])      # the train loop runs upi steps per iteration
        super().__init__(problem, seed, workdir, **cfg)
        self._k = 0
        if self.cfg["init_pkl"]:
            ck = pickle.load(open(self.cfg["init_pkl"], "rb"))
            self.params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
            self.lam = np.asarray(ck["lam"], float)
            self.start = self.cfg["init_pkl"]

    def _build(self, region):
        super()._build(region)
        c = self.cfg
        if c["w3"] == 1.0 and c["w2"] == 1.0 and c["extra_coarse"] == 0:
            return
        base, T = self.mlstep, self.problem.T
        upi, extra = self._upi0, int(c["extra_coarse"])
        scale = {l: (c["w3"] if l >= 3 else c["w2"] if l >= 2 else 1.0) for l in range(T + 1)}
        coarse_on = jnp.asarray([1.0 if l >= 3 else 0.0 for l in range(T + 1)], jnp.float32)
        lv = lambda k: int(k[1:k.index("_")])

        def step(params, opt, batch, w, lr, on):
            # calls 1..upi of an iteration are the normal ML steps, the next `extra` update levels >= 3 only
            if self._k >= upi and extra:
                on = on * coarse_on
            self._k += 1
            new, nopt, info = base(params, opt, batch, w, lr, on)
            new = {k: (jax.tree_util.tree_map(lambda a, b, s=scale[lv(k)]: b + s * (a - b), v, params[k])
                       if scale[lv(k)] != 1.0 else v) for k, v in new.items()}
            return new, nopt, info
        self.mlstep = step

    def refine(self, st, key, phase=2):
        self._k = 0             # called once per iteration (and once in setup) before that iteration's ML steps
        return super().refine(st, key, phase)


register(CCData)
