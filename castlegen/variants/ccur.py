"""ccur: cfast with a bottom-up (fine-first) per-level actor learning-rate
schedule (ideas #7).  Engineering only (constitution: learning rates and
schedules); the objective, estimator, critic, tau and lambda schedules and
generation are cfast's.

- Levels >= `freeze_from` (ground: 3 and 4) have actor lr 0 -- their
  parameters stay at the stochastic init -- until their release time; levels
  below it train from update 1.  Implemented by restoring the frozen levels'
  parameters after every fused cfast update (the Adam moments keep
  accumulating their gradients, so a released level starts with warm moments
  rather than a cold-start step of ~3x lr).
- Release is staged from fine to coarse: level freeze_from at release[0],
  the next level at release[1], ... (fractions of the post-setup training
  time).  Staged, because the idea is that every level should train against
  finer levels that have already adapted; the top level (which locks first in
  cfast, csat report) is released last, against the most-trained downstream.
- The critic trains at every level throughout.  tau anneals as in cfast
  (2 -> 1 over the first 20 % of post-setup time), so the fine levels learn the
  tau = 1 objective they face after release; the frozen coarse levels' draws
  change little with tau (untrained dirt 0.86 at tau 2, 0.88 at tau 1).
- Diagnostics (ccur_diag / ccur_off only; off in ccur so the bench clock is cfast's): csat's per-level/per-channel entropy, forced fraction and bias
  spread every `diag_every` updates, plus line_match and the rho4 fill error
  (ground.stats) of the diagnostic worlds' draws, and the parameters at the
  first release (self.snap_release) for the clamp gate.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from ._cfast_core import train_pass
from .cfast import CFast
from .csat import CSat

_TFN = {}


class CCur(CSat):
    name = "ccur"
    CUR = dict(freeze_from=3, release=(0.35, 0.55))

    def __init__(self, problem, seed, workdir, **cfg):
        cur = {k: cfg.pop(k, v) for k, v in self.CUR.items()}
        super().__init__(problem, seed, workdir, **cfg)
        self.cfg.update(cur)
        T = problem.T
        ff = self.cfg["freeze_from"]
        rel = list(self.cfg["release"])
        assert len(rel) >= T + 1 - ff or ff > T
        # release fraction per level (0 for levels that train from the start)
        self.rel_frac = {l: (rel[l - ff] if l >= ff else 0.0) for l in range(T + 1)}
        self.p_init = jax.device_get(self.params)
        self.snap_release = None

    def _frozen(self, frac):
        return [l for l, r in self.rel_frac.items() if frac < r]

    def _draw_t(self, lay):
        key = id(lay)
        if key not in _TFN:
            _TFN[key] = jax.jit(jax.vmap(lambda pr, s, d, tau, lam: train_pass(lay, pr, s, d, tau, lam,
                                                                               record=False)[0],
                                         in_axes=(None, 0, 0, None, None)))
        return _TFN[key]

    def diag(self, tau=None, lam=None):
        res = super().diag(tau, lam)
        if self.problem.name != "ground":
            return res
        from castlegen.bench.problems.ground import stats
        cfg = self.cfg
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        nb = cfg.get("diag_B", 8)
        seeds = np.arange(900_000, 900_000 + nb)
        des = self._batch(seeds, region)
        tau = self.tau if tau is None else tau
        lam = self.lam if lam is None else lam
        st = jax.device_get(self._draw_t(lay)(self.params, jnp.asarray(seeds, jnp.int32), des,
                                              jnp.float32(tau), jnp.asarray(lam, jnp.float32)))
        ss = [stats({"t": st["t"][b], "fill": np.asarray(des["fill"][b]), "rho4": st["rho4"][b]})
              for b in range(nb)]
        res["line_match"] = float(np.mean([s["line_match"] for s in ss]))
        res["fill_err"] = float(np.mean([s["fill_err"] for s in ss]))
        return res

    def train(self, budget_s, log):
        T = self.problem.T
        tset = {"t": None}
        released = set()

        def wlog(d):
            if d.get("setup_done"):
                tset["t"] = d["t"]
            u = d.get("update")
            if u is not None:
                ts = tset["t"] or 0.0
                frac = (d["t"] - ts) / max(budget_s - ts, 1e-9)
                frozen = self._frozen(frac)
                if frozen:
                    for l in frozen:
                        for c in self.problem.chans:
                            k = f"l{l}_{c.name}"
                            if c.level == l and k in self.params:
                                self.params[k] = jax.tree_util.tree_map(jnp.asarray, self.p_init[k])
                newly = [l for l in range(T + 1) if l not in frozen and l not in released
                         and self.rel_frac[l] > 0]
                if newly:
                    released.update(newly)
                    if self.snap_release is None:
                        self.snap_release = {"params": jax.device_get(self.params), "lam": self.lam.tolist(),
                                             "update": u}
                    d = {**d, "released": newly}
                d = {**d, "frozen": frozen}
                if N and (u == 1 or u % N == 0):         # csat's diagnostics, after the restore
                    ta = time.time()
                    d = {**d, "diag": self.diag()}
                    self._dtime += time.time() - ta
                    d["diag_time"] = self._dtime
            log(d)
        N = self.cfg.get("diag_every", 0)
        self._dtime = 0.0
        CFast.train(self, budget_s, wlog)


register(CCur)


def _cvariant(name, **over):
    cls = type(f"CCur_{name}", (CCur,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


# "ccur" itself (the bench entry) runs without diagnostics (diag_every = 0), so
# its training clock is cfast's; the gate runs use the instrumented copies.
_cvariant("ccur_diag", diag_every=25)                            # ccur + diagnostics every 25 updates
_cvariant("ccur_off", freeze_from=99, release=(), diag_every=25)  # no freezing: cfast + the same diagnostics
