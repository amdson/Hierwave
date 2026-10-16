"""cdimerworm (2026-10-16, dsector follow-up): cfitpen_ml2 on dimers whose population refresh is the worm
(alternating-loop) kernel of _dimer_loop_kernel.py instead of the single-site / block kernels.

Kernel-route amendment: the refresh is a p*-invariant Markov kernel on the population (worm excursions, exact
symmetry test in tests/variants/test_dimer_loop_kernel.py).  A fresh own draw that is not a perfect matching is first
projected onto perfect matchings by `repair` (monomer worm walk; NOT p*-invariant, used as the start only).  The
penalty-only reverse-KL steps (cfitpen: rkl_n, pen_only, lam control) and the 16 ML updates per iteration are kept.
Generation is the unchanged fixed-step pass.  cdimerworm_stage adds the fixed stage (staged = True).
cfg: worm (sweeps per iteration, default 1), worm_lmax (1024), rkl_n (32; 0 = no penalty steps).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from castlegen.bench.variant import VARIANTS, register

from . import _dimer_loop_kernel as dk
from . import cfitpen  # noqa: F401
from .cfitpen import CFitPen
from .cstage import CKPopHardPenStage

WORM = dict(worm=1, worm_lmax=1024, upi=16, hard_sweeps=0, cg_sweeps=0, m_flip=0, tt_Ls=(), m_l2=0, n_fine=0,
            refresh=0.1)


class CDimerWorm(CFitPen):
    name = "cdimerworm"

    def __init__(self, problem, seed, workdir, **cfg):
        assert problem.name == "dimers"
        super().__init__(problem, seed, workdir, **{**WORM, **cfg})
        self._wseed = 1000 * seed + 17

    def refine(self, st, key, phase=2):
        c = self.cfg
        P = np.asarray(st["p"], np.int64)
        self._wseed += 1
        P, info = dk.loop_refresh(P, int(c["worm"]) * P.shape[1] * P.shape[2], lmax=int(c["worm_lmax"]),
                                  seed=self._wseed, repair=True)
        st = {**st, "p": jnp.asarray(P, jnp.int32)}
        d = {"worm_valid_in": info["valid_frac"], "worm_acc": info["acc"], "worm_repaired": info["repaired"]}
        return st, d, self._to_flat(st)


register(CDimerWorm)

_BORROW = ("STAGE_R", "STAGE_F", "REACH_ROUND", "REACH_REFILL", "BATCH", "_stage_RF", "_reach", "_pass")


class CDimerWormStage(CDimerWorm):
    name = "cdimerworm_stage"
    staged = True   # amendment 2026-10-16

    def generate(self, seed, region):
        return CKPopHardPenStage.generate(self, seed, region)

    def info(self):
        d = super().info()
        R, F = self._stage_RF()
        d = dict(d) if isinstance(d, dict) else {"base": d}
        d["stage"] = {"R": R, "F": F}
        return d


for _k in _BORROW:
    setattr(CDimerWormStage, _k, CKPopHardPenStage.__dict__[_k])
register(CDimerWormStage)
