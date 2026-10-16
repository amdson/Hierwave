"""clongfit_stage (2026-10-16 overnight, agent longfit2): clongfit's training (ML only, nctx 64, ml_B 32, no penalty)
plus cstage's fixed stage (Amendment 2026-10-16) as generation.  Pattern: cfitstage.py.  Stage R/F via cfg stage_R (default 30), stage_F (150);
bench with stage_R 15 as set below."""
from __future__ import annotations

from castlegen.bench.variant import VARIANTS, register

from . import clongfit  # noqa: F401  (registers clongfit)
from .cstage import CKPopHardPenStage

_BORROW = ("STAGE_R", "STAGE_F", "REACH_ROUND", "REACH_REFILL", "BATCH", "_stage_RF", "_reach", "_pass")


class CLongFitStage(VARIANTS["clongfit"]):
    name = "clongfit_stage"
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
    setattr(CLongFitStage, _k, CKPopHardPenStage.__dict__[_k])
CLongFitStage.STAGE_R = 15
register(CLongFitStage)
