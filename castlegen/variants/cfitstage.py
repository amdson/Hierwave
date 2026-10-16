"""cfitstage (wave 4 2026-10-16, agent combo): cfitpen_ml2's training + cstage's fixed stage as generation.

cfitpen_ml2_stage = cfitpen_ml2 (ckpop_hard_pen with 16 ML steps per iteration; castlegen/variants/cfitpen.py)
whose generate() is CKPopHardPenStage's (castlegen/variants/cstage.py: keyed-noise v1b-d5 stage, padded by its
declared reach, Amendment 2026-10-16).  Training is untouched; the stage contributes no gradient.
cfg: stage_R (default 30), stage_F (default 150).  On the first generate call the trained params/lam are
pickled to <workdir>/params_combo.pkl (offline stage analysis only; does not change any output).
"""
from __future__ import annotations

import os
import pickle

import jax
import numpy as np

from castlegen.bench.variant import VARIANTS, register

from . import cfitpen  # noqa: F401  (registers cfitpen_ml2)
from .cstage import CKPopHardPenStage, LAM

_BORROW = ("STAGE_R", "STAGE_F", "REACH_ROUND", "REACH_REFILL", "BATCH", "_stage_RF", "_reach", "_pass")


class CFitPenML2Stage(VARIANTS["cfitpen_ml2"]):
    name = "cfitpen_ml2_stage"
    staged = True   # amendment 2026-10-16

    def generate(self, seed, region):
        if not getattr(self, "_saved_params", False):
            self._saved_params = True
            try:
                wd = getattr(self, "workdir", None)
                if wd:
                    os.makedirs(str(wd), exist_ok=True)
                    with open(os.path.join(str(wd), "params_combo.pkl"), "wb") as f:
                        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, self.params),
                                     "lam": np.asarray(self.lam, float)}, f)
            except Exception as e:  # never let the dump affect generation
                print("ccombo: params dump failed:", e)
        return CKPopHardPenStage.generate(self, seed, region)

    def info(self):
        d = super().info()
        R, F = self._stage_RF()
        d = dict(d) if isinstance(d, dict) else {"base": d}
        d["stage"] = {"R": R, "F": F, "lam": LAM, "kind": "v1b d5 (p10)"}
        return d


for _k in _BORROW:
    setattr(CFitPenML2Stage, _k, CKPopHardPenStage.__dict__[_k])

register(CFitPenML2Stage)
