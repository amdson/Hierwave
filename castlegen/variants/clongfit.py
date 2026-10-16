"""clongfit (2026-10-16 overnight, agent longfit): cfitpen_ml2's population and kernels, ML steps only.

No penalty term (rkl_n = 0; the lambda control stays at its init and is inert), actor = ckernel's LN + SiLU,
6x6 window (CK default net="ln", win=(-2, 3): the same actor as cfitpen_ml2 and cln6), larger ML batch and
population.  Meant for a long (hour) run; the driver is reportables/2026-10-16_longfit/scripts/longfit.py.
Engineering freedoms only (step counts, batch sizes, run length).  Objective, kernels, generation unchanged.
"""
from __future__ import annotations

from castlegen.bench.variant import register

from .cfitpen import CFitPen


def _mk(name, **over):
    cls = type(f"CLongFit_{name}", (CFitPen,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


# upi 16 = cfitpen_ml2; nctx/ml_B larger; no penalty steps
_mk("clongfit", upi=16, rkl_n=0, nctx=64, ml_B=32)
_mk("clongfit_small", upi=16, rkl_n=0)          # smoke / speed reference: ml2's nctx 32, ml_B 8, no penalty
