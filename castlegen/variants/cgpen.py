"""cgpen (ideas #21; constitution addendum 2026-10-14 (c), training only): cfast's penalty-only update
with the GRADED violation mass, objective J_gpen = sum_l lam_l E_q[D_l] (D_l the summed per-instance
distance to feasibility of castlegen.graded) estimated by cfast's unbiased pen_only estimator with
ell = lam_l * nvg (closed graded distance per candidate) in place of lam_l * nv.  Generation is cfast's
coloured pass + repair, unchanged (the masks still use the binary closed violations + lookahead).  The
lambda control is cfast's, driven by the binary raw violation rate.  Core: castlegen/variants/_graded_core.py.

    cgpen            CFast with the graded penalty-only update (K = 0, cone returns, critic h1)
    ckpop_hard_gpen  ckpop_hard_pen (ckernel_own.py: persistent own-draw population + hard sweeps + 32
                     penalty-only steps per iteration, own draws only) with the 32 steps = cgpen's update
"""
from __future__ import annotations

from castlegen.bench.variant import VARIANTS, register

from . import _graded_core
from . import ckernel_own  # noqa: F401  (registers the ckpop_* family)
from .cfast import CFast


def _gpen_fns(var, lay):
    key = ("gpen", var.cfg["critic"], var.cfg["sub"], var.cfg.get("net", "tanh"), var.cfg["cone"])
    if not hasattr(lay, "gpen_fns"):
        lay.gpen_fns = {}
    if key not in lay.gpen_fns:
        lay.gpen_fns[key] = _graded_core.build_train(var, lay)
    return lay.gpen_fns[key]


@register
class CGPen(CFast):
    name = "cgpen"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{"pen_only": True, "K": 0, **cfg})
        assert self.cfg["K"] == 0 and self.cfg["pen_only"]

    def train_fns(self, lay):
        return _gpen_fns(self, lay)

    def _build_train(self, lay):
        return _graded_core.build_train(self, lay)


class CKPopHardGPen(VARIANTS["ckpop_hard_pen"]):
    """ckpop_hard_pen whose 32 penalty-only steps per iteration use the graded mass (cgpen's fused update)."""
    name = "ckpop_hard_gpen"

    def train_fns(self, lay):
        return _gpen_fns(self, lay)

    def _build_train(self, lay):
        return _graded_core.build_train(self, lay)


register(CKPopHardGPen)
