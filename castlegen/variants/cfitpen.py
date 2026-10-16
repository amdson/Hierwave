"""cfitpen (wave3 2026-10-16, agent fitpen): ckpop_hard_pen with the ML fit rebalanced against the penalty.

Question: can the pass fit its kernel-refined population (population line_match ~0.64, own draws 0.55)
while the penalty-only term keeps L2 under budget, by giving the ML fit more weight per iteration?
Engineering freedoms only (step counts, learning rates, schedules); objective, kernels, lambda control
(0 <= lam <= 20, constraint 7) and generation unchanged.  Objective: weighted ML on own draws + p*-invariant
kernels, plus cfast pen_only steps (kernel-route amendment + change request #2).

Knobs (defaults = ckpop_hard_pen exactly):
  upi      ML updates per iteration (ckpop_hard_pen: 8)
  ml_lr    ML Adam lr (1e-3)
  pen_every  run the rkl_n penalty steps only every k-th iteration (1 = every iteration)
"""
from __future__ import annotations

import numpy as np

from castlegen.bench.variant import register

from .ckernel_own import CKernelPop, P3

HARD_PEN = dict(hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3], pen_only=True)
FIT = dict(pen_every=1)


class CFitPen(CKernelPop):
    name = "cfitpen"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**P3, **HARD_PEN, **FIT, **cfg})
        self._pen_calls = 0
        self._pen_last = None

    def _ground_stats(self, st):
        """+ offset's per-level implied surface offset (rows; reportables/2026-10-16_offset/scripts/offset.py
        measure): H - sum_rows(rho_l)/2^l minus the designed line, per level-l column, mean over worlds."""
        d = super()._ground_stats(st)
        if self.problem.name != "ground" or "fill" not in st:
            return d
        from castlegen.bench.problems import ground as G
        p = self.problem
        f = np.asarray(st["fill"]); W = np.asarray(st["t"]).shape[-1]
        gs = [G._line_from_fill(np.asarray(f[b], np.int64), W) for b in range(f.shape[0])]
        off = []
        for l in range(p.T + 1):
            k = "t" if l == 0 else f"rho{l}"
            if k not in st:
                off.append(None); continue
            r = np.asarray(st[k], np.float64); h = 1 << l
            off.append(float(np.mean([(p.H - r[b].sum(0) / h) - gs[b].reshape(-1, h).mean(1) for b in range(r.shape[0])])))
        d["offset"] = off
        return d

    def _rkl_steps(self, n):
        k = int(self.cfg.get("pen_every", 1))
        if n > 1:                       # per-iteration call (the setup warm call has n = 1)
            self._pen_calls += 1
            if k > 1 and self._pen_calls % k != 0:
                return self._pen_last if self._pen_last is not None else np.zeros(self.problem.T + 1)
        out = super()._rkl_steps(n)
        self._pen_last = out
        return out


register(CFitPen)


def _fit_variant(name, **over):
    cls = type(f"CFitPen_{name}", (CFitPen,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_fit_variant("cfitpen_ml2", upi=16)          # (b) 2x ML steps per penalty step, same penalty steps per iteration
_fit_variant("cfitpen_mllr2", ml_lr=2e-3)    # (b') 2x ML step size, same step counts (iteration rate unchanged)
_fit_variant("cfitpen_pe2", pen_every=2)     # (b'') penalty steps every 2nd iteration
