"""ckblock (wave 2, agent kblock): ckpop_hard_pen with block-Gibbs sweeps added to the population refresh.

Training-only kernels under the constitution's "Amendment 2026-10-14: kernel-route training" (exact block
Gibbs on small blocks given their boundary); generation is unchanged (cfast's coloured pass).

Everything is ckpop_hard_pen (castlegen/variants/ckernel_own.py: refresh 0.03, phase-2 refine incl. 3 x
(J20, J11) at lam 20, 20 colour-Gibbs sweeps at lam 20, 2 hard single-site sweeps, 8 ML + 32 penalty-only
steps).  Added at the END of each refine call (i.e. before the colour-Gibbs and hard sweeps), kb_rounds
rounds of:
  kb_ss  single-site colour-Gibbs sweeps of all levels (lam = kb_lam; None = hard),
  kb_b11 parent+4-children block sweeps at level 1 (JaxBlockGibbs level 1 depth 1, lam = kb_lam),
  kb_f2  2x2 level-0 tile block sweeps (FineBlockKernel k = 2, random offset from a host rng, lam = kb_lam).
p*-invariance of these kernels is tested in castlegen/kernels tests (p5 / p10 reused them); not re-proved here.
With kb_rounds = 0 the variant is bit-identical to ckpop_hard_pen (the kb key is a fold_in of the refine key,
no other rng is consumed).  Logged per iteration: time_kb (ms per refresh of the kb sweeps), kb_chg_t (tile
cells changed by the kb sweeps), plus the population stats/levels ckpop already logs.
"""
from __future__ import annotations

import time

import jax
import numpy as np

from castlegen.bench.variant import register

from .ckernel_own import P3, CKernelPop

HP = dict(hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3], pen_only=True)
KB = dict(kb_rounds=1, kb_ss=0, kb_b11=1, kb_f2=1, kb_lam=None)


class CKPopHardPenKB(CKernelPop):
    name = "ckpop_hard_pen_kb"
    _BUILT_ATTRS = CKernelPop._BUILT_ATTRS + ("kbJ11", "kbF2")

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**P3, **HP, **KB, **cfg})

    def _build_uncached(self, region):
        from castlegen.kernels.amortised import JaxBlockGibbs
        from castlegen.kernels.fine_block import FineBlockKernel
        super()._build_uncached(region)
        c, p = self.cfg, self.problem
        lam = c["kb_lam"]
        self.kbJ11 = JaxBlockGibbs(p, region, 1, 1, lam=lam) if c["kb_rounds"] and c["kb_b11"] else None
        self.kbF2 = FineBlockKernel(p, region, k=2, lam=lam, seed=self.seed) if c["kb_rounds"] and c["kb_f2"] else None

    def refine(self, st, key, phase=2):
        out, d, v = super().refine(st, key, phase)
        c = self.cfg
        if phase != 2 or c["kb_rounds"] == 0:
            return out, d, v
        from castlegen.kernels import colour_gibbs as cg
        if self.kbF2 is not None and not getattr(self.kbF2, "_kb_warm", False):   # compile every offset once
            for i, J in enumerate(self.kbF2.Js):
                jax.block_until_ready(J.sweep(v, jax.random.PRNGKey(i))[0])
            self.kbF2._kb_warm = True
        t0 = time.time()
        t_before = v
        kk = jax.random.fold_in(key, 7919)
        for _ in range(c["kb_rounds"]):
            if c["kb_ss"]:
                st2 = self._from_flat(v)
                for _ in range(c["kb_ss"]):
                    kk, s = jax.random.split(kk)
                    st2 = cg.sweep_all(self.lay_tr, st2, 1.0, c["kb_lam"], s)
                v = self._to_flat(st2)
            for _ in range(c["kb_b11"]):
                kk, s = jax.random.split(kk)
                v, _ = self.kbJ11.sweep(v, s)
            for _ in range(c["kb_f2"]):
                kk, s = jax.random.split(kk)
                v, _ = self.kbF2.sweep(v, s)
        v = jax.block_until_ready(v)
        d["time_kb_ms"] = 1000 * (time.time() - t0)
        o, n = self.fl.off["t"], int(np.prod(self.fl.shape["t"]))
        d["kb_chg_t"] = float(np.asarray(v[:, o:o + n] != t_before[:, o:o + n]).mean())
        return self._from_flat(v), d, v


register(CKPopHardPenKB)


def _kb(name, **over):
    cls = type(f"CKPopHardPenKB_{name}", (CKPopHardPenKB,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_kb("ckpop_hard_pen_kb0", kb_rounds=0)          # identity check: must equal ckpop_hard_pen
_kb("ckpop_hard_pen_kb4", kb_rounds=4)          # 4 rounds of (J11, F2) per iteration
