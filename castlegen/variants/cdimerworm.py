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

WORM = dict(init=None, worm=1, worm_lmax=1024, upi=16, hard_sweeps=0, cg_sweeps=0, m_flip=0, tt_Ls=(), m_l2=0, n_fine=0,
            refresh=0.1)


class CDimerWorm(CFitPen):
    name = "cdimerworm"

    def __init__(self, problem, seed, workdir, **cfg):
        assert problem.name == "dimers"
        super().__init__(problem, seed, workdir, **{**WORM, **cfg})
        self._wseed = 1000 * seed + 17

    def _build_uncached(self, region):
        """Only what the worm route needs (ML step, energy, generator).  The level-1/2 block chains and coarse flips
        of ckernel_own do not exist at T = 0; the energy uses a 3x3 FineBlockKernel's instance list (as dk_loop)."""
        import jax
        from castlegen.kernels.amortised import make_energy, make_ml_update_drawwise
        from castlegen.kernels.fine_block import FineBlockKernel
        from ._cfast_core import train_pass
        fb = FineBlockKernel(self.problem, region, 3, lam=float(self.cfg["kernel_lam"]), seed=self.seed)
        self.fl = fb.flat
        self.E = make_energy(fb.flat, fb.Ks[0].insts, fine_max=0)
        lay = self.layout(region)
        self.mlstep = make_ml_update_drawwise(lay)
        lam_gen = jnp.asarray(self.lam, jnp.float32)
        self.gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lam_gen, record=False)[0],
                                    in_axes=(None, 0, 0)))
        self.lay_tr = lay

    def _pop_stats(self, st, v):
        s, nv, _ = self.E(v)
        nv = np.asarray(nv)
        o = np.asarray(st["p"]) // 2
        return {"valid": float((nv == 0).mean()), "viol_per_cell": float(nv.mean() / o[0].size),
                "hvd": [float((o == i).mean()) for i in range(3)]}

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
