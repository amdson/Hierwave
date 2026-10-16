"""cstage (ideas #22, wave-2 agent `stage`; reportables/2026-10-15_stage): ckpop_hard_pen with p10's fixed
post-pass stage ("v1b d5") as its generation-time repair.  Training is ckpop_hard_pen's, unchanged.

Generation: the coloured pass draw (`raw`, unrepaired) of ckpop_hard_pen, then
  stage 1, R rounds, each: one single-cell colour-Gibbs sweep of all levels (T..0, every sampled channel,
           tau = 1, graded penalty lam = 20 on every hard factor), one level-1 parent+4-children exact block
           sweep (JaxBlockGibbs depth 1, binary lam = 20) and one 2x2 tile block sweep (FineBlockKernel k = 2)
           at offset (r mod 4) of the fixed offset cycle (no host rng);
  stage 2, every certificate channel (levels 1..T) reset to coordinate-keyed uniform values
           (castlegen.bench.noise, tag below) and refilled by F certificate-only single-cell sweeps (T..1).
`final` = the stage output; the pass's own repair is not used.  Fixed counts throughout (constraint 2); the
objective and training are untouched (constraint 4).

Noise (wave 3, agent stage2, reportables/2026-10-16_stage2): every Gumbel draw of every stage sweep comes from
castlegen.bench.noise keyed by (world seed, phase/round/sweep/channel or block class, level, Y, absolute X, value)
(_cstage_ck.py); the 2x2 block offset cycles r mod 4; counts are fixed.  Open windows: the pass runs on the
region + (pass margin + stage reach) each side, the stage on region + reach each side treated as periodic (the seam
is outside the stage's light cone).  Reach = R x REACH_ROUND + REACH_REFILL, from the schedule's exact light cone
(scripts/reach.py).  Periodic worlds are generated BATCH seeds at a time in one batched stage call and cached.
"""
from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise
from castlegen.bench.problem import Region
from castlegen.bench.variant import VARIANTS, register

from . import _cpass_core as core
from . import ckernel_own  # noqa: F401  (registers ckpop_hard_pen)
from ._cpass_core import Layout

LAM = 20.0
TAG_CERT = 0x57A6E


def _graded_table(t):
    fin = np.argwhere(np.isfinite(t))
    idx = np.indices(t.shape).reshape(t.ndim, -1).T
    d = np.abs(idx[:, None, :] - fin[None, :, :]).sum(-1).min(1)
    return d.reshape(t.shape).astype(np.float64)


def graded_problem(p, lam):
    """Copy of reportables/2026-10-14_cgraded/run.py: every infinite hard entry -> lam x L1 distance to feasibility."""
    q = copy.deepcopy(p)
    for f in q.factors:
        if f.hard:
            inf = np.isinf(f.table)
            f.table = np.where(inf, lam * _graded_table(f.table), f.table)
    return q


class Stage:
    """The fixed stage on a periodic strip of width W (batch of worlds)."""

    def __init__(self, p, W):
        from castlegen.kernels.amortised import JaxBlockGibbs
        from castlegen.kernels.exact_block import Flat
        from castlegen.kernels.fine_block import FineBlockKernel
        self.p, self.W = p, W
        reg = Region(W)
        self.layg = Layout(graded_problem(p, LAM), W, True)
        self.J1 = JaxBlockGibbs(p, reg, 1, 1, lam=LAM)
        self.F2 = FineBlockKernel(p, reg, k=2, lam=LAM)
        self.flat = Flat(p, reg)
        self.order = sorted(p.chans, key=lambda c: self.flat.off[c.name])
        self.certs = [c for c in p.sampled() if c.name.startswith("cert") and c.level >= 1]

    def to_v(self, st):
        n = st[self.order[0].name].shape[0]
        return jnp.concatenate([jnp.asarray(st[c.name]).reshape(n, -1) for c in self.order], 1).astype(jnp.int32)

    def to_st(self, v):
        out = {}
        for c in self.p.chans:
            o = self.flat.off[c.name]; h, w = self.flat.shape[c.name]
            out[c.name] = v[:, o:o + h * w].reshape(v.shape[0], h, w)
        return out

    def run(self, st, seeds, R, F, x0=0):
        """st: {chan: (n, h, w)} incl. designed channels; seeds: (n,) world seeds.  x0: absolute fine x of
        column 0 (a multiple of 64).  Every Gumbel draw is keyed by (seed, phase/round/sweep/channel, level,
        Y, absolute X, value) (_cstage_ck); the block offset cycle is r mod 4; counts are fixed."""
        from . import _cstage_ck as ck
        assert x0 % 64 == 0
        p = self.p
        T = p.T
        seeds = jnp.asarray(np.asarray(seeds, np.int64).astype(np.uint32).astype(np.int32))
        if not hasattr(self, "_jb"):
            self._jb = ck.make_block_sweep(self.J1, self.flat, p)
            self._fb = [ck.make_block_sweep(J, self.flat, p) for J in self.F2.Js]
        lev = list(range(T, -1, -1))
        x0j = jnp.int32(x0)
        for r in range(R):
            for l in lev:
                if self.layg.chans[l]:
                    st = ck.sweep_level(self.layg, st, l, seeds, [ck.tag("c", r, l, c.name) for c in self.layg.chans[l]],
                                        LAM, x0)
            v = self.to_v(st)
            v = self._jb(v, seeds, jnp.asarray([ck.tag("J", r, i) for i in range(len(self.J1.classes))], jnp.int32), x0j)
            i4 = r % len(self.F2.Js)
            J = self.F2.Js[i4]
            v = self._fb[i4](v, seeds, jnp.asarray([ck.tag("F", r, i) for i in range(len(J.classes))], jnp.int32), x0j)
            st = self.to_st(v)
        st = dict(st)
        sd = np.asarray(seeds).astype(np.int64)
        for c in self.certs:
            h, w = st[c.name].shape[1:]
            Y, X = np.meshgrid(np.arange(h), (x0 >> c.level) + np.arange(w), indexing="ij")
            u = noise.hash32(np, sd[:, None, None], TAG_CERT, c.level, Y[None], X[None]) % np.uint32(c.D)
            st[c.name] = jnp.asarray(u.astype(np.int32))
        n = sd.size
        cmask = {c.name: jnp.full(st[c.name].shape, c.name.startswith("cert"), bool)
                 for c in p.sampled() if c.level >= 1}
        for j in range(F):
            for l in range(T, 0, -1):
                if self.layg.chans[l]:
                    st = ck.sweep_level(self.layg, st, l, seeds, [ck.tag("f", j, l, c.name) for c in self.layg.chans[l]],
                                        LAM, x0, mask=cmask)
        return st


def core_sweep_all(lay, st, key, levels, mask=None):
    from castlegen.kernels import colour_gibbs as cg
    return cg.sweep_all(lay, st, 1.0, 0.0, key, levels=levels, mask=mask)


_STAGES = {}


def get_stage(p, W):
    k = (p.name, W)
    if k not in _STAGES:
        _STAGES[k] = Stage(p, W)
    return _STAGES[k]


class CKPopHardPenStage(VARIANTS["ckpop_hard_pen"]):
    """ckpop_hard_pen + the fixed v1b-d5 stage as repair (R, F in cfg: stage_R, stage_F)."""
    name = "ckpop_hard_pen_stage"
    staged = True   # amendment 2026-10-16
    STAGE_R, STAGE_F = 30, 150
    # exact light cone (reportables/2026-10-16_stage2/runs/reach2.log): rounds 1-4 reach 40/104/167/233 fine
    # columns right, 36/70/101/121 left; the refill saturates at +7 (certificates couple only inside a top-level
    # column).  Declared with margin:
    REACH_ROUND, REACH_REFILL = 80, 32

    def _stage_RF(self):
        return int(self.cfg.get("stage_R", self.STAGE_R)), int(self.cfg.get("stage_F", self.STAGE_F))

    BATCH = 32               # periodic worlds: the stage runs on BATCH consecutive seeds in one batched call

    def _reach(self, R, F):
        """Declared one-sided dependency radius of the stage (fine columns), from the exact light cone of its
        schedule (reportables/2026-10-16_stage2/scripts/reach.py): per round <= REACH_ROUND, the certificate
        refill <= REACH_REFILL in total (certificates couple only within a top-level column), rounded up to 64."""
        r = R * self.REACH_ROUND + (self.REACH_REFILL if F else 0)
        return -(-r // 64) * 64

    def _pass(self, seed, ext):
        lay = self.layout(ext)
        des = self.designed(seed, ext)
        raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
        return des, raw

    def generate(self, seed, region):
        from .cpass import window_of
        R, F = self._stage_RF()
        sampled = self.problem.sampled()
        if region.periodic:
            cache = self.__dict__.setdefault("_gen_cache", {})
            ck = (seed, region.W)
            if ck not in cache:
                cache.clear()
                seeds = [seed + i for i in range(self.BATCH)]
                ds, rs = zip(*[self._pass(s, region) for s in seeds])
                st = {**{k: jnp.asarray(np.stack([np.asarray(d[k]) for d in ds])) for k in ds[0]},
                      **{c.name: jnp.asarray(np.stack([np.asarray(r[c.name]) for r in rs]), jnp.int32) for c in sampled}}
                fin = get_stage(self.problem, region.W).run(st, seeds, R, F, x0=0)
                fin = {c.name: np.asarray(fin[c.name]) for c in sampled}
                for i, s in enumerate(seeds):
                    cache[(s, region.W)] = {"raw": {c.name: np.asarray(rs[i][c.name]) for c in sampled},
                                            "final": {c.name: fin[c.name][i] for c in sampled}}
            return cache.pop(ck)
        # open window: pass on region + (pass margin m + stage reach ms) each side, so the pass output is exact on
        # region +- ms; the stage runs on region +- ms treated as periodic (the seam is >= ms from the region,
        # outside the stage's light cone), with absolute colours/tilings (x0 multiple of 64) and keyed noise
        m = core.margin(self.problem, window_of(self.cfg.get("win")))
        ms = self._reach(R, F)
        M = m + ms
        ext = Region(region.W + 2 * M, periodic=False, x0=region.x0 - M)
        des, raw = self._pass(seed, ext)
        st = {**{k: jnp.asarray(v)[None] for k, v in des.items()},
              **{c.name: jnp.asarray(raw[c.name], jnp.int32)[None] for c in sampled}}
        sub = {c.name: st[c.name][..., ((M - ms) >> c.level):((M + region.W + ms) >> c.level)] for c in self.problem.chans}
        fin = get_stage(self.problem, region.W + 2 * ms).run(sub, [seed], R, F, x0=region.x0 - ms)
        out = {}
        for kind, s, o in (("raw", raw, M), ("final", {k: v[0] for k, v in fin.items()}, ms)):
            out[kind] = {c.name: np.asarray(s[c.name])[:, (o >> c.level):(o >> c.level) + (region.W >> c.level)]
                         for c in sampled}
        return out

    def info(self):
        d = super().info()
        R, F = self._stage_RF()
        d = dict(d) if isinstance(d, dict) else {"base": d}
        d["stage"] = {"R": R, "F": F, "lam": LAM, "kind": "v1b d5 (p10)"}
        return d


register(CKPopHardPenStage)


class CKPopHardPenStage120(CKPopHardPenStage):
    """Same stage at R = 120 rounds (reach 120 x 80 + 32 -> 9664 fine columns each side)."""
    name = "ckpop_hard_pen_stage120"
    STAGE_R = 120


register(CKPopHardPenStage120)
