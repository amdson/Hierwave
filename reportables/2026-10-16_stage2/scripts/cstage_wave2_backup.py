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

Noise: the sweeps' Gumbel noise is jax.random keyed by (seed, round / sweep index), i.e. by array position on
the strip, not by absolute coordinate (colour_gibbs and the block kernels draw from one key per sweep).  On a
periodic strip this is a valid fixed-seed draw; across open windows it is not coordinate-consistent.  Open
windows (finite-support check) run the stage on the window + STAGE_PAD columns each side, cut from the pass's
margin-extended window and treated as periodic, then crop.
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

    def run(self, st, seed, R, F, x0=0):
        """st: {chan: (n, h, w)} incl. designed channels.  x0: absolute fine x of column 0 (cert reset keys)."""
        p = self.p
        T = p.T
        key = jax.random.fold_in(jax.random.PRNGKey(0), int(seed) % (2 ** 31))
        lev = list(range(T, -1, -1))
        n = st[self.order[0].name].shape[0]
        for r in range(R):
            k1, k2, k3 = jax.random.split(jax.random.fold_in(key, r), 3)
            st = core_sweep_all(self.layg, st, k1, lev)
            v = self.to_v(st)
            v, _ = self.J1.sweep(v, k2)
            v, _ = self.F2.Js[r % len(self.F2.Js)].sweep(v, k3)
            st = self.to_st(v)
        st = dict(st)
        for c in self.certs:
            h, w = st[c.name].shape[1:]
            Y, X = np.meshgrid(np.arange(h), (x0 >> c.level) + np.arange(w), indexing="ij")
            u = noise.hash32(np, seed, TAG_CERT, c.level, Y, X) % np.uint32(c.D)
            st[c.name] = jnp.asarray(np.broadcast_to(u.astype(np.int32), (n, h, w)))
        cmask = {c.name: jnp.full(st[c.name].shape, c.name.startswith("cert"), bool)
                 for c in p.sampled() if c.level >= 1}
        clev = list(range(T, 0, -1))
        for j in range(F):
            st = core_sweep_all(self.layg, st, jax.random.fold_in(key, 100000 + j), clev, cmask)
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
    STAGE_PAD = 128          # fine columns of stage padding on open windows (multiple of P << T)

    def _stage_RF(self):
        return int(self.cfg.get("stage_R", self.STAGE_R)), int(self.cfg.get("stage_F", self.STAGE_F))

    def generate(self, seed, region):
        from .cpass import window_of
        R, F = self._stage_RF()
        if region.periodic:
            ext, m = region, 0
        else:
            m = core.margin(self.problem, window_of(self.cfg.get("win")))
            ext = Region(region.W + 2 * m, periodic=False, x0=region.x0 - m)
        lay = self.layout(ext)
        des = self.designed(seed, ext)
        raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
        st = {**{k: jnp.asarray(v)[None] for k, v in des.items()},
              **{c.name: jnp.asarray(raw[c.name], jnp.int32)[None] for c in self.problem.sampled()}}
        if region.periodic:
            fin = get_stage(self.problem, ext.W).run(st, seed, R, F, x0=0)
            ms = 0
        else:
            # stage on the region + STAGE_PAD fine columns each side (cut from the pass's extended window),
            # treated as periodic: the full pass margin (2752 columns each side) is too wide for the block kernels
            ms = self.STAGE_PAD
            sub = {c.name: v[..., ((m - ms) >> c.level):((m + region.W + ms) >> c.level)]
                   for c in self.problem.chans for v in [st[c.name]]}
            fin = get_stage(self.problem, region.W + 2 * ms).run(sub, seed, R, F, x0=region.x0 - ms)
        out = {}
        for kind, s, o in (("raw", raw, m), ("final", {k: v[0] for k, v in fin.items()}, ms)):
            out[kind] = {c.name: np.asarray(s[c.name])[:, (o >> c.level):(o >> c.level) + (region.W >> c.level)]
                         for c in self.problem.sampled()}
        return out

    def info(self):
        d = super().info()
        R, F = self._stage_RF()
        d = dict(d) if isinstance(d, dict) else {"base": d}
        d["stage"] = {"R": R, "F": F, "lam": LAM, "kind": "v1b d5 (p10)"}
        return d


register(CKPopHardPenStage)
