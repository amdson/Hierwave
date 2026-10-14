"""cguided: cfast-family variants whose generation adds R guided revising steps (_guided_step.py)
after the coloured pass, then cpass's one-round repair.  Training is the base variant's, unchanged
(the base classes never call generate() during training), so saved parameters of the base variant
can be evaluated with the revised generation without retraining.

    ckpop_hard_pen_gs2 / _gs4   ckpop_hard_pen (change request #2 arm) + 2 / 4 guided steps
    ckpop_hard_rkl_gs2 / _gs4   ckpop_hard_rkl (wave4-6 own-draws arm) + 2 / 4 guided steps

cfg: gs_R (steps), gs_dilate (selection neighbourhood, cells), gs_descend (redraw descendants of
selected coarse cells), gs_repair (cpass repair after the steps; default True).  "raw" is the pass
draw, "final" the revised and repaired state.  Dependency radius: _cpass_core.margin + guided_margin.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import VARIANTS, register

from . import _cpass_core as core
from . import ckernel_own  # noqa: F401  (registers the base variants)
from ._cln_core import window_of
from ._guided_step import guided_margin, guided_revise

GS = dict(gs_R=2, gs_dilate=1, gs_descend=True, gs_repair=True)


def _revise_fn(self, lay):
    key = ("guided", self.cfg["gs_R"], self.cfg["gs_dilate"], bool(self.cfg["gs_descend"]), bool(self.cfg["gs_repair"]))
    fns = getattr(lay, "guided_fns", None)
    if fns is None:
        fns = lay.guided_fns = {}
    if key not in fns:
        R, dil, desc, rep = key[1:]

        def f(params, st, seed):
            st, info = guided_revise(lay, params, st, seed, R, 1.0, dil, desc)
            if rep:
                st = core.repair(lay, st, seed)
            return st, info["changed"]
        fns[key] = jax.jit(f)
    return fns[key]


def _generate(self, seed, region):
    if region.periodic:
        lay = self.layout(region)
        des = self.designed(seed, region)
        raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
        fin, chg = _revise_fn(self, lay)(self.params, raw, seed)
        crop = lambda st, c: np.asarray(st[c.name])
    else:
        m = core.margin(self.problem, window_of(self.cfg.get("win"))) + guided_margin(self.problem, self.cfg["gs_R"], self.cfg["gs_dilate"])
        ext = Region(region.W + 2 * m, periodic=False, x0=region.x0 - m)
        lay = self.layout(ext)
        des = self.designed(seed, ext)
        raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
        fin, chg = _revise_fn(self, lay)(self.params, raw, seed)
        crop = lambda st, c: np.asarray(st[c.name])[:, (m >> c.level):(m >> c.level) + (region.W >> c.level)]
    self.last_changed = np.asarray(chg)
    out = {}
    for kind, st in (("raw", raw), ("final", fin)):
        out[kind] = {c.name: crop(st, c) for c in self.problem.sampled()}
    return out


def _info(self):
    return {**super(type(self), self).info(), "guided": {k: self.cfg[k] for k in GS},
            "guided_margin": guided_margin(self.problem, self.cfg["gs_R"], self.cfg["gs_dilate"])}


def guided_variant(base, name, **over):
    """Register `name`: `base` (a registered variant name) with guided revising generation."""
    Base = VARIANTS[base]
    cls = type(f"Guided_{name}", (Base,), {"name": name, "generate": _generate, "info": _info})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**GS, **over, **cfg})
    cls.__init__ = init
    return register(cls)


guided_variant("ckpop_hard_pen", "ckpop_hard_pen_gs2", gs_R=2)
guided_variant("ckpop_hard_pen", "ckpop_hard_pen_gs4", gs_R=4)
guided_variant("ckpop_hard_rkl", "ckpop_hard_rkl_gs2", gs_R=2)
guided_variant("ckpop_hard_rkl", "ckpop_hard_rkl_gs4", gs_R=4)
