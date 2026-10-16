"""ccertfree (overnight 2026-10-16, agent certfree; ideas #22 target B): the certificate-free pipeline.

Training: cfitpen_ml2's (population + kernels + ML + penalty, castlegen/variants/cfitpen.py) on `ground_cf`
(_certfree_problem.make_ground_cf: ground without certificate channels, honour factors and level >= 1 support; hard factor
left: support0 on tiles).  The coloured pass samples only rho and tiles.
Generation: pass on ground_cf (raw rho + tiles) -> certificates DERIVED bottom-up by the tight map
(_certfree_problem.derive_certs; declared as an abstraction map of the level below, read from the honour factors'
subsets, constraint 6) -> fixed stage on the FULL ground problem (cstage's v1b-d5 sweeps, R rounds, keyed noise) ->
certificates re-derived from the stage's tiles and refilled by F certificate-only sweeps (default 20; cstage: random
start, 150).  raw = pass + derived certificates (no stage).  staged = True (Amendment 2026-10-16); the dropped
certificate channels are a target change accepted by the user for this line (ideas #22, target B).
cfg: stage_R (30), stage_F (20), derive_start (True: refill starts from certificates re-derived from the stage's
tiles; False: keeps the stage-1 certificates).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise
from castlegen.bench.problem import Region
from castlegen.bench.variant import VARIANTS, register

from . import _cpass_core as core
from . import cfitpen  # noqa: F401
from . import cstage
from ._certfree_problem import derive_certs, make_ground_cf

LAM = cstage.LAM


class StageCF(cstage.Stage):
    """cstage.Stage on the full problem whose certificates start derived and are refilled from the derived values."""

    def run(self, st, seeds, R, F, x0=0, derive_start=True):
        from . import _cstage_ck as ck
        assert x0 % 64 == 0
        p, T = self.p, self.p.T
        seeds = jnp.asarray(np.asarray(seeds, np.int64).astype(np.uint32).astype(np.int32))
        if not hasattr(self, "_jb"):
            self._jb = ck.make_block_sweep(self.J1, self.flat, p)
            self._fb = [ck.make_block_sweep(J, self.flat, p) for J in self.F2.Js]
        st = dict(st)
        if "cert1" not in st:
            st.update(derive_certs(p, st["t"], xp=jnp))
        x0j = jnp.int32(x0)
        for r in range(R):
            for l in range(T, -1, -1):
                if self.layg.chans[l]:
                    st = ck.sweep_level(self.layg, st, l, seeds, [ck.tag("c", r, l, c.name) for c in self.layg.chans[l]], LAM, x0)
            v = self.to_v(st)
            v = self._jb(v, seeds, jnp.asarray([ck.tag("J", r, i) for i in range(len(self.J1.classes))], jnp.int32), x0j)
            i4 = r % len(self.F2.Js)
            J = self.F2.Js[i4]
            v = self._fb[i4](v, seeds, jnp.asarray([ck.tag("F", r, i) for i in range(len(J.classes))], jnp.int32), x0j)
            st = self.to_st(v)
        st = dict(st)
        if derive_start and R > 0:
            st.update(derive_certs(p, st["t"], xp=jnp))
        cmask = {c.name: jnp.full(st[c.name].shape, c.name.startswith("cert"), bool) for c in p.sampled() if c.level >= 1}
        for j in range(F):
            for l in range(T, 0, -1):
                if self.layg.chans[l]:
                    st = ck.sweep_level(self.layg, st, l, seeds, [ck.tag("f", j, l, c.name) for c in self.layg.chans[l]],
                                        LAM, x0, mask=cmask)
        return st


_STAGES = {}


def get_stage_cf(p, W):
    k = (p.name, W)
    if k not in _STAGES:
        _STAGES[k] = StageCF(p, W)
    return _STAGES[k]


class CCertFree(VARIANTS["cfitpen_ml2"]):
    name = "ccertfree"
    staged = True       # amendment 2026-10-16
    STAGE_R, STAGE_F = 30, 20
    REACH_ROUND, REACH_REFILL, BATCH = (cstage.CKPopHardPenStage.REACH_ROUND, cstage.CKPopHardPenStage.REACH_REFILL,
                                        cstage.CKPopHardPenStage.BATCH)

    def __init__(self, problem, seed, workdir, **cfg):
        self.full = problem
        super().__init__(make_ground_cf(problem), seed, workdir, **{"init": None, **cfg})

    def _ground_stats(self, st):
        pr = self.problem
        if pr.name != "ground_cf":
            return super()._ground_stats(st)
        pr.name = "ground"                      # ckernel's / cfitpen's ground-only stats read the name
        try:
            return super()._ground_stats(st)
        finally:
            pr.name = "ground_cf"

    def _stage_RF(self):
        return int(self.cfg.get("stage_R", self.STAGE_R)), int(self.cfg.get("stage_F", self.STAGE_F))

    _reach = cstage.CKPopHardPenStage._reach

    def _pass(self, seed, ext):
        lay = self.layout(ext)
        des = self.designed(seed, ext)
        return des, lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))

    def _full_state(self, des, raw):
        """pass output (cf channels) + designed -> jnp full-ground state with derived certificates, batch dim 1 or n."""
        st = {**{k: jnp.asarray(v) for k, v in des.items()}, **{c.name: jnp.asarray(raw[c.name], jnp.int32) for c in self.problem.sampled()}}
        return st

    def generate(self, seed, region):
        R, F = self._stage_RF()
        full = self.full
        sampled = full.sampled()
        ds = bool(self.cfg.get("derive_start", True))
        if region.periodic:
            cache = self.__dict__.setdefault("_gen_cache", {})
            ck = (seed, region.W)
            if ck not in cache:
                cache.clear()
                seeds = [seed + i for i in range(self.BATCH)]
                dr = [self._pass(s, region) for s in seeds]
                st = {k: jnp.asarray(np.stack([np.asarray(d[0][k]) for d in dr])) for k in dr[0][0]}
                st.update({c.name: jnp.asarray(np.stack([np.asarray(d[1][c.name]) for d in dr]), jnp.int32) for c in self.problem.sampled()})
                st.update(derive_certs(full, st["t"], xp=jnp))
                raw = {c.name: np.asarray(st[c.name]) for c in sampled}
                fin = get_stage_cf(full, region.W).run(st, seeds, R, F, x0=0, derive_start=ds) if (R or F) else st
                for i, s in enumerate(seeds):
                    cache[(s, region.W)] = {"raw": {k: v[i] for k, v in raw.items()},
                                            "final": {c.name: np.asarray(fin[c.name])[i] for c in sampled}}
            return cache.pop(ck)
        from .cpass import window_of
        m = core.margin(self.problem, window_of(self.cfg.get("win")))
        ms = self._reach(R, F)
        M = m + ms
        ext = Region(region.W + 2 * M, periodic=False, x0=region.x0 - M)
        des, rawp = self._pass(seed, ext)
        st = self._full_state(des, rawp)
        st = {k: v[None] if v.ndim == 2 else v for k, v in st.items()}
        st.update(derive_certs(full, st["t"], xp=jnp))
        raw = {c.name: np.asarray(st[c.name])[0] for c in sampled}
        sub = {c.name: st[c.name][..., ((M - ms) >> c.level):((M + region.W + ms) >> c.level)] for c in full.chans}
        fin = get_stage_cf(full, region.W + 2 * ms).run(sub, [seed], R, F, x0=region.x0 - ms, derive_start=ds) if (R or F) else sub
        out = {}
        for kind, s, o in (("raw", raw, M), ("final", {c.name: np.asarray(fin[c.name])[0] for c in sampled}, ms)):
            out[kind] = {c.name: s[c.name][:, (o >> c.level):(o >> c.level) + (region.W >> c.level)] for c in sampled}
        return out

    def score_sites(self, x, region):
        s = super().score_sites(x, region)
        for c in self.full.sampled():
            s.setdefault(c.name, np.zeros((self.full.H >> c.level, region.W >> c.level)))
        return s

    def info(self):
        d = super().info()
        R, F = self._stage_RF()
        d = dict(d) if isinstance(d, dict) else {"base": d}
        d["stage"] = {"R": R, "F": F, "lam": LAM, "kind": "v1b d5, derived-cert refill", "certfree": True}
        return d


register(CCertFree)
