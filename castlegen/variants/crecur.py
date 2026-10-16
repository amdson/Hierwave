"""crecur (overnight 2026-10-16, agent recur): constitution amendment 2026-10-16 (b), recursive (K-sample) training
of the coarse levels l >= rec_l0 on top of cfitpen_ml2's fitted fine levels.

Training per iteration = cfitpen_ml2's iteration (population refresh, p*-invariant kernels, 16 weighted-ML steps,
32 pen-only steps, lambda control) with two changes when rec_n > 0:
  * the ML steps and the pen-only steps touch only the fine levels l < rec_l0 (ML level weight 0 at coarse
    levels; pen rkl_levels = fine levels), so the fine levels keep exactly their own objective;
  * rec_n recursive steps (castlegen/variants/_crecur_core.py) train the coarse levels: per coarse draw the
    return is its coarse cone cost plus R_K = -log (1/K) sum_k w_k over K fine completions of the same coarse
    state (cone-local), tau = 1, score function + all-action own-cost term, leave-one-out baseline over worlds.
rec_n = 0 is cfitpen_ml2 unchanged (the control).  init: a params.pkl ({"params", "lam"}) to start from;
lambda is then kept (rkl_lam0 None) and controlled as before (0 <= lam <= 20).
Generation unchanged (the coloured pass); crecur_stage adds cfitstage's fixed stage (staged = True).
"""
from __future__ import annotations

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import VARIANTS, register

from . import cfitpen  # noqa: F401  (registers cfitpen_ml2)
from ._crecur_core import build
from .cpass import adam_init

FITPEN_PARAMS = "reportables/2026-10-16_fitpen/runs/arm_cfitpen_ml2/params.pkl"
REC = dict(rec_l0=3, rec_K=4, rec_n=8, rec_B=8, rec_lr=1e-3, rec_mode="footonly", rec_clip=10.0,
           rec_probe_frac=0.0,   # ideas #14 contexts: this fraction of the rec worlds has L_T clamped to a probe lift
           init_params=None)


class CRecur(VARIANTS["cfitpen_ml2"]):
    name = "crecur"

    def __init__(self, problem, seed, workdir, **cfg):
        cfg = {**REC, **cfg}
        if cfg["rec_n"] > 0:
            cfg.setdefault("rkl_levels", [l for l in range(problem.T + 1) if l < cfg["rec_l0"]])
        if cfg.get("init_params"):
            cfg.setdefault("rkl_lam0", None)
        super().__init__(problem, seed, workdir, **cfg)
        if self.cfg.get("init_params"):
            ck = pickle.load(open(self.cfg["init_params"], "rb"))
            tr = ck["trained"] if "trained" in ck else ck
            assert set(tr["params"]) == set(self.params)
            self.params = jax.tree_util.tree_map(jnp.asarray, tr["params"])
            self.lam = np.asarray(tr["lam"], float)
            self.start = self.cfg["init_params"]
        self.rec_updates = 0
        self.rec_last = None
        self._fseed = 1_500_000_000 + 10_000_000 * seed

    def _build(self, region):
        super()._build(region)
        if self.cfg["rec_n"] > 0:
            orig = self.mlstep
            fm = jnp.asarray([1.0 if l < self.cfg["rec_l0"] else 0.0 for l in range(self.problem.T + 1)], jnp.float32)
            self.mlstep = lambda params, opt, batch, w, lr, on, _o=orig: _o(params, opt, batch, w, lr, on * fm)

    def _rkl_steps(self, n):
        out = super()._rkl_steps(n)
        if self.cfg["rec_n"] > 0:
            self._rec_steps(1 if n <= 1 else self.cfg["rec_n"])
        return out

    def _rec_steps(self, n):
        c, lay = self.cfg, self.lay_tr
        region = Region(lay.W)
        if not hasattr(self, "rfn"):
            self.rfn = build(lay, self.fn["cone"], c["rec_l0"], c["rec_K"], mode=c["rec_mode"], Lc=self.problem.T)
            self.ropt = adam_init({k: self.params[k] for k in self.rfn["ckeys"]})
        B, K = c["rec_B"], c["rec_K"]
        acc = []
        for _ in range(n):
            seeds = np.arange(self.train_seed, self.train_seed + B); self.train_seed += B
            fs = np.arange(self._fseed, self._fseed + B * K).reshape(B, K); self._fseed += B * K
            des = self._batch(seeds, region)
            clamp, cflag = self._probe_clamp(seeds, region)
            self.params, self.ropt, loss, viol, diag, gstd = self.rfn["step"](
                self.params, self.ropt, jnp.asarray(seeds, jnp.int32), jnp.asarray(fs, jnp.int32), des,
                jnp.asarray(self.lam, jnp.float32), jnp.float32(c["rec_lr"]), jnp.float32(c["rec_clip"]), clamp, cflag)
            self.rec_updates += 1
            acc.append({"loss": loss, **diag, **{f"Gstd_{k}": v for k, v in gstd.items()}})
        acc = jax.device_get(acc)
        self.rec_last = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        self.rec_last["rec_updates"] = self.rec_updates

    def _probe_clamp(self, seeds, region):
        """First round(rec_probe_frac * B) worlds: the coarsest level clamped to a probe lift of the same seed
        (problem.probes, jitter variant k uniform in 0-3; designed channels of the same seed, as ckernel's
        _probe_ctx).  None when rec_probe_frac = 0."""
        fr = float(self.cfg["rec_probe_frac"])
        if fr <= 0:
            return None, None
        p, T = self.problem, self.problem.T
        nc = int(round(fr * len(seeds)))
        names = [ch.name for ch in self.lay_tr.chans[T]]
        cl = {n: np.zeros((len(seeds),) + tuple(self.lay_tr.hw[T]), np.int32) for n in names}
        for b in range(nc):
            k = int(self.rng.integers(4))
            pr = p.probes(int(seeds[b]), region, n=k + 1)[k]
            for n in names:
                cl[n][b] = np.asarray(pr[n])
        flag = np.arange(len(seeds)) < nc
        return {n: jnp.asarray(v) for n, v in cl.items()}, jnp.asarray(flag)

    def info(self):
        return {**super().info(), "rec_updates": self.rec_updates,
                "rec": {k: self.cfg[k] for k in REC}}


register(CRecur)


def _rec_variant(name, base=CRecur, **over):
    cls = type(f"CRecur_{name}", (base,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


# crecur_stage: crecur training + cfitstage's fixed stage as generation (Amendment 2026-10-16), staged = True.
from .cfitstage import _BORROW  # noqa: E402
from .cstage import CKPopHardPenStage  # noqa: E402


class CRecurStage(CRecur):
    name = "crecur_stage"
    staged = True

    def generate(self, seed, region):
        if not getattr(self, "_saved_params", False):
            self._saved_params = True
            try:
                wd = getattr(self, "workdir", None)
                if wd:
                    os.makedirs(str(wd), exist_ok=True)
                    with open(os.path.join(str(wd), f"params_recur_s{self.seed}.pkl"), "wb") as f:
                        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, self.params),
                                     "lam": np.asarray(self.lam, float)}, f)
            except Exception as e:  # never let the dump affect generation
                print("crecur: params dump failed:", e)
        return CKPopHardPenStage.generate(self, seed, region)

    def info(self):
        d = super().info()
        R, F = self._stage_RF()
        d["stage"] = {"R": R, "F": F, "kind": "v1b d5 (p10), as cfitpen_ml2_stage"}
        return d


for _k in _BORROW:
    setattr(CRecurStage, _k, CKPopHardPenStage.__dict__[_k])

register(CRecurStage)
