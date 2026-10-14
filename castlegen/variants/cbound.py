"""cbound: cfast with a training-time curriculum that keeps the policy soft while
the penalty gradient dominates (cdim3: under lambda = 20 the first gradient points
at the locally safe action and the shared-weight actor saturates every context
within ~50 steps).  Engineering freedoms only (constitution: optimiser,
schedules, defaults incl. the lambda control); the objective, estimator, critic
and generation are cfast's.  Two schedules, both off by default (cbound with
no options == cfast):

- lambda ramp (lam_start, lam_frac): the penalty actually used by the update
  goes linearly from lam_start to lam0 over the first lam_frac of the
  post-setup training time; the lambda control is frozen during the ramp and
  takes over from lam0 when it ends.  Always 0 <= lambda <= 20 (constraint 7).
- output bound (c0, cmax, c_frac): after every update each actor's last layer
  is projected so that sum_j |W3[j, d]| + |b3[d]| <= c(t) (tanh hidden units,
  so |b_d| <= c for every input); c goes geometrically c0 -> cmax over the
  first c_frac of the post-setup time and stays at cmax.  Projected Adam:
  generation uses the final parameters, which satisfy the final bound.

Diagnostics (cbound_*_diag): csat's per-level entropy / bias spread / dirt and
ccur's line_match every diag_every updates; the dimers orientation fractions
are measured from saved parameters (reportables/2026-10-14_cbound/dimers.py).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.variant import register

from .ccur import CCur
from .cfast import CFast

_PROJ = {}


def _project(params, c):
    def one(p):
        if "W3" not in p:
            return p
        n = jnp.sum(jnp.abs(p["W3"]), 0) + jnp.abs(p["b3"])
        s = jnp.minimum(1.0, c / jnp.maximum(n, 1e-12))
        return {**p, "W3": p["W3"] * s, "b3": p["b3"] * s}
    return {k: one(v) for k, v in params.items()}


_project_j = jax.jit(_project)


class CBound(CCur):
    """Subclasses ccur only for its diagnostics (csat + line_match); no freezing."""
    name = "cbound"
    BND = dict(lam_start=None, lam_frac=0.0, c0=None, cmax=30.0, c_frac=0.5, snap_every=0, whole=False)

    def __init__(self, problem, seed, workdir, **cfg):
        bnd = {k: cfg.pop(k, v) for k, v in self.BND.items()}
        cfg = {"freeze_from": 99, "release": (), "diag_every": 0, **cfg}
        super().__init__(problem, seed, workdir, **cfg)
        self.cfg.update(bnd)
        self.snaps = []

    def bound_at(self, frac):
        cfg = self.cfg
        c = None
        if cfg["c0"] is not None:
            r = 1.0 if cfg["c_frac"] <= 0 else min(1.0, frac / cfg["c_frac"])
            c = float(cfg["c0"] * (cfg["cmax"] / cfg["c0"]) ** r)
        lam_r = None
        if cfg["lam_start"] is not None and frac < cfg["lam_frac"]:
            lam_r = frac / cfg["lam_frac"]
        return c, lam_r

    def train(self, budget_s, log):
        cfg = self.cfg
        tset = {"t": None}
        N = cfg.get("diag_every", 0)
        S = cfg.get("snap_every", 0)
        self._dtime = 0.0
        lam_target = np.array(self.lam, np.float64)          # lam0 per level (0 where no hard factor later)

        def apply(frac):
            c, lam_r = self.bound_at(frac)
            if c is not None:
                self.params = _project_j(self.params, jnp.float32(c))
            if lam_r is not None:
                ls = cfg["lam_start"]
                self.lam = np.where(lam_target > 0, ls + (lam_target - ls) * lam_r, self.lam)
            elif cfg["lam_start"] is not None and not getattr(self, "_ramp_done", False):
                self._ramp_done = True
                self.lam = np.where(lam_target > 0, lam_target, self.lam)
            return c

        # whole-logit tau (q ~ exp((-s - b_raw) / tau)): b_raw / tau is linear in the last layer, so the
        # actor holds the effective layer W_eff = W_raw / tau and every update is mapped back to what Adam
        # on W_raw would do: the Adam step on W_eff is divided by tau, and when tau changes W_eff is
        # rescaled by tau_old / tau_new.  At tau = 1 W_eff = W_raw (generation unchanged).
        prev = {}

        def tau_now():
            ts = tset["t"] or 0.0
            ftr = (time.time() - self._t0 - ts) / max(budget_s - ts, 1e-9)
            return cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - ftr / cfg["anneal"])

        def last(p, f):
            return {k: ({**v, "W3": f(k, "W3", v["W3"]), "b3": f(k, "b3", v["b3"])} if "W3" in v else v)
                    for k, v in p.items()}

        def whole(tau_used, tau_next):
            if prev:
                self.params = last(self.params, lambda k, n, x: prev[k][n] + (x - prev[k][n]) / tau_used)
            r = tau_used / tau_next
            self.params = last(self.params, lambda k, n, x: x * r)
            prev.clear()
            prev.update({k: {"W3": v["W3"], "b3": v["b3"]} for k, v in self.params.items() if "W3" in v})

        self._t0 = time.time()

        def wlog(d):
            if cfg["whole"] and (d.get("setup_done") or d.get("update") is not None):
                if d.get("setup_done"):
                    tset["t"] = d["t"]
                whole(d.get("tau", 1.0), tau_now())
            if d.get("setup_done"):
                tset["t"] = d["t"]
                lam_target[:] = np.where(self.nhard > 0, self.cfg["lam0"], 0.0)
                apply(0.0)
                d = {**d, "lam_start": self.lam.tolist()}
            u = d.get("update")
            if u is not None:
                ts = tset["t"] or 0.0
                frac = (d["t"] - ts) / max(budget_s - ts, 1e-9)
                c = apply(frac)
                d = {**d, "bound": c, "lam_used_next": self.lam.tolist()}
                if N and (u == 1 or u % N == 0):
                    ta = time.time()
                    d = {**d, "diag": self.diag()}
                    self._dtime += time.time() - ta
                    d["diag_time"] = self._dtime
                if S and (u == 1 or u % S == 0):
                    self.snaps.append({"update": u, "params": jax.device_get(self.params),
                                       "lam": self.lam.tolist(), "tau": self.tau})
            log(d)
        CFast.train(self, budget_s, wlog)


register(CBound)


def _bvariant(name, **over):
    cls = type(f"CBound_{name}", (CBound,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


# lambda ramp 1 -> lam0 over the first half of post-setup time (exact gradients, dimers 4x4: see report)
_bvariant("cbound_lam", lam_start=1.0, lam_frac=0.5)
_bvariant("cbound_lam_diag", lam_start=1.0, lam_frac=0.5, diag_every=25)
# last-layer bound 1 -> 30 nats over the first half (projected Adam), alone and with the ramp
_bvariant("cbound_proj", c0=1.0, cmax=30.0, c_frac=0.5)
_bvariant("cbound_projlam", c0=1.0, cmax=30.0, c_frac=0.5, lam_start=1.0, lam_frac=0.5)
_bvariant("cbound_off_diag", diag_every=25)          # cfast + the same diagnostics (reference)
# whole-logit tau 8 -> 1 over the first 60 % of post-setup time (exact gradients: KL 0.031 on dimers 4x4)
_bvariant("cbound_whole8", whole=True, tau0=8.0, anneal=0.6)
_bvariant("cbound_whole8_diag", whole=True, tau0=8.0, anneal=0.6, diag_every=25)
_bvariant("cbound_tau8", tau0=8.0, anneal=0.6)        # control: the same tau schedule on -s only (cfast's)
# both: lambda ramp 1 -> lam0 over the first half and whole-logit tau 8 -> 1 over 60 % (exact: KL 3e-5)
_bvariant("cbound_wholelam", whole=True, tau0=8.0, anneal=0.6, lam_start=1.0, lam_frac=0.5)
_bvariant("cbound_wholelam_diag", whole=True, tau0=8.0, anneal=0.6, lam_start=1.0, lam_frac=0.5, diag_every=25)
