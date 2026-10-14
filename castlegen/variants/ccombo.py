"""ccombo: cstart's mixed start depth with a phase schedule + the LN+SiLU 6 x 6
actor + cbound's whole-logit tau and lambda-ramp schedules (box_waves
"Recommended next steps" 1; ideas #14 mixed start depth, #12/#17 schedules,
cln actor options; all user-approved or allowed freedoms).  Generation is
cfast's, unchanged (the final parameters are mapped back to tau = 1).

Schedules, all functions of ftr = the fraction of post-setup training time
(`schedule(cfg, ftr)`, pure, tested):

- start depth (ideas #14).  Phase 1 (ftr < p1): a fraction frac1 (default 1)
  of each batch are clamped worlds with L drawn from Ls1 (default (3,): levels
  0-2 see only probe-lifted fill contexts, cstart_f1L3_ln6).  Decay
  (p1 <= ftr < decay_end): frac goes linearly frac1 -> frac_end (default 0)
  and clamped worlds draw L from Ls2 (default (3, 4), so level 3 trains under
  a probe level 4 as well).  After decay_end: frac_end (0 = cfast's full-depth
  worlds only: every level, own-draw contexts).  Defaults p1 = 0.35,
  decay_end = 0.8.
- whole-logit tau (cbound `whole`).  q ~ exp((-s - b_raw) / tau); the actor
  stores the effective last layer W_eff = W_raw / tau, every Adam step on it is
  divided by tau, and W_eff is rescaled by tau_old / tau_new when tau changes
  (the same mapping as cbound.py).  tau goes tau0 -> tau1 geometrically
  (tau_geom=True) or linearly over the first `anneal` of the time; defaults
  8 -> 1 geometric over 0.8.  whole=False gives cfast's tau on -s only.
- lambda ramp (cbound `lam_start`).  The lambda used by the update goes
  linearly lam_start -> lam_end over the first lam_frac (defaults 1 -> 20 over
  0.5) at every level with hard factors; the lambda control is frozen during
  the ramp (its violation average keeps updating) and takes over from lam_end
  when it ends.  0 <= lambda <= 20 always (constraint 7).

Variants: ccombo (everything), ccombo_notau (no whole-logit tau: cfast's tau
2 -> 1 over 20 % on -s only), ccombo_nophase (control: start-depth schedule
off, i.e. cfast worlds only, everything else as ccombo).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from .cstart import CStart

COMBO = dict(p1=0.35, decay_end=0.8, frac1=1.0, frac_end=0.0, Ls1=(3,), Ls2=(3, 4),
             whole=True, tau_geom=True, lam_start=1.0, lam_end=20.0, lam_frac=0.5)


def schedule(cfg, ftr):
    """-> dict(frac, Ls, tau, lam_r): the start-depth mixture, temperature and
    lambda-ramp position (None after the ramp) at post-setup time fraction ftr."""
    ftr = max(0.0, float(ftr))
    p1, de = cfg["p1"], cfg["decay_end"]
    if ftr < p1:
        frac, Ls = cfg["frac1"], tuple(cfg["Ls1"])
    else:
        r = 1.0 if de <= p1 else min(1.0, (ftr - p1) / (de - p1))
        frac, Ls = cfg["frac1"] + (cfg["frac_end"] - cfg["frac1"]) * r, tuple(cfg["Ls2"])
    a = 1.0 if cfg["anneal"] <= 0 else min(1.0, ftr / cfg["anneal"])
    t0, t1 = cfg["tau0"], cfg["tau1"]
    tau = t0 * (t1 / t0) ** a if cfg["tau_geom"] else t0 + (t1 - t0) * a
    lam_r = None
    if cfg["lam_start"] is not None and ftr < cfg["lam_frac"]:
        lam_r = ftr / cfg["lam_frac"]
    return {"frac": float(frac), "Ls": Ls, "tau": float(tau), "lam_r": lam_r}


def _last(p, f):
    return {k: ({**v, "W3": f(k, "W3", v["W3"]), "b3": f(k, "b3", v["b3"])} if "W3" in v else v)
            for k, v in p.items()}


@register
class CCombo(CStart):
    name = "ccombo"

    def __init__(self, problem, seed, workdir, **cfg):
        cmb = {k: cfg.pop(k, v) for k, v in COMBO.items()}
        cfg = {"net": "ln", "win": (-2, 3), "tau0": 8.0, "tau1": 1.0, "anneal": 0.8, **cfg}
        super().__init__(problem, seed, workdir, **cfg)
        self.cfg.update(cmb)
        self._sched = schedule(self.cfg, 0.0)
        self.Ls = list(self._sched["Ls"])
        T = problem.T
        self.can_clamp = T >= 1 and problem.probes is not None

    def draw_L(self, B):
        T, s = self.problem.T, self._sched
        Ls_w = np.full(B, T + 1, np.int64)
        Ls = [L for L in s["Ls"] if 1 <= L <= T]
        if self.can_clamp and s["frac"] > 0 and Ls:
            nc = int(round(s["frac"] * B))
            who = self.crng.permutation(B)[:nc]
            Ls_w[who] = self.crng.choice(Ls, size=nc)
        return Ls_w

    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        self.fn = self.train_fns(lay)
        self._measure_baseline(lay, log)
        t_setup = time.time() - t0
        lam_target = np.where(self.nhard > 0, float(cfg["lam_end"] if cfg["lam_start"] is not None
                                                    else cfg["lam0"]), 0.0)
        log({"t": t_setup, "setup_done": True, "combo": {k: cfg[k] for k in COMBO},
             "tau0": cfg["tau0"], "anneal": cfg["anneal"], "can_clamp": self.can_clamp})
        B = cfg["B"]
        tm = {"rec": 0.0, "ctx": 0.0}
        nclamp = np.zeros(T + 2)
        whole = bool(cfg["whole"])
        tau_cur = None                        # the tau the stored W_eff currently corresponds to
        ramp_done = False
        while True:
            if (time.time() - t0) / max(budget_s, 1e-9) >= 1.0:
                break
            ftr = (time.time() - t0 - t_setup) / max(budget_s - t_setup, 1e-9)
            self._sched = s = schedule(cfg, ftr)
            self.Ls = list(s["Ls"])
            self.tau = s["tau"]
            if whole:                         # W_eff = W_raw / tau: rescale to the new tau
                if tau_cur is None:
                    tau_cur = 1.0             # init / last layer is stored at tau = 1
                if tau_cur != self.tau:
                    r = tau_cur / self.tau
                    self.params = _last(self.params, lambda k, n, x: x * r)
                    tau_cur = self.tau
            if s["lam_r"] is not None:
                ls = cfg["lam_start"]
                self.lam = np.where(lam_target > 0, ls + (lam_target - ls) * s["lam_r"], self.lam)
            elif cfg["lam_start"] is not None and not ramp_done:
                ramp_done = True
                self.lam = np.where(lam_target > 0, lam_target, self.lam)
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            des = self._batch(seeds, region)
            tc = time.time()
            Ls_w = self.draw_L(B)
            ctx, Lj = self.contexts(seeds, region, Ls_w)
            for L in Ls_w:
                nclamp[L] += 1
            ta = time.time()
            prev = self.params
            out = self.fn["fused_ctx"](self.params, self.cparams, self.opt, self.copt, self.gs,
                                       jnp.asarray(seeds, jnp.int32), des, tau, lam, cfg["lr"],
                                       cfg["lr_critic"], cfg["clip"],
                                       jax.random.PRNGKey(self.updates + 7919 * self.seed), ctx, Lj)
            _, self.params, self.cparams, self.opt, self.copt, loss, closs, viol, tot, _ = out
            if whole and self.tau != 1.0:     # Adam step on W_raw = tau W_eff: divide the step by tau
                it = self.tau
                self.params = _last(self.params, lambda k, n, x: prev[k][n] + (x - prev[k][n]) / it)
            viol = np.asarray(viol)
            tm["ctx"] += ta - tc
            tm["rec"] += time.time() - ta
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            log({"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                 "critic_loss": float(closs), "cost_per_world": float(np.asarray(tot).mean()),
                 "raw_viol": hf.tolist(), "lam": self.lam.tolist(), "lam_used": np.asarray(lam).tolist(),
                 "frac": s["frac"], "L_worlds": Ls_w.tolist(),
                 "time_rec": tm["rec"], "time_ctx": tm["ctx"]})
        if whole and tau_cur is not None and tau_cur != 1.0:   # back to tau = 1 for generation
            r = tau_cur / 1.0
            self.params = _last(self.params, lambda k, n, x: x * r)
        self.tau = 1.0
        self.timing = {**tm, "setup": t_setup, "total": time.time() - t0, "L_counts": nclamp.tolist()}


def _variant(name, **over):
    cls = type(f"CCombo_{name}", (CCombo,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("ccombo_notau", whole=False, tau0=2.0, anneal=0.2, tau_geom=False)  # cfast's tau (on -s only)
_variant("ccombo_nophase", frac1=0.0, frac_end=0.0)                             # control: no clamped worlds
# fix after the first ground run (levels 3-4 lock into dirt ~300 updates after release, then own-only training
# erases the fine levels' fill-context fit): the clamped share decays to a floor of 0.5 (L in {3, 4}) instead of 0
_variant("ccombo_floor", frac_end=0.5)
