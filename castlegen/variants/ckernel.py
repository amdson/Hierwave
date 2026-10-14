"""ckernel (Sweep 2, coarse half): the amortised kernel-route loop on ALL levels.

Training (constitution amendment 2026-10-14: weighted ML on states produced by
p*-invariant kernels; generation unchanged = cfast's coloured pass):

  contexts   probe lifts (problem.probes, jitter 0-3, designed channels) and
             the sampler's own tau = 1 draws, half each;
  kernels    one cycle, every move p*-invariant (lam = 20 prices violations):
               coarse flips at L in {3, 4}  (castlegen/kernels/coarse_flip.py:
                 cell value from q tempered at tau_c, subtree redrawn from q,
                 exact multiple-try MH ratio, K_f tries),
               exact single-cell Gibbs at L = 4 then L = 3 (JaxBlockGibbs(L, 0)),
               multiple-try q-subtree moves at L = 2 (block_mh, K = 8),
               n_fine cycles of exact fine block Gibbs [(2,0), (1,1)]
                 (kernel-a's blocks, amortised.JaxBlockGibbs);
  update     weighted ML (weights 1) on the refined states, every level
             (amortised.make_ml_update_drawwise: draws outside the pass's
             least-violating set are dropped individually and counted).

Start (cfg init): sweep2a's arm B loop params (reportables/2026-10-14_sweep2a/
runs/armB_s0: levels 0-2 trained by the fine amortised loop, levels 3-4 at
cstart2's init); init=KA_INIT uses kernel-a's ML fit; init=None cfast's init.

Generic: the kernels and the update read only the declared factors, the
layout and the sampler.  Only the context source (probe lifts) is a
problem-supplied valid-state generator, as in kernel-a / sweep2a.
"""
from __future__ import annotations

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from ._cfast_core import train_pass
from .cfast import CFast

KA_INIT = "reportables/2026-10-14_kernel_a/runs/ml_init/params_ground.pkl"
S2A_INIT = "reportables/2026-10-14_sweep2a/runs/armB_s0/params_ground.pkl"
CK = dict(net="ln", win=(-2, 3), init=S2A_INIT, nctx=32, own_frac=0.5, upi=8, ml_B=8, ml_lr=1e-3,
          n_fine=3, m_l2=4, K_l2=8, m_flip=4, K_flip=4, flip_Ls=(3, 4), flip_mode="q", tau_c=2.0,
          kernel_lam=20.0, level_on=None, diag_n=16,
          # wave2-bench (violation pressure; all off by default = sweep2b's ckernel):
          ml_wlam=0.0,        # ML weight of a refined state: exp(-ml_wlam * V) normalised per minibatch
          rkl_n=0,            # cfast fused reverse-KL steps (tau 1, own draws, lambda control) per iteration
          rkl_lr=1e-3, rkl_lam0=None,   # rkl_lam0: restart lambda at cfast's lam0 (None: keep the init's lambda)
          diag_every=2,
          rkl_levels=None,    # levels whose parameters the RKL step may change (None: all); others are restored
          coarse_lam=None,
          rkl_level_scale=None)  # {level: s}: that level's RKL parameter step is scaled by s (Adam: lr x s)    # lam of the coarse kernels (flips L3/L4, coarse Gibbs); None: kernel_lam; inf = infeasible


class CKernel(CFast):
    name = "ckernel"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CK, **cfg})
        self.updates = 0
        init = self.cfg.get("init")
        self.start = "cfast init"
        if init and os.path.exists(init):
            ck = pickle.load(open(init, "rb"))
            pr = ck["trained"]["params"]
            assert set(pr) == set(self.params), "init params do not match the layout"
            self.params = jax.tree_util.tree_map(jnp.asarray, pr)
            self.lam = np.asarray(ck["trained"]["lam"], float)
            self.start = init

    # ------------------------------------------------------------------ kernels
    def _build(self, region):
        from castlegen.kernels import block_mh as bm
        from castlegen.kernels import coarse_flip as cf
        from castlegen.kernels.amortised import JaxBlockGibbs, make_energy, make_ml_update_drawwise
        p, c = self.problem, self.cfg
        lamk = float(c["kernel_lam"])
        lamg = None if np.isinf(lamk) else lamk        # lam = inf: violating completions infeasible (logged as kernel_lam)
        self.J20 = JaxBlockGibbs(p, region, 2, 0, lam=lamg)
        self.J11 = JaxBlockGibbs(p, region, 1, 1, lam=lamg)
        lamc = lamk if c.get("coarse_lam") is None else float(c["coarse_lam"])
        self.Gc = {L: JaxBlockGibbs(p, region, L, 0, lam=None if np.isinf(lamc) else lamc, chunk=4096)
                   for L in sorted(c["flip_Ls"], reverse=True)}
        self.fl = self.J11.K.flat
        self.E = make_energy(self.fl, self.J11.K.insts, fine_max=2)
        lay = self.layout(region)
        self.mtm = bm.make_chain(lay, [2], 1.0, lamk, c["K_l2"])
        self.flip = cf.make_flip_chain(lay, list(c["flip_Ls"]), 1.0, c["tau_c"], c["flip_mode"], lamc, c["K_flip"])
        self.mlstep = make_ml_update_drawwise(lay)
        lam_gen = jnp.asarray(self.lam, jnp.float32)
        self.gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lam_gen, record=False)[0],
                                    in_axes=(None, 0, 0)))
        self.lay_tr = lay

    def _to_flat(self, st):
        n = next(iter(st.values())).shape[0]
        return jnp.concatenate([jnp.asarray(st[ch.name]).reshape(n, -1) for ch in self.problem.chans], 1).astype(jnp.int32)

    def _from_flat(self, v):
        fl = self.fl
        return {ch.name: v[:, fl.off[ch.name]:fl.off[ch.name] + int(np.prod(fl.shape[ch.name]))].reshape((-1,) + fl.shape[ch.name])
                for ch in self.problem.chans}

    def _probe_ctx(self, n, region):
        p, out = self.problem, []
        for _ in range(n):
            s = self.ctx_seed; self.ctx_seed += 1
            k = int(self.rng.integers(4))
            out.append({**p.probes(s, region, n=k + 1)[k], **p.designed_values(s, region)})
        return {ch.name: jnp.asarray(np.stack([x[ch.name] for x in out]), jnp.int32) for ch in p.chans}

    def _own_ctx(self, n, region):
        p = self.problem
        ss = np.arange(self.ctx_seed, self.ctx_seed + n); self.ctx_seed += n
        ds = [p.designed_values(int(s), region) for s in ss]
        dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
        st = self.gen(self.params, jnp.asarray(ss, jnp.int32), dj)
        return {ch.name: jnp.asarray(st[ch.name] if ch.name in st else dj[ch.name], jnp.int32) for ch in p.chans}

    def _ground_stats(self, st):
        """dirt fraction and line_match (problem.stats) of a batch."""
        if self.problem.name != "ground":
            return {}
        t, f = np.asarray(st["t"]), np.asarray(st["fill"])
        ss = [self.problem.stats({"t": t[b], "fill": f[b], "rho4": np.zeros_like(f[b])}) for b in range(t.shape[0])]
        return {"dirt": float(np.mean([s["dirt_frac"] for s in ss])), "line_match": float(np.mean([s["line_match"] for s in ss]))}

    def _level_viol(self, st, lo, hi, region):
        """Per-level violated hard fraction and violated count per factor (bench energy.py) of worlds lo:hi."""
        from castlegen.bench import energy as en
        p = self.problem
        x = jax.device_get({c.name: st[c.name][lo:hi] for c in p.chans})
        viol, nh, fac = np.zeros(p.T + 1), np.zeros(p.T + 1), {}
        for b in range(hi - lo):
            E = en.energy(p, {k: np.asarray(v[b]) for k, v in x.items()}, region)
            for l in range(p.T + 1):
                viol[l] += E.viol[l]; nh[l] += E.nhard[l]
            for k, (_, v, _) in E.per_factor.items():
                if v:
                    fac[k] = fac.get(k, 0) + v
        return {"frac": (viol / np.maximum(nh, 1)).tolist(), "per_world": (viol / (hi - lo)).tolist(), "factors": fac}

    def _rkl_setup(self, log):
        lay = self.lay_tr
        self.fn = self.train_fns(lay)
        self._measure_baseline(lay, log)
        if self.cfg["rkl_lam0"] is not None:
            self.lam = np.where(self.nhard > 0, float(self.cfg["rkl_lam0"]), 0.0)

    def _rkl_steps(self, n):
        """n of cfast's fused reverse-KL updates (tau = 1) on the sampler's own draws, with cfast's
        adaptive lambda (<= lam_max = 20) driven by the own-draw raw violation rate."""
        cfg, region = self.cfg, Region(self.lay_tr.W)
        out = []
        for _ in range(n):
            seeds = np.arange(self.train_seed, self.train_seed + cfg["B"]); self.train_seed += cfg["B"]
            des = self._batch(seeds, region)
            lam = jnp.asarray(self.lam, jnp.float32)
            res = self.fn["fused"](self.params, self.cparams, self.opt, self.copt, self.gs, jnp.asarray(seeds, jnp.int32),
                                   des, jnp.float32(1.0), lam, cfg["rkl_lr"], cfg["lr_critic"], cfg["clip"],
                                   jax.random.PRNGKey(self.updates + 7919 * self.seed + 1))
            p_old = self.params
            _, self.params, self.cparams, self.opt, self.copt, loss, closs, viol, tot, _ = res
            if cfg.get("rkl_levels") is not None:
                keep = set(cfg["rkl_levels"])
                self.params = {k: (v if int(k[1:k.index("_")]) in keep else p_old[k]) for k, v in self.params.items()}
            if cfg.get("rkl_level_scale"):
                sc = {int(l): float(x) for l, x in cfg["rkl_level_scale"].items()}
                lv = lambda k: int(k[1:k.index("_")])
                self.params = {k: (jax.tree_util.tree_map(lambda a, b, s=sc[lv(k)]: b + s * (a - b), v, p_old[k])
                                   if lv(k) in sc else v) for k, v in self.params.items()}
            viol = np.asarray(viol)
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.rkl_updates += 1
            out.append(hf)
        return np.mean(out, 0)

    def refine(self, st, key):
        """One kernel cycle on a batch of states (dict).  Returns (states, diagnostics)."""
        c, d = self.cfg, {}
        kf, k4, k2, kg = jax.random.split(key, 4)
        n = st["t"].shape[0]
        if c["m_flip"] > 0:
            st, rec = self.flip(self.params, jax.random.split(kf, n), st, c["m_flip"])
            rec = jax.device_get(rec)
            for L in c["flip_Ls"]:
                m = rec["L"] == L
                d[f"acc_flip_L{L}"] = float(rec["acc"][m].mean()) if m.any() else None
                d[f"medlogr_flip_L{L}"] = float(np.median(rec["log_r"][m])) if m.any() else None
        v = self._to_flat(st)
        for L, G in self.Gc.items():
            k4, sub = jax.random.split(k4)
            v, ch = G.sweep(v, sub)
            d[f"chg_gibbs_L{L}"] = float(np.asarray(ch).mean()) / G.K.nb
        st = self._from_flat(v)
        if c["m_l2"] > 0:
            st, rec = self.mtm(self.params, jax.random.split(k2, n), st, c["m_l2"])
            rec = jax.device_get(rec)
            d["acc_mtm_L2"] = float(rec["acc"].mean())
            d["medlogr_mtm_L2"] = float(np.median(rec["log_r"]))
        v = self._to_flat(st)
        chf = []
        for _ in range(c["n_fine"]):
            for J in (self.J20, self.J11):
                kg, sub = jax.random.split(kg)
                v, ch = J.sweep(v, sub)
                chf.append(float(np.asarray(ch).mean()) / J.K.nb)
        d["chg_gibbs_20"], d["chg_gibbs_11"] = (float(np.mean(chf[0::2])), float(np.mean(chf[1::2]))) if chf else (None, None)
        return self._from_flat(v), d, v

    # ------------------------------------------------------------------ training
    def train(self, budget_s, log):
        from castlegen.kernels.logq import ml_opt_init
        t0 = time.time()
        c, p = self.cfg, self.problem
        region = Region(c["W_train"] or p.W_train)
        cells = p.H * region.W
        self._build(region)
        t_build = time.time() - t0
        self.ctx_seed = 20_000_000 + 1_000_000 * self.seed
        opt = ml_opt_init(self.params)
        on = jnp.asarray(c["level_on"] or [1.0] * (p.T + 1), jnp.float32)
        n_own = int(round(c["nctx"] * c["own_frac"]))
        n_prb = c["nctx"] - n_own
        key = jax.random.PRNGKey(4242 + self.seed)
        # warm compiles (same shapes as the loop)
        parts = ([self._probe_ctx(n_prb, region)] if n_prb else []) + ([self._own_ctx(n_own, region)] if n_own else [])
        st = {k: jnp.concatenate([q[k] for q in parts]) for k in parts[0]}
        t_ctx = time.time() - t0
        st2, _, v = self.refine(st, key)
        self.E(v)
        t_ref = time.time() - t0
        idx = np.arange(c["ml_B"])
        _ = self.mlstep(self.params, opt, {k: x[idx] for k, x in st2.items()}, jnp.ones(c["ml_B"]), jnp.float32(c["ml_lr"]), on)
        self.rkl_updates = 0
        if c["rkl_n"] > 0:
            self._rkl_setup(log)
            self._rkl_steps(1)          # warm compile (one real step)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True, "start": self.start, "t_build": t_build, "t_ctx": t_ctx, "t_refine_compiled": t_ref})
        it = 0
        while time.time() - t0 < budget_s:
            it += 1
            ti = time.time()
            parts = ([self._probe_ctx(n_prb, region)] if n_prb else []) + ([self._own_ctx(n_own, region)] if n_own else [])
            st = {k: jnp.concatenate([q[k] for q in parts]) for k in parts[0]}
            v0 = self._to_flat(st)
            s0, nv0, _ = self.E(v0)
            rec = {"iter": it, "own_raw": self._ground_stats({k: x[n_prb:] for k, x in st.items()})}
            key, sub = jax.random.split(key)
            tr = time.time()
            st, kd, v = self.refine(st, sub)
            s1, nv1, _ = self.E(v)
            s0, s1, nv0, nv1 = (np.asarray(x) / cells for x in (s0, s1, nv0, nv1))
            rec.update(kd)
            rec["own_refined"] = self._ground_stats({k: x[n_prb:] for k, x in st.items()})
            rec["soft_own"], rec["soft_own_refined"] = float(s0[n_prb:].mean()), float(s1[n_prb:].mean())
            if n_prb:
                rec["soft_probe"], rec["soft_probe_refined"] = float(s0[:n_prb].mean()), float(s1[:n_prb].mean())
            rec["viol_own"], rec["viol_own_refined"] = float(nv0[n_prb:].mean()), float(nv1[n_prb:].mean())
            if c["diag_every"] and it % c["diag_every"] == 1 % c["diag_every"]:
                st0 = parts and {k: jnp.concatenate([q[k] for q in parts]) for k in parts[0]}
                rec["lv_own_raw"] = self._level_viol(st0, n_prb, c["nctx"], region)
                rec["lv_own_refined"] = self._level_viol(st, n_prb, c["nctx"], region)
                if n_prb:
                    rec["lv_probe_refined"] = self._level_viol(st, 0, n_prb, region)
            nvw = np.asarray(nv1) * cells           # violated hard instances per refined state
            tm = time.time()
            losses, ninf, nfeas, ndraw, wl = [], 0, 0, 0, []
            for u in range(c["upi"]):
                idx = self.rng.choice(c["nctx"], c["ml_B"], replace=False)
                vw = nvw[idx]
                w = np.exp(-c["ml_wlam"] * (vw - vw.min()))
                w = w / w.sum() * c["ml_B"]
                wl.append(float((w ** 2).sum() and w.sum() ** 2 / (w ** 2).sum()))
                self.params, opt, info = self.mlstep(self.params, opt, {k: x[idx] for k, x in st.items()},
                                                     jnp.asarray(w, jnp.float32), jnp.float32(c["ml_lr"]), on)
                losses.append(float(info["loss"]) / cells)
                ninf += int(info["n_infeasible"]); nfeas += int(info["n_feasible"])
                self.updates += 1
            rec.update({"t": time.time() - t0, "update": self.updates, "loss_per_cell": float(np.mean(losses)),
                        "infeasible_draws_per_state": ninf / (c["upi"] * c["ml_B"]),
                        "feasible_state_frac": nfeas / (c["upi"] * c["ml_B"]),
                        "time_ctx": tr - ti, "time_refine": tm - tr, "time_ml": time.time() - tm,
                        "ml_ess": float(np.mean(wl)), "refined_viol_states_frac": float((nvw > 0).mean())})
            if c["rkl_n"] > 0:
                tk = time.time()
                rec["rkl_raw_viol"] = self._rkl_steps(c["rkl_n"]).tolist()
                rec["lam"] = self.lam.tolist(); rec["rkl_updates"] = self.rkl_updates
                rec["time_rkl"] = time.time() - tk
            log(rec)
        self.timing = {"setup": t_setup, "total": time.time() - t0, "iters": it}

    def info(self):
        return {**super().info(), "start": self.start, "rkl_updates": getattr(self, "rkl_updates", 0)}


def _variant(name, **over):
    cls = type(f"CKernel_{name}", (CKernel,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


register(CKernel)
# wave2-bench arms (upper bound: probe-lift contexts, outside the frozen constitution, change request #1)
_variant("ckernel_w20", ml_wlam=20.0)
_variant("ckernel_rkl", ml_wlam=20.0, rkl_n=16, rkl_lam0=10.0)
_variant("ckernel_rkl32", ml_wlam=20.0, rkl_n=32, rkl_lam0=10.0)   # rkl_lr 1e-3
_variant("ckernel_rkl48", ml_wlam=20.0, rkl_n=48, rkl_lam0=10.0)   # rkl_lr 1e-3
# wave2-bench2: level-split objectives, coarse kernels at lam = inf (infeasible)
_variant("ckernel_split", ml_wlam=20.0, rkl_n=32, rkl_lam0=10.0, level_on=[1.0, 1.0, 1.0, 0.0, 0.0], rkl_levels=[2, 3, 4],
         coarse_lam=float("inf"))
_variant("ckernel_fine_rkl", ml_wlam=20.0, rkl_n=32, rkl_lam0=10.0, rkl_levels=[0, 1, 2], coarse_lam=float("inf"))
_variant("ckernel_rkl32inf", ml_wlam=20.0, rkl_n=32, rkl_lam0=10.0, coarse_lam=float("inf"))
_variant("ckernel_rkl123", ml_wlam=20.0, rkl_n=32, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3], coarse_lam=float("inf"))
# wave3-bench: RKL on L0-3 at lr 1e-3 plus a weaker L4 RKL step (scale s = lr_L4 / 1e-3)
_variant("ckernel_rkl4w1", ml_wlam=20.0, rkl_n=32, rkl_lam0=10.0, coarse_lam=float("inf"), rkl_level_scale={4: 0.1})
_variant("ckernel_rkl4w3", ml_wlam=20.0, rkl_n=32, rkl_lam0=10.0, coarse_lam=float("inf"), rkl_level_scale={4: 0.3})
_variant("ckernel_inf", ml_wlam=20.0, kernel_lam=float("inf"))
