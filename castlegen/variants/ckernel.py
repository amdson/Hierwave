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
          kernel_lam=20.0, level_on=None, diag_n=16)


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
        lamk = c["kernel_lam"]
        self.J20 = JaxBlockGibbs(p, region, 2, 0, lam=lamk)
        self.J11 = JaxBlockGibbs(p, region, 1, 1, lam=lamk)
        self.Gc = {L: JaxBlockGibbs(p, region, L, 0, lam=lamk, chunk=4096) for L in sorted(c["flip_Ls"], reverse=True)}
        self.fl = self.J11.K.flat
        self.E = make_energy(self.fl, self.J11.K.insts, fine_max=2)
        lay = self.layout(region)
        self.mtm = bm.make_chain(lay, [2], 1.0, lamk, c["K_l2"])
        self.flip = cf.make_flip_chain(lay, list(c["flip_Ls"]), 1.0, c["tau_c"], c["flip_mode"], lamk, c["K_flip"])
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
            tm = time.time()
            losses, ninf, nfeas, ndraw = [], 0, 0, 0
            for u in range(c["upi"]):
                idx = self.rng.choice(c["nctx"], c["ml_B"], replace=False)
                self.params, opt, info = self.mlstep(self.params, opt, {k: x[idx] for k, x in st.items()},
                                                     jnp.ones(c["ml_B"]), jnp.float32(c["ml_lr"]), on)
                losses.append(float(info["loss"]) / cells)
                ninf += int(info["n_infeasible"]); nfeas += int(info["n_feasible"])
                self.updates += 1
            rec.update({"t": time.time() - t0, "update": self.updates, "loss_per_cell": float(np.mean(losses)),
                        "infeasible_draws_per_state": ninf / (c["upi"] * c["ml_B"]),
                        "feasible_state_frac": nfeas / (c["upi"] * c["ml_B"]),
                        "time_ctx": tr - ti, "time_refine": tm - tr, "time_ml": time.time() - tm})
            log(rec)
        self.timing = {"setup": t_setup, "total": time.time() - t0, "iters": it}

    def info(self):
        return {**super().info(), "start": self.start}


def _variant(name, **over):
    cls = type(f"CKernel_{name}", (CKernel,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


register(CKernel)
