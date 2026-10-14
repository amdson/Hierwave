"""ckernel_own (wave2-explore): ckernel with contexts = the sampler's OWN draws only.

Inside the amendment 2026-10-14 (training only; generation = cfast's coloured
pass, unchanged).  Objective: weighted ML on states produced by p*-invariant
kernels applied to own tau = 1 draws, and on annealed multiple-try candidates
with their importance weights.  No probe lifts are used for training.

Per iteration (nctx own draws):
  phase 1 (t < phase1_s, curriculum): MTM q-subtree moves at levels
    `mtm_Ls_1` (default L1-L2), n_fine cycles of exact fine Gibbs; ML on the
    levels `ml_levels_1` only (default 0-2), so the fine levels fit their own
    contexts first;
  phase 2: + coarse flips at flip_Ls (tracked: old/new value of the cell's
    first channel, so full -> partial moves are counted), exact single-cell
    coarse Gibbs, MTM at mtm_Ls_2, fine Gibbs; ML on all levels;
  optional (an_n > 0, phase 2): annealed level-an_L subtree candidates
    (explore.make_annealed_candidates: x0 ~ q, tau an_tau_hi -> 1, lambda
    schedule, hard final sweeps; exact AIS weights with q as the start
    proposal) on an_n refined own draws, K = an_K tries each; per ML update
    an_B candidates are drawn in proportion to their self-normalised weights
    (importance resampling) and added to the batch.
Generic: kernels and update read only the declared factors, the layout and q.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from .ckernel import CKernel

_BUILT = {}     # wave5-pop: per-process memo of built kernels (see CKernelOwn._build)

OWN = dict(own_frac=1.0, phase1_s=60.0, mtm_Ls_1=(1, 2), mtm_Ls_2=(2,), ml_levels_1=(0, 1, 2),
           flip_Ls=(3, 4), K_flip=16, flip_mode="q", tau_c=2.0, m_flip=4,
           an_n=0, an_L=4, an_K=8, an_steps=64, an_tau_hi=8.0, an_lam_kind="const", an_B=4, an_full_support=True,
           tt_Ls=(), tt_m=4, tt_tau_hi=3.0, tt_n=16, tt_lam_hi=None)


class CKernelOwn(CKernel):
    name = "ckernel_own"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**OWN, **cfg})

    _BUILT_ATTRS = ("J20", "J11", "Gc", "fl", "E", "mtm", "flip", "mlstep", "gen", "lay_tr",
                    "mtm1", "mtm2", "full_val", "tt", "ann")

    def _build(self, region):
        """wave5-pop: the built kernels / jitted callables are memoised per process (key: problem,
        width, full cfg, the init lambda baked into gen), so a second instance with the same
        configuration (the bench's seed 1, or a warm-up instance) reuses the traced functions
        instead of re-tracing ~35 s of them.  Pure compile caching: no state is shared."""
        key = (self.problem.name, region.W, repr(sorted((k, repr(v)) for k, v in self.cfg.items() if k not in ("max_iters", "cache_all"))),
               tuple(np.round(np.asarray(self.lam, float), 9).tolist()))
        hit = _BUILT.get(key)
        if hit is not None:
            for a, x in hit.items():
                setattr(self, a, x)
            return
        self._build_uncached(region)
        _BUILT[key] = {a: getattr(self, a) for a in self._BUILT_ATTRS if hasattr(self, a)}

    def _build_uncached(self, region):
        from castlegen.kernels import block_mh as bm
        from castlegen.kernels import explore as ex
        super()._build(region)
        c, lay = self.cfg, self.lay_tr
        lamk = c["kernel_lam"]
        self.mtm1 = bm.make_chain(lay, list(c["mtm_Ls_1"]), 1.0, lamk, c["K_l2"])
        self.mtm2 = bm.make_chain(lay, list(c["mtm_Ls_2"]), 1.0, lamk, c["K_l2"])
        self.flip = ex.make_flip_chain_tracked(lay, list(c["flip_Ls"]), 1.0, c["tau_c"], c["flip_mode"], lamk, c["K_flip"])
        self.full_val = {L: lay.chans[L][0].D - 1 for L in c["flip_Ls"]}
        self.tt = {L: ex.make_tempered_transition(lay, L, c["tt_tau_hi"], c["tt_n"], lamk, c["tt_lam_hi"]) for L in c["tt_Ls"]}
        if c["an_n"] > 0:
            self.ann = ex.make_annealed_candidates(lay, c["an_L"], c["an_K"], c["an_tau_hi"], c["an_steps"], lamk,
                                                   c["an_lam_kind"],
                                                   full_support=c["an_full_support"])

    def refine(self, st, key, phase=2):
        c, d = self.cfg, {}
        kf, k4, k2, kg = jax.random.split(key, 4)
        n = st["t"].shape[0]
        if phase == 2 and c["m_flip"] > 0:
            st, rec = self.flip(self.params, jax.random.split(kf, n), st, c["m_flip"])
            rec = jax.device_get(rec)
            for L in c["flip_Ls"]:
                m = rec["L"] == L
                full = self.full_val[L]
                d[f"acc_flip_L{L}"] = float(rec["acc"][m].mean()) if m.any() else None
                d[f"medlogr_flip_L{L}"] = float(np.median(rec["log_r"][m])) if m.any() else None
                d[f"n_flip_L{L}"] = int(m.sum())
                d[f"n_full_L{L}"] = int((m & (rec["v_old"] == full)).sum())
                d[f"full2part_L{L}"] = int((m & rec["acc"] & (rec["v_old"] == full) & (rec["v_new"] < full)).sum())
                d[f"part2full_L{L}"] = int((m & rec["acc"] & (rec["v_old"] < full) & (rec["v_new"] == full)).sum())
                pure = m & (rec["f_old"] == rec["f_max"])
                d[f"n_pure_L{L}"] = int(pure.sum())
                d[f"pure2part_L{L}"] = int((pure & rec["acc"] & (rec["f_new"] < rec["f_max"])).sum())
                d[f"chg_val_L{L}"] = int((m & rec["acc"] & (rec["v_old"] != rec["v_new"])).sum())
        v = self._to_flat(st)
        if phase == 2:
            for L, G in self.Gc.items():
                k4, sub = jax.random.split(k4)
                v, ch = G.sweep(v, sub)
                d[f"chg_gibbs_L{L}"] = float(np.asarray(ch).mean()) / G.K.nb
            st = self._from_flat(v)
            c0 = self.lay_tr.chans[0][0].name
            for L, f in self.tt.items():
                h, w = self.lay_tr.hw[L]
                acc_l, pure_n, p2p, la_l = [], 0, 0, []
                for _ in range(c["tt_m"]):
                    cells = np.stack([self.rng.integers(h, size=n), self.rng.integers(w, size=n)], 1)
                    t_old = np.asarray(st[c0])
                    k4, sub = jax.random.split(k4)
                    st, la, acc, _ = f(sub, st, jnp.asarray(cells, jnp.int32))
                    acc = np.asarray(acc); t_new = np.asarray(st[c0])
                    H0, W0 = t_old.shape[1:]
                    fo = t_old.reshape(n, H0 >> L, 1 << L, W0 >> L, 1 << L).sum((2, 4))[np.arange(n), cells[:, 0], cells[:, 1]]
                    fn = t_new.reshape(n, H0 >> L, 1 << L, W0 >> L, 1 << L).sum((2, 4))[np.arange(n), cells[:, 0], cells[:, 1]]
                    fmax = (1 << L) ** 2
                    pure = fo == fmax
                    pure_n += int(pure.sum()); p2p += int((pure & acc & (fn < fmax)).sum())
                    acc_l.append(acc.mean()); la_l.append(np.median(np.asarray(la)))
                d[f"acc_tt_L{L}"] = float(np.mean(acc_l)); d[f"medloga_tt_L{L}"] = float(np.mean(la_l))
                d[f"n_pure_tt_L{L}"] = pure_n; d[f"pure2part_tt_L{L}"] = p2p
        mtm = self.mtm1 if phase == 1 else self.mtm2
        if c["m_l2"] > 0:
            st, rec = mtm(self.params, jax.random.split(k2, n), st, c["m_l2"])
            rec = jax.device_get(rec)
            for L in sorted(set(rec["L"].ravel().tolist())):
                m = rec["L"] == L
                d[f"acc_mtm_L{L}"] = float(rec["acc"][m].mean())
        v = self._to_flat(st)
        for _ in range(c["n_fine"]):
            for J in (self.J20, self.J11):
                kg, sub = jax.random.split(kg)
                v, ch = J.sweep(v, sub)
        return self._from_flat(v), d, v

    def _anneal(self, st, key):
        from castlegen.kernels.explore import snis
        c, lay = self.cfg, self.lay_tr
        h, w = lay.hw[c["an_L"]]
        idx = self.rng.choice(st["t"].shape[0], c["an_n"], replace=False)
        sub = {k: x[idx] for k, x in st.items()}
        cells = np.stack([self.rng.integers(h, size=c["an_n"]), self.rng.integers(w, size=c["an_n"])], 1)
        cand, logw, valid, info = self.ann(self.params, key, sub, jnp.asarray(cells, jnp.int32))
        wn, ess = snis(jax.device_get(logw))
        valid = np.asarray(valid)
        full = lay.chans[c["an_L"]][0].D - 1
        rname = lay.chans[c["an_L"]][0].name
        v_old = np.asarray(sub[rname])[np.arange(c["an_n"]), cells[:, 0], cells[:, 1]]
        v_new = np.asarray(cand[rname]).reshape((c["an_n"], c["an_K"]) + cand[rname].shape[1:])[
            np.arange(c["an_n"]), :, cells[:, 0], cells[:, 1]]
        c0 = lay.chans[0][0].name
        sh = c["an_L"]
        t_old = np.asarray(sub[c0]); t_new = np.asarray(cand[c0]).reshape((c["an_n"], c["an_K"]) + t_old.shape[1:])
        H0, W0 = t_old.shape[1:]
        fo = t_old.reshape(c["an_n"], H0 >> sh, 1 << sh, W0 >> sh, 1 << sh).sum((2, 4))[np.arange(c["an_n"]), cells[:, 0], cells[:, 1]]
        fn = t_new.reshape(c["an_n"], c["an_K"], H0 >> sh, 1 << sh, W0 >> sh, 1 << sh).sum((3, 5))[np.arange(c["an_n"]), :, cells[:, 0], cells[:, 1]]
        fmax = (1 << sh) ** 2 * (lay.chans[0][0].D - 1)
        pure = fo == fmax
        p2p = (pure[:, None] & (fn < fmax))
        # weighted (SNIS) fraction of contexts whose cell went full -> partial
        f2p = ((v_old[:, None] == full) & (v_new < full)) * wn
        d = {"an_valid": float(valid.mean()), "an_ctx_any_valid": float(valid.any(1).mean()),
             "an_ess_mean": float(ess[ess > 0].mean()) if (ess > 0).any() else 0.0,
             "an_full_old": float((v_old == full).mean()),
             "an_full2part_w": float(f2p.sum() / max((v_old == full).sum(), 1)),
             "an_pure_old": float(pure.mean()), "an_pure2part_w": float((p2p * wn).sum() / max(pure.sum(), 1)),
             "an_pure2part_valid_unw": float((p2p & valid).sum() / max((pure[:, None] & valid).sum(), 1)),
             "an_full2part_any": float((((v_old[:, None] == full) & (v_new < full)) & valid).any(1).mean())}
        p = wn.ravel()
        return cand, p, d

    def train(self, budget_s, log):
        from castlegen.kernels.logq import ml_opt_init
        t0 = time.time()
        c, p = self.cfg, self.problem
        region = Region(c["W_train"] or p.W_train)
        cells = p.H * region.W
        self._build(region)
        self.ctx_seed = 20_000_000 + 1_000_000 * self.seed
        opt = ml_opt_init(self.params)
        on_all = jnp.ones(p.T + 1, jnp.float32)
        on_1 = jnp.asarray([1.0 if l in c["ml_levels_1"] else 0.0 for l in range(p.T + 1)], jnp.float32)
        n = c["nctx"]
        key = jax.random.PRNGKey(4242 + self.seed)
        st = self._own_ctx(n, region)
        for ph in (1, 2):                                   # warm compiles, both phases
            st2, _, v = self.refine(st, key, ph)
        self.E(v)
        idx = np.arange(c["ml_B"])
        _ = self.mlstep(self.params, opt, {k: x[idx] for k, x in st2.items()}, jnp.ones(c["ml_B"]), jnp.float32(c["ml_lr"]), on_all)
        if c["an_n"] > 0:
            cand, pw, _ = self._anneal(st2, key)
            bi = np.arange(c["ml_B"] + c["an_B"]) % n
            _ = self.mlstep(self.params, opt, {k: jnp.concatenate([st2[k][:c["ml_B"]], cand[k][:c["an_B"]]]) for k in st2},
                            jnp.ones(c["ml_B"] + c["an_B"]), jnp.float32(c["ml_lr"]), on_all)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True, "start": self.start})
        t_train0 = time.time()
        it = 0
        while time.time() - t0 < budget_s + t_setup:
            it += 1
            phase = 1 if time.time() - t_train0 < c["phase1_s"] else 2
            ti = time.time()
            st = self._own_ctx(n, region)
            v0 = self._to_flat(st)
            s0, nv0, _ = self.E(v0)
            rec = {"iter": it, "phase": phase, "own_raw": self._ground_stats(st)}
            key, sub, ka = jax.random.split(key, 3)
            st, kd, v = self.refine(st, sub, phase)
            s1, nv1, _ = self.E(v)
            s0, s1, nv0, nv1 = (np.asarray(x) / cells for x in (s0, s1, nv0, nv1))
            rec.update(kd)
            rec["own_refined"] = self._ground_stats(st)
            rec["soft_own"], rec["soft_own_refined"] = float(s0.mean()), float(s1.mean())
            rec["viol_own"], rec["viol_own_refined"] = float(nv0.mean()), float(nv1.mean())
            tm = time.time()
            use_an = phase == 2 and c["an_n"] > 0
            if use_an:
                cand, pw, ad = self._anneal(st, ka)
                rec.update(ad)
                rec["time_anneal"] = time.time() - tm
                use_an = pw.sum() > 0
            on = on_1 if phase == 1 else on_all
            losses, ninf = [], 0
            for u in range(c["upi"]):
                idx = self.rng.choice(n, c["ml_B"], replace=False)
                batch = {k: x[idx] for k, x in st.items()}
                wts = np.ones(c["ml_B"])
                if use_an:
                    j = self.rng.choice(pw.size, c["an_B"], p=pw / pw.sum())
                    batch = {k: jnp.concatenate([batch[k], cand[k][j]]) for k in batch}
                    wts = np.ones(c["ml_B"] + c["an_B"])
                elif c["an_n"] > 0:                   # keep one compiled shape
                    j = self.rng.choice(n, c["an_B"], replace=False)
                    batch = {k: jnp.concatenate([batch[k], st[k][j]]) for k in batch}
                    wts = np.concatenate([np.ones(c["ml_B"]), np.zeros(c["an_B"])])
                self.params, opt, info = self.mlstep(self.params, opt, batch, jnp.asarray(wts, jnp.float32),
                                                     jnp.float32(c["ml_lr"]), on)
                losses.append(float(info["loss"]) / cells)
                ninf += int(info["n_infeasible"])
                self.updates += 1
            rec.update({"t": time.time() - t0, "t_train": time.time() - t_train0, "update": self.updates,
                        "loss_per_cell": float(np.mean(losses)), "infeasible_draws_per_state": ninf / (c["upi"] * c["ml_B"]),
                        "time_refine": tm - ti, "time_ml": time.time() - tm})
            log(rec)
        self.timing = {"setup": t_setup, "total": time.time() - t0, "iters": it}


register(CKernelOwn)


POP = dict(phase1_s=0.0, m_flip=0, tt_Ls=(4, 3), tt_tau_hi=2.0, tt_n=16, tt_m=4, refresh=0.1, n_cycles=1, cg_sweeps=0,
           hard_sweeps=0, lv_n=8, lv_every=2, clock_setup=True,
           max_iters=None,        # wave5-pop: stop after this many iterations (warm-up instances only)
           cache_all=True)        # wave5-pop: persist every compile (jax min compile time 0) during train


class CKernelPop(CKernelOwn):
    """Persistent own-started chains (wave3-explore).  A population of nctx states,
    initialised from own tau = 1 draws, is kept across iterations: each iteration
    replaces round(refresh * nctx) members (oldest first) by fresh own draws, applies
    n_cycles kernel cycles (phase-2 refine of ckernel_own) to the whole population,
    and fits weighted ML (weights 1) on it.  Every member is an own draw followed by
    p*-invariant kernels (amendment 2026-10-14); the chain just runs longer."""
    name = "ckernel_pop"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**POP, **cfg})

    def _pop_stats(self, st, v):
        d = self._ground_stats(st)
        s, nv, _ = self.E(v)
        nv = np.asarray(nv)
        d["valid"] = float((nv == 0).mean())
        d["viol_per_cell"] = float(nv.mean() / (self.problem.H * st["t"].shape[2]))
        T = self.problem.T
        cn = f"cert{T}"
        if cn in st:
            Bv = np.asarray(self.problem.chan[cn].views["B"])
            b = Bv[np.asarray(st[cn])]
            h = 1 << T
            d["B_top_row"] = float(b[:, 0].mean() / h)
            d["B_rows"] = (b.mean((0, 2)) / h).round(3).tolist()
        return d

    def train(self, budget_s, log):
        from castlegen.kernels.logq import ml_opt_init
        t0 = time.time()
        c, p = self.cfg, self.problem
        if c["cache_all"]:
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        region = Region(c["W_train"] or p.W_train)
        cells = p.H * region.W
        self._build(region)
        self.ctx_seed = 20_000_000 + 1_000_000 * self.seed
        opt = ml_opt_init(self.params)
        on = jnp.ones(p.T + 1, jnp.float32)
        n = c["nctx"]
        nref = max(1, int(round(c["refresh"] * n)))
        key = jax.random.PRNGKey(4242 + self.seed)
        pop = self._own_ctx(n, region)
        age = np.zeros(n, int)
        fresh = self._own_ctx(nref, region)                 # warm the refresh shape
        self._pop_stats(fresh, self._to_flat(fresh))
        _ = {k: x.at[jnp.asarray(np.arange(nref))].set(fresh[k]) for k, x in pop.items()}
        if c["lv_n"] > 0:                                   # warm the logging shape
            self._own_ctx(c["lv_n"], region)
        pop, _, v = self.refine(pop, key, 2)
        self.E(v)
        self._pop_stats(pop, v)
        if c["cg_sweeps"] > 0 or c["hard_sweeps"] > 0:
            from castlegen.kernels import colour_gibbs as cg
            if c["cg_sweeps"] > 0:
                cg.sweep_all(self.lay_tr, pop, 1.0, c["kernel_lam"], key)
            if c["hard_sweeps"] > 0:
                cg.sweep_all(self.lay_tr, pop, 1.0, None, key)
        idx = np.arange(c["ml_B"])
        _ = self.mlstep(self.params, opt, {k: x[idx] for k, x in pop.items()}, jnp.ones(c["ml_B"]), jnp.float32(c["ml_lr"]), on)
        self.rkl_updates = 0
        if c["rkl_n"] > 0:
            self._rkl_setup(log)
            self._rkl_steps(1)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True, "start": self.start, "refresh": nref})
        it = 0
        t_end = budget_s if c["clock_setup"] else budget_s + t_setup
        while time.time() - t0 < t_end and (c["max_iters"] is None or it < c["max_iters"]):
            it += 1
            ti = time.time()
            fresh = self._own_ctx(nref, region)
            rec = {"iter": it, "own_raw": self._pop_stats(fresh, self._to_flat(fresh))}
            old = np.argsort(-age)[:nref]
            oj = jnp.asarray(old)
            pop = {k: x.at[oj].set(fresh[k]) for k, x in pop.items()}
            age[old] = 0
            key, sub = jax.random.split(key)
            kd = {}
            for _ in range(c["n_cycles"]):
                sub, s2 = jax.random.split(sub)
                pop, kd, v = self.refine(pop, s2, 2)
            if c["cg_sweeps"] > 0:                       # speed-2's colour Gibbs, all levels, lam = kernel_lam
                from castlegen.kernels import colour_gibbs as cg
                for _ in range(c["cg_sweeps"]):
                    sub, s3 = jax.random.split(sub)
                    pop = cg.sweep_all(self.lay_tr, pop, 1.0, c["kernel_lam"], s3)
                v = self._to_flat(pop)
            if c["hard_sweeps"] > 0:                     # lam = inf: violating values excluded (ML targets valid)
                from castlegen.kernels import colour_gibbs as cg
                for _ in range(c["hard_sweeps"]):
                    sub, s4 = jax.random.split(sub)
                    pop = cg.sweep_all(self.lay_tr, pop, 1.0, None, s4)
                v = self._to_flat(pop)
            age += 1
            rec.update(kd)
            rec["pop"] = self._pop_stats(pop, v)
            rec["pop_age_mean"] = float(age.mean())
            tm = time.time()
            losses, ninf = [], 0
            for u in range(c["upi"]):
                idx = self.rng.choice(n, c["ml_B"], replace=False)
                self.params, opt, info = self.mlstep(self.params, opt, {k: x[idx] for k, x in pop.items()},
                                                     jnp.ones(c["ml_B"]), jnp.float32(c["ml_lr"]), on)
                losses.append(float(info["loss"]) / cells)
                ninf += int(info["n_infeasible"])
                self.updates += 1
            rec.update({"t": time.time() - t0, "t_train": time.time() - t0 - t_setup, "update": self.updates,
                        "loss_per_cell": float(np.mean(losses)), "infeasible_draws_per_state": ninf / (c["upi"] * c["ml_B"]),
                        "time_refine": tm - ti, "time_ml": time.time() - tm})
            if c["rkl_n"] > 0:
                tk = time.time()
                rec["rkl_raw_viol"] = self._rkl_steps(c["rkl_n"]).tolist()
                rec["lam"] = self.lam.tolist(); rec["rkl_updates"] = self.rkl_updates
                rec["time_rkl"] = time.time() - tk
            if c["lv_n"] > 0 and it % c["lv_every"] == 1 % c["lv_every"]:
                own8 = self._own_ctx(c["lv_n"], region)
                rec["own8"] = self._ground_stats(own8)
                rec["lv_own"] = self._level_viol(own8, 0, c["lv_n"], region)["frac"]
                rec["lv_pop"] = self._level_viol(pop, 0, n, region)["frac"]
            log(rec)
        self.pop = pop
        self.timing = {"setup": t_setup, "total": time.time() - t0, "iters": it}


register(CKernelPop)

P3 = dict(refresh=0.03, cg_sweeps=20)


def _pop_variant(name, **over):
    cls = type(f"CKernelPop_{name}", (CKernelPop,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**P3, **over, **cfg})
    cls.__init__ = init
    return register(cls)


# wave4-pop: violation pressure, own draws only (inside the frozen constitution)
_pop_variant("ckpop_hard", hard_sweeps=2)
_pop_variant("ckpop_hard_rkl", hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3])

# wave5-pop: reverse-KL pressure on L4 too, the L4 parameter step scaled down (wave2-bench's rkl4w3 recipe)
_pop_variant("ckpop_rkl4w3", hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3, 4],
             rkl_level_scale={4: 0.3})
_pop_variant("ckpop_rkl4w1", hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3, 4],
             rkl_level_scale={4: 0.1})

# wave6-pop: more reverse-KL pressure on L1-L3 (parameter step x2 at L1-L3, Adam lr 1e-3 -> effective 2e-3), own draws only
_pop_variant("ckpop_rkl123x2", hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3],
             rkl_level_scale={1: 2.0, 2: 2.0, 3: 2.0})
_pop_variant("ckpop_rkl123x2_cg40", hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3],
             rkl_level_scale={1: 2.0, 2: 2.0, 3: 2.0}, cg_sweeps=40)

# change request #2 (adopted 2026-10-14): ckpop_hard_rkl with the penalty-only term (cfast pen_only: the 32 steps per
# iteration are the unbiased estimator of grad sum_l lam_l E_q[H_l], no designed-energy or entropy term; same lr,
# levels and lambda control).  Own draws only.  ckpop_pen_all also prices L4 (harmless now: no entropy pull there).
_pop_variant("ckpop_hard_pen", hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3], pen_only=True)
_pop_variant("ckpop_pen_all", hard_sweeps=2, rkl_n=32, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3, 4], pen_only=True)
# cr2-penalty follow-ups: less penalty pressure (half the steps; half the lr) to recover line_match inside the budget
_pop_variant("ckpop_pen16", hard_sweeps=2, rkl_n=16, rkl_lr=1e-3, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3], pen_only=True)
_pop_variant("ckpop_pen_lr5", hard_sweeps=2, rkl_n=32, rkl_lr=5e-4, rkl_lam0=10.0, rkl_levels=[0, 1, 2, 3], pen_only=True)
