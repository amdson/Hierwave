"""cpass: the reference variant, "Coloured-pass sampler: final implementation
spec" as written (notes/bench.md; core in _cpass_core.py).

Training (one update): B worlds drawn on the periodic training strip with
records; returns G_i = sum of every later step's cost (the conservative
superset of D_i: all later draws); one coarsest-level draw per world gets K
coupled replays under the world's noise (enumerated when |A_i| <= K); actor
loss with the critic as control variate; critic fitted afterwards on the same
batch; bounded penalty steps; tau 2 -> 1 over the first 20 % of the budget.
Generation: the pass at tau = 1, then one repair round."""
from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.variant import Variant, register

from . import _cpass_core as core

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cpass"))

CFG = dict(B=8, K=4, width=64, lr=1e-3, lr_critic=1e-3, lam0=10.0, lam_max=20.0, eta=0.01,
           alpha=0.05, tau0=2.0, tau1=1.0, anneal=0.2, clip=10.0, baseline_worlds=4)


# -------------------------------------------------------------------- Adam
def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": z, "t": 0}


def adam_step(p, g, s, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = s["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, s["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, s["v"], g)
    p = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps),
                               p, m, v)
    return p, {"m": m, "v": v, "t": t}


def clip_by_level(g, levels, c):
    """Clip each level's gradient (all its channels' networks) to norm c."""
    out = dict(g)
    for l in levels:
        ks = [k for k in g if k.startswith(f"l{l}_")]
        n = jnp.sqrt(sum(jnp.sum(x * x) for k in ks for x in jax.tree_util.tree_leaves(g[k])))
        s = jnp.minimum(1.0, c / (n + 1e-12))
        for k in ks:
            out[k] = jax.tree_util.tree_map(lambda x: x * s, g[k])
    return out


@register
class CPass(Variant):
    name = "cpass"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir)
        self.cfg = {**CFG, **cfg}
        self.lays = {}
        lay = self.layout(Region(problem.W_train))
        T = problem.T
        key = jax.random.PRNGKey(seed)
        self.params, self.cparams = {}, {}
        n_extra = 1 + (T + 1) + 1                  # tau, lambda per level, step fraction
        for l in range(T + 1):
            for c in lay.chans[l]:
                key, k1, k2 = jax.random.split(key, 3)
                self.params[core.key_of(l, c.name)] = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), c.D,
                                                                     self.cfg["width"])
                self.cparams[core.key_of(l, c.name)] = core.mlp_init(
                    k2, lay.feat_dim(l, core.CRITIC_WIN, n_extra), c.D, self.cfg["width"])
        self.opt, self.copt = adam_init(self.params), adam_init(self.cparams)
        self.lam = np.full(T + 1, self.cfg["lam0"])
        self.vbar = np.zeros(T + 1)
        self.eps = np.full(T + 1, 1e-4)
        self.tau = self.cfg["tau0"]
        self.gscale = None
        self.rng = np.random.default_rng(seed)
        self.train_seed = 1000 * seed
        self.nhard = None
        self.updates = 0

    # ------------------------------------------------------------- layouts
    def layout(self, region):
        key = (region.W, region.periodic, region.x0)
        if key not in self.lays:
            lay = core.Layout(self.problem, region.W, region.periodic, region.x0)
            lay.jit_gen = jax.jit(lambda params, seed, des, lam: core.coloured_pass(
                lay, params, seed, des, 1.0, lam)[0])
            lay.jit_repair = jax.jit(lambda st, seed: core.repair(lay, st, seed))
            lay.jit_score = jax.jit(lambda params, des, forced, lam: core.coloured_pass(
                lay, params, 0, des, 1.0, lam, forced=forced)[2])
            self.lays[key] = lay
        return self.lays[key]

    def designed(self, seed, region):
        return {k: jnp.asarray(v, jnp.int32) for k, v in self.problem.designed_values(seed, region).items()}

    # -------------------------------------------------------------- training
    def _build_train(self, lay):
        T, K = self.problem.T, self.cfg["K"]
        levels = list(range(T + 1))
        topkeys = [core.key_of(T, c.name) for c in lay.chans[T]]
        ndraw = {l: float(sum(int(np.prod(lay.nyx[l])) for _ in lay.chans[l]) * core.NCOL) for l in levels}
        nval = {}
        for l in levels:
            cnt = 0
            for k in range(core.NCOL):
                _, _, v = lay.sites(l, k)
                cnt += int(np.asarray(v).sum())
            nval[l] = cnt * len(lay.chans[l])

        def gen_rec(params, seed, des, tau, lam, extra):
            st, tot, _, rec = core.coloured_pass(lay, params, seed, des, tau, lam, record=True, critic_extra=extra)
            return st, tot, rec

        def gen_tot(params, seed, des, tau, lam, ov):
            return core.coloured_pass(lay, params, seed, des, tau, lam, override=ov)[1]

        lay.jit_rec = jax.jit(jax.vmap(gen_rec, in_axes=(None, 0, 0, None, None, None)))
        lay.jit_tot = jax.jit(jax.vmap(gen_tot, in_axes=(None, 0, 0, None, None, 0)))

        def actor_loss(params, cparams, recs, G, tau, lam, br, gs):
            total = 0.0
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    b = core.mlp(params[key], r["feat"])                          # (B, 16, n, D)
                    logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, -1)
                    q = jnp.exp(logq)
                    lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                    ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                    Q = jax.lax.stop_gradient(core.mlp(cparams[key], r["cfeat"])) * gs
                    t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                    la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    t2 = la * jax.lax.stop_gradient(G[key][..., None] - Qa)
                    keep = r["valid"]
                    if key in topkeys:                                            # the branched draw is replaced
                        keep = keep & ~br["sel"][key]
                        bl = branch_loss(key, r, b, logq, q, ell, Q, br)
                        lev = lev + jnp.sum(bl)
                    lev = lev + jnp.sum(jnp.where(keep, t1 + t2, 0.0))
                total = total + lev / nval[l]
            return total / recs[topkeys[0]]["a"].shape[0]

        def branch_loss(key, r, b, logq, q, ell, Q, br):
            Bn = r["a"].shape[0]
            ix = jnp.arange(Bn)
            k, j = br["k"], br["j"]
            on = br["chan"] == topkeys.index(key)                                # (B,)
            qq, el, QQ, mk = q[ix, k, j], ell[ix, k, j], Q[ix, k, j], r["mask"][ix, k, j]
            lq = jnp.where(mk, logq[ix, k, j], 0.0)                              # (B, D); no -inf to gather
            # worlds branched on another channel carry that channel's actions: neutralise
            # them, or out-of-range gathers put NaN into the gradient through the where below
            acts = jnp.where(on[:, None], jnp.clip(br["acts"], 0, el.shape[-1] - 1), 0)
            Gk, enum = br["Gk"], br["enum"]                                      # (B, K), (B,)
            # enumeration: sum_a q(a) [ell(a) + G^a]
            Gfull = jnp.zeros_like(el).at[ix[:, None], acts].set(Gk)
            le = jnp.sum(jnp.where(mk, qq * jax.lax.stop_gradient(el + Gfull), 0.0), -1)
            # sampled actions: all-action term + leave-one-out coupled term
            t1 = jnp.sum(jnp.where(mk, qq * jax.lax.stop_gradient(el + QQ), 0.0), -1)
            rk = Gk - jnp.take_along_axis(QQ, acts, -1)
            beta = (jnp.sum(rk, -1, keepdims=True) - rk) / (K - 1)
            ls = t1 + jnp.mean(jnp.take_along_axis(lq, acts, -1) * jax.lax.stop_gradient(rk - beta), -1)
            return jnp.where(on, jnp.where(enum, le, ls), 0.0)

        def critic_loss(cparams, recs, G, br, gs):
            total = 0.0
            for l in levels:
                lev = 0.0
                for c in lay.chans[l]:
                    key = core.key_of(l, c.name)
                    r = recs[key]
                    Q = core.mlp(cparams[key], r["cfeat"])
                    Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                    err = (Qa - G[key][..., None] / gs) ** 2
                    lev = lev + jnp.sum(jnp.where(r["valid"], err, 0.0))
                    if key in topkeys:
                        ix = jnp.arange(Q.shape[0])
                        on = br["chan"] == topkeys.index(key)
                        acts = jnp.where(on[:, None], jnp.clip(br["acts"], 0, Q.shape[-1] - 1), 0)
                        Qk = jnp.take_along_axis(Q[ix, br["k"], br["j"]], acts, -1)
                        e2 = jnp.sum((Qk - br["Gk"] / gs) ** 2, -1)
                        lev = lev + jnp.sum(jnp.where(on, e2, 0.0))
                total = total + lev / nval[l]
            return total / recs[topkeys[0]]["a"].shape[0]

        def train_step(params, cparams, opt, copt, recs, G, tau, lam, br, gs, lr, lrc, clip):
            loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, br, gs)
            g = clip_by_level(g, levels, clip)
            params, opt = adam_step(params, g, opt, lr)
            closs, gc = jax.value_and_grad(critic_loss)(cparams, recs, G, br, gs)
            gc = clip_by_level(gc, levels, clip)
            cparams, copt = adam_step(cparams, gc, copt, lrc)
            return params, cparams, opt, copt, loss, closs

        lay.jit_step = jax.jit(train_step)
        lay.topkeys, lay.nval = topkeys, nval
        lay.actor_loss, lay.critic_loss = actor_loss, critic_loss     # exposed for the gradient test

    def _returns(self, lay, totals):
        """G for every step: the sum of all strictly later steps' costs."""
        after = jnp.sum(totals, -1, keepdims=True) - jnp.cumsum(totals, -1)        # (B, nsteps)
        G = {}
        for l in lay.levels:
            for ci, c in enumerate(lay.chans[l]):
                steps = np.array([lay.step(l, k, ci) for k in range(core.NCOL)])
                G[core.key_of(l, c.name)] = after[:, steps]
        return G, after

    def _branches(self, lay, recs, after, seeds, des, tau, lam):
        """Pick one coarsest draw per world; K coupled replays."""
        T, K, B = self.problem.T, self.cfg["K"], len(seeds)
        tk = lay.topkeys
        sel = {k: np.zeros(recs[k]["a"].shape, bool) for k in tk}
        chan, kk, jj = np.zeros(B, int), np.zeros(B, int), np.zeros(B, int)
        acts, enum = np.zeros((B, K), int), np.zeros(B, bool)
        for b in range(B):
            cand = [(ci, k, j) for ci, key in enumerate(tk)
                    for k, j in zip(*np.nonzero(np.asarray(recs[key]["valid"][b])))]
            ci, k, j = cand[self.rng.integers(len(cand))]
            chan[b], kk[b], jj[b] = ci, k, j
            sel[tk[ci]][b, k, j] = True
            r = recs[tk[ci]]
            mask = np.asarray(r["mask"][b, k, j])
            A = np.nonzero(mask)[0]
            if len(A) <= K:
                enum[b] = True
                acts[b] = np.concatenate([A, np.full(K - len(A), A[0])])
            else:
                lq = np.asarray(r["logq"][b, k, j], float)
                p = np.exp(lq - lq.max()) * mask
                p /= p.sum()
                acts[b, 0] = int(r["a"][b, k, j])
                acts[b, 1:] = self.rng.choice(len(p), K - 1, p=p)
        rep = lambda x: jnp.repeat(x, K, axis=0)
        ov = (jnp.asarray(np.repeat(kk, K)), jnp.asarray(np.repeat(chan, K)), jnp.asarray(np.repeat(jj, K)),
              jnp.asarray(acts.reshape(-1), jnp.int32), jnp.ones(B * K, bool))
        tot = lay.jit_tot(self.params, rep(seeds), {k: rep(v) for k, v in des.items()}, tau, lam, ov)
        aft = jnp.sum(tot, -1, keepdims=True) - jnp.cumsum(tot, -1)
        steps = np.array([lay.step(T, kk[b], chan[b]) for b in range(B)])
        Gk = np.asarray(aft).reshape(B, K, -1)[np.arange(B), :, steps]
        g0 = np.asarray(after)[np.arange(B), steps]     # replaying the original action reproduces G
        assert np.allclose(Gk[~enum, 0], g0[~enum], rtol=1e-3, atol=1e-1), (Gk[~enum, 0], g0[~enum])
        br = {"sel": {k: jnp.asarray(v) for k, v in sel.items()}, "chan": jnp.asarray(chan), "k": jnp.asarray(kk),
              "j": jnp.asarray(jj), "acts": jnp.asarray(acts), "Gk": jnp.asarray(Gk, jnp.float32),
              "enum": jnp.asarray(enum)}
        return br

    def _measure_baseline(self, lay, log):
        """Untrained raw and repaired violation fractions -> eps_l (spec, before training)."""
        T = self.problem.T
        region = Region(self.problem.W_train)
        raw, fin, n = np.zeros(T + 1), np.zeros(T + 1), np.zeros(T + 1)
        for i in range(self.cfg["baseline_worlds"]):
            seed = 500_000 + 1000 * self.seed + i
            x = self.generate(seed, region)
            des = self.problem.designed_values(seed, region)
            for kind, acc in (("raw", raw), ("final", fin)):
                E = en.energy(self.problem, {**des, **x[kind]}, region)
                for l in range(T + 1):
                    acc[l] += E.viol[l]
                    if kind == "raw":
                        n[l] += E.nhard[l]
        self.nhard = n / self.cfg["baseline_worlds"]
        rf, ff = raw / np.maximum(n, 1), fin / np.maximum(n, 1)
        rho = np.where(rf > 0, ff / np.maximum(rf, 1e-12), 1.0)
        self.eps = np.clip(1e-4 / np.maximum(rho, 1e-12), 1e-4, 1e-3)
        log({"baseline_raw": rf.tolist(), "baseline_final": ff.tolist(), "rho": rho.tolist(),
             "eps": self.eps.tolist()})
        self.baseline = {"raw": rf.tolist(), "final": ff.tolist(), "eps": self.eps.tolist()}

    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(self.problem.W_train)
        lay = self.layout(region)
        self._measure_baseline(lay, log)
        self._build_train(lay)
        B = cfg["B"]
        while True:
            frac = (time.time() - t0) / max(budget_s, 1e-9)
            if frac >= 1.0:
                break
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - frac / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            desl = [self.designed(int(s), region) for s in seeds]
            des = {k: jnp.stack([d[k] for d in desl]) for k in desl[0]}
            extra = jnp.concatenate([jnp.array([self.tau]), lam]).astype(jnp.float32)
            st, tot, recs = lay.jit_rec(self.params, jnp.asarray(seeds, jnp.int32), des, tau, lam, extra)
            G, after = self._returns(lay, tot)
            if self.gscale is None:                        # critic output scale, fixed after the first batch
                self.gscale = float(max(np.std(np.asarray(after)), 1.0))
            br = self._branches(lay, recs, after, jnp.asarray(seeds, jnp.int32), des, tau, lam)
            self.params, self.cparams, self.opt, self.copt, loss, closs = lay.jit_step(
                self.params, self.cparams, self.opt, self.copt, recs, G, tau, lam, br,
                jnp.float32(self.gscale), cfg["lr"], cfg["lr_critic"], cfg["clip"])
            # penalties: raw violated fraction per level from the closing draws
            viol = np.zeros(T + 1)
            for l in range(T + 1):
                for c in lay.chans[l]:
                    r = recs[core.key_of(l, c.name)]
                    nv_a = np.take_along_axis(np.asarray(r["nv"]), np.asarray(r["a"])[..., None], -1)[..., 0]
                    viol[l] += float((nv_a * np.asarray(r["valid"])).sum()) / B
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            log({"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                 "critic_loss": float(closs), "cost_per_world": float(np.asarray(tot).sum(-1).mean()),
                 "raw_viol": hf.tolist(), "lam": self.lam.tolist(),
                 "enum_frac": float(np.asarray(br["enum"]).mean())})

    # ------------------------------------------------------------ generation
    def generate(self, seed, region):
        T = self.problem.T
        if region.periodic:
            lay = self.layout(region)
            des = self.designed(seed, region)
            raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
            fin = lay.jit_repair(raw, seed)
            crop = lambda st, c: np.asarray(st[c.name])
        else:
            m = core.margin(self.problem)
            ext = Region(region.W + 2 * m, periodic=False, x0=region.x0 - m)
            lay = self.layout(ext)
            des = self.designed(seed, ext)
            raw = lay.jit_gen(self.params, seed, des, jnp.asarray(self.lam, jnp.float32))
            fin = lay.jit_repair(raw, seed)
            crop = lambda st, c: np.asarray(st[c.name])[:, (m >> c.level):(m >> c.level) + (region.W >> c.level)]
        out = {}
        for kind, st in (("raw", raw), ("final", fin)):
            out[kind] = {c.name: crop(st, c) for c in self.problem.sampled()}
        return out

    def score(self, x, region):
        assert region.periodic
        lay = self.layout(region)
        des = {c.name: jnp.asarray(x[c.name], jnp.int32) for c in self.problem.chans if c.designed is not None}
        forced = {c.name: jnp.asarray(x[c.name], jnp.int32) for c in self.problem.sampled()}
        return float(lay.jit_score(self.params, des, forced, jnp.asarray(self.lam, jnp.float32)))

    def info(self):
        return {"updates": self.updates, "lam": self.lam.tolist(), "tau": self.tau,
                "baseline": getattr(self, "baseline", None), "margin": core.margin(self.problem),
                "cfg": self.cfg}
