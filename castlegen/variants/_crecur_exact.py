"""Guard (1) of amendment 2026-10-16 (b): the K-sample recursive cost against exact subtree free energies
on the enumerable two-level problem (_gradcheck.tiny_problem: T = 1, 16 binary fine cells t, 4 coarse
cells c with 3 values), and the coarse-gradient check.

For each coarse state c (reachable by the pass):
  E_fine(c, t)  = sum of the fine (level-0) draws' closed costs soft + lam_0 nv  (cost identity),
  F_supp(c)     = -log sum_{t: q(t|c) > 0} exp(-E_fine)    (what any estimator with q-proposals converges to),
  F_all(c)      = -log sum_{all 2^16 t} exp(-E_fine)       (the true subtree free energy),
  KL(c)         = E_q[log q(t|c) + E_fine] - F_supp(c)     (= the exact bias of R_1).
R_K = -log (1/K) sum_k exp(-E_fine(c, t_k) - log q_fine(t_k|c)), t_k from the split pass (castlegen/
variants/_crecur_core.split_pass: one coarse seed, K fine seeds).  Fine models: (a) exact conditionals
(t_k ~ p*_supp(.|c) by enumeration; and p*_all), (b) the pass's own fine level at random parameters,
(b2) the same with the fine output layer x3 (deliberately worse).
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region

from . import _cpass_core as core
from . import _gradcheck as gc
from ._cfast_cone import Cone
from ._crecur_core import build, split_pass


def setup(lam=(2.0, 3.0), pseed=1, fine_scale=1.0):
    from .cfast import CFast
    p = gc.tiny_problem()
    var = CFast(p, 0, "/tmp/crecur_gc")
    params = gc.randomize(var.params, jax.random.PRNGKey(pseed))
    if fine_scale != 1.0:
        params = {k: ({n: (x * fine_scale if n in ("W3", "b3") else x) for n, x in v.items()} if k.startswith("l0_")
                      else v) for k, v in params.items()}
    var.params = params
    lay = var.layout(Region(8))
    return p, var, lay, np.asarray(lam, np.float32)


def forced_terms(lay, params, X, lam, chunk=8192):
    """Per state: E_fine, log q_fine, E_coarse, log q_coarse (from forced records)."""
    lamj = jnp.asarray(lam)

    def one(forced):
        _, _, _, rec = core.coloured_pass(lay, params, 0, {}, 1.0, lamj, forced=forced, record=True)
        out = []
        for l in (0, 1):
            e, lq = 0.0, 0.0
            for c in lay.chans[l]:
                r = rec[core.key_of(l, c.name)]
                a = r["a"][..., None]
                s = jnp.take_along_axis(r["soft"], a, -1)[..., 0]
                nv = jnp.take_along_axis(r["nv"], a, -1)[..., 0]
                la = jnp.take_along_axis(r["logq"], a, -1)[..., 0]
                e = e + jnp.sum(jnp.where(r["valid"], s + lamj[l] * nv, 0.0))
                lq = lq + jnp.sum(jnp.where(r["valid"], la, 0.0))
            out += [e, lq]
        return jnp.stack(out)

    f = jax.jit(jax.vmap(one))
    n = len(next(iter(X.values())))
    res = [np.asarray(f({k: jnp.asarray(v[s:s + chunk]) for k, v in X.items()}), np.float64) for s in range(0, n, chunk)]
    r = np.concatenate(res)
    return {"Ef": r[:, 0], "lqf": r[:, 1], "Ec": r[:, 2], "lqc": r[:, 3]}


def exact_tables(lay, params, lam, all_t=()):
    """all_t: indices into the sorted coarse states for which F_all (2^16 fine states) is enumerated."""
    X = gc.reachable_states(lay)
    tr = forced_terms(lay, params, X, lam)
    ckey = [tuple(c.ravel()) for c in X["c"]]
    cs = sorted(set(ckey))
    ci = np.array([cs.index(c) for c in ckey])
    out = {"X": X, "terms": tr, "cs": cs, "ci": ci}
    nc = len(cs)
    F_supp, KL, qc, Ec, lqc = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
    for j in range(nc):
        m = ci == j
        Ef, lqf = tr["Ef"][m], tr["lqf"][m]
        F_supp[j] = -np.logaddexp.reduce(-Ef)
        qf = np.exp(lqf)
        assert abs(qf.sum() - 1) < 1e-4, qf.sum()
        KL[j] = float(qf @ (lqf + Ef)) - F_supp[j]
        Ec[j], lqc[j] = tr["Ec"][m][0], tr["lqc"][m][0]
        qc[j] = np.exp(lqc[j])
    out.update(F_supp=F_supp, KL=KL, qc=qc, Ec=Ec, lqc=lqc)
    F_all = np.full(nc, np.nan)
    if len(all_t):
        # every t (2^16) under each chosen c
        T = np.array(list(itertools.product([0, 1], repeat=16)), np.int32).reshape(-1, *X["t"].shape[1:])
        for j in all_t:
            c = cs[j]
            Xc = {"c": np.broadcast_to(np.asarray(c, np.int32).reshape(X["c"].shape[1:]), (len(T),) + X["c"].shape[1:]).copy(),
                  "t": T}
            Ef = forced_terms(lay, params, Xc, lam)["Ef"]
            F_all[j] = -np.logaddexp.reduce(-Ef[np.isfinite(Ef)])
    out["F_all"] = F_all
    return out


def mc_fine_sums(lay, params, lam, n_c, K, seed0=1_000_000):
    """n_c coarse draws, K fine completions each, through split_pass.  Returns coarse states (n_c, ...) and the
    per-replica fine cost sums (n_c, K) = E_fine + log q_fine (tau = 1)."""
    lamj = jnp.asarray(lam)
    f = jax.jit(jax.vmap(lambda s, fs: split_pass(lay, params, s, fs, {}, 1.0, lamj, 1, record=False)))
    cseeds = np.repeat(np.arange(seed0, seed0 + n_c), K).astype(np.int32)
    fseeds = (np.arange(n_c * K) + 7 * seed0 + 13).astype(np.int32)
    C, Fs = [], []
    for s in range(0, len(cseeds), 4096):
        st, costs = f(jnp.asarray(cseeds[s:s + 4096]), jnp.asarray(fseeds[s:s + 4096]))
        C.append(np.asarray(st["c"]))
        Fs.append(sum(np.asarray(v, np.float64).sum((1, 2)) for k, v in costs.items() if k.startswith("l0_")))
    C = np.concatenate(C).reshape((n_c, K) + C[0].shape[1:])
    assert (C == C[:, :1]).all(), "coarse state differs across the K replicas of one coarse seed"
    return C[:, 0], np.concatenate(Fs).reshape(n_c, K)


def R_K(fsum, K):
    """-log (1/K) sum_k exp(-fsum_k) over the first K replicas."""
    x = -fsum[:, :K]
    return -(np.logaddexp.reduce(x, 1) - np.log(K))


def bias_table(ex, cstate, fsum, Ks=(1, 4, 16)):
    cs, ci_of = ex["cs"], {c: i for i, c in enumerate(ex["cs"])}
    idx = np.array([ci_of[tuple(c.ravel())] for c in cstate])
    rows = {}
    for K in Ks:
        R = R_K(fsum, K)
        err = R - ex["F_supp"][idx]
        # q(c)-weighted mean bias = plain mean over coarse draws; per-c means for the 5 most frequent c
        rows[K] = {"bias_mean": float(err.mean()), "bias_se": float(err.std() / np.sqrt(len(err))),
                   }
    top = np.argsort(-np.bincount(idx, minlength=len(cs)))[:5]
    per_c = []
    for j in top:
        m = idx == j
        d = {"c": list(map(int, cs[j])), "n": int(m.sum()), "q_c": float(ex["qc"][j]), "F_supp": float(ex["F_supp"][j]),
             "F_all": float(ex["F_all"][j]), "KL_exact": float(ex["KL"][j])}
        for K in Ks:
            d[f"bias_K{K}"] = float((R_K(fsum[m], K) - ex["F_supp"][j]).mean())
        per_c.append(d)
    rows["exact_KL_mean"] = float(np.mean(ex["KL"][idx]))
    return rows, per_c


def exact_sampler_bias(ex, n_c, K, rng, which="supp"):
    """(a) exact fine conditionals: t_k ~ p*(t | c) by enumeration, w_k = exp(-E_fine - log p*(t_k|c))."""
    tr, ci = ex["terms"], ex["ci"]
    qc = ex["qc"] / ex["qc"].sum()
    js = rng.choice(len(qc), n_c, p=qc)
    errs = {k: [] for k in (1, 4, 16)}
    for j in js:
        Ef = tr["Ef"][ci == j]
        lp = -Ef - np.logaddexp.reduce(-Ef)
        tk = rng.choice(len(Ef), K, p=np.exp(lp))
        fsum = (Ef[tk] + lp[tk])[None]
        for k in errs:
            errs[k].append(float(R_K(fsum, k)[0] - ex["F_supp"][j]))
    return {k: (float(np.mean(v)), float(np.max(np.abs(v)))) for k, v in errs.items()}


# -------------------------------------------------------------- gradient
def exact_coarse_grads(lay, params, lam, ex):
    """Exact gradients wrt the coarse network of (i) the joint J = E_q[E + log q] (what R_1 targets) and
    (ii) the marginal J_m = sum_c q(c) (E_c + log q(c) + F_supp(c)) (what R_K, K -> inf, targets)."""
    X, ci = ex["X"], ex["ci"]
    lamj = jnp.asarray(lam)
    Xj = {k: jnp.asarray(v) for k, v in X.items()}
    tr = ex["terms"]
    E = tr["Ef"] + tr["Ec"]

    def lq_all(pc):
        pr = {**params, **pc}
        return jax.vmap(lambda f: core.coloured_pass(lay, pr, 0, {}, 1.0, lamj, forced=f)[2])(Xj)

    def lqc_all(pc):
        pr = {**params, **pc}

        def one(f):
            _, _, _, rec = core.coloured_pass(lay, pr, 0, {}, 1.0, lamj, forced=f, record=True)
            r = rec["l1_c"]
            la = jnp.take_along_axis(r["logq"], r["a"][..., None], -1)[..., 0]
            return jnp.sum(jnp.where(r["valid"], la, 0.0))
        return jax.vmap(one)(Xj)

    pc = {"l1_c": params["l1_c"]}
    lq, vjp = jax.vjp(lq_all, pc)
    lq = np.asarray(lq, np.float64)
    q = np.exp(lq)
    gJ, = vjp(jnp.asarray(q * (E + lq), jnp.float32))
    # marginal: one representative state per c
    rep = np.array([np.flatnonzero(ci == j)[0] for j in range(len(ex["cs"]))])
    lqc, vjpc = jax.vjp(lqc_all, pc)
    lqc = np.asarray(lqc, np.float64)[rep]
    qc = np.exp(lqc)
    wts = np.zeros(len(q))
    wts[rep] = qc * (ex["Ec"] + lqc + ex["F_supp"])
    gM, = vjpc(jnp.asarray(wts, jnp.float32))
    return gJ, gM, {"J": float(q @ (E + lq)), "J_m": float(qc @ (ex["Ec"] + lqc + ex["F_supp"]))}


def estimator_grads(lay, params, lam, K, M, B, seed0, mode="cone"):
    """M batches of B worlds: the crecur coarse gradient, x nval[1] (estimates grad J at K = 1)."""
    cone = Cone(lay, lay.actor_win)
    fn = build(lay, cone, 1, K, mode=mode)
    lamj = jnp.asarray(lam)
    est = []
    for m in range(M):
        cs = jnp.arange(seed0 + m * B, seed0 + (m + 1) * B, dtype=jnp.int32)
        fs = jnp.asarray((np.arange(B * K) + 3 * seed0 + 977 + m * B * K).reshape(B, K), jnp.int32)
        g, G, st = fn["grad"](params, cs, fs, {}, lamj)
        g = jax.tree_util.tree_map(lambda x: x * fn["nval"][1], g)
        est.append(gc.flat(g))
    return np.array(est)
