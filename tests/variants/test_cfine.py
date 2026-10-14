"""cfine: windowed-replay faithfulness and unbiasedness with fine-level branching.

1. Windowed replays are faithful: for selected draws at the windowed levels,
   the windowed replay's final state equals the full-world override replay's
   final state bit for bit (every cell of every sampled channel), its cone
   return equals the full replay's cone-box sum, and the batched windowed
   returns equal creplay's batched full-world returns for the same draws and
   actions.  (Run on CPU: JAX_PLATFORMS=cpu.)
2. Gradient check (the _gradcheck machinery): on the tiny enumerable problem
   the mean of cfine's actor gradient equals the exact grad J with the fine
   level branched (windowed and full replays, enumerated and sampled,
   non-forced subsets), the existing coarse configurations, and zero and
   wrong critics."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants._cfast_core import train_pass
from castlegen.variants.cfine import CFine


def _setup(prob, W, **cfg):
    var = CFine(prob, 0, "/tmp/cfine_test", W_train=W, width=16, **cfg)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    lay = var.layout(Region(W))
    return var, lay, var.train_fns(lay)


@pytest.mark.parametrize("name,W,cfg", [("toy_support", 64, dict(Kr=3, blevs=(1, 0), R=(6, 12), wlev=1, chunk=64)),
                                        ("ground", 64, dict(Kr=3, blevs=(2, 1, 0), R=(3, 4, 6), wlev=2, chunk=64))])
def test_window_faithful(name, W, cfg):
    prob = problems.get(name)
    var, lay, fn = _setup(prob, W, **cfg)
    seeds = np.array([11, 12], np.int32)
    des = var._batch(seeds, Region(W))
    js = jnp.asarray(seeds)
    tau, lam = jnp.float32(1.3), jnp.full(prob.T + 1, 4.0, jnp.float32)
    st, recs, viol, tot = fn["run_pass"](var.params, js, des, tau, lam)
    final = {c.name: st[c.name] for c in prob.chans if c.designed is None}
    cone = fn["cone"]
    rec_fn = jax.jit(jax.vmap(lambda s, d, o: train_pass(lay, var.params, s, d, tau, lam, override=o)))
    nchk = 0
    assert fn["wlevels"]
    for l in fn["wlevels"]:
        bt = fn["btab"][l]
        sel = fn["select"](recs, l, jax.random.PRNGKey(l))
        Gw = np.asarray(jax.jit(lambda s: fn["replay"](var.params, js, des, tau, lam, l, s, final))(sel))
        Gf = np.asarray(jax.jit(lambda s: fn["replay"](var.params, js, des, tau, lam, l, s, None))(sel))
        real = np.asarray(sel["w"]) > 0
        assert real.any()
        assert np.allclose(Gw[real], Gf[real], rtol=1e-4, atol=1e-3), (name, l, np.abs(Gw - Gf).max())
        ci, kk, jj, acts = (np.asarray(sel[f]) for f in ("ci", "kk", "jj", "acts"))
        B, R, K = acts.shape
        one = jax.jit(lambda s, d, fi, c_, k_, j_, v_: fn["wreplay_one"](var.params, s, d, fi, tau, lam, l, c_, k_, j_, v_,
                                                                         ret_state=True))
        rng = np.random.default_rng(l)
        for b in range(B):
            for r in rng.choice(np.flatnonzero(real[b]), min(int(real[b].sum()), 3), replace=False):
                for k in range(K):
                    step = int(bt["steps"][ci[b, r], kk[b, r]])
                    ov = (jnp.asarray([step]), jnp.asarray([int(jj[b, r])]), jnp.asarray([int(acts[b, r, k])]),
                          jnp.ones(1, bool))
                    sf, rf = rec_fn(js[b:b + 1], {q: v[b:b + 1] for q, v in des.items()}, ov)
                    gw, sw = one(js[b], {q: v[b] for q, v in des.items()}, {q: v[b] for q, v in final.items()},
                                 int(ci[b, r]), int(kk[b, r]), int(jj[b, r]), int(acts[b, r, k]))
                    for c in final:
                        assert np.array_equal(np.asarray(sw[c]), np.asarray(sf[c][0])), (name, l, c)
                    Gfull = cone.returns({q: x["cost"] for q, x in rf.items()})[bt["keys"][ci[b, r]]]
                    g_ref = float(Gfull[0, kk[b, r], jj[b, r]])
                    assert float(gw) == pytest.approx(g_ref, rel=1e-4, abs=1e-3)
                    assert Gw[b, r, k] == pytest.approx(g_ref, rel=1e-4, abs=1e-3)
                    nchk += 1
    assert nchk > 0


# ------------------------------------------------------------ gradient check
TAU, LAM = 1.5, np.array([2.0, 3.0])


@pytest.fixture(scope="module")
def exact():
    p = gc.tiny_problem()
    var = CFine(p, 0, "/tmp/cfine_gc", Kr=4, blevs=(1, 0), R=(None, None))
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    ex = gc.Exact(var, LAM, reference=False, ref_check=200)
    J, g, info = ex.J_and_grad(var.params, TAU)
    return p, var.params, var.cparams, ex, J, g


def _estimate(p, params, cparams, M, B, seed0, **cfg):
    var = CFine(p, 0, "/tmp/cfine_gc", **cfg)
    var.params, var.cparams = params, cparams
    lay = var.layout(Region(8))
    fn = var.train_fns(lay)
    gs = {k: jnp.float32(1.0) for k in params}
    tau, lam = jnp.float32(TAU), jnp.asarray(LAM, jnp.float32)
    est, cost, enum = [], [], []
    for m in range(M):
        seeds = jnp.arange(seed0 + m * B, seed0 + (m + 1) * B, dtype=jnp.int32)
        g, tot, diag = fn["agrad"](params, cparams, gs, seeds, {}, tau, lam, jax.random.PRNGKey(seed0 + m))
        g = {k: jax.tree_util.tree_map(lambda x: x * fn["nval"][int(k[1:].split("_")[0])], v) for k, v in g.items()}
        est.append(gc.flat(g))
        cost.append(np.asarray(tot))
        enum.append({l: float(d["enum_frac"]) for l, d in diag.items()})
    return np.array(est), np.concatenate(cost), enum


CONFIGS = {
    "fine_enum_win": dict(Kr=4, blevs=(0,), R=(None,), wlev=0),               # fine level only, all enumerated, windowed
    "fine_nf_win": dict(Kr=2, blevs=(0,), R=(3,), wlev=0),                    # non-forced subset of the fine level
    "both_nf_win": dict(Kr=2, blevs=(1, 0), R=(2, 3), wlev=1),                # both levels windowed, subsets, LOO at top
    "both_nf_full": dict(Kr=2, blevs=(1, 0), R=(2, 3), wlev=-1),              # same with full-world replays
    "coarse_loo": dict(Kr=2, blevs=(1,), R=(None,), wlev=-1),                 # creplay's loo_top
    "uniform_sel": dict(Kr=2, blevs=(1, 0), R=(2, 5), wlev=1, nonforced=False),  # creplay's selection, windowed
    "wrong_critic": dict(Kr=2, blevs=(1, 0), R=(3, 4), wlev=1),
}


@pytest.mark.parametrize("cname", list(CONFIGS))
def test_cfine_unbiased(exact, cname):
    p, params, cparams, ex, J, g = exact
    if cname == "wrong_critic":
        cparams = gc.randomize(cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    est, cost, enum = _estimate(p, params, cparams, 300, 16, 20_000, **CONFIGS[cname])
    assert abs(cost.mean() - J) < 4 * cost.std() / np.sqrt(len(cost))
    gf = gc.flat(g)
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (cname, k, z)
        assert z["hotelling_p"] > 1e-4, (cname, k, z)
        assert abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.01, (cname, k, z)
