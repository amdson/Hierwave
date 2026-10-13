"""creplay: replay faithfulness and unbiasedness.

1. Replays (batched over world, draw, action, chunked) are faithful: every
   cost outside the replaced draw's cone is bit-identical to the original
   pass, every draw outside it too, replaying the drawn action reproduces the
   original cone return, and the replayed cone return equals the brute-force
   sum over the cone's members.  (Run on CPU: JAX_PLATFORMS=cpu.)
2. Gradient check (spec test 5, with the _gradcheck machinery): on the tiny
   enumerable problem the mean of creplay's actor gradient equals the exact
   grad J, with branches enumerated, sampled (leave-one-out), on a random
   subset of fine draws, and with a wrong fixed critic."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _cpass_core as core
from castlegen.variants import _gradcheck as gc
from castlegen.variants._cfast_core import train_pass
from castlegen.variants.creplay import CReplay


def _setup(prob, W, **cfg):
    var = CReplay(prob, 0, "/tmp/creplay_test", W_train=W, width=16, **cfg)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    lay = var.layout(Region(W))
    return var, lay, var.train_fns(lay)


@pytest.mark.parametrize("name,W,cfg", [("toy_support", 64, dict(Kr=3, R=(None, 16))),
                                        ("columns", 64, dict(Kr=3, R=(12, 6), chunk=16)),
                                        ("ground", 64, dict(Kr=3, R=(12, 6), chunk=16)),
                                        ("dimers", None, dict(Kr=3, blev=1, R=(10,), chunk=16))])
def test_replay_faithful(name, W, cfg):
    prob = problems.get(name)
    W = W or prob.W_train
    var, lay, fn = _setup(prob, W, **cfg)
    seeds = np.array([11, 12], np.int32)
    des = var._batch(seeds, Region(W))
    js = jnp.asarray(seeds)
    tau, lam = jnp.float32(1.3), jnp.full(prob.T + 1, 4.0, jnp.float32)
    st, recs, viol, tot = fn["run_pass"](var.params, js, des, tau, lam)
    costs0 = jax.device_get({k: r["cost"] for k, r in recs.items()})
    acts0 = jax.device_get({k: r["a"] for k, r in recs.items()})
    valid = jax.device_get({k: r["valid"] for k, r in recs.items()})
    cone = fn["cone"]
    G0all = cone.returns({k: r["cost"] for k, r in recs.items()})
    nchk = 0
    for l in fn["blevels"]:
        bt = fn["btab"][l]
        sel = fn["select"](recs, l, jax.random.PRNGKey(l))
        Gk = np.asarray(fn["replay"](var.params, js, des, tau, lam, l, sel))
        ci, kk, jj, acts, enum = (np.asarray(sel[f]) for f in ("ci", "kk", "jj", "acts", "enum"))
        B, R, K = acts.shape
        for b in range(B):
            for r in range(R):
                key = bt["keys"][ci[b, r]]
                g0 = float(G0all[key][b, kk[b, r], jj[b, r]])
                if not enum[b, r]:
                    assert acts[b, r, 0] == acts0[key][b, kk[b, r], jj[b, r]]
                    assert Gk[b, r, 0] == pytest.approx(g0, rel=1e-5, abs=1e-3)
                # action set is legal: inside A
                m = np.asarray(recs[key]["mask"][b, kk[b, r], jj[b, r]])
                assert m[acts[b, r]].all()
        # a few (world, draw, action) replays in full: outside the cone nothing moves
        rng = np.random.default_rng(l)
        picks = [(b, r, k) for b in range(B) for r in rng.choice(R, min(R, 3), replace=False)
                 for k in range(K)]
        step = jnp.asarray([int(bt["steps"][ci[b, r], kk[b, r]]) for b, r, k in picks])
        site = jnp.asarray([int(jj[b, r]) for b, r, k in picks])
        val = jnp.asarray([int(acts[b, r, k]) for b, r, k in picks])
        sd = jnp.asarray([int(seeds[b]) for b, r, k in picks])
        dd = {kk_: v[np.array([b for b, r, k in picks])] for kk_, v in des.items()}
        cr = jax.device_get(jax.jit(fn["replay_costs"])(var.params, sd, dd, tau, lam, step, site, val))
        rec_fn = jax.jit(jax.vmap(lambda s, d, o: train_pass(lay, var.params, s, d, tau, lam, override=o)[1]))
        ar = jax.device_get(rec_fn(sd, dd, (step, site, val, jnp.ones(len(picks), bool))))
        for i, (b, r, k) in enumerate(picks):
            key = bt["keys"][ci[b, r]]
            mem = cone.member(key, int(kk[b, r]), int(jj[b, r]))
            brute = 0.0
            for key2 in costs0:
                out = ~mem[key2] & valid[key2][b]
                if key2 == key:
                    out[kk[b, r], jj[b, r]] = False                       # the replaced draw itself
                # draws: bit-identical; costs: equal up to float32 round-off (XLA's reduction
                # order in the MLP depends on the vmapped batch size)
                assert np.array_equal(ar[key2]["a"][i][out], acts0[key2][b][out]), (name, key2)
                assert np.allclose(cr[key2][i][out], costs0[key2][b][out], rtol=1e-5, atol=1e-5), (name, key2)
                assert np.allclose(ar[key2]["cost"][i][out], costs0[key2][b][out], rtol=1e-5, atol=1e-5)
                brute += float(np.asarray(cr[key2][i], np.float64)[mem[key2]].sum())
            assert Gk[b, r, k] == pytest.approx(brute, rel=1e-4, abs=1e-2)
            nchk += 1
    assert nchk > 0


# ------------------------------------------------------------ gradient check
TAU, LAM = 1.5, np.array([2.0, 3.0])


@pytest.fixture(scope="module")
def exact():
    p = gc.tiny_problem()
    var = CReplay(p, 0, "/tmp/creplay_gc", Kr=4, blev=2, R=(None, None))
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    ex = gc.Exact(var, LAM, reference=False, ref_check=200)
    J, g, info = ex.J_and_grad(var.params, TAU)
    return p, var.params, var.cparams, ex, J, g


def _estimate(p, params, cparams, M, B, seed0, **cfg):
    var = CReplay(p, 0, "/tmp/creplay_gc", **cfg)
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
    "enum_both": dict(Kr=4, blev=2, R=(None, None)),            # every draw branched, all enumerated
    "loo_top": dict(Kr=2, blev=1, R=(None,)),                   # top level sampled (|A| up to 3 > 2), LOO
    "loo_subset": dict(Kr=2, blev=2, R=(2, 5)),                 # random subsets at both levels
    "wrong_critic": dict(Kr=2, blev=2, R=(3, 6)),
}


@pytest.mark.parametrize("cname", list(CONFIGS))
def test_creplay_unbiased(exact, cname):
    p, params, cparams, ex, J, g = exact
    if cname == "wrong_critic":
        cparams = gc.randomize(cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    est, cost, enum = _estimate(p, params, cparams, 300, 16, 20_000, **CONFIGS[cname])
    assert abs(cost.mean() - J) < 4 * cost.std() / np.sqrt(len(cost))
    if cname == "loo_top":
        assert np.mean([e["l1"] for e in enum]) < 0.9            # some draws really are sampled
    gf = gc.flat(g)
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (cname, k, z)
        assert z["hotelling_p"] > 1e-4, (cname, k, z)
        assert abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.01, (cname, k, z)
