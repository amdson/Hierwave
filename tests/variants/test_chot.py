"""chot: eps-hot exploration with importance weights.

1. eps = 0 reproduces cfast exactly: train_pass_hot draws what train_pass
   draws, and chot_off's fused update equals cfast's.
2. Replays are faithful with hot draws on: the hot coins are keyed by the
   coordinate noise, so replacing one draw changes nothing outside its cone
   (draws, costs, coins and ratios), and replaying the drawn value reproduces
   the pass bit for bit.
3. Gradient check (spec test 5, _gradcheck machinery), tiny enumerable
   problem, tau = 1, eps = 0.3, tau_b = 3, hot draws at every level and at the
   top level only, zero and wrong critics: the mean weighted gradient equals
   the exact grad J.  Negative control: hot draws without the weights are
   rejected."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _chot_core
from castlegen.variants import _cpass_core as core
from castlegen.variants import _gradcheck as gc
from castlegen.variants._cfast_core import train_pass
from castlegen.variants.cfast import CFast
from castlegen.variants.chot import CHot

TAU, LAM = 1.0, np.array([2.0, 3.0])


@pytest.mark.parametrize("name", ["toy_support", "ground"])
def test_eps0_matches_cfast_pass(name):
    prob = problems.get(name)
    var = CHot(prob, 0, "/tmp/chot_t", W_train=64)
    params = gc.randomize(var.params, jax.random.PRNGKey(3))
    lay = var.layout(Region(64))
    des = {k: v[0] for k, v in var._batch(np.array([7]), Region(64)).items()}
    lam = jnp.full(prob.T + 1, 4.0, jnp.float32)
    st0, r0 = train_pass(lay, params, 7, des, 1.3, lam)
    st1, r1 = _chot_core.train_pass_hot(lay, params, 7, des, 1.3, lam, jnp.zeros(prob.T + 1), 3.0)
    for c in prob.sampled():
        assert np.array_equal(np.asarray(st0[c.name]), np.asarray(st1[c.name]))
    for k in r0:
        assert np.array_equal(np.asarray(r0[k]["cost"]), np.asarray(r1[k]["cost"]))
        assert float(jnp.abs(r1[k]["logrho"]).max()) == 0.0


def test_off_matches_cfast_update():
    prob = problems.get("toy_support")
    a = CFast(prob, 0, "/tmp/chot_t", W_train=64)
    b = CHot(prob, 0, "/tmp/chot_t", W_train=64, eps=0.0)
    params = gc.randomize(a.params, jax.random.PRNGKey(4))
    lay = a.layout(Region(64))
    fa, fb = a.train_fns(lay), b.train_fns(lay)
    seeds = jnp.arange(5, dtype=jnp.int32)
    des = a._batch(np.arange(5), Region(64))
    tau, lam = jnp.float32(1.2), jnp.asarray(a.lam, jnp.float32)
    gs = {k: jnp.float32(2.0) for k in params}
    oa = fa["fused"](params, a.cparams, a.opt, a.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
                     jax.random.PRNGKey(9))
    ob = fb["fused_hot"](params, b.cparams, b.opt, b.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
                         jax.random.PRNGKey(9), jnp.zeros(prob.T + 1))
    for x, y in zip(jax.tree_util.tree_leaves(oa[1]), jax.tree_util.tree_leaves(ob[1])):
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)
    assert np.isclose(float(oa[5]), float(ob[5]), rtol=1e-5)
    assert np.allclose(np.asarray(ob[10]), 1.0)


@pytest.mark.parametrize("name,width,nrep", [("toy_support", 64, 16), ("ground", 64, 10), ("dimers", None, 10)])
def test_replay_faithful_hot(name, width, nrep):
    prob = problems.get(name)
    W = width or prob.W_train
    var = CHot(prob, 0, "/tmp/chot_t", W_train=W)
    params = gc.randomize(var.params, jax.random.PRNGKey(1))
    lay = var.layout(Region(W))
    des = {k: v[0] for k, v in var._batch(np.array([3]), Region(W)).items()}
    cone = var.train_fns(lay)["cone"]
    lam = jnp.full(prob.T + 1, 5.0, jnp.float32)
    eps = jnp.full(prob.T + 1, 0.3, jnp.float32)
    f = jax.jit(jax.vmap(lambda o: _chot_core.train_pass_hot(lay, params, 3, des, 1.0, lam, eps, 3.0,
                                                              override=o)[1]))
    rng = np.random.default_rng(1)
    keys = list(cone.boxes)
    r0 = jax.device_get(f(tuple(jnp.asarray([x]) for x in (0, 0, 0, False))))
    draws, ovs = [], []
    for _ in range(nrep):
        key = keys[rng.integers(len(keys))]
        l = int(key[1:key.index("_")])
        cname = key[key.index("_") + 1:]
        ci = [c.name for c in lay.chans[l]].index(cname)
        while True:
            k, j = rng.integers(core.NCOL), rng.integers(lay.nsite(l))
            if bool(np.asarray(lay.sites(l, k)[2])[j]):
                break
        draws.append((key, k, j))
        ovs.append((lay.step(l, k, ci), j, int(rng.integers(prob.chan[cname].D)), True))
        # replaying the drawn value reproduces the pass
        draws.append((key, k, j))
        ovs.append((lay.step(l, k, ci), j, int(r0[key]["a"][0][k, j]), True))
    recs = jax.device_get(f(tuple(jnp.asarray(x) for x in zip(*ovs))))
    nhot = sum(int(r0[k]["hot"][0].sum()) for k in r0)
    assert nhot > 0
    nchanged = 0
    for i, (key, k, j) in enumerate(draws):
        mem = cone.member(key, k, j)
        same = i % 2 == 1
        for key2, rec in recs.items():
            v = r0[key2]["valid"][0]
            assert np.array_equal(rec["hot"][i], r0[key2]["hot"][0])            # coins never move
            ch = ((rec["a"][i] != r0[key2]["a"][0])
                  | ~np.isclose(rec["cost"][i], r0[key2]["cost"][0], rtol=1e-5, atol=1e-5)
                  | ~np.isclose(rec["logrho"][i], r0[key2]["logrho"][0], rtol=1e-5, atol=1e-5)) & v
            if same:
                assert not ch.any(), (name, key2)
                continue
            if key2 == key:
                ch[k, j] = False
            nchanged += int(ch.sum())
            assert not (ch & ~mem[key2]).any(), (name, key, key2)
    assert nchanged > 0


# ------------------------------------------------------------ gradient check
@pytest.fixture(scope="module")
def exact():
    p = gc.tiny_problem()
    var = CHot(p, 0, "/tmp/chot_gc", tau_b=3.0)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    ex = gc.Exact(var, LAM, reference=False, ref_check=200)
    J, g, info = ex.J_and_grad(var.params, TAU)
    return p, var, ex, J, g


def _estimate(var, cparams, M, B, seed0, eps, neg=False):
    lay = var.layout(Region(8))
    fn = _chot_core.build_train(var, lay, neg=neg)
    gs = {k: jnp.float32(1.0) for k in var.params}
    tau, lam = jnp.float32(TAU), jnp.asarray(LAM, jnp.float32)
    est, cost, Ws = [], [], []
    for m in range(M):
        seeds = jnp.arange(seed0 + m * B, seed0 + (m + 1) * B, dtype=jnp.int32)
        g, tot, W = fn["agrad"](var.params, cparams, gs, seeds, {}, tau, lam, jax.random.PRNGKey(seed0 + m),
                                jnp.asarray(eps, jnp.float32))
        g = {k: jax.tree_util.tree_map(lambda x: x * fn["nval"][int(k[1:].split("_")[0])], v) for k, v in g.items()}
        est.append(gc.flat(g))
        cost.append(np.asarray(tot) * np.asarray(W))
        Ws.append(np.asarray(W))
    return np.array(est), np.concatenate(cost), np.concatenate(Ws)


CASES = {"all_zero": ([0.3, 0.3], "zero"), "all_wrong": ([0.3, 0.3], "wrong"),
         "top_zero": ([0.0, 0.3], "zero"), "top_wrong": ([0.0, 0.3], "wrong")}


@pytest.mark.parametrize("case", list(CASES))
def test_chot_unbiased(exact, case):
    p, var, ex, J, g = exact
    eps, critic = CASES[case]
    cparams = var.cparams
    if critic == "wrong":
        cparams = gc.randomize(cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    est, cost, W = _estimate(var, cparams, 300, 16, 40_000, eps)
    assert abs(W.mean() - 1) < 4 * W.std() / np.sqrt(len(W))           # E_b[W] = 1
    assert abs(cost.mean() - J) < 4 * cost.std() / np.sqrt(len(cost))   # weighted cost estimates J
    gf = gc.flat(g)
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        print(case, k, {x: round(z[x], 3) for x in ("proj_z", "rand_z", "hotelling_p", "proj_ratio", "rel_noise")})
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (case, k, z)
        assert z["hotelling_p"] > 1e-4, (case, k, z)
        assert abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.01, (case, k, z)


def test_unweighted_hot_rejected(exact):
    """NEGATIVE CONTROL: hot draws with the weights dropped estimate the
    behaviour policy's gradient, not grad J; must be rejected."""
    p, var, ex, J, g = exact
    est, _, _ = _estimate(var, var.cparams, 300, 16, 70_000, [0.3, 0.3], neg=True)
    zs = {k: gc.ztest(est[:, s], gc.flat(g)[s], 16) for k, s in gc.block_slices(g).items()}
    print({k: (round(z["hotelling_p"], 8), round(z["proj_z"], 2)) for k, z in zs.items()})
    assert any(z["hotelling_p"] < 1e-6 or abs(z["proj_z"]) > 6 for z in zs.values()), zs
