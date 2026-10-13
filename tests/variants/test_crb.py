"""crb: Rao-Blackwellised cone returns.

1. expected_cost equals the brute-force sum over each draw's least-violating set.
2. rb=False reproduces cfast's gradient exactly (same worlds, same subsample).
3. Gradient check (spec test 5, _gradcheck machinery): on the tiny enumerable
   problem the mean of crb's actor gradient equals the exact grad J (zero and
   wrong critic); a deliberately biased estimator (expectation under the
   uniform distribution over A instead of q) is rejected."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _crb_core
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cfast import CFast
from castlegen.variants.crb import CRB

TAU, LAM = 1.5, np.array([2.0, 3.0])


def test_expected_cost_brute():
    prob = problems.get("toy_support")
    var = CRB(prob, 0, "/tmp/crb_t", W_train=64)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(3))
    lay = var.layout(Region(64))
    fn = var.train_fns(lay)
    lam = jnp.linspace(3.0, 5.0, prob.T + 1).astype(jnp.float32)
    from castlegen.variants._cfast_core import train_pass
    _, recs = train_pass(lay, var.params, 11, {}, 1.3, lam)
    for key, r in recs.items():
        l = fn["level_of"][key]
        r1 = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), r)
        got = np.asarray(_crb_core.expected_cost(r, 1.3, lam[l]))
        m, v = r1["mask"].astype(bool), r1["valid"].astype(bool)
        ref = np.zeros(got.shape)
        for k in range(16):
            for j in range(m.shape[1]):
                if not v[k, j]:
                    continue
                A = np.flatnonzero(m[k, j])
                lq = r1["logq"][k, j, A]
                ref[k, j] = np.sum(np.exp(lq) * (r1["soft"][k, j, A] + float(lam[l]) * r1["nv"][k, j, A]
                                                 + 1.3 * lq))
        assert np.allclose(got, ref, rtol=1e-4, atol=1e-4), key


def test_rb_off_matches_cfast():
    prob = problems.get("toy_support")
    a = CFast(prob, 0, "/tmp/crb_t", W_train=64)
    b = CRB(prob, 0, "/tmp/crb_t", W_train=64, rb=False)
    params = gc.randomize(a.params, jax.random.PRNGKey(4))
    lay = a.layout(Region(64))
    fa, fb = a.train_fns(lay), b.train_fns(lay)
    seeds = jnp.arange(5, dtype=jnp.int32)
    des = a._batch(np.arange(5), Region(64))
    tau, lam = jnp.float32(1.2), jnp.asarray(a.lam, jnp.float32)
    gs = {k: jnp.float32(2.0) for k in params}
    ones = {k: jnp.float32(1.0) for k in params}
    oa = fa["fused"](params, a.cparams, a.opt, a.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
                     jax.random.PRNGKey(9))
    ob = fb["fused"](params, b.cparams, b.opt, b.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
                     jax.random.PRNGKey(9))
    for x, y in zip(jax.tree_util.tree_leaves(oa[1]), jax.tree_util.tree_leaves(ob[1])):
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)
    assert np.isclose(float(oa[5]), float(ob[5]), rtol=1e-5)


# ------------------------------------------------------------ gradient check
@pytest.fixture(scope="module")
def exact():
    p = gc.tiny_problem()
    var = CRB(p, 0, "/tmp/crb_gc")
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    ex = gc.Exact(var, LAM, reference=False, ref_check=200)
    J, g, info = ex.J_and_grad(var.params, TAU)
    return p, var, ex, J, g


def _estimate(var, cparams, M, B, seed0, rb=True, neg=None):
    lay = var.layout(Region(8))
    fn = _crb_core.build_train(var, lay, rb=rb, neg=neg)
    gs = {k: jnp.float32(1.0) for k in var.params}
    tau, lam = jnp.float32(TAU), jnp.asarray(LAM, jnp.float32)
    est, cost = [], []
    for m in range(M):
        seeds = jnp.arange(seed0 + m * B, seed0 + (m + 1) * B, dtype=jnp.int32)
        g, tot = fn["agrad"](var.params, cparams, gs, seeds, {}, tau, lam, jax.random.PRNGKey(seed0 + m))
        g = {k: jax.tree_util.tree_map(lambda x: x * fn["nval"][int(k[1:].split("_")[0])], v) for k, v in g.items()}
        est.append(gc.flat(g))
        cost.append(np.asarray(tot))
    return np.array(est), np.concatenate(cost)


@pytest.mark.parametrize("critic", ["zero", "wrong"])
def test_crb_unbiased(exact, critic):
    p, var, ex, J, g = exact
    cparams = var.cparams
    if critic == "wrong":
        cparams = gc.randomize(cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    est, cost = _estimate(var, cparams, 300, 16, 30_000)
    assert abs(cost.mean() - J) < 4 * cost.std() / np.sqrt(len(cost))
    gf = gc.flat(g)
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (critic, k, z)
        assert z["hotelling_p"] > 1e-4, (critic, k, z)
        assert abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.01, (critic, k, z)


def test_negative_control_rejected(exact):
    """Expectation under uniform-over-A instead of q: biased, must be rejected."""
    p, var, ex, J, g = exact
    est, _ = _estimate(var, var.cparams, 300, 16, 60_000, neg="uniform")
    zs = {k: gc.ztest(est[:, s], gc.flat(g)[s], 16) for k, s in gc.block_slices(g).items()}
    assert any(z["hotelling_p"] < 1e-6 or abs(z["proj_z"]) > 6 for z in zs.values()), zs
