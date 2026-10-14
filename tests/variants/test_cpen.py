"""pen_only (constitution change request #2, adopted by the user 2026-10-14): cfast's fused
update with every draw's cost = lam_l * violations only, i.e. the same unbiased estimator
(all-action + score-function terms, cone returns, critic as control variate) applied to
J_pen = sum_l lam_l E_q[H_l] instead of E_q[S + sum_l lam_l H_l + tau log q].

1. Gradient check (spec test 5 machinery) on the tiny enumerable problem: the mean of the
   pen_only actor gradient equals the exact grad J_pen, with a zero and a wrong critic; the
   draw costs sum to lam . H(x) per world (cost identity).
2. Power: the same estimator is rejected against the full grad J.
3. pen_only changes costs, not draws: same states as cfast, tot = lam . violations (toy_support).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cfast import CFast

TAU, LAM = 1.5, np.array([2.0, 3.0])


@pytest.fixture(scope="module")
def exact():
    p = gc.tiny_problem()
    var = CFast(p, 0, "/tmp/cpen_gc", pen_only=True)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    ex = gc.Exact(var, LAM, reference=False, ref_check=200)
    # exact grad of J_pen = sum_x q(x) lam . H(x)
    X = {k: jnp.asarray(a) for k, a in ex.X.items()}
    lq, vjp = jax.vjp(lambda pr: ex._lq(pr, X, TAU), var.params)
    q = np.exp(np.asarray(lq, np.float64))
    Hl = ex.H @ LAM
    J_pen = float(np.sum(q * Hl))
    g_pen, = vjp(jnp.asarray(q * Hl, jnp.float32))
    J, g, info = ex.J_and_grad(var.params, TAU)
    assert abs(float(q.sum()) - 1) < 1e-4 and info["p_viol"] > 0.1
    return p, var, ex, J_pen, g_pen, g


def _estimate(var, cparams, M, B, seed0):
    lay = var.layout(Region(8))
    fn = var.train_fns(lay)
    gs = {k: jnp.float32(1.0) for k in var.params}
    tau, lam = jnp.float32(TAU), jnp.asarray(LAM, jnp.float32)
    est, cost, states, seeds_all = [], [], [], []
    for m in range(M):
        seeds = jnp.arange(seed0 + m * B, seed0 + (m + 1) * B, dtype=jnp.int32)
        st, recs, G, viol, tot, _ = fn["rec"](var.params, seeds, {}, tau, lam)
        g = fn["actor_grad"](var.params, cparams, recs, G, tau, lam, gs)
        g = {k: jax.tree_util.tree_map(lambda x: x * fn["nval"][int(k[1:].split("_")[0])], v) for k, v in g.items()}
        est.append(gc.flat(g))
        cost.append(np.asarray(tot))
        if m < 2:
            states.append(jax.device_get(st))
    return np.array(est), np.concatenate(cost), states


def test_cost_identity(exact):
    """Summed draw costs of a world = lam . H(x) (reference energy)."""
    p, var, ex, J_pen, g_pen, g = exact
    est, cost, states = _estimate(var, var.cparams, 2, 16, 90_000)
    region = Region(8)
    i = 0
    for st in states:
        for b in range(16):
            E = en.energy(p, {c.name: np.asarray(st[c.name][b]) for c in p.sampled()}, region)
            assert np.isclose(cost[i], LAM @ [E.viol[l] for l in range(p.T + 1)], atol=1e-3), (i, cost[i], E.viol)
            i += 1


@pytest.mark.parametrize("critic", ["zero", "wrong"])
def test_pen_only_unbiased(exact, critic):
    p, var, ex, J_pen, g_pen, g = exact
    cparams = var.cparams
    if critic == "wrong":
        cparams = gc.randomize(cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    est, cost, _ = _estimate(var, cparams, 300, 16, 10_000)
    assert abs(cost.mean() - J_pen) < 4 * cost.std() / np.sqrt(len(cost))   # the costs estimate J_pen
    gf = gc.flat(g_pen)
    for k, s in gc.block_slices(g_pen).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        print(critic, k, {x: round(z[x], 3) for x in ("proj_z", "rand_z", "hotelling_p", "proj_ratio", "rel_noise")})
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (critic, k, z)
        assert z["hotelling_p"] > 1e-4, (critic, k, z)
        assert abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.01, (critic, k, z)


def test_power_against_full_objective(exact):
    """The test has power: the pen_only estimator is NOT the gradient of the full J."""
    p, var, ex, J_pen, g_pen, g = exact
    est, _, _ = _estimate(var, var.cparams, 300, 16, 50_000)
    zs = {k: gc.ztest(est[:, s], gc.flat(g)[s], 16) for k, s in gc.block_slices(g).items()}
    print({k: (round(z["hotelling_p"], 8), round(z["proj_z"], 2)) for k, z in zs.items()})
    assert any(z["hotelling_p"] < 1e-6 or abs(z["proj_z"]) > 6 for z in zs.values()), zs


def test_same_draws_cost_is_penalty():
    prob = problems.get("toy_support")
    a = CFast(prob, 0, "/tmp/cpen_t", W_train=64)
    b = CFast(prob, 0, "/tmp/cpen_t", W_train=64, pen_only=True)
    params = gc.randomize(a.params, jax.random.PRNGKey(4))
    lay = a.layout(Region(64))
    fa, fb = a.train_fns(lay), b.train_fns(lay)
    assert fa is not fb
    seeds = jnp.arange(5, dtype=jnp.int32)
    des = a._batch(np.arange(5), Region(64))
    tau, lam = jnp.float32(1.2), jnp.asarray([3.0, 5.0, 7.0][:prob.T + 1], jnp.float32)
    gs = {k: jnp.float32(2.0) for k in params}
    oa = fa["fused"](params, a.cparams, a.opt, a.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(9))
    ob = fb["fused"](params, b.cparams, b.opt, b.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(9))
    for c in prob.sampled():
        assert np.array_equal(np.asarray(oa[0][c.name]), np.asarray(ob[0][c.name]))     # same draws
    assert np.allclose(np.asarray(oa[7]), np.asarray(ob[7]))                            # same violations
    viol = np.asarray(ob[7])                                                             # per-world mean, per level
    assert np.isclose(float(np.asarray(ob[8]).sum()), 5 * float(np.asarray(lam) @ viol), rtol=1e-4, atol=1e-3)
    assert not np.isclose(float(np.asarray(oa[8]).sum()), float(np.asarray(ob[8]).sum()))  # the full cost differs
    assert float(oa[5]) != float(ob[5])                                                  # and so does the loss
