"""crevise (ideas #19, Amendment 2026-10-14 (b)): revising generation trained by the path KL.

(a) R = 0 is cfast bit for bit: same draws, same per-world costs, same parameters after one fused update.
(b) Cost identity: the summed per-draw costs of a world equal S(x_0) + lam . H(x_0) + tau (log Q(traj)
    + sum_masked log D), with S, H from the reference energy of the final state.
(c) Gradient check (spec test 5 machinery) on a tiny enumerable problem with the erase coins fixed: the mean
    estimator gradient (all-action + score-function terms, cone returns, critic as control variate) equals
    the exact gradient of the path KL computed by enumerating every trajectory, with a zero and a wrong
    critic; the draw costs average to the exact J.  Negative control: returns without the revising steps'
    costs (a dropped cost term, constraint 3) are rejected.  Also on the two-level gradcheck problem with a
    revising step at the coarse level (one erased cell).
(d) Bound gap on the tiny problem: path KL >= exact reverse KL of x_0 (both without log Z), reported.
(e) Finite support: replacing one draw changes nothing outside its dependency cone (replaying the drawn
    value changes nothing at all); two overlapping open windows agree on their overlap, raw and final.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _cpass_core as core
from castlegen.variants import _gradcheck as gc
from castlegen.variants._crevise_core import revise_pass
from castlegen.variants._crevise_gradcheck import ExactPath, enumerate_trajectories, tiny_flat_problem
from castlegen.variants.cfast import CFast
from castlegen.variants.crevise import CRevise

TAU, LAM = 1.5, np.array([2.0, 3.0])


# ------------------------------------------------------------------ (a) R = 0 = cfast
def test_r0_matches_cfast():
    prob = problems.get("toy_support")
    a = CFast(prob, 0, "/tmp/crv_t", W_train=64)
    b = CRevise(prob, 0, "/tmp/crv_t", W_train=64, R=0)
    params = gc.randomize(a.params, jax.random.PRNGKey(4))
    lay = a.layout(Region(64))
    fa, fb = a.train_fns(lay), b.train_fns(lay)
    seeds = jnp.arange(5, dtype=jnp.int32)
    des = a._batch(np.arange(5), Region(64))
    tau, lam = jnp.float32(1.2), jnp.asarray([3.0, 5.0, 7.0], jnp.float32)
    gs = {k: jnp.float32(2.0) for k in params}
    oa = fa["fused"](params, a.cparams, a.opt, a.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(9))
    ob = fb["fused"](params, b.cparams, b.opt, b.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(9))
    for c in prob.sampled():
        assert np.array_equal(np.asarray(oa[0][c.name]), np.asarray(ob[0][c.name]))
    assert np.array_equal(np.asarray(oa[8]), np.asarray(ob[8]))            # per-world costs
    assert np.array_equal(np.asarray(oa[7]), np.asarray(ob[7]))            # violations
    for x, y in zip(jax.tree_util.tree_leaves(oa[1]), jax.tree_util.tree_leaves(ob[1])):
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-7)
    assert np.isclose(float(oa[5]), float(ob[5]), rtol=1e-6)
    # score falls back to cpass's exact log q
    x = {c.name: np.asarray(oa[0][c.name][0]) for c in prob.sampled()}
    x = {**{k: np.asarray(v[0]) for k, v in des.items()}, **x}
    b.params = params
    a.params = params
    assert np.isclose(a.score(x, Region(64)), b.score(x, Region(64)), rtol=1e-6)


# ------------------------------------------------------------------ (b) cost identity
@pytest.mark.parametrize("name,R,rho,guide", [("toy_support", 1, 0.5, False), ("ground", 2, 0.4, True),
                                              ("dimers", 2, 0.5, True)])
def test_cost_identity(name, R, rho, guide):
    prob = problems.get(name)
    W = 64 if name != "dimers" else prob.W_train
    var = CRevise(prob, 0, "/tmp/crv_t", W_train=W, R=R, rho=rho, guide=guide)
    params = gc.randomize(var.params, jax.random.PRNGKey(2))
    lay = var.layout(Region(W))
    fn = var.train_fns(lay)
    seeds = jnp.arange(3, dtype=jnp.int32)
    des = var._batch(np.arange(3), Region(W))
    tau, lam = jnp.float32(1.3), jnp.asarray(np.linspace(2.0, 6.0, prob.T + 1), jnp.float32)
    st, recs, G, viol, tot = fn["rec"](params, seeds, des, tau, lam)
    st = jax.device_get(st)
    logD = fn["logD"]
    nrev = 0
    for b in range(3):
        d = {k: np.asarray(v[b]) for k, v in des.items()}
        xs = {c.name: np.asarray(st[c.name][b]) for c in prob.sampled()}
        E = en.energy(prob, {**d, **xs}, Region(W))
        lq = sum(float(np.sum(np.where(np.asarray(recs[k]["valid"][b]), np.take_along_axis(
            np.asarray(recs[k]["logq"][b]), np.asarray(recs[k]["a"][b])[..., None], -1)[..., 0], 0.0))) for k in recs)
        ld = sum(logD[k] * float(np.sum(np.asarray(recs[k]["valid"][b]))) for k in recs)
        nrev += sum(int(np.sum(np.asarray(recs[k]["valid"][b]))) for k in recs if "@rev" in k)
        H = np.array([E.viol[l] for l in range(prob.T + 1)])
        pred = E.soft + float(np.asarray(lam) @ H) + float(tau) * (lq + ld)
        assert np.isclose(float(tot[b]), pred, rtol=1e-5, atol=2e-3), (b, float(tot[b]), pred)
        # violations of the final state from the telescoped dv
        hv = np.array([sum(float(np.sum(np.asarray(recs[k]["dv"][b]))) for k in recs if fn["level_of"][k] == l)
                       for l in range(prob.T + 1)])
        assert np.array_equal(hv, H), (hv, H)
    assert nrev > 0


# ------------------------------------------------------------------ (c) gradient check
@pytest.fixture(scope="module")
def flat():
    p = tiny_flat_problem()
    var = CRevise(p, 0, "/tmp/crv_gc", R=1, rho=0.5, guide=True)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1), scale=0.3)      # a soft policy: the revisions matter
    var.cparams = gc.zero_critic(var.cparams)
    lay = var.layout(Region(8))
    sch = var.sched(lay)
    forced, final, masks = enumerate_trajectories(sch, 0.5, 3)
    ex = ExactPath(var, TAU, LAM[:1], 3, forced, final)
    J, g, info = ex.J_and_grad(var.params)
    return var, ex, J, g, info


def _estimate(var, cparams, M, B, seed0, mask_seed, tau, lam, neg=None):
    lay = var.layout(Region(var.problem.W_train))
    fn = var.train_fns(lay)
    sch, cone = fn["sched"], fn["cone"]
    rho, share, guide = float(var.cfg["rho"]), bool(var.cfg["share"]), bool(var.cfg["guide"])
    tau_, lam_ = jnp.float32(tau), jnp.asarray(lam, jnp.float32)

    def rec(params, seeds):
        st, recs = jax.vmap(lambda s: revise_pass(sch, params, s, {}, tau_, lam_, rho, share, mask_seed=mask_seed,
                                                  guide=guide))(seeds)
        costs = {k: r["cost"] for k, r in recs.items()}
        if neg == "rev":                          # NEGATIVE CONTROL: the revising steps' costs leave the returns
            costs = {k: (jnp.zeros_like(c) if "@rev" in k else c) for k, c in costs.items()}
        elif neg == "all":                        # NEGATIVE CONTROL: no returns at all (myopic, all-action only)
            costs = {k: jnp.zeros_like(c) for k, c in costs.items()}
        tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
        return recs, cone.returns(costs), tot

    rec = jax.jit(rec)
    gs = {k: jnp.float32(1.0) for k in cparams}
    est, cost = [], []
    for m in range(M):
        seeds = jnp.arange(seed0 + m * B, seed0 + (m + 1) * B, dtype=jnp.int32)
        recs, G, tot = rec(var.params, seeds)
        g = fn["actor_grad"](var.params, cparams, recs, G, tau_, lam_, gs)
        g = {k: jax.tree_util.tree_map(lambda x: x * fn["nval"][int(k[1:].split("_")[0])], v) for k, v in g.items()}
        est.append(gc.flat(g))
        cost.append(np.asarray(tot))
    return np.array(est), np.concatenate(cost)


def test_flat_enumeration(flat):
    var, ex, J, g, info = flat
    assert abs(info["sum_q"] - 1) < 1e-4, info["sum_q"]          # the trajectories are exhaustive and normalised
    assert ex.N > 1000


@pytest.mark.parametrize("critic", ["zero", "wrong"])
def test_flat_unbiased(flat, critic):
    var, ex, J, g, info = flat
    cparams = var.cparams
    if critic == "wrong":
        cparams = gc.randomize(cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    est, cost = _estimate(var, cparams, 300, 16, 20_000, 3, TAU, LAM[:1])
    assert abs(cost.mean() - J) < 4 * cost.std() / np.sqrt(len(cost)), (cost.mean(), J)
    gf = gc.flat(g)
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        print(critic, k, {x: round(z[x], 3) for x in ("proj_z", "rand_z", "hotelling_p", "proj_ratio", "rel_noise")})
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (critic, k, z)
        assert z["hotelling_p"] > 1e-4, (critic, k, z)
        assert abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.01, (critic, k, z)


@pytest.mark.parametrize("neg", ["rev", "all"])
def test_flat_negative_control(flat, neg):
    """Returns that drop the revising steps' costs (or every later cost) are biased estimators: rejected."""
    var, ex, J, g, info = flat
    est, _ = _estimate(var, var.cparams, 300, 16, 60_000, 3, TAU, LAM[:1], neg=neg)
    zs = {k: gc.ztest(est[:, s], gc.flat(g)[s], 16) for k, s in gc.block_slices(g).items()}
    print({k: (round(z["hotelling_p"], 8), round(z["proj_z"], 2)) for k, z in zs.items()})
    assert any(z["hotelling_p"] < 1e-6 or abs(z["proj_z"]) > 6 for z in zs.values()), zs


def test_flat_bound_gap(flat):
    """(d) path KL >= exact reverse KL of x_0 (log Z cancels); the per-trajectory cost identity holds."""
    var, ex, J, g, info = flat
    r = ex.reverse_kl_parts(var.params, tau=1.0)
    print(r)
    assert r["ident_max"] < 1e-3, r["ident_max"]
    assert r["gap"] >= -1e-6, r
    assert abs(r["sum_q"] - 1) < 1e-4


def test_two_level_unbiased():
    """Hierarchy: gradcheck's two-level problem, one revising step at the coarse level (one erased cell)."""
    p = gc.tiny_problem()
    var = CRevise(p, 0, "/tmp/crv_gc2", R=[0, 1], rho=0.5, guide=False)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    lay = var.layout(Region(8))
    forced, final, masks = enumerate_trajectories(var.sched(lay), 0.5, 2)
    assert sum(int(m.sum()) for m in masks.values()) == 1
    ex = ExactPath(var, TAU, LAM, 2, forced, final)
    J, g, info = ex.J_and_grad(var.params)
    assert abs(info["sum_q"] - 1) < 1e-4
    est, cost = _estimate(var, var.cparams, 200, 16, 30_000, 2, TAU, LAM)
    assert abs(cost.mean() - J) < 4 * cost.std() / np.sqrt(len(cost)), (cost.mean(), J)
    gf = gc.flat(g)
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        print("two_level", k, {x: round(z[x], 3) for x in ("proj_z", "rand_z", "hotelling_p", "proj_ratio", "rel_noise")})
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (k, z)
        assert z["hotelling_p"] > 1e-4, (k, z)
    r = ex.reverse_kl_parts(var.params, tau=1.0)
    print(r)
    assert r["gap"] >= -1e-6 and r["ident_max"] < 1e-3


# ------------------------------------------------------------------ (e) finite support
@pytest.mark.parametrize("name,width,R,nrep,guide", [("toy_support", 64, 1, 16, False), ("dimers", None, 2, 12, True),
                                                     ("ground", 64, 1, 8, True), ("ground", 64, 1, 8, False)])
def test_changes_inside_cone(name, width, R, nrep, guide):
    prob = problems.get(name)
    W = width or prob.W_train
    var = CRevise(prob, 0, "/tmp/crv_t", W_train=W, R=R, rho=0.5, guide=guide)
    params = gc.randomize(var.params, jax.random.PRNGKey(1))
    lay = var.layout(Region(W))
    fn = var.train_fns(lay)
    sch, cone = fn["sched"], fn["cone"]
    des = {k: v[0] for k, v in var._batch(np.array([3]), Region(W)).items()}
    lam = jnp.full(prob.T + 1, 5.0, jnp.float32)
    f = jax.jit(jax.vmap(lambda o: revise_pass(sch, params, 3, des, 1.0, lam, 0.5, True, override=o, guide=guide)[1]))
    rng = np.random.default_rng(1)
    keys = list(cone.boxes)
    r0 = jax.device_get(f(tuple(jnp.asarray([x]) for x in (0, 0, 0, False))))
    draws, ovs = [], []
    for _ in range(nrep):
        key = keys[rng.integers(len(keys))]
        l, ci, cname, m, _ = sch.groups[key]
        while True:
            k, j = rng.integers(core.NCOL), rng.integers(lay.nsite(l))
            if bool(r0[key]["valid"][0][k, j]):
                break
        draws.append((key, k, j))
        ovs.append((sch.step(key, k), j, int(rng.integers(prob.chan[cname].D)), True))
        draws.append((key, k, j))
        ovs.append((sch.step(key, k), j, int(r0[key]["a"][0][k, j]), True))     # replaying the drawn value
    recs = jax.device_get(f(tuple(jnp.asarray(x) for x in zip(*ovs))))
    nchanged, nrev = 0, 0
    for i, (key, k, j) in enumerate(draws):
        mem = cone.member(key, k, j)
        same = i % 2 == 1
        for key2, rec in recs.items():
            v = r0[key2]["valid"][0]
            assert np.array_equal(rec["valid"][i], v)                        # the erase coins never move
            ch = ((rec["a"][i] != r0[key2]["a"][0])
                  | ~np.isclose(rec["cost"][i], r0[key2]["cost"][0], rtol=1e-5, atol=1e-5)) & v
            if same:
                assert not ch.any(), (name, key, key2)
                continue
            if key2 == key:
                ch[k, j] = False
            nchanged += int(ch.sum())
            if "@rev" in key2:
                nrev += int(ch.sum())
            assert not (ch & ~mem[key2]).any(), (name, key, k, j, key2, np.argwhere(ch & ~mem[key2])[:5])
    assert nchanged > 0 and nrev > 0


@pytest.mark.parametrize("name,R,guide", [("toy_support", 1, False), ("dimers", 2, True), ("ground", 1, True)])
def test_open_windows_agree(name, R, guide):
    prob = problems.get(name)
    var = CRevise(prob, 0, "/tmp/crv_t", R=R, rho=0.5, guide=guide)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(6))
    W = 64
    s = 1 << prob.T
    a = var.generate(11, Region(W, periodic=False, x0=0))
    b = var.generate(11, Region(W, periodic=False, x0=W // 2))
    for kind in ("raw", "final"):
        for c in prob.sampled():
            w = W >> c.level
            o = (W // 2) >> c.level
            assert np.array_equal(a[kind][c.name][:, o:], b[kind][c.name][:, :w - o]), (name, kind, c.name)
