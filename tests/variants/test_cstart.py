"""cstart: mixed start depths (ideas #14).

1. Full depth (L = T + 1) is cfast bit for bit: train_pass_ctx draws, costs
   and records equal train_pass's; the fused update with no clamped worlds
   equals cfast's fused update exactly (params, critic, loss).
2. A clamped world reproduces the diagnostic clamped pass
   (_creplay_clamp.clamped_pass): same state at every level, same costs below L.
3. The loss on clamped worlds includes only draws below L: with every world
   clamped at L, an update leaves the actor and critic parameters of levels
   >= L exactly unchanged and moves those below; clamped draws are invalid and
   cost 0 in the records.
4. No hierarchy (dimers, T = 0) or no probes: no clamped worlds."""
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
from castlegen.variants._creplay_clamp import clamped_pass
from castlegen.variants._cstart_core import train_pass_ctx
from castlegen.variants.cfast import CFast
from castlegen.variants.cstart import CStart

W = 64


def _setup(name, cls=CStart, **cfg):
    prob = problems.get(name)
    var = cls(prob, 0, "/tmp/cstart_t", W_train=W, **cfg)
    params = gc.randomize(var.params, jax.random.PRNGKey(3))
    lay = var.layout(Region(W))
    return prob, var, params, lay


@pytest.mark.parametrize("name", ["toy_support", "ground"])
def test_full_depth_pass_is_cfast(name):
    prob, var, params, lay = _setup(name)
    des = {k: v[0] for k, v in var._batch(np.array([7]), Region(W)).items()}
    lam = jnp.full(prob.T + 1, 4.0, jnp.float32)
    ctx, Lj = var.contexts(np.array([7]), Region(W), np.array([prob.T + 1]))
    ctx = {k: v[0] for k, v in ctx.items()}
    st0, r0 = train_pass(lay, params, 7, des, 1.3, lam)
    st1, r1 = train_pass_ctx(lay, params, 7, des, 1.3, lam, ctx, Lj[0])
    for c in prob.sampled():
        assert np.array_equal(np.asarray(st0[c.name]), np.asarray(st1[c.name]))
    for k in r0:
        for f in r0[k]:
            assert np.array_equal(np.asarray(r0[k][f]), np.asarray(r1[k][f])), (k, f)
        assert not bool(np.asarray(r1[k]["clamped"]).any())


@pytest.mark.parametrize("name", ["toy_support", "ground"])
def test_off_update_is_cfast(name):
    prob, var, params, lay = _setup(name, frac=0.0)
    a = CFast(prob, 0, "/tmp/cstart_t", W_train=W)
    fa, fb = a.train_fns(lay), var.train_fns(lay)
    seeds_np = np.arange(5)
    Ls = var.draw_L(5)
    assert (Ls == prob.T + 1).all()
    ctx, Lj = var.contexts(seeds_np, Region(W), Ls)
    seeds = jnp.asarray(seeds_np, jnp.int32)
    des = a._batch(seeds_np, Region(W))
    tau, lam = jnp.float32(1.2), jnp.asarray(a.lam, jnp.float32)
    gs = {k: jnp.float32(2.0) for k in params}
    oa = fa["fused"](params, a.cparams, a.opt, a.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
                     jax.random.PRNGKey(9))
    ob = fb["fused_ctx"](params, a.cparams, a.opt, a.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
                         jax.random.PRNGKey(9), ctx, Lj)
    for i in (1, 2, 3, 4, 7, 8):                       # params, cparams, opt, copt, viol, tot
        for x, y in zip(jax.tree_util.tree_leaves(oa[i]), jax.tree_util.tree_leaves(ob[i])):
            assert np.array_equal(np.asarray(x), np.asarray(y)), i
    assert float(oa[5]) == float(ob[5]) and float(oa[6]) == float(ob[6])


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_clamped_world_matches_clamped_pass(L):
    prob, var, params, lay = _setup("ground")
    des = {k: v[0] for k, v in var._batch(np.array([11]), Region(W)).items()}
    lam = jnp.full(prob.T + 1, 4.0, jnp.float32)
    ctx, Lj = var.contexts(np.array([11]), Region(W), np.array([L]))
    ctx = {k: v[0] for k, v in ctx.items()}
    clamp = {c.name: ctx[c.name] for c in prob.sampled() if c.level >= L}
    st0, costs0, _ = clamped_pass(lay, params, 11, des, 1.0, lam, clamp)
    st1, r1 = train_pass_ctx(lay, params, 11, des, 1.0, lam, ctx, Lj[0])
    for c in prob.sampled():
        assert np.array_equal(np.asarray(st0[c.name]), np.asarray(st1[c.name])), c.name
        if c.level >= L:
            assert np.array_equal(np.asarray(st1[c.name]), np.asarray(ctx[c.name]))
    for l in range(prob.T + 1):
        for c in lay.chans[l]:
            k = core.key_of(l, c.name)
            if l < L:
                assert np.allclose(np.asarray(costs0[k]), np.asarray(r1[k]["cost"]), rtol=1e-6, atol=1e-5)
                assert not bool(np.asarray(r1[k]["clamped"]).any())
            else:
                assert bool(np.asarray(r1[k]["clamped"]).all())
                assert not bool(np.asarray(r1[k]["valid"]).any())
                assert float(jnp.abs(r1[k]["cost"]).max()) == 0.0


@pytest.mark.parametrize("L", [1, 3])
def test_loss_only_below_L(L):
    prob, var, params, lay = _setup("ground")
    fb = var.train_fns(lay)
    seeds_np = np.arange(4)
    ctx, Lj = var.contexts(seeds_np, Region(W), np.full(4, L))
    des = var._batch(seeds_np, Region(W))
    tau, lam = jnp.float32(1.2), jnp.asarray(var.lam, jnp.float32)
    gs = {k: jnp.float32(2.0) for k in params}
    cp = gc.randomize(var.cparams, jax.random.PRNGKey(5))
    out = fb["fused_ctx"](params, cp, var.opt, var.copt, gs, jnp.asarray(seeds_np, jnp.int32), des, tau,
                          lam, 1e-2, 1e-2, 10.0, jax.random.PRNGKey(2), ctx, Lj)
    for old, new in ((params, out[1]), (cp, out[2])):
        for k in old:
            l = int(k[1:k.index("_")])
            same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
                       zip(jax.tree_util.tree_leaves(old[k]), jax.tree_util.tree_leaves(new[k])))
            assert same == (l >= L), (k, L)
    assert np.isfinite(float(out[5])) and np.isfinite(float(out[6]))


def test_no_hierarchy_no_clamp():
    prob = problems.get("dimers")
    var = CStart(prob, 0, "/tmp/cstart_t")
    assert not var.can_clamp
    assert (var.draw_L(8) == prob.T + 1).all()
