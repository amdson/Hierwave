"""cfast cone returns: every draw or cost that changes when one draw is
replaced lies inside that draw's cone, the cone sums equal brute-force sums,
and the cfast training pass draws exactly what cpass draws."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_cone import Cone
from castlegen.variants._cfast_core import train_pass


def _setup(name, seed=0, width=None):
    prob = problems.get(name)
    W = width or prob.W_train
    lay = core.Layout(prob, W, True)
    key = jax.random.PRNGKey(seed)
    params = {}
    for l in range(prob.T + 1):
        for c in lay.chans[l]:
            key, k1, k2 = jax.random.split(key, 3)
            p = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), c.D, 16)
            p["W3"] = jax.random.normal(k2, p["W3"].shape)          # make the bias read its window
            params[core.key_of(l, c.name)] = p
    des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(seed, Region(W)).items()}
    return prob, lay, params, des


def _replays(lay, params, des, ovs, seed=3):
    lam = jnp.full(lay.T + 1, 5.0, jnp.float32)
    f = jax.jit(jax.vmap(lambda o: train_pass(lay, params, seed, des, 1.0, lam, override=o)[1]))
    o = tuple(jnp.asarray(x) for x in zip(*ovs))
    return f(o)


CASES = [("toy_support", 64, 24), ("toy_modes", 64, 24), ("dimers", None, 16), ("columns", 64, 16),
         ("ground", 64, 12)]


@pytest.mark.parametrize("name,width,nrep", CASES)
def test_changes_inside_cone(name, width, nrep):
    prob, lay, params, des = _setup(name, width=width)
    cone = Cone(lay, core.ACTOR_WIN)
    rng = np.random.default_rng(1)
    keys = list(cone.boxes)
    draws, ovs = [], [(0, 0, 0, False)]
    for _ in range(nrep):
        key = keys[rng.integers(len(keys))]
        l = int(key[1:key.index("_")])
        cname = key[key.index("_") + 1:]
        ci = [c.name for c in lay.chans[l]].index(cname)
        while True:
            k, j = rng.integers(core.NCOL), rng.integers(lay.nsite(l))
            if bool(np.asarray(lay.sites(l, k)[2])[j]):
                break
        D = prob.chan[cname].D
        draws.append((key, k, j))
        ovs.append((lay.step(l, k, ci), j, int(rng.integers(D)), True))
    recs = _replays(lay, params, des, ovs)
    nchanged = 0
    for r, (key, k, j) in enumerate(draws, 1):
        mem = cone.member(key, k, j)
        for key2, rec in recs.items():
            a0, a1 = np.asarray(rec["a"][0]), np.asarray(rec["a"][r])
            c0, c1 = np.asarray(rec["cost"][0]), np.asarray(rec["cost"][r])
            ch = ((a0 != a1) | ~np.isclose(c0, c1, rtol=1e-5, atol=1e-5)) & np.asarray(rec["valid"][0])
            if key2 == key:
                ch[k, j] = False                                     # the replaced draw itself
            nchanged += int(ch.sum())
            bad = ch & ~mem[key2]
            assert not bad.any(), (name, key, k, j, key2, np.argwhere(bad)[:5])
    assert nchanged > 0


@pytest.mark.parametrize("name,width", [("toy_support", 64), ("dimers", None), ("ground", 64)])
def test_cone_sums_match_brute_force(name, width):
    prob, lay, params, des = _setup(name, width=width)
    cone = Cone(lay, core.ACTOR_WIN)
    recs = _replays(lay, params, des, [(0, 0, 0, False)])
    costs = {k: v["cost"] for k, v in recs.items()}
    G = cone.returns(costs)
    rng = np.random.default_rng(2)
    for key in cone.boxes:
        for _ in range(4):
            l = int(key[1:key.index("_")])
            k, j = rng.integers(core.NCOL), rng.integers(lay.nsite(l))
            mem = cone.member(key, k, j)
            brute = sum(float(np.asarray(costs[k2][0], np.float64)[m].sum()) for k2, m in mem.items())
            assert np.isclose(float(G[key][0, k, j]), brute, rtol=1e-4, atol=1e-2), (key, k, j)


@pytest.mark.parametrize("name", ["toy_support", "columns"])
def test_train_pass_matches_cpass(name):
    prob, lay, params, des = _setup(name)
    lam = jnp.full(lay.T + 1, 5.0, jnp.float32)
    st0 = core.coloured_pass(lay, params, 7, des, 1.0, lam)[0]
    st1, recs = train_pass(lay, params, 7, des, 1.0, lam)
    for c in prob.sampled():
        assert np.array_equal(np.asarray(st0[c.name]), np.asarray(st1[c.name]))
    tot = core.coloured_pass(lay, params, 7, des, 1.0, lam)[1]
    assert np.isclose(float(tot.sum()), float(sum(r["cost"].sum() for r in recs.values())), rtol=1e-4)
