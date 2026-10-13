"""cmulti: P = 1 is cfast's pass; the per-draw costs telescope to the bound's
integrand; every change caused by replacing one draw lies in its cone; open
windows agree on their overlap (finite support with the P-pass margin)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass
from castlegen.variants._cmulti_core import MCone, Sched, multi_pass


def _setup(name, P, seed=0, width=None):
    prob = problems.get(name)
    W = width or prob.W_train
    lay = core.Layout(prob, W, True)
    sch = Sched(lay, P)
    key = jax.random.PRNGKey(seed)
    params, rparams = {}, {}
    for k, (l, ci, cn, p, kind) in sch.groups.items():
        key, k1, k2 = jax.random.split(key, 3)
        m = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), prob.chan[cn].D, 16)
        m["W3"] = jax.random.normal(k2, m["W3"].shape)
        (params if kind == "draw" else rparams)[k] = m
    des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(seed, Region(W)).items()}
    return prob, lay, sch, params, rparams, des


@pytest.mark.parametrize("name", ["toy_support", "columns"])
def test_p1_is_cfast(name):
    prob, lay, sch, params, rparams, des = _setup(name, 1, width=64)
    lam = jnp.full(lay.T + 1, 5.0, jnp.float32)
    st0, rec0 = train_pass(lay, params, 7, des, 1.0, lam)
    st1, rec1 = multi_pass(sch, params, rparams, 7, des, 1.0, lam)
    for c in prob.sampled():
        assert np.array_equal(np.asarray(st0[c.name]), np.asarray(st1[c.name]))
    for k in rec0:
        assert np.allclose(np.asarray(rec0[k]["cost"]), np.asarray(rec1[k]["cost"]), atol=1e-5)


@pytest.mark.parametrize("name,P", [("toy_support", 2), ("dimers", 3), ("columns", 2)])
def test_telescoping(name, P):
    prob, lay, sch, params, rparams, des = _setup(name, P, width=64)
    region = Region(lay.W)
    lam0 = jnp.zeros(lay.T + 1, jnp.float32)
    st, recs = multi_pass(sch, params, rparams, 11, des, 1.0, lam0)
    x = {**{k: np.asarray(v) for k, v in des.items()}, **{c.name: np.asarray(st[c.name]) for c in prob.sampled()}}
    E = en.energy(prob, x, region)
    tot = float(sum(np.asarray(r["cost"], np.float64).sum() for r in recs.values()))
    nv = int(sum(np.asarray(recs[k]["dv"]).sum() for k in sch.draw_keys()))
    assert nv == int(sum(E.viol.values())), (nv, E.viol)
    lq = sum(float(np.where(np.asarray(recs[k]["valid"]), np.take_along_axis(
        np.asarray(recs[k]["logq"]), np.asarray(recs[k]["a"])[..., None], -1)[..., 0], 0).sum())
        for k in sch.draw_keys())
    rt = sum(float(np.asarray(recs[k]["cost"]).sum()) for k in sch.r_keys())
    assert rt > 0
    assert np.isclose(tot - lq - rt, E.soft, rtol=1e-4, atol=1e-2), (tot - lq - rt, E.soft)


CASES = [("toy_support", 2, 64, 20), ("dimers", 2, None, 16), ("columns", 2, 64, 16), ("toy_modes", 3, 64, 12)]


@pytest.mark.parametrize("name,P,width,nrep", CASES)
def test_changes_inside_cone(name, P, width, nrep):
    prob, lay, sch, params, rparams, des = _setup(name, P, width=width)
    cone = MCone(sch, core.ACTOR_WIN)
    lam = jnp.full(lay.T + 1, 5.0, jnp.float32)
    rng = np.random.default_rng(1)
    dk = sch.draw_keys()
    draws, ovs = [], [(0, 0, 0, False)]
    for _ in range(nrep):
        key = dk[rng.integers(len(dk))]
        l, ci, cn, p, _ = sch.groups[key]
        while True:
            k, j = rng.integers(core.NCOL), rng.integers(lay.nsite(l))
            if bool(np.asarray(lay.sites(l, k)[2])[j]):
                break
        draws.append((key, k, j))
        ovs.append((sch.step(key, k), j, int(rng.integers(prob.chan[cn].D)), True))
    o = tuple(jnp.asarray(x) for x in zip(*ovs))
    f = jax.jit(jax.vmap(lambda ov: multi_pass(sch, params, rparams, 3, des, 1.0, lam, override=ov)[1]))
    recs = f(o)
    nchanged = 0
    for r, (key, k, j) in enumerate(draws, 1):
        mem = cone.member(key, k, j)
        for key2, rec in recs.items():
            c0, c1 = np.asarray(rec["cost"][0]), np.asarray(rec["cost"][r])
            ch = ~np.isclose(c0, c1, rtol=1e-5, atol=1e-5)
            if "a" in rec:
                ch |= np.asarray(rec["a"][0]) != np.asarray(rec["a"][r])
            ch &= np.asarray(rec["valid"][0])
            if key2 == key:
                ch[k, j] = False
            nchanged += int(ch.sum())
            bad = ch & ~mem[key2]
            assert not bad.any(), (name, key, k, j, key2, np.argwhere(bad)[:5])
    assert nchanged > 0


def test_cone_sums_match_brute_force():
    prob, lay, sch, params, rparams, des = _setup("toy_support", 2, width=64)
    cone = MCone(sch, core.ACTOR_WIN)
    lam = jnp.full(lay.T + 1, 5.0, jnp.float32)
    _, recs = multi_pass(sch, params, rparams, 3, des, 1.0, lam)
    costs = {k: v["cost"][None] for k, v in recs.items()}
    G = cone.returns(costs)
    rng = np.random.default_rng(2)
    for key in sch.draw_keys():
        l = sch.groups[key][0]
        for _ in range(4):
            k, j = rng.integers(core.NCOL), rng.integers(lay.nsite(l))
            mem = cone.member(key, k, j)
            brute = sum(float(np.asarray(costs[k2][0], np.float64)[m].sum()) for k2, m in mem.items())
            assert np.isclose(float(G[key][0, k, j]), brute, rtol=1e-4, atol=1e-2), (key, k, j)


def test_finite_support():
    from castlegen.variants.cmulti import CMulti
    prob = problems.get("toy_support")
    v = CMulti(prob, 0, "/tmp", P=2)
    for p in list(v.params.values()) + list(v.rparams.values()):
        p["W3"] = jax.random.normal(jax.random.PRNGKey(5), p["W3"].shape)
    W = prob.W_eval
    half = W // 2 >> prob.T << prob.T
    ga = v.generate(17, Region(W, periodic=False, x0=0))
    gb = v.generate(17, Region(W, periodic=False, x0=half))
    for kind in ("raw", "final"):
        for c in prob.sampled():
            s = half >> c.level
            a = np.asarray(ga[kind][c.name])[:, s:]
            b = np.asarray(gb[kind][c.name])[:, :a.shape[1]]
            assert np.array_equal(a, b), (kind, c.name)
