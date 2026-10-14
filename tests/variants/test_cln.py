"""cln / cln6: selectable actor network (LayerNorm + SiLU) and actor window
(6 x 6).  Defaults are unchanged; the cone of the 6 x 6 window contains every
change; overlapping open windows agree (finite support, margin from the
window); the training pass draws what cpass's pass draws."""
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
from castlegen.variants._cln_core import ln_mlp_init, win_reach, window

W6 = window(-2, 3)


def _params(lay, prob, net="ln", seed=0):
    key = jax.random.PRNGKey(seed)
    params = {}
    for l in range(prob.T + 1):
        for c in lay.chans[l]:
            key, k1, k2 = jax.random.split(key, 3)
            init = ln_mlp_init if net == "ln" else core.mlp_init
            p = init(k1, lay.feat_dim(l, lay.actor_win), c.D, 16)
            p["W3"] = jax.random.normal(k2, p["W3"].shape)          # make the bias read its window
            params[core.key_of(l, c.name)] = p
    return params


def test_defaults_unchanged():
    from castlegen.bench.variant import load_variants
    prob = problems.get("toy_modes")
    vs = load_variants()
    a, b = vs["cfast"](prob, 0, "/tmp"), vs["cln6"](prob, 0, "/tmp")
    la, lb = a.layout(Region(prob.W_train)), b.layout(Region(prob.W_train))
    assert la.actor_win == core.ACTOR_WIN and lb.actor_win == W6
    assert core.margin(prob) == core.margin(prob, core.ACTOR_WIN)
    assert win_reach(core.ACTOR_WIN) <= 3 and win_reach(W6) == 3
    k = core.key_of(0, la.chans[0][0].name)
    assert set(a.params[k]) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert "g1" in b.params[k]
    assert b.params[k]["W1"].shape[0] > a.params[k]["W1"].shape[0]
    # same seed, same tanh init as before (mlp_init with the same key)
    key = jax.random.PRNGKey(0)
    key, k1, _ = jax.random.split(key, 3)
    l0 = 0
    k0 = core.key_of(l0, la.chans[l0][0].name)
    ref = core.mlp_init(k1, la.feat_dim(l0, core.ACTOR_WIN), la.chans[l0][0].D, 64)
    assert all(np.array_equal(np.asarray(ref[n]), np.asarray(a.params[k0][n])) for n in ref)


@pytest.mark.parametrize("name,width,nrep", [("toy_modes", 64, 16), ("toy_support", 64, 16),
                                             ("ground", 64, 12), ("columns", 64, 10)])
def test_changes_inside_cone_w6(name, width, nrep):
    prob = problems.get(name)
    lay = core.Layout(prob, width, True, win=W6)
    params = _params(lay, prob)
    des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(0, Region(width)).items()}
    cone = Cone(lay, lay.actor_win)
    rng = np.random.default_rng(1)
    keys = list(cone.boxes)
    ovs, info = [(0, 0, 0, False)], []
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
        ovs.append((lay.step(l, k, ci), int(j), int(rng.integers(D)), True))
        info.append((key, k, j))
    lam = jnp.full(lay.T + 1, 5.0, jnp.float32)
    f = jax.jit(jax.vmap(lambda o: train_pass(lay, params, 3, des, 1.0, lam, override=o)[1]))
    o = tuple(jnp.asarray(x) for x in zip(*ovs))
    recs = jax.device_get(f(o))
    nchanged = 0
    for r, (key, k, j) in enumerate(info, 1):
        mem = cone.member(key, k, j)
        for key2, rec in recs.items():
            a0, a1 = np.asarray(rec["a"][0]), np.asarray(rec["a"][r])
            c0, c1 = np.asarray(rec["cost"][0]), np.asarray(rec["cost"][r])
            ch = ((a0 != a1) | ~np.isclose(c0, c1, rtol=1e-5, atol=1e-5)) & np.asarray(rec["valid"][0])
            if key2 == key:
                ch[k, j] = False
            nchanged += int(ch.sum())
            bad = ch & ~mem[key2]
            assert not bad.any(), (name, key, k, j, key2, np.argwhere(bad)[:5])
    assert nchanged > 0


def test_cone_w6_wider_than_w4():
    prob = problems.get("toy_modes")
    a = Cone(core.Layout(prob, 64, True), core.ACTOR_WIN)
    b = Cone(core.Layout(prob, 64, True, win=W6), W6)
    assert all(rb[0] >= ra[0] and rb[1] >= ra[1] for ra, rb in zip(a.reach, b.reach))
    assert any(rb != ra for ra, rb in zip(a.reach, b.reach))


@pytest.mark.parametrize("name", ["toy_support", "columns"])
def test_train_pass_matches_cpass_w6(name):
    prob = problems.get(name)
    lay = core.Layout(prob, prob.W_train, True, win=W6)
    params = _params(lay, prob)
    des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(0, Region(prob.W_train)).items()}
    lam = jnp.full(lay.T + 1, 5.0, jnp.float32)
    st0, tot, _, _ = core.coloured_pass(lay, params, 7, des, 1.0, lam)
    st1, recs = train_pass(lay, params, 7, des, 1.0, lam)
    for c in prob.sampled():
        assert np.array_equal(np.asarray(st0[c.name]), np.asarray(st1[c.name]))
    assert np.isclose(float(tot.sum()), float(sum(r["cost"].sum() for r in recs.values())), rtol=1e-4)


@pytest.mark.parametrize("vname,name", [("cln6", "toy_modes"), ("cln6", "ground"), ("cln", "toy_support")])
def test_finite_support(vname, name):
    from castlegen.bench.variant import load_variants
    prob = problems.get(name)
    v = load_variants()[vname](prob, 0, "/tmp")
    lay = v.layout(Region(prob.W_train))
    v.params = _params(lay, prob, "ln", 5)
    W = 128
    half = 64
    ga = v.generate(77, Region(W, periodic=False, x0=0))
    gb = v.generate(77, Region(W, periodic=False, x0=half))
    for kind in ("raw", "final"):
        for c in prob.sampled():
            s = half >> c.level
            a = np.asarray(ga[kind][c.name])[:, s:]
            b = np.asarray(gb[kind][c.name])[:, :a.shape[1]]
            assert (a == b).all(), (kind, c.name)
