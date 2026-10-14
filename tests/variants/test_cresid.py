"""cresid: count-residual features.  Values match a direct computation, the
cfast cone still contains every change (the features read only the parent and
the 2 x 2 block, inside ACTOR_WIN), and overlapping open windows agree
(finite support) with a bias that reads its features."""
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


def _params(lay, prob, seed=0):
    key = jax.random.PRNGKey(seed)
    params = {}
    for l in range(prob.T + 1):
        for c in lay.chans[l]:
            key, k1, k2 = jax.random.split(key, 3)
            p = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), c.D, 16)
            p["W3"] = jax.random.normal(k2, p["W3"].shape)
            params[core.key_of(l, c.name)] = p
    return params


def test_off_by_default():
    prob = problems.get("ground")
    a, b = core.Layout(prob, 64, True), core.Layout(prob, 64, True, resid=True)
    for l in range(prob.T):
        n = len(b.counts[l])
        assert n > 0
        assert b.feat_dim(l, core.ACTOR_WIN) == a.feat_dim(l, core.ACTOR_WIN) + 6 * n


def test_values():
    prob = problems.get("ground")
    W = 64
    lay = core.Layout(prob, W, True, resid=True)
    rng = np.random.default_rng(0)
    from castlegen.bench.problems.ground import lift
    t = (np.arange(64)[:, None] >= rng.integers(10, 50, size=W)[None, :]).astype(np.int64)
    full = lift(t)
    st = {c.name: jnp.asarray(full.get(c.name, np.zeros(prob.shape(c.name, Region(W)), np.int64)), jnp.int32)
          for c in prob.chans}
    # level 2: hide rho2 at children k in {1, 3} of every block (the "unknown" ones)
    l = 2
    r2 = np.array(full["rho2"])
    h, w = r2.shape
    hid = np.zeros_like(r2, bool)
    hid[:, 1::2] = True
    st["rho2"] = jnp.asarray(np.where(hid, -1, r2), jnp.int32)
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    Y, X = jnp.asarray(Y.ravel()), jnp.asarray(X.ravel())
    f = np.asarray(lay._resid_feats(st, l, Y, X))
    i = [j for j, e in enumerate(lay.counts[l]) if e[0] == "rho2"][0]
    g = f[:, 6 * i:6 * i + 6]
    Yn, Xn = np.asarray(Y), np.asarray(X)
    par = full["rho3"][Yn >> 1, Xn >> 1]
    known_sum = r2[2 * (Yn >> 1), 2 * (Xn >> 1)] + r2[2 * (Yn >> 1) + 1, 2 * (Xn >> 1)]
    assert np.allclose(g[:, 0], 1) and np.allclose(g[:, 1], 0)
    assert np.allclose(g[:, 2], (par - known_sum) / 64)
    assert np.allclose(g[:, 3], 0.5)
    assert np.allclose(g[:, 4], np.clip((par - known_sum) / 2 / 16, -2, 2))
    assert np.allclose(g[:, 5], par / 64)


@pytest.mark.parametrize("name,width,nrep", [("ground", 64, 12), ("toy_modes", 64, 16), ("columns", 64, 12)])
def test_changes_inside_cone(name, width, nrep):
    prob = problems.get(name)
    lay = core.Layout(prob, width, True, resid=True)
    params = _params(lay, prob)
    des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(0, Region(width)).items()}
    cone = Cone(lay, core.ACTOR_WIN)
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


@pytest.mark.parametrize("name", ["toy_modes", "ground"])
def test_finite_support(name):
    from castlegen.bench.variant import load_variants
    prob = problems.get(name)
    v = load_variants()["cresid"](prob, 0, "/tmp")
    lay = v.layout(Region(prob.W_train))
    v.params = _params(lay, prob, 5)
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
