"""cconv: the patch trunk (loss) equals the dense trunk (pass); the loss
recomputes the pass's log q; every draw or cost changed by replacing one draw
lies in that draw's cone (conv window); scoring matches the pass; the open-
window generation agrees on overlaps (finite support)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _cconv_core as cc
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_cone import Cone

R = 3


def _setup(name, seed=0, width=None, periodic=True, x0=0):
    prob = problems.get(name)
    W = width or prob.W_train
    lay = core.Layout(prob, W, periodic, x0)
    net = cc.ConvNet(lay, R, 16, 8)
    params = net.init(jax.random.PRNGKey(seed))
    key = jax.random.PRNGKey(seed + 1)
    for k in list(params):                              # nonzero heads: the bias reads the trunk
        if k.endswith(cc.TRUNK):
            continue
        hp = dict(params[k])
        for f in ("W", "w"):
            key, k1 = jax.random.split(key)
            hp[f] = jax.random.normal(k1, hp[f].shape)
        params[k] = hp
    reg = Region(W, periodic=periodic, x0=x0)
    des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(seed, reg).items()}
    return prob, lay, net, params, des


def _lam(lay):
    return jnp.full(lay.T + 1, 5.0, jnp.float32)


@pytest.mark.parametrize("name,periodic", [("toy_support", True), ("ground", True), ("ground", False),
                                           ("dimers", True)])
def test_patch_equals_dense(name, periodic):
    prob, lay, net, params, des = _setup(name, width=64 if name == "ground" else None, periodic=periodic,
                                         x0=-64 if not periodic else 0)
    st, _ = cc.conv_pass(lay, net, params, 3, des, 1.0, _lam(lay))
    steps = {}
    for l in range(prob.T + 1):
        steps.update(net.set_step(l))
    rng = np.random.default_rng(0)
    for l in lay.levels:
        for ci, c in enumerate(lay.chans[l]):
            for k in rng.choice(core.NCOL, 3, replace=False):
                s = lay.step(l, int(k), ci)
                before = {n: (jnp.where(steps[n] < s, v, -1) if n in steps else v) for n, v in st.items()}
                Y, X, valid = lay.sites(l, int(k))
                h, w = lay.hw[l]
                fd = net.dense_feat(params, before, l)[jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)]
                fp = net.patch_feat(params, st, l, Y, X, s, steps)
                v = np.asarray(valid)
                assert np.allclose(np.asarray(fd)[v], np.asarray(fp)[v], atol=2e-4), (l, c.name, k)


@pytest.mark.parametrize("name", ["toy_support", "ground"])
def test_loss_recomputes_pass_logq(name):
    prob, lay, net, params, des = _setup(name, width=64 if name == "ground" else None)
    st, recs = cc.conv_pass(lay, net, params, 5, des, 1.0, _lam(lay))
    steps = {}
    for l in range(prob.T + 1):
        steps.update(net.set_step(l))
    for l in lay.levels:
        for ci, c in enumerate(lay.chans[l]):
            r = recs[core.key_of(l, c.name)]
            for k in (0, 7, 15):
                Y, X, valid = lay.sites(l, k)
                f = net.patch_feat(params, st, l, Y, X, lay.step(l, k, ci), steps)
                b = net.head(params[core.key_of(l, c.name)], f, lay.emb[c.name], r["soft"][k], r["nv"][k],
                             r["look"][k])
                lq = jax.nn.log_softmax(jnp.where(r["mask"][k], -r["soft"][k] - b, -jnp.inf), -1)
                v = np.asarray(valid)
                a = np.asarray(r["a"][k])
                got = np.take_along_axis(np.asarray(lq), a[:, None], 1)[:, 0]
                want = np.take_along_axis(np.asarray(r["logq"][k]), a[:, None], 1)[:, 0]
                assert np.allclose(got[v], want[v], atol=1e-3), (l, c.name, k)


def test_score_matches_pass():
    prob, lay, net, params, des = _setup("toy_support")
    st, recs = cc.conv_pass(lay, net, params, 5, des, 1.0, _lam(lay))
    want = sum(float(jnp.sum(jnp.where(r["valid"], jnp.take_along_axis(r["logq"], r["a"][..., None], -1)[..., 0],
                                       0.0))) for r in recs.values())
    forced = {c.name: st[c.name] for c in prob.sampled()}
    _, got = cc.conv_pass(lay, net, params, 0, des, 1.0, _lam(lay), forced=forced)
    assert np.isclose(float(got), want, rtol=1e-4, atol=1e-3)


CASES = [("toy_support", 64, 24), ("toy_modes", 64, 16), ("dimers", None, 16), ("columns", 64, 12),
         ("ground", 64, 10)]


@pytest.mark.parametrize("name,width,nrep", CASES)
def test_changes_inside_cone(name, width, nrep):
    prob, lay, net, params, des = _setup(name, width=width)
    cone = Cone(lay, cc.conv_window(R))
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
    f = jax.jit(jax.vmap(lambda o: cc.conv_pass(lay, net, params, 3, des, 1.0, _lam(lay), override=o)[1]))
    recs = f(tuple(jnp.asarray(x) for x in zip(*ovs)))
    nchanged = 0
    for r, (key, k, j) in enumerate(draws, 1):
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


def test_conv_window_covers_trunk_reach():
    """The window holds every level-l cell within R of the site and its parent."""
    win = cc.conv_window(R)
    same, par = set(win.same), set(win.parent)
    for Y in range(8, 10):
        for X in range(8, 10):
            by, bx = 2 * (Y >> 1), 2 * (X >> 1)
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    assert (Y + dy - by, X + dx - bx) in same
                    assert (((Y + dy) >> 1) - (Y >> 1), ((X + dx) >> 1) - (X >> 1)) in par


@pytest.mark.parametrize("name", ["toy_support", "ground"])
def test_open_windows_agree(name):
    """Two overlapping open windows from the same seed agree on the overlap (the
    bench's finite-support check, through the variant with the conv margin)."""
    from castlegen.variants.cconv import CConv
    prob = problems.get(name)
    v = CConv(prob, 0, None, r=R, cw=16, hc=8, B=2)
    key = jax.random.PRNGKey(9)
    for k in list(v.params):                             # an untrained but non-trivial actor
        if k.endswith(cc.TRUNK):
            continue
        hp = dict(v.params[k])
        key, k1 = jax.random.split(key)
        hp["W"] = jax.random.normal(k1, hp["W"].shape)
        v.params[k] = hp
    u = 1 << (prob.T + 2)
    a = v.generate(77, Region(4 * u, periodic=False, x0=0))
    b = v.generate(77, Region(4 * u, periodic=False, x0=2 * u))
    for kind in ("raw", "final"):
        for c in prob.sampled():
            o = (2 * u) >> c.level
            assert np.array_equal(a[kind][c.name][:, o:], b[kind][c.name][:, :o]), (kind, c.name)
