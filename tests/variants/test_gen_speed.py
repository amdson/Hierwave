"""speed-3: the fast generation pass (hoisted geometry and noise, encoded context images) equals the old one
(kept in reportables/2026-10-14_speed3/_cfast_core_old.py and _cpass_core_old.py): same worlds on fixed seeds,
same records, totals, log q, forced scores and overrides; periodic and strip layouts."""
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cfast_core as new_fc
from castlegen.variants import _cpass_core as core

HERE = os.path.join(os.path.dirname(__file__), "..", "..", "reportables", "2026-10-14_speed3")


def _load(fname):
    nm = "castlegen.variants._old_" + fname[:-3]
    sp = importlib.util.spec_from_file_location(nm, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(sp)
    m.__package__ = "castlegen.variants"
    sys.modules[nm] = m
    sp.loader.exec_module(m)
    return m


old_fc, old_cp = _load("_cfast_core_old.py"), _load("_cpass_core_old.py")


def _same(a, b):
    """Every leaf bit-identical (draws, masks, features, logits, costs)."""
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _close(a, b):
    """coloured_pass totals / log q are float sums over a step: equal up to summation order (<= ~5e-7 abs
    observed on CPU); integer leaves (the drawn worlds) bit-identical."""
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype.kind in "iub":
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=2e-5)


def _params(p, lay, seed=0):
    key = jax.random.PRNGKey(seed)
    out = {}
    for l in range(p.T + 1):
        for c in lay.chans[l]:
            key, k2, k3 = jax.random.split(key, 3)
            pr = core.mlp_init(k2, lay.feat_dim(l, lay.actor_win), c.D, 16)
            pr["W3"] = jax.random.normal(k3, pr["W3"].shape) * 0.3
            out[core.key_of(l, c.name)] = pr
    return out


def _des(p, seeds, region):
    ds = [p.designed_values(int(s), region) for s in seeds]
    return {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}


CASES = [("toy_support", 16, True, 0), ("toy_support", 24, False, 8), ("toy_modes", 16, True, 0),
         ("dimers", 16, True, 0), ("ground", 64, True, 0), ("ground", 48, False, 16)]


@pytest.mark.parametrize("name,W,periodic,x0", CASES)
def test_generation_equal(name, W, periodic, x0):
    p = get(name)
    region = Region(W, periodic=periodic, x0=x0)
    lay = core.Layout(p, W, periodic, x0)
    params = _params(p, lay)
    lam = jnp.full(p.T + 1, 5.0)
    seeds = np.arange(7, 10)
    des = _des(p, seeds, region)
    js = jnp.asarray(seeds, jnp.int32)
    tau = jnp.float32(1.3)
    # train_pass: recorded and not, with and without an override
    ov = (jnp.asarray(0), jnp.asarray(0), jnp.asarray(0), jnp.asarray(True))
    for rec in (False, True):
        for o in (None, ov):
            def run(fc):
                f = jax.vmap(lambda s, d: fc.train_pass(lay, params, s, d, tau, lam, override=o, record=rec))
                return jax.device_get(f(js, des))
            a, b = run(old_fc), run(new_fc)
            _same(a, b)
    # coloured_pass (the deployed generator): draws, totals, log q; then score the draws (forced)
    def gen(cp):
        f = jax.vmap(lambda s, d: cp.coloured_pass(lay, params, s, d, tau, lam)[:3])
        return jax.device_get(f(js, des))
    a, b = gen(old_cp), gen(core)
    _close(a, b)
    forced = {c.name: a[0][c.name] for c in p.sampled()}
    def score(cp):
        f = jax.vmap(lambda d, fo: cp.coloured_pass(lay, params, 0, d, tau, lam, forced=fo)[1:3])
        return jax.device_get(f(des, forced))
    a2, b2 = score(old_cp), score(core)
    _close(a2, b2)
    # scoring the draws gives their own log q
    np.testing.assert_allclose(b2[1], a[2], rtol=1e-5, atol=1e-4)
