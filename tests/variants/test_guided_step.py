"""Guided revising step (_guided_step.py) and the jitter-repair baseline (_jitter_repair.py):
exact violation maps, determinism, locality of the step, finite support, repair invariant."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants import _cpass_core as core
from castlegen.variants._guided_step import (HardMaps, guided_margin, guided_revise, guided_step,
                                             select_cells, violation_field)
from castlegen.variants._jitter_repair import jitter_repair


def _params(prob, lay, seed=0):
    key = jax.random.PRNGKey(seed)
    params = {}
    for l in range(prob.T + 1):
        for c in lay.chans[l]:
            key, k1, k2 = jax.random.split(key, 3)
            p = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), c.D, 16)
            p["W3"] = 0.3 * jax.random.normal(k2, p["W3"].shape)        # the bias reads its window
            params[core.key_of(l, c.name)] = p
    return params


def _setup(name="toy_support", W=32):
    prob = problems.get(name)
    lay = core.Layout(prob, W, True)
    params = _params(prob, lay)
    des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(0, Region(W)).items()}
    lam = jnp.zeros(prob.T + 1)
    gen = jax.jit(lambda s: core.coloured_pass(lay, params, s, des, 1.0, lam)[0])
    return prob, lay, params, des, gen


@pytest.mark.parametrize("name,W", [("toy_support", 32), ("ground", 64), ("columns", 64)])
def test_maps_match_energy(name, W):
    prob = problems.get(name)
    lay = core.Layout(prob, W, True)
    hm = HardMaps(lay)
    des = prob.designed_values(0, Region(W))
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = {**des, **{c.name: rng.integers(0, c.D, lay.hw[c.level]) for c in prob.sampled()}}
        E = en.energy(prob, x, Region(W))
        st = {k: jnp.asarray(v, jnp.int32) for k, v in x.items()}
        pl = np.asarray(hm.per_level(st))
        assert [int(pl[l]) for l in range(prob.T + 1)] == [E.viol[l] for l in range(prob.T + 1)]
        f = violation_field(hm, st)
        assert sum(int(np.asarray(v).sum()) for v in f.values()) >= int(pl.sum())
        assert (int(pl.sum()) == 0) == all(not np.asarray(v).any() for v in f.values())


def test_step_is_deterministic_and_local():
    prob, lay, params, des, gen = _setup()
    hm = HardMaps(lay)
    st = gen(3)
    field = violation_field(hm, st)
    sel = select_cells(lay, field, 1, True)
    step = jax.jit(lambda st, s: guided_step(lay, params, st, field, s, 1))
    a, lq_a, n_a = step(st, 3)
    b, lq_b, n_b = step(st, 3)
    c, _, _ = step(st, 4)
    for ch in prob.sampled():
        A, B, C, S0 = (np.asarray(x[ch.name]) for x in (a, b, c, st))
        assert np.array_equal(A, B)
        unsel = ~np.asarray(sel[ch.level])
        assert np.array_equal(A[unsel], S0[unsel]), "unselected cells must not change"
    assert int(n_a) == sum(int((np.asarray(a[ch.name]) != np.asarray(st[ch.name])).sum()) for ch in prob.sampled())
    assert any(not np.array_equal(np.asarray(a[ch.name]), np.asarray(c[ch.name])) for ch in prob.sampled())
    assert np.isfinite(float(lq_a)) and float(lq_a) <= 0.0


def test_revise_from_valid_state_is_identity():
    prob, lay, params, des, gen = _setup()
    hm = HardMaps(lay)
    x = prob.probes(0, Region(32))[0]                        # a valid state: empty field, nothing selected
    st = {**des, **{k: jnp.asarray(v, jnp.int32) for k, v in x.items()}}
    assert int(hm.total(st)) == 0
    out, info = guided_revise(lay, params, st, 5, 2)
    for ch in prob.sampled():
        assert np.array_equal(np.asarray(out[ch.name]), np.asarray(st[ch.name]))
    assert int(np.asarray(info["changed"]).sum()) == 0


def test_finite_support_two_windows():
    """Two overlapping open windows of the same infinite world agree exactly on their overlap
    after the pass + 1 guided step, padded by _cpass_core.margin + guided_margin."""
    prob = problems.get("toy_support")
    W, R = 32, 1
    m = core.margin(prob) + guided_margin(prob, R)
    outs = []
    lays = {}
    for x0 in (0, 16):
        ext = Region(W + 2 * m, periodic=False, x0=x0 - m)
        lay = core.Layout(prob, ext.W, False, ext.x0)
        params = _params(prob, core.Layout(prob, W, True))
        des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(7, ext).items()}
        lam = jnp.zeros(prob.T + 1)
        raw = core.coloured_pass(lay, params, 7, des, 1.0, lam)[0]
        fin, _ = guided_revise(lay, params, raw, 7, R)
        outs.append({c.name: np.asarray(fin[c.name])[:, (m >> c.level):(m >> c.level) + (W >> c.level)]
                     for c in prob.sampled()})
        lays[x0] = lay
    for c in prob.sampled():
        s = 16 >> c.level
        a = outs[0][c.name][:, s:]
        b = outs[1][c.name][:, :a.shape[1]]
        assert np.array_equal(a, b), c.name


def test_jitter_repair_never_increases_violations():
    prob, lay, params, des, gen = _setup()
    hm = HardMaps(lay)
    jr = jax.jit(lambda st, s: jitter_repair(lay, st, s, 2))
    rep = jax.jit(lambda st, s: core.repair(lay, st, s))
    for s in range(4):
        st = gen(10 + s)
        a, b, c = int(hm.total(st)), int(hm.total(rep(st, 10 + s))), int(hm.total(jr(st, 10 + s)))
        assert b <= a and c <= b, (a, b, c)
    out1, out2 = jr(st, 13), jr(st, 13)
    assert all(np.array_equal(np.asarray(out1[k]), np.asarray(out2[k])) for k in out1)
