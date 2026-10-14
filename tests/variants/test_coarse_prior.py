"""_coarse_prior: generic restricted-energy Gibbs for target-like coarse states."""
import numpy as np
import pytest

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _coarse_prior as cpr

CASES = [("ground", 4), ("ground", 3), ("columns", 3), ("columns", 1), ("toy_support", 1),
         ("toy_modes", 1)]


@pytest.mark.parametrize("name,L", CASES)
@pytest.mark.parametrize("init", ["min", "max", "mix"])
def test_hard_coarse_factors_satisfied(name, L, init):
    p = get(name)
    reg = Region(p.W_train)
    x = cpr.sample_coarse_batch(p, reg, L, np.arange(3), sweeps=3, init=init)
    assert (cpr.hard_violations(p, x, reg, L) == 0).all()
    for c in p.chans:
        if c.level >= L:
            assert x[c.name].shape == (3, p.H >> c.level, reg.W >> c.level)
            assert x[c.name].min() >= 0 and x[c.name].max() < c.D
        else:
            assert c.name not in x


def test_repairs_an_invalid_start():
    p = get("columns")
    reg = Region(p.W_train)
    rng = np.random.default_rng(0)
    start = {c.name: rng.integers(0, c.D, (2, p.H >> c.level, reg.W >> c.level))
             for c in p.chans if c.level >= 2}
    assert (cpr.hard_violations(p, start, reg, 2) > 0).all()
    x = cpr.sample_coarse_batch(p, reg, 2, [5, 6], sweeps=2, start=start)
    assert (cpr.hard_violations(p, x, reg, 2) == 0).all()


@pytest.mark.parametrize("name,L", [("ground", 3), ("columns", 2), ("toy_support", 1)])
def test_deterministic_per_seed_and_batch_independent(name, L):
    p = get(name)
    reg = Region(p.W_train)
    a = cpr.sample_coarse_batch(p, reg, L, [11, 12], sweeps=2, init="min")
    b = cpr.sample_coarse_batch(p, reg, L, [12], sweeps=2, init="min")
    c = cpr.sample_coarse_batch(p, reg, L, [11, 12], sweeps=2, init="min")
    for k in a:
        assert np.array_equal(a[k], c[k])
        assert np.array_equal(a[k][1], b[k][0])
    assert any(not np.array_equal(a[k][0], a[k][1]) for k in a if p.chan[k].designed is None)


def test_chain_move_exact_on_toy_support():
    """Restricted to levels >= 1 the only factor is rho2 = sum rho1 (hard): the
    target is uniform over rho1 in 0..4 per cell; one chain move is an exact draw."""
    p = get("toy_support")
    reg = Region(p.W_train)
    x = cpr.sample_coarse_batch(p, reg, 1, np.arange(64), sweeps=1, init="min")
    freq = np.bincount(x["rho1"].ravel(), minlength=5) / x["rho1"].size
    assert np.abs(freq - 0.2).max() < 0.02


def test_clamp_interface():
    p = get("ground")
    cl = cpr.coarse_clamps(p, [1, 2, 3], L=3, sweeps=2)
    assert len(cl) == 3
    for d in cl:
        assert set(d) == {"rho4", "cert4", "rho3", "cert3"}
        assert d["rho3"].dtype == np.int32 and d["rho3"].shape == (8, 16)
    bat = cpr.coarse_clamp_batch(p, [1, 2, 3], L=3, sweeps=2)
    assert np.array_equal(bat["cert3"][1], cl[1]["cert3"])


def test_contexts_mixed_depth():
    p = get("ground")
    seeds, Ls = [1, 2, 3, 4], [3, 4, 5, 3]
    ctx = cpr.coarse_contexts(p, seeds, Ls, sweeps=2)
    assert set(ctx) == {c.name for c in p.sampled() if c.level >= 1}
    assert (ctx["rho3"][1] == 0).all() and (ctx["rho4"][2] == 0).all()
    one = cpr.coarse_clamp_batch(p, [1, 4], L=3, sweeps=2)
    assert np.array_equal(ctx["cert3"][[0, 3]], one["cert3"])
