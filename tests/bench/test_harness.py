"""Harness self-tests: noise, the reference energy against the generic local
conditional (every factor kind), and the greedy variant's contracts."""
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench import noise
from castlegen.bench.problem import Chan, Problem, Region, count, pair, unary
from castlegen.variants._local import cond
from castlegen.variants.greedy import Greedy, coloured_pass

INF = np.inf


def tiny():
    """Every factor kind: unary, same-level pairs with pads and within,
    parent pair by position, counts over subsets, a designed channel."""
    rng = np.random.default_rng(0)
    D0, D1 = 3, 4
    chans = [
        Chan("c1", 1, D1, views={"lo": [0, 0, 1, 1]}),
        Chan("g0", 0, 2, designed=lambda s, Y, X, p: ((X + Y + s) % 3 == 0).astype(int)),
        Chan("t0", 0, D0, views={"on": [0, 1, 1]}),
    ]
    tb = rng.normal(size=(D0, D0))
    tb[2, 0] = INF
    tp = rng.normal(size=(4, D0, 2))
    tp[1, 1, 1] = INF
    tc = rng.normal(size=(2, 3))
    tc[0, 2] = INF
    tg = rng.normal(size=(D0, 2))
    factors = [
        unary(("t0", "id"), rng.normal(size=D0), name="u"),
        pair(("t0", "id"), ("t0", "id"), (1, 0), tb, pad_b=1, pad_a=0, name="vert"),
        pair(("t0", "id"), ("t0", "id"), (0, 1), rng.normal(size=(D0, D0)), within=1, name="horiz"),
        pair(("t0", "id"), ("c1", "lo"), (0, 0), tp, by_pos=True, name="parent"),
        pair(("t0", "id"), ("g0", "id"), (0, 0), tg, name="designed"),
        pair(("c1", "id"), ("c1", "id"), (0, 1), rng.normal(size=(D1, D1)), name="c1h"),
        count(("t0", "on"), ("c1", "lo"), tc, subset=(0, 1), name="cnt"),
    ]
    return Problem("tiny", T=1, H=6, chans=chans, factors=factors, W_train=8, W_eval=8)


def random_sample(p, region, rng):
    x = p.designed_values(3, region)
    for c in p.sampled():
        x[c.name] = rng.integers(0, c.D, p.shape(c.name, region))
    return x


def test_noise_numpy_matches_jax():
    jnp = pytest.importorskip("jax.numpy")
    Y = np.arange(-3, 3)[:, None]
    X = np.arange(-40, 40, 3)[None, :]
    a = noise.hash32(np, 7, 11, 2, Y, X, 5)
    b = np.asarray(noise.hash32(jnp, 7, 11, 2, jnp.asarray(Y), jnp.asarray(X), 5))
    assert (a == b).all()
    u = noise.uniform(np, 1, 2, 0, np.zeros((200, 200), int), np.arange(200)[None, :] + np.zeros((200, 1), int)
                      + 1000 * np.arange(200)[:, None])
    assert 0 < u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.005


@pytest.mark.parametrize("periodic", [True, False])
def test_local_conditional_matches_energy(periodic):
    """With every cell known, cond's candidate energies differ exactly as the
    reference energy does when the site's value changes."""
    p = tiny()
    region = Region(8, periodic=periodic, x0=0 if periodic else 4)
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = random_sample(p, region, rng)
        for c in p.sampled():
            h, w = p.shape(c.name, region)
            Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            soft, nv = cond(p, x, region, c.name, Y.ravel(), X.ravel())
            for i in rng.choice(Y.size, 6, replace=False):
                y0, x0 = Y.ravel()[i], X.ravel()[i]
                ref_s, ref_v = [], []
                for t in range(c.D):
                    xt = {k: v.copy() for k, v in x.items()}
                    xt[c.name][y0, x0] = t
                    E = en.energy(p, xt, region)
                    ref_s.append(E.soft)
                    ref_v.append(sum(E.viol.values()))
                np.testing.assert_allclose(soft[i] - soft[i][0], np.array(ref_s) - ref_s[0], atol=1e-9)
                np.testing.assert_array_equal(nv[i] - nv[i][0], np.array(ref_v) - ref_v[0])


def test_energy_counts_instances():
    p = tiny()
    region = Region(8)
    x = random_sample(p, region, np.random.default_rng(2))
    E = en.energy(p, x, region)
    n = {k: v[2] for k, v in E.per_factor.items()}
    assert n["u"] == 6 * 8
    assert n["vert"] == 6 * 8 + 8          # a's rows (bottom row padded) plus the top row read through pad_a
    assert n["horiz"] == 6 * 4             # within level-1 blocks only
    assert n["cnt"] == 3 * 4
    assert E.nhard[0] == n["vert"] + n["parent"] + n["cnt"]


def test_greedy_contracts():
    p = tiny()
    v = Greedy(p, 0, "/tmp")
    region = Region(16)
    g = v.generate(5, region)
    x = {**p.designed_values(5, region), **g["raw"]}
    _, lq = coloured_pass(p, 5, region)
    assert np.isclose(v.score(x, region), lq)
    # finite support: overlapping open windows agree
    a = v.generate(9, Region(16, periodic=False, x0=0))["raw"]
    b = v.generate(9, Region(16, periodic=False, x0=8))["raw"]
    for c in p.sampled():
        s = 8 >> c.level
        np.testing.assert_array_equal(a[c.name][:, s:], b[c.name][:, :a[c.name].shape[1] - s])
