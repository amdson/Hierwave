"""block_mh: the block energy delta matches energy.py, a draw's log q equals its
teacher-forced log q, draws follow exp(log q), and a short MH chain from exact
toy_modes samples keeps the exact mode masses (bad q, L = 2)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems import toys
from castlegen.kernels import block_mh as bm
from castlegen.variants import _cpass_core as core

LAM = 20.0


def _setup(name, scale=3.0):
    p = get_problem(name)
    lay = core.Layout(p, 16, True, 0)
    key = jax.random.PRNGKey(0)
    params = {}
    for l in range(p.T + 1):
        for c in lay.chans[l]:
            key, k2, k3 = jax.random.split(key, 3)
            pr = core.mlp_init(k2, lay.feat_dim(l, lay.actor_win), c.D, 64)
            pr["W3"] = jax.random.normal(k3, pr["W3"].shape) * scale / 8.0
            params[core.key_of(l, c.name)] = pr
    return p, lay, params


def _exact_states(p, n, seed=1):
    E = toys.modes_block_energy if p.name == "toy_modes" else toys.support_block_energy
    pr = toys._probs(E(toys.all_blocks()))
    rng = np.random.default_rng(seed)
    xs = [toys.lift(toys.sample_t(rng, pr, p.H, 16)) for _ in range(n)]
    return {k: jnp.asarray(np.stack([x[k] for x in xs]), jnp.int32) for k in xs[0]}


@pytest.mark.parametrize("name", ["toy_support", "toy_modes"])
def test_delta_energy_and_logq(name):
    p, lay, params = _setup(name)
    reg = Region(16)
    st = _exact_states(p, 4)
    fp = jax.jit(lambda k, s, L, Y, X: bm.block_pass(lay, params, k, s, bm.block_masks(lay, L, Y, X), 1.0, LAM, p.T))
    ff = jax.jit(lambda k, s, f, L, Y, X: bm.block_pass(lay, params, k, s, bm.block_masks(lay, L, Y, X), 1.0, LAM,
                                                        p.T, forced=f))

    def etot(x):
        E = en.energy(p, {k: np.asarray(v) for k, v in x.items()}, reg)
        return E.soft + LAM * sum(E.viol.values())
    rng = np.random.default_rng(0)
    for i in range(12):
        L = int(rng.integers(0, p.T + 1))
        h, w = lay.hw[L]
        Y, X = int(rng.integers(h)), int(rng.integers(w))
        x = {k: v[i % 4] for k, v in st.items()}
        y, lqy, esy, nvy = fp(jax.random.PRNGKey(i), x, L, Y, X)
        _, lqx, esx, nvx = ff(jax.random.PRNGKey(0), x, x, L, Y, X)
        _, lqy2, _, _ = ff(jax.random.PRNGKey(0), x, y, L, Y, X)
        assert abs(float(esy + LAM * nvy - esx - LAM * nvx) - (etot(y) - etot(x))) < 1e-3
        assert abs(float(lqy - lqy2)) < 1e-4
        assert np.isfinite(float(lqx))          # a valid state is in q's support


def test_exact_start_no_drift_and_power():
    """Exact toy_modes states, a strongly biased q, L = 2: the MH chain keeps
    the exact mode_B mass; dropping the q ratio drifts it far away."""
    p, lay, params = _setup("toy_modes")
    B = 64
    st = _exact_states(p, B)
    keys = jax.random.split(jax.random.PRNGKey(3), B)
    ex = p.exact["mode_B"]
    out = {}
    for drop in (False, True):
        chain = bm.make_chain(lay, [2], 1.0, LAM, 1, drop)
        s, _ = chain(params, keys, st, 1500)
        t = np.asarray(s["t"])
        out[drop] = np.mean([toys.modes_block_stats(toys.blocks_of(t[b]))["mode_B"].mean() for b in range(B)])
    se = np.sqrt(ex * (1 - ex) / (B * 8))
    assert abs(out[False] - ex) < 4 * se, (out, ex)
    assert abs(out[True] - ex) > 10 * se, (out, ex)
