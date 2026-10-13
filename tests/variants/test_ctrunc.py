"""ctrunc: the training chain draws exactly the generator's worlds, cost grids
add up to the pass's per-step totals, box / down sums, and full-depth
downstream maps equal the realised costs below."""
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.variants import _ctrunc_core as tc
from castlegen.variants.ctrunc import CTrunc


def _variant(depth=1):
    p = get("toy_support")
    v = CTrunc(p, 0, "/tmp", depth=depth, cells=1024)
    lay = v.layout(Region(p.W_train))
    v._build_trunc(lay)
    return p, v, lay


def test_box_and_down():
    a = np.random.default_rng(0).normal(size=(2, 6, 8))
    b = np.asarray(tc.box(jnp.asarray(a), 1))
    ref = np.zeros_like(a)
    for y in range(6):
        for x in range(8):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if 0 <= y + dy < 6:
                        ref[:, y, x] += a[:, y + dy, (x + dx) % 8]
    assert np.allclose(b, ref, atol=1e-5)
    d = np.asarray(tc.down(jnp.asarray(a)))
    assert np.allclose(d.sum((1, 2)), a.sum((1, 2)), atol=1e-5) and d.shape == (2, 3, 4)


def test_chain_matches_generator_and_totals():
    p, v, lay = _variant()
    T = p.T
    tau, lam = jnp.float32(1.3), jnp.asarray(v.lam, jnp.float32)
    extra = jnp.concatenate([jnp.array([1.3]), lam]).astype(jnp.float32)
    seeds = np.arange(7, 7 + v.B[T])
    out = v._chain(lay, seeds, tau, lam, extra, jnp.ones(T + 1))
    # world 0 reaches level 0: same draws and same per-level cost as the full pass
    des = v.designed(int(seeds[0]), Region(p.W_train))
    st, totals, _, _ = core.coloured_pass(lay, v.params, int(seeds[0]), des, tau, lam)
    for l in range(T + 1):
        Cl = np.asarray(out[l]["C"][0])                           # (S_l, h, w), steps k * nchan + ci
        steps = [lay.step(l, k, ci) for k in range(core.NCOL) for ci in range(len(lay.chans[l]))]
        assert np.allclose(Cl.sum((1, 2)), np.asarray(totals)[steps], rtol=1e-4, atol=1e-3)
    for c in p.sampled():                                         # the chain's world 0 is the generator's draw
        assert np.array_equal(np.asarray(out[0]["st_out"][c.name][0]), np.asarray(st[c.name])), c.name


def test_full_depth_downstream_is_realised():
    p, v, lay = _variant(depth=99)
    T = p.T
    tau, lam = jnp.float32(1.0), jnp.asarray(v.lam, jnp.float32)
    extra = jnp.concatenate([jnp.array([1.0]), lam]).astype(jnp.float32)
    seeds = np.arange(3, 3 + v.B[T])
    out = v._chain(lay, seeds, tau, lam, extra, jnp.ones(T + 1))
    U = v._downstream(lay, out, T + 1)
    n0 = v.B[0]
    for l in range(1, T + 1):
        real = sum(np.asarray(out[m]["C"][:n0]).sum((1, 2, 3)) for m in range(l))
        assert np.allclose(np.asarray(U[l][:n0]).sum((1, 2)), real, rtol=1e-4, atol=1e-2)
        # worlds that stopped above level 0 fall back to the value estimate
        nb = v.B[l - 1]
        assert np.allclose(np.asarray(U[l][nb:]), np.asarray(out[l]["Vm"][nb:]))


def test_returns_local_and_later_only():
    """G of the last step at a level with zero downstream is a box sum of nothing."""
    p, v, lay = _variant()
    l = 0
    S = core.NCOL * len(lay.chans[l])
    h, w = lay.hw[l]
    C = jnp.ones((1, S, h, w))
    U = jnp.zeros((1, h, w))
    G, _ = lay.ret[l](C, U)
    key = core.key_of(l, lay.chans[l][-1].name)
    g = np.asarray(G[key][0])                                   # (16, n)
    assert np.allclose(g[-1], 0.0)                              # nothing drawn after the last step
    R = v.cfg["R"]
    assert g[0].max() <= (S - 1) * (2 * R + 1) ** 2 + 1e-3
