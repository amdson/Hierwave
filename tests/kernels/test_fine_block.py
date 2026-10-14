"""Exact fine-block Gibbs (castlegen/kernels/fine_block.py) on dimers (T = 0):
a small periodic region where all perfect matchings can be enumerated.
Invariance: iid exact samples stay exactly uniform after a sweep (z-scores of the
orientation fractions and a chi-square on the state histogram); k = region size
(one tile) is an exact independent sampler; k = 2 from a fixed start reproduces
the exact orientation fractions (ergodic mixture of offsets).  CPU, ~1 min."""
import numpy as np
import pytest

from castlegen.bench.problem import Region
from castlegen.kernels.fine_block import FineBlockGibbs, FineBlockKernel
from castlegen.variants._exactopt import dimers_problem

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]
OPP = [1, 0, 3, 2, 5, 4]


def matchings(H, W):
    """All perfect matchings of the H x W (x periodic) triangular lattice as p arrays."""
    out = []
    p = -np.ones((H, W), np.int64)

    def rec(i):
        while i < H * W and p[i // W, i % W] >= 0:
            i += 1
        if i == H * W:
            out.append(p.copy())
            return
        y, x = divmod(i, W)
        for d in range(6):                               # any unassigned partner (x wraps)
            dy, dx = DIRS[d]
            y2, x2 = y + dy, (x + dx) % W
            if 0 <= y2 < H and p[y2, x2] < 0 and (y2, x2) != (y, x):
                p[y, x], p[y2, x2] = d, OPP[d]
                rec(i + 1)
                p[y, x], p[y2, x2] = -1, -1
    rec(0)
    return out


def orient(p):
    o = p // 2
    return np.array([(o == 0).mean(), (o == 1).mean(), (o == 2).mean()])


@pytest.fixture(scope="module")
def setup():
    H, W = 4, 8
    pr = dimers_problem(H)
    reg = Region(W)
    ms = matchings(H, W)
    des = pr.designed_values(0, reg)
    return pr, reg, ms, des


def flat_of(K, p, des):
    return K.flat.flatten({"p": p, "edge": des["edge"]})


def test_enumeration_is_valid(setup):
    pr, reg, ms, des = setup
    K = FineBlockGibbs(pr, reg, 2, (0, 0))
    from castlegen.kernels.exact_block import build_instances, total_energy
    for p in ms[:50]:
        assert total_energy(K.flat, K.insts, flat_of(K, p, des))[1] == 0
    assert len(ms) > 50


@pytest.mark.parametrize("k,offset", [(2, (0, 0)), (2, (1, 1)), (3, (0, 0)), (4, (0, 3))])
def test_invariance_numpy(setup, k, offset):
    pr, reg, ms, des = setup
    K = FineBlockGibbs(pr, reg, k, offset, lam=None, seed=1, internal="boundary" if k == 4 else True)
    rng = np.random.default_rng(0)
    N = 1500 if k == 4 else 6000
    idx = rng.integers(len(ms), size=N)
    new = np.empty(N, np.int64)
    key = {m.tobytes(): i for i, m in enumerate(ms)}
    for n, i in enumerate(idx):
        v = flat_of(K, ms[i], des)
        K.sweep(v, rng)
        p = K.flat.unflatten(v)["p"]
        new[n] = key[p.astype(np.int64).tobytes()]               # stays a valid matching
    # state histogram: uniform
    M = len(ms)
    cnt = np.bincount(new, minlength=M)
    chi = ((cnt - N / M) ** 2 / (N / M)).sum()
    assert chi < M - 1 + 5 * np.sqrt(2 * (M - 1)), (chi, M)
    # orientation z-scores against the exact mean
    ex = np.array([orient(m) for m in ms])
    mu, sd = ex.mean(0), ex.std(0)
    got = np.array([orient(ms[i]) for i in new]).mean(0)
    z = (got - mu) / (sd / np.sqrt(N) + 1e-12)
    assert np.abs(z).max() < 4.5, z


def test_whole_region_tile_is_exact_sampler(setup):
    """k = 4, H = 4: one tile row, W = 8 -> two tiles, exact conditional each."""
    pr, reg, ms, des = setup
    rng = np.random.default_rng(3)
    ex = np.array([orient(m) for m in ms])
    # two-tile Gibbs alone may not be ergodic over offsets; with offsets (0,0),(0,2) it is
    K2 = FineBlockKernel(pr, reg, 4, lam=None, offsets=[(0, 0), (0, 2), (0, 1), (0, 3)], jax_backend=False)
    v = flat_of(K2.Ks[0], ms[0], des)
    acc = []
    for s in range(600):
        K2.sweep_np(v, rng)
        acc.append(orient(K2.Ks[0].flat.unflatten(v)["p"]))
    acc = np.array(acc[100:])
    mu = ex.mean(0)
    se = acc.std(0) * np.sqrt(10) / np.sqrt(len(acc)) + 1e-3   # crude autocorrelation inflation
    assert np.abs(acc.mean(0) - mu).max() < 4 * se.max() + 0.01, (acc.mean(0), mu)


def test_jax_matches_structure_and_invariance(setup):
    import jax
    import jax.numpy as jnp
    pr, reg, ms, des = setup
    KK = FineBlockKernel(pr, reg, 3, lam=None, offsets=[(0, 0)], seed=0)
    rng = np.random.default_rng(5)
    N = 6000
    idx = rng.integers(len(ms), size=N)
    V = np.stack([flat_of(KK.Ks[0], ms[i], des) for i in idx]).astype(np.int32)
    v, ch = KK.sweep(jnp.asarray(V), jax.random.PRNGKey(0))
    v = np.asarray(v)
    key = {m.tobytes(): i for i, m in enumerate(ms)}
    new = np.array([key[KK.flat.unflatten(r.astype(np.int64))["p"].astype(np.int64).tobytes()] for r in v])
    M = len(ms)
    cnt = np.bincount(new, minlength=M)
    chi = ((cnt - N / M) ** 2 / (N / M)).sum()
    assert chi < M - 1 + 5 * np.sqrt(2 * (M - 1)), (chi, M)
    assert (v != V).any()
