"""Exact block-Gibbs kernel (castlegen/kernels/exact_block.py): energies,
exactness and invariance on the toys.  CPU only, about 1 min."""
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.exact_block import ExactBlockGibbs, total_energy


@pytest.mark.parametrize("name", ["toy_support", "toy_modes", "ground"])
def test_instance_energy_matches_reference(name):
    p = get(name)
    reg = Region(p.W_train)
    K = ExactBlockGibbs(p, reg, level=1)
    x = {**p.probes(5, reg)[1], **p.designed_values(5, reg)}
    E = en.energy(p, x, reg)
    s, nv = total_energy(K.flat, K.insts, K.flat.flatten(x))
    assert abs(s - E.soft) < 1e-6 and nv == sum(E.viol.values())


@pytest.mark.parametrize("name", ["toy_support", "ground"])
def test_block_energy_deltas(name):
    """Conditional energies of every completion = full-energy differences."""
    p = get(name)
    reg = Region(p.W_train)
    K = ExactBlockGibbs(p, reg, level=1)
    x = {**p.probes(2, reg)[3], **p.designed_values(2, reg)}
    v = K.flat.flatten(x)
    rng = np.random.default_rng(1)
    s0, n0 = total_energy(K.flat, K.insts, v)
    for k in range(K.ncol):
        soft, nv = K.energies(v, k)
        blocks, _ = K.cls[k]
        for i in rng.choice(blocks.size, 2):
            b = blocks[i]
            cur = np.where((K.comps == v[K.BV[b]]).all(1))[0][0]
            for c in rng.choice(K.comps.shape[0], 4):
                w = v.copy()
                w[K.BV[b]] = K.comps[c]
                s1, n1 = total_energy(K.flat, K.insts, w)
                assert abs((s1 - s0) - (soft[i, c] - soft[i, cur])) < 1e-8
                assert (n1 - n0) == nv[i, c] - nv[i, cur]


def _stats_run(p, kernels, v0s, n_sweeps, burn, rng):
    acc = []
    for v0 in v0s:
        v = v0.copy()
        for s in range(n_sweeps):
            for K in kernels:
                K.sweep(v, rng)
            if s >= burn:
                acc.append(p.stats(kernels[0].flat.unflatten(v)))
    return {k: np.array([a[k] for a in acc]) for k in acc[0]}


@pytest.mark.parametrize("name", ["toy_support", "toy_modes"])
def test_full_block_gibbs_reproduces_exact(name):
    """Level-2 blocks with all descendants (pruned enumeration): from an all-zero
    (invalid) start, sweeps reproduce problem.exact."""
    p = get(name)
    reg = Region(p.W_train)
    K = ExactBlockGibbs(p, reg, level=2, max_joint=1000)
    assert not K.shared
    rng = np.random.default_rng(0)
    v0 = np.zeros(K.flat.n, np.int64)
    st = _stats_run(p, [K], [v0], 100, 1, rng)
    for k, ex in p.exact.items():
        se = st[k].std() / np.sqrt(len(st[k])) + 1e-3
        assert abs(st[k].mean() - ex) < 4 * se, (k, st[k].mean(), ex, se)


@pytest.mark.parametrize("name", ["toy_support", "toy_modes"])
def test_fine_block_kernel_invariance(name):
    """Level-1 blocks (rho1 + 4 tiles, not ergodic alone): started from exact
    samples (the toys' probes are exact iid draws), the stats do not drift."""
    p = get(name)
    reg = Region(p.W_train)
    K = ExactBlockGibbs(p, reg, level=1, seed=3)
    rng = np.random.default_rng(4)
    v0s = [K.flat.flatten(p.probes(s, reg)[0]) for s in range(60)]
    st = _stats_run(p, [K], v0s, 10, 0, rng)
    moved = sum(K.sweep(v.copy(), rng) for v in v0s[:5])
    assert moved > 0
    for k, ex in p.exact.items():
        x = st[k].reshape(60, 10)[:, -1]                  # last sweep, 60 independent chains
        se = x.std() / np.sqrt(len(x)) + 1e-3
        assert abs(x.mean() - ex) < 4 * se, (k, x.mean(), ex, se)
