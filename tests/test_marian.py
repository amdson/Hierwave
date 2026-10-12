"""Marian's 3D module set and the layered 3D infinite sampler."""
import numpy as np
import pytest

from castlegen import marian
from castlegen.infinite import BlockSchedule, Schedule, World, generate, u01

pytestmark = pytest.mark.skipif(not (marian.Path("cache/marian/Assets/ModuleData.asset").exists()),
                                reason="cache/marian not downloaded")


@pytest.fixture(scope="module")
def ms():
    return marian.load()


@pytest.fixture(scope="module")
def model(ms):
    return marian.tile_model(ms)


FLAT = Schedule.anneal(S=3, beta0=1.0, gamma0=2.0, quench=1)
BLOCK = BlockSchedule.anneal(B=4, Sb=1, K=2, restart=False, shift=True, beta0=1.0, gamma0=2.0, quench=1)
BLOCK2 = BlockSchedule.anneal(B=4, Sb=2, K=1, restart=False, shift=True, beta0=1.0, gamma0=2.0, quench=1)


def test_loader(ms):
    assert ms.T == 454
    assert ms.names[0] == "Empty R0" and ms.weight[0] == 10.0
    assert len(set(ms.prototype.tolist())) == 121
    assert set(ms.rotation.tolist()) == {0, 1, 2, 3}
    # above Empty: Empty or a bridge (bitset bytes 0 and 10 of its Up set)
    up = {ms.names[i] for i in np.nonzero(ms.allowed[4, 0])[0]}
    assert up == {"Empty R0", "Bridge R0", "Bridge R1", "Bridge R3"}


def test_table_symmetry(ms):
    """The bitsets are nearly symmetric; the mismatches come from
    SimplifyNeighborData pruning one side only, so every listed pair still
    fits by its face connectors."""
    asym = ms.asymmetry()
    assert asym[0] == asym[3] and asym[2] == asym[5] and asym[1] == asym[4]
    assert sum(asym) < 0.002 * 6 * ms.T ** 2
    rng = np.random.default_rng(0)
    for d in range(6):
        a_s, b_s = np.nonzero(ms.allowed[d])
        for k in rng.choice(a_s.size, 200, replace=False):
            assert ms.fits(a_s[k], d, b_s[k])


def test_model_tables(ms, model):
    for d in range(6):
        assert (model.F[d] == model.F[(d + 3) % 6].T).all()
        ok = ms.allowed[d] & ms.allowed[(d + 3) % 6].T
        assert (model.F[d].T.astype(bool) == ~ok).all()
    dom = ms.layer_domains()
    raw = ms.boundary_ok()
    assert dom.shape == (6, 454) and (dom <= raw).all() and dom.any(1).all()
    assert not dom[0, ms.air()]          # nothing floats on layer 0
    assert dom[5, ms.air()]


@pytest.mark.parametrize("sched", [FLAT, BLOCK, BLOCK2], ids=["flat", "block", "block_sb2"])
def test_overlap_exact(model, sched):
    a = generate(model, sched, 7, -5, -9, 12, 14)
    b = generate(model, sched, 7, 1, -3, 13, 10)
    assert a.shape == (6, 12, 14)
    np.testing.assert_array_equal(a[:, 6:, 6:], b[:, :6, :8])


@pytest.mark.parametrize("sched", [FLAT, BLOCK], ids=["flat", "block"])
def test_chunk_order(model, sched):
    C = 8
    w1 = World(model, sched, seed=3, C=C)
    keys = [(cy, cx) for cy in range(-1, 2) for cx in range(-2, 1)]
    rng = np.random.default_rng(1)
    for k in rng.permutation(len(keys)):
        w1.chunk(*keys[k])
    w2 = World(model, sched, seed=3, C=C)
    r2 = w2.region(-C, -2 * C, 3 * C, 3 * C)
    r1 = w1.region(-C, -2 * C, 3 * C, 3 * C)
    assert r1.shape == (6, 3 * C, 3 * C)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(w1.region(-3, -5, 4, 6), r2[:, C - 3:C + 1, 2 * C - 5:2 * C + 1])


@pytest.mark.parametrize("sched", [FLAT, BLOCK], ids=["flat", "block"])
def test_no_misfits(model, sched):
    g = generate(model, sched, 11, -20, 30, 24, 24)
    assert model.violations(g) == (0, 0)
    assert (g <= model.T).all()


def test_layer_in_hash():
    """Different layers draw different uniforms at the same column and step."""
    from castlegen.infinite import _key3
    assert u01(5, 2, 3, _key3(4, 0)) != u01(5, 2, 3, _key3(4, 1))
