import numpy as np
import pytest

from castlegen import export_models as ex
from castlegen import hwsampler as hs
from castlegen.infinite import generate, u01


@pytest.fixture(scope="module")
def city():
    m, s = ex.city_model(S=6, quench=2)
    return m, s


def _load(tmp_path, name, m, s):
    return hs.Model.load(ex.write_hwm(str(tmp_path / f"{name}.hwm"), m, s))


def test_hash_matches():
    for seed in (0, 9, (1 << 64) - 3):
        for y, x, h in ((-5, 7, 0), (-(1 << 40), -1, 129), (3, 1 << 33, 7)):
            assert hs.u01(seed, y, x, h) == u01(seed, y, x, h)


def test_biomes_bit_exact(tmp_path):
    m, s = ex.biomes_model(S=10)
    M = _load(tmp_path, "b", m, s)
    assert M.T == 8 and M.pad == 20
    for seed, y0, x0, H, W in ((3, -37, -20, 40, 50), ((1 << 64) - 1, 5, -3, 17, 9)):
        assert np.array_equal(M.tiles(seed, y0, x0, H, W), generate(m, s, seed, y0, x0, H, W))


def test_biomes_soft_gamma_bit_exact(tmp_path):
    from castlegen.infinite import Schedule
    m, _ = ex.biomes_model()
    s = Schedule(np.linspace(0.2, 1.0, 6), np.linspace(0.5, 3.0, 6))   # no gamma = inf sweep
    M = _load(tmp_path, "b2", m, s)
    assert np.array_equal(M.tiles(1, -9, -30, 24, 24), generate(m, s, 1, -9, -30, 24, 24))


def test_city_bit_exact(tmp_path, city):
    m, s = city
    M = _load(tmp_path, "c", m, s)
    assert M.T == m.T and M.pad == 2 * s.S + 1
    a = generate(m, s, 5, -17, -20, 20, 24)
    assert np.array_equal(M.tiles(5, -17, -20, 20, 24), a)
    assert (a == m.T).any()                       # wildcards exercised


def test_city_hot_bit_exact(tmp_path):
    """beta0 0.1: contradictions in the quench take the full lexicographic path."""
    m, _ = ex.city_model()
    from castlegen.infinite import Schedule
    s = Schedule.anneal(S=5, quench=2)
    M = _load(tmp_path, "ch", m, s)
    assert np.array_equal(M.tiles(11, -3, -12, 14, 16), generate(m, s, 11, -3, -12, 14, 16))


def test_world_equals_tiles(tmp_path):
    m, s = ex.biomes_model(S=6)
    M = _load(tmp_path, "w", m, s)
    w = hs.World(M, seed=4, C=8)
    for k in [(1, 0), (-1, -1), (0, 1), (0, 0), (-1, 0), (1, 1)]:
        w.region(k[0] * 8 + 2, k[1] * 8 + 3, 3, 4)
    full = w.region(-13, -11, 30, 29)
    assert np.array_equal(full, M.tiles(4, -13, -11, 30, 29))
    assert np.array_equal(full, generate(m, s, 4, -13, -11, 30, 29))
    assert w.n_chunks >= 16


def test_world_city(tmp_path, city):
    m, s = city
    M = _load(tmp_path, "wc", m, s)
    w = hs.World(M, seed=2, C=8)
    w.region(-8, 8, 8, 8)
    w.region(0, 0, 4, 4)
    assert np.array_equal(w.region(-8, 0, 16, 16), M.tiles(2, -8, 0, 16, 16))


def test_city_fast_deterministic(tmp_path, city):
    m, s = city
    M = _load(tmp_path, "cf", m, s)
    M.set_fast(True)
    a = M.tiles(5, -17, -20, 20, 24)
    assert np.array_equal(a, M.tiles(5, -17, -20, 20, 24))
    assert np.array_equal(a[4:, 6:], M.tiles(5, -13, -14, 16, 18))   # overlapping requests agree
    w = hs.World(M, seed=5, C=4)
    w.region(-16, -20, 4, 4)
    assert np.array_equal(w.region(-17, -20, 20, 24), a)
    assert m.energy(a)[1] == 0                                       # wildcard pass: no misfits
