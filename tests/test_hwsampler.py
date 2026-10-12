import os

import numpy as np
import pytest

from castlegen import export_models as ex
from castlegen import hwsampler as hs
from castlegen.infinite import generate, u01


@pytest.fixture(scope="module")
def city():
    m, s = ex.city_flat_model(S=6, quench=2)
    return m, s


@pytest.fixture(scope="module")
def city_m():
    return ex.city_socket_model()


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


def test_city_hot_bit_exact(tmp_path, city_m):
    """beta0 0.1: contradictions in the quench take the full lexicographic path."""
    m = city_m
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


def test_reads_hwm1(tmp_path):
    m, s = ex.biomes_model(S=4)
    p = ex.write_hwm(str(tmp_path / "b.hwm"), m, s)
    raw = open(p, "rb").read()
    old = tmp_path / "b1.hwm"
    old.write_bytes(b"HWM1" + raw[4:20] + raw[36:])          # drop the block words
    assert np.array_equal(hs.Model.load(str(old)).tiles(2, -5, -6, 9, 7), generate(m, s, 2, -5, -6, 9, 7))


@pytest.mark.parametrize("B,Sb,K,shift,restart,beta0,region", [
    (4, 2, 3, True, False, 1.0, (-13, -9, 11, 14)),
    (4, 2, 3, False, True, 1.0, (-5, -22, 9, 13)),
    (6, 2, 2, True, True, 0.1, (-7, 3, 10, 8)),            # hot: contradictions in the quench
    (4, 3, 2, True, False, 0.3, (17, -31, 6, 9)),
])
def test_city_block_bit_exact(tmp_path, city_m, B, Sb, K, shift, restart, beta0, region):
    from castlegen.infinite import BlockSchedule
    s = BlockSchedule.anneal(B=B, Sb=Sb, K=K, shift=shift, restart=restart, beta0=beta0, gamma0=2.0, quench=1)
    M = _load(tmp_path, "cb", city_m, s)
    assert M.B == B and M.S == Sb * K and M.pad == (2 * Sb + 1) * B
    assert np.array_equal(M.tiles(9, *region), generate(city_m, s, 9, *region))


def test_city_block_no_wildcard_bit_exact(tmp_path, city_m):
    from castlegen.infinite import BlockSchedule, SocketModel
    m = SocketModel(city_m.u, city_m.sN, city_m.sE, city_m.sS, city_m.sW)
    s = BlockSchedule.anneal(B=4, Sb=2, K=2, shift=True, restart=False, beta0=1.0, gamma0=2.0, quench=1)
    M = _load(tmp_path, "cbn", m, s)
    assert np.array_equal(M.tiles(3, -6, -2, 8, 9), generate(m, s, 3, -6, -2, 8, 9))


def test_city_block_world_and_fast(tmp_path, city_m):
    from castlegen.infinite import BlockSchedule
    s = BlockSchedule.anneal(B=4, Sb=2, K=3, shift=True, restart=False, beta0=1.0, gamma0=2.0, quench=1)
    M = _load(tmp_path, "cbw", city_m, s)
    w = hs.World(M, seed=6, C=8)
    w.region(-8, 8, 8, 8)
    w.region(3, -5, 2, 2)
    a = w.region(-9, -7, 18, 22)
    assert np.array_equal(a, M.tiles(6, -9, -7, 18, 22))
    assert np.array_equal(a, generate(city_m, s, 6, -9, -7, 18, 22))
    M.set_fast(True)
    f = M.tiles(6, -9, -7, 18, 22)
    assert np.array_equal(f[3:, 5:], M.tiles(6, -6, -2, 15, 17))
    wf = hs.World(M, seed=6, C=4)
    wf.region(0, 0, 4, 4)
    assert np.array_equal(wf.region(-9, -7, 18, 22), f)
    assert city_m.energy(f)[1] == 0


# ------------------------------------------------------------------ layered 3D (TileModel3D, kind 2)
def _toy3d(lu=False, wild=True, seed=0, T=14, L=4, dens=0.55):
    from castlegen.infinite import TileModel3D
    rng = np.random.default_rng(seed)
    F = rng.random((6, T, T)) < dens
    F[:, 0, 0] = False                         # tile 0 fits itself everywhere: no empty world
    dom = rng.random((L, T)) < 0.7
    dom[:, 0] = True
    return TileModel3D([f"t{i}" for i in range(T)], rng.random(T) * 2, F, dom, np.zeros((T, 3)),
                       wild=wild, lu=rng.normal(size=(L, T)) if lu else None)


@pytest.mark.parametrize("lu,wild", [(False, True), (True, True), (True, False)])
def test_3d_flat_bit_exact(tmp_path, lu, wild):
    from castlegen.infinite import Schedule
    m = _toy3d(lu=lu, wild=wild)
    s = Schedule.anneal(S=6, beta0=0.3, gamma0=1.0, quench=2)
    M = _load(tmp_path, "t3f", m, s)
    assert M.kind == 2 and M.L == m.L and M.T == m.T and M.pad == 2 * s.S + wild
    for seed, reg in ((4, (-11, -7, 9, 13)), ((1 << 64) - 5, (6, -30, 7, 5))):
        a = generate(m, s, seed, *reg)
        assert a.shape == (m.L,) + reg[2:]
        assert np.array_equal(M.tiles(seed, *reg), a)


@pytest.mark.parametrize("B,Sb,K,shift,restart,lu", [
    (4, 2, 3, True, False, True),
    (4, 2, 2, False, True, False),
    (6, 2, 2, True, True, True),
])
def test_3d_block_bit_exact(tmp_path, B, Sb, K, shift, restart, lu):
    from castlegen.infinite import BlockSchedule
    m = _toy3d(lu=lu, seed=B)
    s = BlockSchedule.anneal(B=B, Sb=Sb, K=K, shift=shift, restart=restart, beta0=0.5, gamma0=1.0, quench=1)
    M = _load(tmp_path, "t3b", m, s)
    assert M.B == B and M.pad == (2 * Sb + 1) * B
    for reg in ((-13, -9, 11, 14), (17, -31, 6, 9)):
        assert np.array_equal(M.tiles(9, *reg), generate(m, s, 9, *reg))


def test_3d_world_equals_tiles(tmp_path):
    from castlegen.infinite import BlockSchedule
    m = _toy3d(lu=True)
    s = BlockSchedule.anneal(B=4, Sb=2, K=2, shift=True, restart=False, beta0=0.5, gamma0=1.0, quench=1)
    M = _load(tmp_path, "t3w", m, s)
    w = hs.World(M, seed=6, C=8)
    w.region(-8, 8, 8, 8)
    w.region(3, -5, 2, 2)
    a = w.region(-9, -7, 18, 22)
    assert a.shape == (m.L, 18, 22)
    assert np.array_equal(a, M.tiles(6, -9, -7, 18, 22))
    assert np.array_equal(a, generate(m, s, 6, -9, -7, 18, 22))
    M.set_fast(True)                           # ignored for 3D models
    assert np.array_equal(M.tiles(6, -9, -7, 18, 22), a)


_MARIAN = os.path.exists("cache/marian/Assets/ModuleData.asset")


@pytest.mark.skipif(not _MARIAN, reason="cache/marian assets missing")
@pytest.mark.parametrize("air", [None, (1.0, 3.0, 10.0, 30.0, 60.0, 100.0)])
def test_marian_block_bit_exact(tmp_path, air):
    from castlegen import marian
    from castlegen.infinite import BlockSchedule
    m = marian.tile_model(marian.load(), air_weights=air)
    assert (m.lu is None) == (air is None)
    s = BlockSchedule.anneal(B=4, Sb=2, K=3, shift=True, restart=False, beta0=1.0, gamma0=2.0, quench=1)
    M = _load(tmp_path, "mb", m, s)
    reg = (-6, -9, 7, 10)
    a = generate(m, s, 7, *reg)
    assert np.array_equal(M.tiles(7, *reg), a)
    w = hs.World(M, seed=7, C=4)
    w.region(-4, -8, 2, 2)
    assert np.array_equal(w.region(*reg), a)
