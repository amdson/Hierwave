import numpy as np

from castlegen.infinite import Schedule, World, biomes, generate, roads, u01


def test_noise_is_positional():
    assert u01(3, -7, 11, 5) == u01(3, -7, 11, 5)
    assert len({u01(3, y, x, 1) for y in range(-3, 3) for x in range(-3, 3)}) == 36


def test_overlapping_requests_agree():
    m, s = biomes(), Schedule.anneal(S=12)
    a = generate(m, s, 1, -40, -25, 50, 60)
    b = generate(m, s, 1, -20, 0, 50, 60)
    assert np.array_equal(a[20:, 25:], b[:30, :35])
    assert np.array_equal(a, generate(m, s, 1, -40, -25, 50, 60, pad=s.pad + 7))


def test_chunk_order_independent():
    m, s = roads(), Schedule.anneal(S=10)
    w = World(m, s, seed=4, C=16)
    for k in [(1, 0), (-1, -1), (0, 1), (0, 0), (-1, 0), (1, 1), (0, -1), (-1, 1), (1, -1)]:
        w.chunk(*k)
    assert np.array_equal(w.region(-16, -16, 48, 48), generate(m, s, 4, -16, -16, 48, 48))


def test_quench_removes_violations_on_biomes():
    m = biomes()
    g = generate(m, Schedule.anneal(S=32), 2, 0, 0, 64, 64)
    assert m.energy(g)[1] == 0
