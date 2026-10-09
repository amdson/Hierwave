import numpy as np

from castlegen import bitgibbs as BG


def test_sample_matches_exact_distribution():
    rng = np.random.default_rng(0)
    for S, unit in ((40, np.log(2.0)), (150, 0.3)):
        L = rng.integers(0, 6, (5, S))
        L[rng.random(L.shape) < 0.05] = -1
        legal = rng.random(S) < 0.9
        cw = BG.weights(unit)
        p = BG.exact(L, legal, cw)
        n = 400_000
        x = BG.sample_many(BG.stacks(L, 4), BG.masks(legal), cw, 3, n)
        assert (x >= 0).all()
        f = np.bincount(x, minlength=S) / n
        assert f[p == 0].sum() == 0
        z = np.abs(f - p)[p > 0] / np.sqrt(p * (1 - p) / n)[p > 0]
        assert z.max() < 5


def test_no_legal_tile_returns_minus_one():
    L = np.full((2, 10), -1)
    x = BG.sample_many(BG.stacks(L, 4), BG.masks(np.ones(10, bool)), BG.weights(), 0, 5)
    assert (x == -1).all()


def test_pair_tables_orientation():
    """pair[d][n] holds tile t's energy with neighbour n on its side d."""
    rng = np.random.default_rng(1)
    S = 8
    Eh, Ev = rng.integers(0, 4, (S, S)) * np.log(2.0), rng.integers(0, 4, (S, S)) * np.log(2.0)
    pair = BG.pair_tables(Eh, Ev, np.log(2.0), 4)
    for d, E in enumerate((Ev, Eh.T, Ev.T, Eh)):
        for n in range(S):
            want = BG.quantize(E[n], np.log(2.0), 4)
            got = sum(((pair[d, n, b, 0] >> np.arange(S, dtype=np.uint64)) & np.uint64(1)).astype(int) << b
                      for b in range(4))
            assert (got == want).all()
