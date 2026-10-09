import numpy as np

from castlegen.legacy import blockconn as BC, objects as OB, tileset

TS = tileset.load("macro")
LAY = {"size": [24, 24], "objects": {"box": {"size": [3, 4], "place": [{"cell": "altar", "at": [1, 1]}]}},
       "place": [{"object": "box", "at": [5, 6]}, {"object": "box", "at": [15, 2]}]}


def test_channel_and_census():
    tabs = OB.tables(TS, LAY)
    O = OB.channel(TS, LAY, tabs)
    T = np.full(O.shape, TS.WALL, np.int64)
    T[O >= 0] = tabs[3][O[O >= 0]]
    assert OB.broken(O, tabs[4]) == 0
    assert OB.census(O, T, tabs) == {"box": (2, 0), "tiles off template": 0}
    O2 = O.copy()
    O2[6, 7] = -1                                                           # a hole: its 4 neighbours each break a check
    assert OB.broken(O2, tabs[4]) == 4
    assert OB.census(O2, T, tabs)["box"] == (1, 11)


def test_coarse_term_matches_tile_rule():
    """coarse_term = the tile rule's broken checks across a cell's seams, for
    windows pasted from an exemplar channel."""
    tabs = OB.tables(TS, LAY)
    Oex = OB.channel(TS, LAY, tabs)
    m, h = 24, 4

    class L:                                                                # the JointLevel fields coarse_term reads
        R = C = 3
    L.h, L.m = h, m
    rng = np.random.default_rng(0)
    for _ in range(20):
        L.U = rng.integers(0, m * m, (3, 3))
        L.U[1, 2] = L.U[1, 1] + h if rng.random() < 0.5 else L.U[1, 2]      # sometimes coherent
        f = OB.coarse_term(L, Oex, tabs[4], 1.0)
        cand = rng.integers(0, m * m, (1, 6))
        got = f(np.array([1]), np.array([1]), cand)[0]
        for j, u in enumerate(cand[0]):
            U = L.U.copy()
            U[1, 1] = u
            uy, ux = np.divmod(np.repeat(np.repeat(U, h, 0), h, 1), m)
            ly, lx = np.mgrid[:3 * h, :3 * h] % h
            Om = Oex.ravel()[((uy - h // 2 + ly) % m) * m + (ux - h // 2 + lx) % m]
            nbr, want = tabs[4], 0
            for y in range(h, 2 * h):
                for x in range(h, 2 * h):
                    for d, (dy, dx) in enumerate(OB.DIRS):
                        ny, nx = y + dy, x + dx
                        if h <= ny < 2 * h and h <= nx < 2 * h:
                            continue                                        # inside p's window: not a seam
                        a, b = Om[y, x], Om[ny, nx]
                        want += (nbr[a, d] >= 0 and b != nbr[a, d]) + (nbr[b, OB.OPP[d]] >= 0 and a != nbr[b, OB.OPP[d]])
            assert got[j] == want


def test_map_sweep_keeps_and_draws_objects():
    tabs = OB.tables(TS, LAY)
    O = OB.channel(TS, LAY, tabs)
    T = np.full(O.shape, TS.WALL, np.int64)
    T[O >= 0] = tabs[3][O[O >= 0]]
    O[16, 3], T[16, 3] = -1, TS.WALL                                        # a hole in the second copy
    O[2, 20] = tabs[2][0] + 1 * 4 + 1                                       # a lone cell
    T[2, 20] = tabs[3][O[2, 20]]
    S = TS.n_sig
    t = TS.np_tables
    Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
    allowed = np.ones(S, bool)
    soft = np.ones(S, bool)
    soft[np.unique(tabs[3])] = False
    soft[TS.WALL] = True
    btabs = BC.tables(TS)
    labs = np.zeros((3, 3, 8), np.int64)
    rng = np.random.default_rng(0)
    for _ in range(40):
        BC.map_sweep(T, 8, rng.permutation(T.size), rng.gumbel(size=(T.size, S + 5)), Eh, Ev, logz, allowed,
                     np.zeros(T.shape + (S,)), labs, np.ones(BC.MLAB + 1, np.int64), *btabs, 0, np.ones(4), 0.5,
                     0.0, TS.WALL, O, tabs[3], tabs[4], soft, 1000.0)
    c = OB.census(O, T, tabs)
    assert c["tiles off template"] == 0 and OB.broken(O, tabs[4]) == 0
    assert c["box"] == (2, 0)                                               # the hole refilled, the lone cell gone
