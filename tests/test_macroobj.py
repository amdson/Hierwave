import numpy as np

from castlegen import macroobj as MO


def _occ_of(C, S):
    """occ rebuilt from the labels; None if two footprints overlap."""
    H, W = S.BY * MO.BS, S.BX * MO.BS
    occ = np.full(H * W, -1, np.int64)
    for o in np.nonzero(S.lab >= 0)[0]:
        k = S.lab[o]
        ay, ax = MO._anchor(o, C.tb, S.st)
        yy, xx = np.nonzero(C.KC[k, :C.KH[k], :C.KW[k]] != MO.VOID)
        idx = ((ay + yy) % H) * W + (ax + xx) % W
        if (occ[idx] >= 0).any():
            return None
        occ[idx] = o
    return occ


def _exemplar(C, EB=24, seed=0):
    X = MO.State(EB, EB)
    k0 = int(np.nonzero((C.fam == C.fams.index("hall")) & C.FULL)[0][0])
    MO.grow(k0, EB * EB, C.weight.astype(np.float64), C.fam, C.FULL, C.tb, X.st, seed, 200000)
    left = MO.close(C.SEAL, C.tb, X.st, seed, 50)
    return X, left


def test_attachment_table_is_symmetric():
    C = MO.load()
    pairs = set()
    for k2 in range(C.K):
        for j2 in range(C.PN[k2]):
            s = k2 * C.MAXP + j2
            for a in range(C.ATTP[s], C.ATTP[s + 1]):
                pairs.add((k2, j2, int(C.ATTK[a]), int(C.ATTJ[a])))
    assert pairs and all((k, j, k2, j2) in pairs for k2, j2, k, j in pairs)


def test_seal_walls_up_one_port():
    C = MO.load()
    for k in range(C.K):
        for j in range(C.PN[k]):
            k2 = C.SEAL[k, j]
            if k2 < 0:
                assert C.PN[k] == 1
                continue
            assert C.PN[k2] == C.PN[k] - 1 and C.fam[k2] == C.fam[k]
            diff = C.KC[k] != C.KC[k2]
            assert diff.sum() == 1 and diff[C.PY[k, j], C.PX[k, j]] and C.KC[k2, C.PY[k, j], C.PX[k, j]] == MO.WALL


def test_exemplar_is_closed_and_connected():
    C = MO.load()
    X, left = _exemplar(C)
    m = MO.measure(C, X)
    assert left == 0 and m["unmatched"] == 0 and m["objects"] > 50 and m["comps"] == 1
    assert np.array_equal(_occ_of(C, X), X.occ)


def test_sweeps_keep_footprints_disjoint_and_end_hard():
    C = MO.load()
    G = len(C.fams)
    S = MO.State(24, 24)
    MO.sweeps(np.concatenate([np.linspace(0.5, 4, 10), np.full(10, 1000.0)]),
              np.concatenate([np.geomspace(3, 1, 10), np.ones(10)]), np.full(G, -1.0), C.fam, C.tb, S.st, 3)
    assert np.array_equal(_occ_of(C, S), S.occ)
    m = MO.measure(C, S)
    assert m["objects"] > 0 and m["unmatched"] == 0


def test_coarse_paste_and_sweeps_consistent():
    C = MO.load()
    G = len(C.fams)
    X, _ = _exemplar(C)
    S = MO.State(32, 32)
    U = np.random.default_rng(0).integers(0, 24, (4, 4, 2)).astype(np.int64)
    MO.paste_all(U, 8, X.st, 24, C.tb, S.st)
    MO.coarse_sweeps(U, 8, np.array([2.0, 4.0]), np.array([1.0, 0.5]), 6, np.full(G, -0.6), C.fam, X.st, 24,
                     C.tb, S.st, 1)
    assert np.array_equal(_occ_of(C, S), S.occ)


def test_forest_protection_never_splits():
    """With new islands forbidden, the protected spanning forest keeps the
    number of components from growing: leaves may retract, structures merge."""
    C = MO.load("macro_dense")
    G = len(C.fams)
    X, _ = _exemplar(C)
    S = MO.State(32, 32)
    U = np.random.default_rng(0).integers(0, 24, (4, 4, 2)).astype(np.int64)
    MO.paste_all(U, 8, X.st, 24, C.tb, S.st)
    MO.sweeps(np.full(3, 1000.0), np.ones(3), np.full(G, -2.0), C.fam, C.tb, S.st, 1)
    before = MO.measure(C, S)
    cn = MO.conn_state(S.lab.size, C.MAXP, beta=2.0, gamma=1e9)
    for seed in range(4):
        MO.sweeps(np.full(1, 1000.0), np.full(1, 2.0), np.full(G, -2.0), C.fam, C.tb, S.st, seed, cn=cn)
        after = MO.measure(C, S)
        assert after["comps"] <= before["comps"]
        before = after
    assert np.array_equal(_occ_of(C, S), S.occ)
