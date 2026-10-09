import numpy as np

from castlegen.legacy import macroobj as MO, macrocontact as MC


def _setup(n=12):
    C = MO.load("macro_dense")
    return C, MC.tables(C), MO.State(n, n)


def _footprints_disjoint(C, ct, S):
    H, W = S.BY * MO.BS, S.BX * MO.BS
    cover = np.zeros(H * W, int)
    for o in np.nonzero(S.lab >= 0)[0]:
        k = S.lab[o]
        ay, ax = MC._anchor(o, ct, S.st)
        yy, xx = np.nonzero(C.KC[k, :C.KH[k], :C.KW[k]] != MO.VOID)
        cells = ((ay + yy) % H) * W + (ax + xx) % W
        assert (S.occ[cells] == o).all()
        cover[cells] += 1
    assert cover.max() <= 1


def test_slots_on_door_sides_only():
    C, ct, _ = _setup()
    for k in ct.FULLK:
        sides = set(C.PD[k, :C.PN[k]].tolist())
        assert set(ct.SD[k, :ct.SN[k]].tolist()) <= sides
        for j in range(C.PN[k]):                           # every door is a slot
            assert ct.SLOT[k, C.PY[k, j], C.PX[k, j]] == C.PD[k, j]


def test_sweeps_disjoint_and_contacts_symmetric():
    C, ct, S = _setup()
    MC.sweeps(np.full(4, 2.0), np.ones(4), np.zeros(len(C.fams)), C.fam, ct, S.st, 0)
    assert (S.lab >= 0).sum() > 20
    _footprints_disjoint(C, ct, S)
    nb, cnt = np.empty(64, np.int64), np.empty(64, np.int64)
    c = {}
    for o in np.nonzero(S.lab >= 0)[0]:
        ay, ax = MC._anchor(o, ct, S.st)
        n = MC._contacts(S.lab[o], ay, ax, ct, S.st, nb, cnt)
        for i in range(n):
            c[o, nb[i]] = cnt[i]
    assert c and all(c.get((p, o)) == v for (o, p), v in c.items())


def test_forest_never_splits():
    C, ct, S = _setup()
    MC.sweeps(np.full(4, 2.0), np.ones(4), np.zeros(len(C.fams)), C.fam, ct, S.st, 0)
    cn = MC.conn_state(S.lab.size, 8.0, 8.0, 0.0, 0, 3)
    for s in range(4):
        MC._snapshot(ct, S.st, cn)
        members = {}
        for o in np.nonzero(S.lab >= 0)[0]:
            members.setdefault(MC._find(cn.cu, o), []).append(o)
        MC.sweeps(np.full(1, 2.0), np.ones(1), np.zeros(len(C.fams)), C.fam, ct, S.st, 10 + s, cn=cn)
        _footprints_disjoint(C, ct, S)
        # rooms of one snapshot component that survive stay in one component
        par = {o: o for o in np.nonzero(S.lab >= 0)[0]}
        def f(a):
            while par[a] != a:
                a = par[a]
            return a
        for o, p, _ in MC.contact_graph(ct, S):
            par[f(o)] = f(p)
        for ms in members.values():
            alive = [o for o in ms if S.lab[o] >= 0]
            assert len({f(o) for o in alive}) <= 1
