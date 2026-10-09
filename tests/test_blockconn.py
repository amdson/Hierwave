import numpy as np

from castlegen import blockconn as BC, tileset


def _ref(T, lab, grp, tabs, mode):
    """Reference of blockconn.parts: BFS components, then classes as extra nodes."""
    node, Dh, Dv, sock = tabs
    k = T.shape[0]
    comp = -np.ones((k, k), int)
    nc = 0
    for y0 in range(k):
        for x0 in range(k):
            if node[T[y0, x0]] and comp[y0, x0] < 0:
                comp[y0, x0], st = nc, [(y0, x0)]
                while st:
                    y, x = st.pop()
                    for ny, nx, j in ((y, x + 1, Dh[T[y, x], T[y, x + 1]] if x + 1 < k else 0),
                                      (y, x - 1, Dh[T[y, x - 1], T[y, x]] if x else 0),
                                      (y + 1, x, Dv[T[y, x], T[y + 1, x]] if y + 1 < k else 0),
                                      (y - 1, x, Dv[T[y - 1, x], T[y, x]] if y else 0)):
                        if j and node[T[ny, nx]] and comp[ny, nx] < 0:
                            comp[ny, nx] = nc
                            st.append((ny, nx))
                nc += 1
    out, hh = np.zeros(4), k // 2
    touch = {}                                                             # class -> components with its openings
    for q in range(8):
        d, i = divmod(q, 2)
        cells = [((0, s), (s, k - 1), (k - 1, s), (s, 0))[d] for s in range(i * hh, (i + 1) * hh)]
        op = [c for c in cells if node[T[c]] and sock[d, T[c]]]
        if lab[q] == 0:
            out[0] += (len(op) > 0) if mode == 0 else len(op)
        else:
            out[1] += not op
            if op and lab[q] != BC.OPEN:
                touch.setdefault(int(lab[q]), set()).update(comp[c] for c in op)
    cl = [({l}, set(c)) for l, c in touch.items()]                         # merge classes sharing a component
    merged = True
    while merged:
        merged = False
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                if cl[a][1] & cl[b][1]:
                    la, ca = cl[a]
                    lb, cb = cl.pop(b)
                    cl[a] = (la | lb, ca | cb)
                    merged = True
                    break
            if merged:
                break
    for g in set(grp[l] for l in touch):
        n = sum(any(grp[l] == g for l in ls) for ls, _ in cl)
        out[2] += (n > 1) if mode == 0 else n - 1
    if all(l in (0, BC.OPEN) for l in lab) and any(l for l in lab):        # a sink: one component
        out[3] = (nc > 1) if mode == 0 else max(nc - 1, 0)
        return out
    good = set().union(*[c for _, c in cl]) if cl else set()
    orph = [c for c in range(nc) if c not in good]
    if mode == 3:                                                          # distance to a class port's edge + size / 2
        cells = [((0, s), (s, k - 1), (k - 1, s), (s, 0))[q // 2] for q in range(8) if 0 < lab[q] < BC.OPEN
                 for s in range(q % 2 * hh, (q % 2 + 1) * hh)]
        for c in orph:
            ys, xs = np.nonzero(comp == c)
            out[3] += 0.5 * len(ys) + min(abs(y - cy) + abs(x - cx) for y, x in zip(ys, xs) for cy, cx in cells)
        return out
    out[3] = (len(orph) > 0) if mode == 0 else len(orph) if mode == 1 else np.isin(comp, orph).sum()
    return out


def test_parts_match_reference():
    ts = tileset.load("demo")
    tabs = BC.tables(ts)
    E = np.load("cache/castle_ex.npy").astype(np.int64)
    rng = np.random.default_rng(0)
    for trial in range(60):
        k = 8 if trial % 2 else 16
        y, x = rng.integers(0, 64, 2)
        T = np.ascontiguousarray(np.roll(E, (-y, -x), (0, 1))[:k, :k])
        if trial % 3 == 0:                                                 # scramble some tiles
            T[rng.random((k, k)) < 0.2] = rng.integers(0, ts.n_sig - 1)
        lab = (rng.integers(0, 6, 8) if trial % 4 else BC.pattern(T, tabs)) if trial % 5 else rng.choice([0, BC.OPEN], 8)
        grp = np.ones(BC.MLAB + 1, np.int64) if trial % 2 else rng.integers(1, 3, BC.MLAB + 1)
        for mode in (0, 1, 2, 3):
            np.testing.assert_allclose(BC.parts(T, lab, grp, *tabs, mode), _ref(T, lab, grp, tabs, mode),
                                       err_msg=f"{trial} {mode}")


def test_own_pattern_scores_only_orphans():
    ts = tileset.load("demo")
    tabs = BC.tables(ts)
    E = np.load("cache/castle_ex.npy").astype(np.int64)
    T = np.ascontiguousarray(E[:16, :16])
    p = BC.parts(T, BC.pattern(T, tabs), np.ones(BC.MLAB + 1, np.int64), *tabs, 1)
    assert p[0] == 0 and p[1] == 0


def test_cand_scores():
    ts = tileset.load("demo")
    tabs = BC.tables(ts)
    E = np.load("cache/castle_ex.npy").astype(np.int64)
    T = np.ascontiguousarray(E[8:24, 8:24])
    lab = BC.pattern(T, tabs)
    w = np.array([1.0, 2.0, 3.0, 4.0])
    py, px = np.array([0, 7, 15]), np.array([3, 7, 0])
    cand = np.random.default_rng(1).integers(0, ts.n_sig - 1, (3, 20))
    grp = np.ones(BC.MLAB + 1, np.int64)
    got = BC.cand_scores(T, py, px, cand, lab, grp, *tabs, 1, w)
    for i in range(3):
        for j in range(20):
            B = T.copy()
            B[py[i], px[i]] = cand[i, j]
            assert np.isclose(got[i, j], w @ BC.parts(B, lab, grp, *tabs, 1))


def test_site_scores_match_parts():
    ts = tileset.load("demo")
    tabs = BC.tables(ts)
    E = np.load("cache/castle_ex.npy").astype(np.int64)
    rng = np.random.default_rng(2)
    allowed = np.ones(ts.n_sig, bool)
    w = np.array([1.0, 2.0, 3.0, 5.0])
    for trial in range(40):
        y, x = rng.integers(0, 64, 2)
        T = np.ascontiguousarray(np.roll(E, (-y, -x), (0, 1))[:16, :16])
        if trial % 3 == 0:
            T[rng.random((16, 16)) < 0.2] = rng.integers(0, ts.n_sig - 1)
        lab = rng.integers(0, 6, 8) if trial % 5 else rng.choice([0, BC.OPEN], 8)
        grp = np.ones(BC.MLAB + 1, np.int64) if trial % 2 else rng.integers(1, 3, BC.MLAB + 1)
        sites = [(0, 0), (0, 15), (15, 0), (15, 15), (0, 7), (8, 15)] + [tuple(rng.integers(0, 16, 2)) for _ in range(4)]
        for mode in (0, 1, 2, 3):
            for py, px in sites:
                got = BC.site_scores(T, py, px, lab, grp, *tabs, mode, w, allowed)
                for t in range(ts.n_sig):
                    B = T.copy()
                    B[py, px] = t
                    assert np.isclose(got[t], w @ BC.parts(B, lab, grp, *tabs, mode)), (trial, mode, py, px, t)
