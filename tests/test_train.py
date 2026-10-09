"""Bidirectional packing (below rows) and the generic training pieces, on a
hand-built three-level model: tile h=1 D=3, mid h=2 D=3, top h=4 D=2."""
import numpy as np
import pytest

from castlegen.channels.core import Channel, Factor, Model
from castlegen.channels import train

NAMES = ["tt_h", "tt_v", "tm", "cnt", "ttop", "mt", "mid_h", "mid_u", "top_v"]


def hand_model(seed=0, H=8, W=8):
    rng = np.random.default_rng(seed)
    tile = Channel("tile", 1, 3).add_view("id", np.arange(3)).add_view("par", [0, 1, 0], 2)
    mid = Channel("mid", 2, 3).add_view("id", np.arange(3))
    top = Channel("top", 4, 2).add_view("id", np.arange(2))
    R = lambda *s: rng.normal(size=s)
    fs = [Factor.pair(("tile", "id"), ("tile", "id"), (0, 1), R(3, 3), pad_b=1, pad_a=2, name="tt_h"),
          Factor.pair(("tile", "id"), ("tile", "id"), (1, 0), R(3, 3), name="tt_v"),
          Factor.pair(("tile", "id"), ("mid", "id"), (0, 0), R(3, 3), name="tm"),
          Factor.count(("tile", "par"), ("mid", "id"), R(3, 5), name="cnt"),
          Factor.pair(("tile", "par"), ("top", "id"), (0, 0), R(2, 2), name="ttop"),
          Factor.pair(("mid", "id"), ("top", "id"), (0, 0), R(3, 2), name="mt"),
          Factor.pair(("mid", "id"), ("mid", "id"), (0, 1), R(3, 3), pad_b=0, pad_a=1, name="mid_h"),
          Factor.unary(("mid", "id"), R(3), name="mid_u"),
          Factor.pair(("top", "id"), ("top", "id"), (1, 0), R(2, 2), name="top_v")]
    m = Model(H, W, [top, mid, tile], fs)
    for c in m.channels:
        c.grid[:] = rng.integers(0, c.D, c.grid.shape)
    return m


def full_energy(m):
    """The joint, by plain loops: pairs at (a at p, b at q) with pad_b for an
    off-grid q, and pad_a for a same-level b cell whose a partner is off."""
    E = 0.0
    for f in m.factors:
        A = m.chan(f.a[0])
        va = A.views[f.a[1]]
        ga = A.grid
        if f.kind == 2:
            E += sum(f.table[va[v]] for v in ga.ravel())
            continue
        B = m.chan(f.b[0])
        vb = B.views[f.b[1]]
        gb = B.grid
        if f.kind == 1:
            r = B.h // A.h
            for i in range(gb.shape[0]):
                for j in range(gb.shape[1]):
                    s = va[ga[i * r:(i + 1) * r, j * r:(j + 1) * r]].sum()
                    E += f.table[vb[gb[i, j]], s]
            continue
        for y in range(ga.shape[0]):
            for x in range(ga.shape[1]):
                qy, qx = y * A.h // B.h + f.off[0], x * A.h // B.h + f.off[1]
                if 0 <= qy < gb.shape[0] and 0 <= qx < gb.shape[1]:
                    E += f.table[va[ga[y, x]], vb[gb[qy, qx]]]
                elif f.pad_b >= 0:
                    E += f.table[va[ga[y, x]], f.pad_b]
        if A.h == B.h and f.pad_a >= 0:
            for y in range(gb.shape[0]):
                for x in range(gb.shape[1]):
                    py, px = y - f.off[0], x - f.off[1]
                    if not (0 <= py < ga.shape[0] and 0 <= px < ga.shape[1]):
                        E += f.table[f.pad_a, vb[gb[y, x]]]
    return E


@pytest.mark.parametrize("home", ["tile", "mid", "top"])
def test_below_site_energies_match_joint(home):
    m = hand_model(1)
    rng = np.random.default_rng(2)
    g = m.chan(home).grid
    D = m.chan(home).D
    for _ in range(6):
        y, x = rng.integers(0, g.shape[0]), rng.integers(0, g.shape[1])
        e = m.site_energies(home, y, x, below=True)
        old = g[y, x]
        tot = []
        for t in range(D):
            g[y, x] = t
            tot.append(full_energy(m))
        g[y, x] = old
        np.testing.assert_allclose(e[:, None] - e[None, :], np.subtract.outer(tot, tot), atol=1e-9)


def test_counts_energy_identity():
    """E = sum over factors of <table, counts>, the joint by plain loops."""
    m = hand_model(3)
    N = train.counts(m, NAMES)
    E = sum((f.table * N[f.name]).sum() for f in m.factors)
    assert np.isclose(E, full_energy(m))
    assert N["cnt"].sum() == m.chan("mid").grid.size
    assert N["tt_h"].sum() == 56 + 8 + 8                      # 56 pairs, 8 pad_b, 8 pad_a
    assert N["mid_u"].sum() == 16 and N["top_v"].sum() == 2


def test_counts_numpy():
    m = hand_model(4)
    N = train.counts(m, ["tm", "top_v", "mid_u"])
    t, md, tp = m.chan("tile").grid, m.chan("mid").grid, m.chan("top").grid
    ref = np.zeros((3, 3))
    for y in range(8):
        for x in range(8):
            ref[t[y, x], md[y // 2, x // 2]] += 1
    np.testing.assert_array_equal(N["tm"], ref)
    ref = np.zeros((2, 2))
    ref[tp[0, 0], tp[1, 0]] += 1
    ref[tp[0, 1], tp[1, 1]] += 1
    np.testing.assert_array_equal(N["top_v"], ref)
    np.testing.assert_array_equal(N["mid_u"], np.bincount(md.ravel(), minlength=3))


@pytest.mark.parametrize("home", ["tile", "mid", "top"])
def test_rb_gap(home):
    m = hand_model(5)
    c = m.chan(home)
    g = c.grid
    onehot = np.zeros(g.shape + (c.D,))
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    onehot[yy, xx, g] = 1.0
    gap = train.rb_gap(m, home, NAMES, onehot)
    for n in NAMES:
        assert np.all(gap[n] == 0), n
    before = train.counts(m, NAMES)
    rng = np.random.default_rng(6)
    sites = [(0, 0), (g.shape[0] - 1, g.shape[1] - 1)] + [tuple(rng.integers(0, s) for s in g.shape) for _ in range(4)]
    for y, x in sites:
        t = (g[y, x] + 1) % c.D
        p = onehot.copy()
        p[y, x] = 0
        p[y, x, t] = 1
        gap = train.rb_gap(m, home, NAMES, p)
        old = g[y, x]
        g[y, x] = t
        after = train.counts(m, NAMES)
        g[y, x] = old
        for n in NAMES:
            np.testing.assert_allclose(gap[n], after[n] - before[n], atol=1e-12, err_msg=f"{home} {n}")
    # mixed probabilities: the gap is linear in probs
    pr = train.site_probs(m, home, below=True)
    assert np.allclose(pr.sum(-1), 1)
    gap = train.rb_gap(m, home, NAMES, pr)
    ref = {n: np.zeros_like(before[n]) for n in NAMES}
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            old = g[y, x]
            for t in range(c.D):
                g[y, x] = t
                after = train.counts(m, NAMES)
                for n in NAMES:
                    ref[n] += pr[y, x, t] * (after[n] - before[n])
            g[y, x] = old
    for n in NAMES:
        np.testing.assert_allclose(gap[n], ref[n], atol=1e-9, err_msg=f"{home} {n}")


def test_site_probs_fixed_and_softmax():
    m = hand_model(7)
    mid = m.chan("mid")
    mid.fixed[1, 2] = True
    pr = train.site_probs(m, "mid", T=2.0, below=True)
    assert pr[1, 2, mid.grid[1, 2]] == 1.0 and pr[1, 2].sum() == 1.0
    e = m.site_energies("mid", 0, 3, below=True) / 2.0
    w = np.exp(-(e - e.min()))
    np.testing.assert_allclose(pr[0, 3], w / w.sum())


def test_step():
    th = {"a": np.zeros((3, 3)), "u": np.zeros(3)}
    gap = {"a": np.diag([1.0, 0, 0]), "u": np.array([0.0, -1, 0])}
    new = train.step(th, gap, eta=0.5)
    for k in new:
        assert abs(new[k].mean()) < 1e-12
    assert new["a"][0, 0] > new["a"][1, 1]                   # over-produced gains energy
    assert new["u"][1] < new["u"][0]
    pri = {"a": np.ones((3, 3)), "u": np.zeros(3)}
    new2 = train.step(new, {}, prior=pri, lam_prior=0.5)
    np.testing.assert_allclose(new2["a"], 0.5 * new["a"] - (0.5 * new["a"]).mean())
    new3 = train.step(th, gap, eta=1.0, scale={"a": 2.0, "u": 1.0})
    np.testing.assert_allclose(new3["a"], 0.5 * (gap["a"] - gap["a"].mean()))


def test_bidirectional_sweep_and_total_energy():
    m = hand_model(8)
    P = m.compile("mid", below=True)
    assert set(P.fac[:, 0]) == {0, 2, 3, 4}
    from castlegen.channels import kernel
    Pb = m.compile("mid", below=False)
    tb = kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.cert, P.joins, P.delta)
    ta = kernel.total_energy(Pb.home, Pb.grids, Pb.hs, Pb.views, Pb.fac, Pb.tabs, Pb.cert, Pb.joins, Pb.delta)
    assert tb == ta                                          # below rows are not double counted
    assert m.sweep("top", 3, seed=1, below=True) == 0
    assert m.sweep("mid", 3, seed=1, below=True) == 0


# compile(home) rows of the hand model before the below packing was added
OLD_FAC = {
    "tile": [[0, 2, 2, 2, 0, 1, 1, 0], [0, 2, 2, 2, 0, -1, 2, 1], [0, 2, 2, 2, 1, 0, -1, 2],
             [0, 2, 2, 2, -1, 0, -1, 3], [0, 1, 2, 1, 0, 0, -1, 4], [1, 1, 3, 1, 0, 0, -1, 5],
             [0, 0, 3, 0, 0, 0, -1, 6]],
    "mid": [[0, 0, 1, 0, 0, 0, -1, 0], [0, 1, 1, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, -1, 1, 2],
            [2, -1, 1, -1, 0, 0, -1, 3]],
    "top": [[0, 0, 0, 0, 1, 0, -1, 0], [0, 0, 0, 0, -1, 0, -1, 1]],
}


@pytest.mark.parametrize("home", ["tile", "mid", "top"])
def test_compile_unchanged_without_below(home):
    m = hand_model(9)
    P = m.compile(home)
    np.testing.assert_array_equal(P.fac, np.array(OLD_FAC[home], np.int64))
    assert P.src.shape == (P.fac.shape[0],) and P.transposed.shape == (P.fac.shape[0],)
    for f, row in enumerate(P.fac):
        T = m.factors[P.src[f]].table
        T = T.T if P.transposed[f] else T
        np.testing.assert_array_equal(P.tabs[row[7]], T.reshape(T.shape[0], -1))


def test_potts_below_site_energies():
    """The same check on the Potts model, if present: mid and top sites see the joint."""
    potts = pytest.importorskip("castlegen.channels.potts")
    Pz = potts.Potts(1, 1, BM=2)
    top, mid, tile = Pz.channels()
    rng = np.random.default_rng(0)
    th = {k: rng.normal(size=v.shape) for k, v in Pz.theta0().items()}
    m = Pz.model(top, mid, tile, th)
    for c in m.channels:
        c.grid[:] = rng.integers(0, c.D, c.grid.shape)
    for home in ("mid", "top"):
        g, D = m.chan(home).grid, m.chan(home).D
        for y in range(g.shape[0]):
            for x in range(g.shape[1]):
                e = m.site_energies(home, y, x, below=True)
                old, tot = g[y, x], []
                for t in range(D):
                    g[y, x] = t
                    tot.append(full_energy(m))
                g[y, x] = old
                np.testing.assert_allclose(e - e[0], np.array(tot) - tot[0], atol=1e-9)
