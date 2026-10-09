import itertools

import numpy as np

from castlegen.channels import induce
from castlegen.channels.circles import Circles
from castlegen.channels.core import Channel, Factor, Model
from castlegen.channels.potts import Potts
from castlegen.channels.sampler import Sampler

INF = np.inf
D = 8
ALLOW = np.ones((D, 3), bool)                   # par 0: every value; 1: 0..4; 2: only 3
ALLOW[5:, 1] = False
ALLOW[:, 2] = False
ALLOW[3, 2] = True


def toy(par, hard=True, soft_lam=None, seed=0):
    """c (h = 1, D = 8) under a fixed parent par (h = 2, D = 3): a parent row
    (INF off ALLOW, or soft_lam there), random same-level pairs and unary."""
    r = np.random.default_rng(seed)
    par = np.asarray(par, np.int32)
    H, W = 2 * par.shape[0], 2 * par.shape[1]
    p = Channel("par", 2, 3).add_view("val", np.arange(3))
    c = Channel("c", 1, D).add_view("self", np.arange(D))
    p.grid, p.fixed = par.copy(), np.ones(par.shape, bool)
    c.grid = r.integers(0, D, (H, W)).astype(np.int32)
    pt = r.normal(size=(D, 3)) * 0.3
    pt[~ALLOW] = INF if hard else (soft_lam if soft_lam is not None else 3.0)
    fac = [Factor.pair(("c", "self"), ("c", "self"), (0, 1), r.normal(size=(D, D)) * 0.7, name="h"),
           Factor.pair(("c", "self"), ("c", "self"), (1, 0), r.normal(size=(D, D)) * 0.7, name="v"),
           Factor.unary(("c", "self"), r.normal(size=D) * 0.5, name="u"),
           Factor.pair(("c", "self"), ("par", "val"), (0, 0), pt, name="mask")]
    return Model(H, W, [p, c], fac), c


def rand_state(chans, seed):
    r = np.random.default_rng(seed)
    for ch in chans:
        ch.grid[:] = r.integers(0, ch.D, ch.grid.shape)


def potts_model(extra=()):
    P = Potts(2, 2)
    chans = P.channels()
    rand_state(chans, 1)
    r = np.random.default_rng(2)
    th = {k: r.normal(size=v.shape) for k, v in P.theta0().items()}
    return Model(P.H, P.W, list(chans), P.designed_factors() + P.learned_factors(th) + list(extra)), chans


def circles_model(kappa=4.0, extra=()):
    with np.errstate(divide="ignore"):
        C = Circles(2, 2, kappa=kappa)
    chans = C.channels()
    rand_state(chans[:3], 3)
    r = np.random.default_rng(4)
    th = {k: r.normal(size=v.shape) * 0.3 for k, v in C.theta0().items()}
    return C.model(chans, th, extra), chans


# ---------------------------------------------------------------- identity
def test_wrapper_identity():
    for build in (potts_model, circles_model):
        m1, c1 = build()
        m2, c2 = build()
        for home in ("top", "mid", "tile"):
            m1.sweep(home, 4, seed=11)
            Sampler(m2, home).sweep(4, seed=11)
            for a, b in zip(c1, c2):
                assert np.array_equal(a.grid, b.grid), (build.__name__, home, a.name)
        assert m1.energy("tile") == Sampler(m2, "tile").energy()


# ---------------------------------------------------------- hard rows first
def test_hard_rows_first():
    P = Potts(2, 2)
    pal = Factor.pair(("mid", "col"), ("top", "pal"), (0, 0), np.where(P.PAL.T, 0.0, INF), name="palhard")
    hdisc = np.zeros((Circles(2, 2).D, 2))
    hdisc[1:, 0] = INF                                  # an object only where the slot says so
    cdisc = Factor.pair(("mid", "self"), ("slot", "val"), (0, 0), hdisc, name="slothard")
    for (m, chans), home in ((potts_model([pal]), "mid"), (circles_model(INF, [cdisc]), "mid"),
                             (circles_model(INF, [cdisc]), "tile")):
        S = Sampler(m, home)
        assert S.P.nhard >= 1
        assert all(np.isposinf(S.P.tabs[t]).any() for t in S.P.fac[:S.P.nhard, 7])
        assert not any(np.isposinf(S.P.tabs[t]).any() for t in S.P.fac[S.P.nhard:, 7])
        for s in range(3):
            rand_state([c for c in chans if not c.fixed.all()], 10 + s)
            g = m.chan(home).grid
            for y in range(g.shape[0]):
                for x in range(g.shape[1]):
                    a, b = m.site_energies(home, y, x), S.site_energies(y, x)
                    assert np.array_equal(np.isinf(a), np.isinf(b))
                    assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-12, atol=1e-12)
            assert S.energy() == m.energy(home) or np.isclose(S.energy()[0], m.energy(home)[0])


# --------------------------------------------------------- cap invariance
def _marginals(K, sweeps, seed):
    m, c = toy([[0, 1], [2, 0]])
    S = Sampler(m, "c", K=K, seed=seed)
    S.init()
    par = np.repeat(np.repeat(m.chan("par").grid, 2, 0), 2, 1)
    S.sweep(200)
    rec = np.zeros((sweeps, 2 * D + 1))
    for i in range(sweeps):
        S.sweep(1)
        g = c.grid
        rec[i, :D] = np.bincount(g[par == 0], minlength=D) / (par == 0).sum()
        rec[i, D:2 * D] = np.bincount(g[par == 1], minlength=D) / (par == 1).sum()
        rec[i, 2 * D] = (g[:, 1:] == g[:, :-1]).mean()
    nb = 50
    b = rec[: sweeps // nb * nb].reshape(nb, -1, rec.shape[1]).mean(axis=1)
    return b.mean(axis=0), b.std(axis=0, ddof=1) / np.sqrt(nb)


def test_cap_invariance_marginals():
    m0, s0 = _marginals(None, 40000, 1)
    m4, s4 = _marginals(4, 40000, 2)
    z = np.abs(m0 - m4) / np.sqrt(s0 ** 2 + s4 ** 2 + 1e-12)
    assert z.max() < 3.0, z
    assert np.all(m4[D + 5:2 * D] == 0)                 # never an inadmissible value


def test_cap_single_site_chi2():
    m, c = toy([[0, 1], [2, 0]], seed=5)
    c.fixed = np.ones(c.grid.shape, bool)
    c.fixed[1, 1] = False
    c.grid[1, 1] = 0
    S = Sampler(m, "c", K=3, seed=3)
    e = S.site_energies(1, 1)
    p = np.exp(-(e - e.min()))
    p /= p.sum()
    n, thin = 4000, 10
    h = np.zeros(D)
    for _ in range(n):
        S.sweep(thin)
        h[c.grid[1, 1]] += 1
    chi2 = ((h - n * p) ** 2 / (n * p)).sum()
    assert chi2 < 24.3, (chi2, h / n, p)                # df 7, p = 0.001
    assert S.candidates(1, 1).shape == (3,) and S.candidates(1, 1)[0] == c.grid[1, 1]


def test_cap_exact_2x2():
    """Site (0, 0) marginal of a 2 x 2 grid against enumeration, K = None / 4 / 2."""
    m, c = toy([[0]], seed=9)
    S = Sampler(m, "c")
    states = np.array(list(itertools.product(range(D), repeat=4)))
    lw = np.empty(len(states))
    for i, st in enumerate(states):
        c.grid[:] = st.reshape(2, 2)
        lw[i] = -S.energy()[0]
    w = np.exp(lw - lw.max())
    exact = np.bincount(states[:, 0], weights=w / w.sum(), minlength=D)
    for K in (None, 4, 2):
        S = Sampler(m, "c", K=K, seed=1)
        S.sweep(100)
        hist = np.zeros((50, D))
        for b in range(50):
            for _ in range(2000):
                S.sweep(1)
                hist[b, c.grid[0, 0]] += 1
        hist /= 2000
        z = np.abs(hist.mean(0) - exact) / (hist.std(0, ddof=1) / np.sqrt(50))
        assert z.max() < 3.0, (K, z)


def test_candidates_admissible():
    m, c = toy([[0, 1], [2, 0]])
    S = Sampler(m, "c")
    assert np.array_equal(S.candidates(0, 0), np.arange(D))
    assert np.array_equal(S.candidates(0, 2), np.arange(5))
    assert np.array_equal(S.candidates(2, 0), [3])


# ----------------------------------------------------------------- dormancy
def test_dormancy():
    m, c = toy([[2, 0], [1, 2]])
    S = Sampler(m, "c", hb=2)
    frac = S.init()
    dorm = np.zeros((4, 4), bool)
    dorm[:2, :2] = dorm[2:, 2:] = True
    assert frac == 0.5 and np.array_equal(S.dormant, dorm)
    assert np.all(c.grid[dorm] == 3) and not c.fixed.any()
    assert np.array_equal(S.active, [[False, True], [True, False]])
    before = c.grid.copy()
    S.sweep(20)
    assert np.array_equal(c.grid[dorm], before[dorm]) and not np.array_equal(c.grid, before)
    soft = Model(m.H, m.W, [m.chan("par"), c], toy([[2, 0], [1, 2]], hard=False)[0].factors)
    S.relax(20, soft)
    assert np.array_equal(c.grid[dorm], before[dorm])

    m, c = toy([[2, 2], [2, 2]])
    S = Sampler(m, "c", hb=2)
    assert S.init() == 1.0 and not S.active.any()
    e0, g0 = S.energy(), c.grid.copy()
    assert S.sweep(10) == 0
    assert S.energy() == e0 and np.array_equal(c.grid, g0)


# --------------------------------------------------------------------- AIS
def test_ais_matches_induce():
    m1, c1 = toy([[0, 1], [1, 0]], hard=False)
    m2, c2 = toy([[0, 1], [1, 0]], hard=False)
    lz1, lw1 = induce.ais_log_z(m1, "c", K=8, M=4, seed=5)
    lz2, lw2 = Sampler(m2, "c").ais_log_z(K=8, M=4, seed=5)
    assert lz1 == lz2 and np.array_equal(lw1, lw2)


def test_ais_capped_hard():
    """AIS on the capped kernel against exact log Z of a 2 x 2 window under par = 1 (5^4 states)."""
    m, c = toy([[1]], seed=7)
    S = Sampler(m, "c", K=3)
    ws = []
    for st in itertools.product(range(D), repeat=4):
        c.grid[:] = np.array(st).reshape(2, 2)
        e, nv = S.energy()
        if nv == 0:
            ws.append(-e)
    exact = induce.logmeanexp(ws) + np.log(len(ws))
    lz, lw = S.ais_log_z(K=64, M=32, seed=1)
    assert abs(lz - exact) < 4 * induce.ais_se(lw) + 0.05, (lz, exact)


# ------------------------------------------------- admitted lists (C1, A_p)
def _biome_obj(K, seed, adm, fams_seed=5):
    from castlegen.channels.circles_biome import CirclesBiome, Forward, random_families, random_pair_masks
    rng = np.random.default_rng(fams_seed)
    C = CirclesBiome(2, 2, families=random_families(5, rng), masks=random_pair_masks(5, rng))
    th = C.theta0()
    th["bio_u"] = -C.bio_u0
    F = Forward(C, th, seed=1, use_sampler=True)
    F.build()
    F.biome.grid[:] = np.random.default_rng(seed).integers(0, C.P, F.biome.grid.shape)
    F.biome.grid[0, 0] = 0                              # one none block: dormant
    C.paint_allow(F.biome, F.allow)
    return C, F, Sampler(F.model, "obj", K=K, hb=C.BT, seed=seed, adm=adm)


def test_adm_identity():
    """sweep_adm = sweep_cap draw for draw: Potts / Circles with parent hard rows, the toy, the biome."""
    P = Potts(2, 2)
    pal = Factor.pair(("mid", "col"), ("top", "pal"), (0, 0), np.where(P.PAL.T, 0.0, INF), name="palhard")
    hdisc = np.zeros((Circles(2, 2).D, 2))
    hdisc[1:, 0] = INF
    cdisc = Factor.pair(("mid", "self"), ("slot", "val"), (0, 0), hdisc, name="slothard")
    builds = [(lambda: potts_model(), ("top", "mid", "tile")), (lambda: potts_model([pal]), ("mid",)),
              (lambda: circles_model(), ("top", "mid", "tile")), (lambda: circles_model(INF, [cdisc]), ("mid", "tile")),
              (lambda: toy([[0, 1], [2, 0]]), ("c",))]
    for build, homes in builds:
        for home in homes:
            for K in (None, 2, 4):
                (m1, c1), (m2, c2) = build(), build()
                if K and m1.compile(home).cert[4]:
                    continue
                S1, S2 = Sampler(m1, home, K=K, seed=3), Sampler(m2, home, K=K, seed=3, adm=True)
                assert S1.init() == S2.init() and np.array_equal(S1.dormant, S2.dormant)
                c1, c2 = ((c1,), (c2,)) if hasattr(c1, "grid") else (c1, c2)
                for a, b in zip(c1, c2):
                    assert np.array_equal(a.grid, b.grid)
                S1.sweep(4, seed=11)
                S2.sweep(4, seed=11)
                for a, b in zip(c1, c2):
                    assert np.array_equal(a.grid, b.grid), (home, K, a.name)
    for K in (None, 8):
        (_, _, S1), (_, F2, S2) = _biome_obj(K, 4, False), _biome_obj(K, 4, True)
        S1.init(), S2.init()
        S1.sweep(10, seed=7), S2.sweep(10, seed=7)
        assert np.array_equal(S1.hc.grid, S2.hc.grid), K
        assert len(S2.adm_ptr) - 1 <= 8 and len(S2.sib_rows) > 0


def test_adm_lists_and_dormancy():
    C, F, S = _biome_obj(8, 2, True)
    frac = S.init()
    ok = S.admissible()
    for y in range(ok.shape[0]):
        for x in range(ok.shape[1]):
            a = S.adm_id[y, x]
            assert np.array_equal(S.adm_idx[S.adm_ptr[a]:S.adm_ptr[a + 1]], np.flatnonzero(ok[y, x]))
            assert S.candidates(y, x)[0] == S.hc.grid[y, x] and ok[y, x][S.candidates(y, x)[1:]].all()
    ln = np.diff(S.adm_ptr)[S.adm_id]
    assert frac > 0 and np.array_equal(S.dormant, ln == 1) and np.array_equal(S.dormant, C.dormant_of(F.allow))
    assert np.unique(S.adm_id).size == len(S.adm_ptr) - 1 <= C.P   # one list per biome value at most
    g0 = S.hc.grid.copy()
    S.sweep(20)
    assert np.array_equal(S.hc.grid[S.dormant], g0[S.dormant]) and not np.array_equal(S.hc.grid, g0)
    assert ok[np.arange(ok.shape[0])[:, None], np.arange(ok.shape[1])[None, :], S.hc.grid].all()


def _biome_monitor(K, adm, seed, n=6000, nb=30):
    C, F, S = _biome_obj(K, 4, adm)
    S.init()
    S.rng = np.random.default_rng(seed)
    S.sweep(100)
    rec = np.zeros((n, C.NF + 3))
    for i in range(n):
        S.sweep(1)
        g = S.hc.grid
        rec[i, :C.NF + 1] = np.bincount(C.FAM[g].ravel(), minlength=C.NF + 1) / g.size
        rec[i, C.NF + 1:] = (g % 2).mean(), S.energy()[0]
    b = rec.reshape(nb, -1, rec.shape[1]).mean(axis=1)
    return b.mean(axis=0), b.std(axis=0, ddof=1) / np.sqrt(nb)


def test_adm_long_run_biome():
    for K in (None, 8):
        m0, s0 = _biome_monitor(K, False, 21)
        m1, s1 = _biome_monitor(K, True, 22)
        z = np.abs(m0 - m1) / np.sqrt(s0 ** 2 + s1 ** 2 + 1e-12)
        assert (s0[1:] > 0).sum() >= 3 and z.max() < 4.0, (K, m0, m1, z)
