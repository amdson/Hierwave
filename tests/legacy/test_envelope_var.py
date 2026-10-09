import itertools

import numpy as np
import pytest

from castlegen.legacy import hier, texsyn, tileset
from castlegen.legacy.envelope_var import EnvelopePromise, InfeasibleError, ffbs, window_costs
from castlegen.legacy.promise import CoordPromiseCoupling, Tables
from castlegen.legacy.quantities import envelope as EN

G = 4
LG = EN.lang()


@pytest.fixture(scope="module")
def ts():
    return tileset.load("cliffs")


def heightmap(ts, H, n, rng):
    solid = np.flatnonzero(ts.solid)
    air = int(np.flatnonzero(~ts.solid)[0])
    rows = np.arange(n)[:, None]
    return np.where(rows >= n - np.asarray(H)[None, :], rng.choice(solid, (n, n)), air).astype(np.int32)


def random_H(rng, n, kind):
    if kind == "walk":
        H = np.cumsum(rng.integers(-3, 4, n)) + n // 2
    elif kind == "steps":
        H = np.repeat(rng.integers(G, n + 1, n // 4), 4)
    elif kind == "mountain":                      # low plain with one tall narrow peak
        H = np.full(n, G + 2)
        c = rng.integers(0, n - 6)
        H[c:c + 6] = n - 3
    elif kind == "valley":                        # high plateau cut by a deep gap
        H = np.full(n, n - 5)
        c = rng.integers(0, n - 8)
        H[c:c + 8] = G
    else:
        H = rng.integers(G, n + 1, n)
    return np.clip(H, G, n)


def random_tables(rng, hs, scale=1.0):
    V = LG.V
    tab = Tables.flat(V, hs)
    for h in hs:
        tab.u[h] = scale * rng.normal(size=V)
        tab.pair[h] = scale * rng.normal(size=(2, V, V))
        tab.parent[h] = scale * rng.normal(size=(4, V, V))
    return tab


def run(v, n, seed):
    ctx = hier.Ctx(seed)
    hier.run([v], [], n, ctx)
    return {lv.h: lv.vars[v.name] for lv in ctx.levels if v.name in lv.vars}


def check_consistent(v, grids):
    hs = sorted(grids)
    for h in hs:
        par = grids.get(2 * h)
        assert LG.consistent(grids[h], par, v.G, h), f"level {h} inconsistent"
        assert np.isfinite(v.level_energy(grids[h], par, h))


# ------------------------------------------------------------------ FFBS
def _brute(unary, trans, T):
    N, R, V = unary.shape
    states = np.array(list(itertools.product(range(V), repeat=R)))
    E = unary[0][np.arange(R), states].sum(1)
    for y in range(R - 1):
        E = E + trans[y][states[:, y], states[:, y + 1]]
    return states, E


@pytest.mark.parametrize("T", [1.0, 0.5, 0.0])
def test_ffbs_matches_brute_force(T):
    rng = np.random.default_rng(0)
    R, V, N = 4, 5, 40000
    un = rng.normal(size=(R, V))
    un[rng.random((R, V)) < 0.2] = np.inf
    trans = [np.where(rng.random((V, V)) < 0.3, np.inf, rng.normal(size=(V, V))) for _ in range(R - 1)]
    unary = np.broadcast_to(un, (N, R, V)).copy()
    g = hier.gumbel(rng.random((N, R, V)).astype(np.float32).clip(1e-7, 1 - 1e-7))
    out = ffbs(unary, lambda y: trans[y][None], lambda y, prev: trans[y][prev], T, g)
    states, E = _brute(unary[:1], trans, 1.0)
    fin = np.isfinite(E)
    assert fin.any()
    if T == 0:
        assert np.array_equal(out[0], states[np.argmin(E)])
        assert (out == out[0]).all()
        return
    p = np.exp(-(E[fin] - E[fin].min()) / T)
    p /= p.sum()
    code = (out * V ** np.arange(R)[::-1]).sum(1)
    scode = (states[fin] * V ** np.arange(R)[::-1]).sum(1)
    emp = np.array([(code == c).mean() for c in scode])
    assert emp.sum() == pytest.approx(1.0)                 # never an infeasible state
    assert 0.5 * np.abs(emp - p).sum() < 0.03


def test_ffbs_empty_set_raises():
    un = np.zeros((1, 2, 3))
    trans = np.full((3, 3), np.inf)
    with pytest.raises(InfeasibleError):
        ffbs(un, lambda y: trans[None], lambda y, prev: trans[prev], 1.0, np.zeros((1, 2, 3)))


def test_level_conditional_is_exact(ts):
    """Everything clamped except two stacked children of one parent column:
    one smoothing sweep draws them from the exact level conditional
    (brute force over all 100^2 pairs with level_energy)."""
    rng = np.random.default_rng(3)
    n, K = 64, 16
    tiles = heightmap(ts, random_H(rng, n, "walk"), n, rng)
    vals, _ = EN.pyramid(ts, tiles, K, G, 32)
    free = [(0, 0), (1, 0)]
    clamps = {(32, y, x): int(vals[32][y, x]) for y, x in np.ndindex(2, 2)}
    clamps.update({(16, y, x): int(vals[16][y, x]) for y, x in np.ndindex(4, 4) if (y, x) not in free})
    tab = random_tables(rng, [16, 32], scale=1.0)
    v = EnvelopePromise(K=K, h_top=32, tables=tab, T=1.0, sweeps=1, top_sweeps=0, ground=G, clamps=clamps)
    base = vals[16].copy()
    E = np.full((LG.V, LG.V), np.inf)
    for a, b in itertools.product(range(LG.V), repeat=2):
        base[0, 0], base[1, 0] = a, b
        E[a, b] = v.level_energy(base, vals[32], 16)
    p = np.exp(-(E - E[np.isfinite(E)].min()))
    p /= p.sum()
    cnt = np.zeros((LG.V, LG.V))
    S = 1500
    for seed in range(S):
        g = run(v, n, seed)[16]
        cnt[g[0, 0], g[1, 0]] += 1
    assert cnt[~np.isfinite(E)].sum() == 0
    assert 0.5 * np.abs(cnt / S - p).sum() < 0.08, (np.isfinite(E).sum(), 0.5 * np.abs(cnt / S - p).sum())


# -------------------------------------------------------------- whole runs
@pytest.mark.parametrize("T", [1.0, 0.0])
@pytest.mark.parametrize("ground", [0, G])
@pytest.mark.parametrize("n,K,h_top", [(128, 16, 64), (64, 8, 64)])
def test_runs_are_consistent(T, ground, n, K, h_top):
    rng = np.random.default_rng(7)
    hs = [K * 2 ** j for j in range(5)]
    for tab in (None, random_tables(rng, hs, 2.0)):
        v = EnvelopePromise(K=K, h_top=h_top, tables=tab, T=T, sweeps=2, top_sweeps=3, ground=ground)
        grids = run(v, n, seed=1)
        assert sorted(grids) == [h for h in hs if h <= h_top]
        check_consistent(v, grids)


def test_top_level_is_whole_map():
    v = EnvelopePromise(K=16, h_top=64, T=1.0, ground=G)
    check_consistent(v, run(v, 64, seed=2))       # 1 x 1 top level (a single-cell chain)


def test_determinism():
    rng = np.random.default_rng(1)
    tab = random_tables(rng, [16, 32, 64])
    a = run(EnvelopePromise(K=16, h_top=64, tables=tab, ground=G), 128, seed=5)
    b = run(EnvelopePromise(K=16, h_top=64, tables=tab, ground=G), 128, seed=5)
    c = run(EnvelopePromise(K=16, h_top=64, tables=tab, ground=G), 128, seed=6)
    assert all(np.array_equal(a[h], b[h]) for h in a)
    assert not all(np.array_equal(a[h], c[h]) for h in a)


def test_sampler_uses_tables():
    tab = Tables.flat(LG.V, [16, 32, 64])
    target = int(LG.encode([LG.F, LG.F], [LG.F, LG.F]))          # full blocks are always feasible
    for h in (16, 32, 64):
        tab.u[h] = np.full(LG.V, 6.0)
        tab.u[h][target] = 0.0
    g = run(EnvelopePromise(K=16, h_top=64, tables=tab, T=1.0, ground=G), 128, seed=0)[16]
    assert (g == target).mean() > 0.6


# ------------------------------------------------------------------- clamps
def _clamp_battery(ts):
    rng = np.random.default_rng(11)
    n, K = 128, 16
    out = []
    for kind in ("mountain", "valley", "walk", "steps", "iid"):
        tiles = heightmap(ts, random_H(rng, n, kind), n, rng)
        vals, _ = EN.pyramid(ts, tiles, K, G, 64)
        for levels_, frac in (((16,), 0.15), ((32,), 0.3), ((16, 32), 0.1), ((64,), 0.5)):
            cl = {}
            for h in levels_:
                R = n // h
                for y, x in np.ndindex(R, R):
                    if rng.random() < frac:
                        cl[(h, y, x)] = int(vals[h][y, x])
            out.append(cl)
    # a mountain top high up: a single full block in the top row of level K
    out.append({(16, 0, 3): int(LG.encode([LG.F, LG.F], [LG.F, LG.F]))})
    # a valley: an empty block right above the ground row
    out.append({(16, 6, 5): 0})
    # random single clamps, each value feasible on its own
    for _ in range(12):
        h = int(rng.choice([16, 32]))
        y, x = rng.integers(0, n // h, 2)
        ok = LG.valid(np.arange(LG.V), y == n // h - 1, G, h)
        out.append({(h, int(y), int(x)): int(rng.choice(np.flatnonzero(ok)))})
    return out


def test_clamps_are_honoured_and_beta_avoids_dead_ends(ts):
    rng = np.random.default_rng(2)
    tab = random_tables(rng, [16, 32, 64])
    for i, cl in enumerate(_clamp_battery(ts)):
        v = EnvelopePromise(K=16, h_top=64, tables=tab, T=1.0 if i % 2 else 0.0, sweeps=1, top_sweeps=2,
                            ground=G, clamps=cl)
        grids = run(v, 128, seed=i)
        for (h, y, x), val in cl.items():
            assert grids[h][y, x] == val
        check_consistent(v, grids)


def test_infeasible_clamps_raise():
    up = int(LG.encode([0, 0], [1, 1]))           # some matter, not full
    lo = int(LG.encode([0, 0], [1, 1]))           # not full below it: the seam rule fails
    # inside one parent: beta finds F empty before sampling
    v = EnvelopePromise(K=16, h_top=64, ground=0, clamps={(16, 0, 0): up, (16, 1, 0): lo})
    with pytest.raises(InfeasibleError, match="clamps infeasible"):
        run(v, 64, 0)
    # across two parents: beta cannot see it, the level-K column chain raises
    v = EnvelopePromise(K=16, h_top=64, ground=0, clamps={(16, 1, 0): up, (16, 2, 0): lo})
    with pytest.raises(InfeasibleError):
        run(v, 64, 0)
    # an empty block on the ground row is invalid
    v = EnvelopePromise(K=16, h_top=64, ground=G, clamps={(16, 3, 1): 0})
    with pytest.raises(InfeasibleError):
        run(v, 64, 0)


# --------------------------------------------------------------- chi
def _brute_window(s, cy, cx, h, lg):
    m = s.shape[0]
    rows, cols = (np.arange(h) + cy - h // 2) % m, (np.arange(h) + cx - h // 2) % m
    w = s[np.ix_(rows, cols)]
    t = np.array([next((i for i in range(h) if not w[h - 1 - i, c]), h) for c in range(h)])
    viol = (w.sum(0) > t).sum()
    wi = h // lg.I
    lo = lg.bins([t[i * wi:(i + 1) * wi].min() for i in range(lg.I)], h)
    hi = lg.bins([t[i * wi:(i + 1) * wi].max() for i in range(lg.I)], h)
    return lo, hi, viol


def test_window_costs_match_brute_force(ts):
    rng = np.random.default_rng(4)
    m = 32
    Es = [heightmap(ts, random_H(rng, m, "walk"), m, rng), rng.integers(0, ts.n_sig, (m, m)).astype(np.int32)]
    for E in Es:
        s = ts.solid[E]
        for h in (8, 16, 32):
            c = window_costs(ts, E, h, LG)
            assert c.shape == (m * m, LG.V) and c.dtype == np.float32
            for cy, cx in rng.integers(0, m, (25, 2)):
                lo, hi, viol = _brute_window(s, cy, cx, h, LG)
                want = np.abs(lo[None] - LG.LO).sum(1) + np.abs(hi[None] - LG.HI).sum(1) + min(viol, 4)
                assert np.allclose(c[cy * m + cx], want)
                assert c[cy * m + cx].argmin() == LG.encode(lo, hi)


def test_coupled_run_with_coordinates(ts):
    """Plugs in next to texsyn.CoordVar and CoordPromiseCoupling exactly
    like the pipeline's promise variables."""
    rng = np.random.default_rng(5)
    m = 32
    E = heightmap(ts, random_H(rng, m, "walk"), m, rng)
    an = texsyn.Analysis(ts, E, n_pca=8)
    coord = texsyn.CoordVar(an, [1, 1, 0, 0, 0, 0], kappa=4.0)
    prom = EnvelopePromise(K=8, h_top=32, T=1.0, sweeps=2, top_sweeps=3, ground=G)
    cost = {h: prom.window_costs(ts, E, h) for h in (8, 16, 32)}
    cp = CoordPromiseCoupling(coord, prom, cost, 2.0, an.m)
    ctx = hier.Ctx(3)
    S = hier.run([prom, coord], [cp], 64, ctx).vars["coord"]
    grids = {lv.h: lv.vars[prom.name] for lv in ctx.levels if prom.name in lv.vars}
    check_consistent(prom, grids)
    tiles = E[S[..., 0], S[..., 1]]
    sat = prom.satisfaction(ts, tiles, ctx.levels)
    assert set(sat) == {8, 16, 32} and all(0.0 <= f <= 1.0 for f in sat.values())
    # chi enters the promise side: at T = 0 with flat tables the top level (2 x 2, columns independent)
    # is the exact constrained argmin of the chi energy of each column
    prom2 = EnvelopePromise(K=8, h_top=32, T=0.0, sweeps=2, top_sweeps=3, ground=G, lam=100.0)
    ctx2 = hier.Ctx(3)
    hier.run([prom2, coord], [CoordPromiseCoupling(coord, prom2, cost, 1.0, an.m)], 64, ctx2)
    lv = next(l for l in ctx2.levels if l.h == 32)
    S32, P32 = lv.vars["coord"], lv.vars[prom2.name]
    F = prom2.feasible(64)[0][32]
    for x in range(2):
        c0, c1 = (cost[32][S32[y, x, 0] * m + S32[y, x, 1]].astype(float) for y in (0, 1))
        tot = c0[:, None] + c1[None, :]
        tot[~(LG.COMPAT & F[0, x][:, None] & F[1, x][None, :])] = np.inf
        assert c0[P32[0, x]] + c1[P32[1, x]] == pytest.approx(tot.min())


def test_beta_stress_feasible_clamp_sets(ts):
    """Random clamp sets read off valid maps (always jointly feasible), at
    random levels and densities: never an empty set mid-run."""
    rng = np.random.default_rng(0)
    for trial in range(40):
        n, K = [(128, 16), (128, 8)][trial % 2]
        kind = ["mountain", "valley", "walk", "steps", "iid"][trial % 5]
        tiles = heightmap(ts, random_H(rng, n, kind), n, rng)
        vals, _ = EN.pyramid(ts, tiles, K, G, 64)
        frac = rng.choice([0.05, 0.2, 0.5])
        hs = [h for h in vals if rng.random() < 0.6] or [K]
        cl = {(h, y, x): int(vals[h][y, x]) for h in hs for y, x in np.ndindex(*vals[h].shape) if rng.random() < frac}
        v = EnvelopePromise(K=K, h_top=64, tables=random_tables(rng, [K * 2 ** j for j in range(5)]),
                            T=float(trial % 3 > 0), sweeps=1, top_sweeps=2, ground=G, clamps=cl)
        grids = run(v, n, trial)
        assert all(grids[h][y, x] == val for (h, y, x), val in cl.items())
        check_consistent(v, grids)


def test_ffbs_ignores_infinite_gumbel():
    """hier.noise can return exactly 1.0 (float32 rounding), i.e. a +inf
    Gumbel; an infeasible value must still never be drawn."""
    un = np.array([[[np.inf, 0.0, 0.0]]])
    g = np.array([[[np.inf, 0.0, 0.0]]])
    assert ffbs(un, None, None, 1.0, g)[0, 0] != 0
