import itertools

import numpy as np
import pytest

from castlegen.legacy import refheights as RH
from castlegen.legacy import tileset
from castlegen.legacy.quantities import envelope as EN


@pytest.fixture(scope="module")
def ts():
    return tileset.load("cliffs")


def names(ts, *ks):
    return [RH._sig(ts, k) for k in ks]


def forbid_all_but(ts, keep):
    f = np.ones(ts.n_sig, bool)
    f[names(ts, *keep)] = False
    return f


# ------------------------------------------------------------- exact enumeration
N, GR = 4, 1                         # 4 x 4 torus, bottom row band
KEEP = ("air", "stone", "soil")
T_, MU = 1.0, -0.5


@pytest.fixture(scope="module")
def exact(ts):
    """Every valid 4x4 map over air/stone/soil with its p_G probability."""
    air, stone, soil = names(ts, *KEEP)
    free = (N - GR) * N
    alph = np.array([air, stone, soil])
    idx = np.array(list(itertools.product(range(3), repeat=free)), np.int8)
    maps = np.empty((len(idx), N, N), np.int32)
    maps[:, :N - GR] = alph[idx].reshape(-1, N - GR, N)
    maps[:, N - GR:] = stone
    sol = ts.solid[maps]
    below = np.roll(sol, -1, 1)
    below[:, N - GR - 1] = True
    ok = ~(sol[:, :N - GR] & ~below[:, :N - GR]).any((1, 2))
    maps = maps[ok]
    Eh, Ev, logz = ts.np_tables["Eh"], ts.np_tables["Ev"], ts.np_tables["logz"]
    pair = Eh[maps, np.roll(maps, -1, 2)].sum((1, 2)) + Ev[maps, np.roll(maps, -1, 1)].sum((1, 2))
    lw = -pair / T_ + logz[maps].sum((1, 2)) + MU * ts.solid[maps].sum((1, 2))
    p = np.exp(lw - lw.max())
    s = ts.solid[maps] | (np.arange(N) >= N - GR)[None, :, None]
    H = np.cumprod(s[:, ::-1], 1).sum(1)
    return maps, p / p.sum(), H


def hkey(H):
    return (np.asarray(H) - GR) @ (N ** np.arange(N))


def run_marginals(ts, chain, sweeps, columns=True):
    Hc = np.zeros(N ** N)
    mat = np.zeros((N, N, ts.n_sig))
    for _ in range(sweeps):
        chain.sweep(columns=columns)
        Hc[hkey(RH.heights(ts, chain.t, GR))] += 1
        mat[np.arange(N)[:, None], np.arange(N)[None, :], chain.t] += 1
    return Hc / sweeps, mat / sweeps


def exact_marginals(ts, maps, p, H, mask=None):
    if mask is not None:
        maps, p, H = maps[mask], p[mask] / p[mask].sum(), H[mask]
    Hp = np.bincount(hkey(H), weights=p, minlength=N ** N)
    mat = np.zeros((N, N, ts.n_sig))
    for s in range(ts.n_sig):
        mat[..., s] = ((maps == s) * p[:, None, None]).sum(0)
    return Hp, mat


def test_log_weight_matches_enumeration(ts, exact):
    maps, p, _ = exact
    i, j = 0, len(maps) // 2
    d = RH.log_weight(ts, maps[i], T_, MU) - RH.log_weight(ts, maps[j], T_, MU)
    assert np.isclose(d, np.log(p[i] / p[j]))


@pytest.mark.parametrize("columns", [False, True])
def test_heat_bath_is_exact(ts, exact, columns):
    maps, p, H = exact
    Hp, mat_p = exact_marginals(ts, maps, p, H)
    ch = RH.Chain(ts, RH.band_map(ts, N, GR), GR, T_, MU, forbid_all_but(ts, KEEP), seed=3, D=2)
    ch.run(200, columns=columns)
    Hq, mat_q = run_marginals(ts, ch, 20000 if columns else 40000, columns)
    # noise at these lengths: TV ~0.02-0.04; a sampler at mu - 0.05 gives TV ~0.1,
    # dropping the reverse proposal probability of the column move TV ~0.5
    assert 0.5 * np.abs(Hp - Hq).sum() < 0.06                 # 256 height vectors
    assert np.abs(mat_p - mat_q).max() < 0.04                 # per-cell signature marginals
    # per-column height marginal
    hp = np.array([np.bincount((H[:, c] - GR), weights=p, minlength=N) for c in range(N)])
    Hs = np.array(list(itertools.product(range(N), repeat=N)))[:, ::-1]      # digit c of the key
    hq = np.array([np.bincount(Hs[:, c], weights=Hq, minlength=N) for c in range(N)])
    assert np.abs(hp - hq).max() < 0.04


def test_conditioned_is_exact(ts, exact):
    maps, p, H = exact
    lg = EN.lang()
    vals = RH.block_values(H, N, 2, lg)                        # (M, 2, 2)
    target = vals[np.argmax(p), 1, 0]
    clamps = {(2, 1, 0): int(target)}
    Hp, mat_p = exact_marginals(ts, maps, p, H, vals[:, 1, 0] == target)
    ch = RH.sample_conditioned(ts, clamps, n=N, ground=GR, T=T_, mu=MU, sweeps=100,
                               forbid=forbid_all_but(ts, KEEP), seed=5, return_chain=True)
    ch.D = 2
    Hq, mat_q = run_marginals(ts, ch, 20000)
    assert Hq[Hp == 0].sum() == 0                             # never leaves the clamped set
    assert 0.5 * np.abs(Hp - Hq).sum() < 0.06
    assert np.abs(mat_p - mat_q).max() < 0.04


# ----------------------------------------------------------------- larger maps
def test_valid_after_every_sweep_and_block_values(ts):
    for init, anneal in (("band", 0), ("flat", 5)):
        ch = RH.Chain(ts, RH.band_map(ts, 32, 4, H0=4 if init == "band" else 20), 4, T=0.9, mu=0.1, seed=1)
        ch.run(30, T_hot=2.0, anneal=anneal, check=True)
        vals, ok = EN.pyramid(ts, ch.t, 8, 4, 32)
        H = RH.heights(ts, ch.t, 4)
        for h, v in vals.items():
            assert ok[h].all()
            assert np.array_equal(RH.block_values(H, 32, h), v)


def test_conditioned_sampler_honours_clamps(ts):
    rng = np.random.default_rng(0)
    n, G = 32, 4
    H0 = np.clip(np.cumsum(rng.integers(-2, 3, n)) + 14, G, n)
    src = RH.from_heights(ts, H0, n, G)
    vals, _ = EN.pyramid(ts, src, 8, G, 32)
    clamps = {(8, 1, 2): int(vals[8][1, 2]), (8, 2, 0): int(vals[8][2, 0]), (16, 1, 1): int(vals[16][1, 1])}
    ch = RH.sample_conditioned(ts, clamps, n=n, ground=G, T=0.9, sweeps=0, seed=2, return_chain=True)
    assert ch.clamps_ok()
    moved = 0
    for _ in range(40):
        before = RH.heights(ts, ch.t, G)
        ch.sweep()
        assert RH.violations(ts, ch.t, G) == 0 and ch.clamps_ok()
        moved += int((RH.heights(ts, ch.t, G) != before).sum())
    assert moved > 0
    v2, _ = EN.pyramid(ts, ch.t, 8, G, 32)
    for (h, by, bx), v in clamps.items():
        assert v2[h][by, bx] == v


def test_stats_and_compare(ts):
    maps = [RH.sample(ts, 32, s, 0.9, 4, sweeps=20, mu=0.1) for s in range(3)]
    a = RH.stats_many(ts, maps, 4, K=8, h_top=32)
    b = RH.stats_many(ts, [RH.band_map(ts, 32, 4, H0=10)] * 2, 4, K=8, h_top=32)
    c = RH.compare(a, a)
    assert c["H_hist_tv"] == 0 and c["solid"]["diff"] == 0 and all(v == 0 for v in c["promise_tv"].values())
    d = RH.compare(a, b)
    assert d["material_tv"] > 0 and set(d["promise_tv"]) == {8, 16, 32}
    assert a[0]["violations"] == 0 and a[0]["slope_hist"].sum() == 32
    grids = RH.mine(ts, maps, 8, 4, 32)
    tab = RH.tables(ts, maps, K=8, ground=4, h_top=32)
    assert set(grids) == {8, 16, 32} and tab.V == EN.lang().V
