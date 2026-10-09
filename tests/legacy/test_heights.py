import itertools
import time
from collections import Counter, deque

import numpy as np
import pytest

from castlegen.legacy import exemplar as ex
from castlegen.legacy import heights as HT
from castlegen.legacy import tileset
from castlegen.legacy.quantities import envelope as EN
from castlegen.legacy.quantities import support as SP


@pytest.fixture(scope="module")
def ts():
    return tileset.load("cliffs")


def heightmap(ts, H, n, rng):
    """Valid map: column c solid (random solid material) on rows >= n - H[c]."""
    solid = np.flatnonzero(ts.solid)
    air = int(np.flatnonzero(~ts.solid)[0])
    rows = np.arange(n)[:, None]
    return np.where(rows >= n - np.asarray(H)[None, :], rng.choice(solid, (n, n)), air).astype(np.int32)


def random_H(rng, n, G, kind):
    if kind == "walk":
        H = np.cumsum(rng.integers(-3, 4, n)) + n // 2
    elif kind == "steps":
        H = np.repeat(rng.integers(G, n + 1, n // 4), 4)
    else:
        H = rng.integers(G, n + 1, n)
    return np.clip(H, G, n)


def level_grid(ts, x, K, G, lg=None):
    return EN.pyramid(ts, x, K, G, h_top=K, lg=lg)[0][K]


def texture(ts, kind, ref, rng):
    n = ref.shape[0]
    if kind == "random":
        return rng.integers(0, ts.n_sig - 1, (n, n)).astype(np.int32)           # never the gate
    if kind == "noisy":
        flip = rng.random((n, n)) < 0.15
        return np.where(flip, rng.integers(0, ts.n_sig - 1, (n, n)), ref).astype(np.int32)
    # contradicting: solid exactly where the reference is air, and vice versa
    air = int(np.flatnonzero(~ts.solid)[0])
    return np.where(ts.solid[ref], air, ex.sig_by_name(ts, "soil")).astype(np.int32)


def check_exact(ts, x, P, K, G):
    assert SP.support_violations(ts, x, G) == 0
    assert EN.satisfaction(ts, x, {K: P}, K, G)[K] == 1.0
    assert (x[-G:] == ex.sig_by_name(ts, "stone")).all()


# ----------------------------------------------------------------- language
def test_constraints_agree_with_language_consistency(ts):
    """Realisability (exact, via heights) == lg.consistent on mutated grids."""
    rng = np.random.default_rng(1)
    for I, Q in ((2, 2), (2, 4)):
        lg = EN.lang(I, Q)
        for _ in range(60):
            n, K, G = 32, int(rng.choice([8, 16])), int(rng.choice([1, 2, 4]))
            x = heightmap(ts, random_H(rng, n, G, rng.choice(["walk", "steps", "iid"])), n, rng)
            P = level_grid(ts, x, K, G, lg)
            HT.constraints(P, K, G, n, lg)                   # an abstraction is always realisable
            for _ in range(5):
                P2 = P.copy()
                P2[tuple(rng.integers(0, n // K, 2))] = rng.integers(lg.V)
                try:
                    HT.constraints(P2, K, G, n, lg)
                    real = True
                except ValueError:
                    real = False
                assert real == lg.consistent(P2, G=G, k=K)


def test_inconsistent_grid_raises(ts):
    rng = np.random.default_rng(2)
    n, K, G = 64, 16, 4
    x = heightmap(ts, random_H(rng, n, G, "walk"), n, rng)
    P = level_grid(ts, x, K, G).copy()
    lg = EN.lang()
    P[0, 0] = lg.encode([lg.F, lg.F], [lg.F, lg.F])             # full top block over whatever is below
    P[1, 0] = lg.encode([0, 0], [0, 0])                         # ... an empty block
    assert not lg.consistent(P, G=G, k=K)
    with pytest.raises(ValueError):
        HT.sample_tiles(ts, x, P, K, G)


# ------------------------------------------------------------- projection
def test_project_is_identity_on_a_valid_texture(ts):
    rng = np.random.default_rng(3)
    n, K, G = 64, 16, 4
    ref = heightmap(ts, random_H(rng, n, G, "walk"), n, rng)
    P = level_grid(ts, ref, K, G)
    x, st = HT.project(ts, ref, P, K, G)
    assert st["texture_cost"] == 0 and st["solidity_flips"] == 0
    assert np.array_equal(x[:-G], ref[:-G])
    check_exact(ts, x, P, K, G)


def test_projection_is_optimal_on_small_instances(ts):
    """The DP projection attains the brute-force minimum texture cost."""
    rng = np.random.default_rng(4)
    n, K, G = 8, 4, 1
    for _ in range(20):
        ref = heightmap(ts, random_H(rng, n, G, "iid"), n, rng)
        P = level_grid(ts, ref, K, G)
        X = rng.integers(0, ts.n_sig - 1, (n, n))
        cost = HT.texture_cost(ts, X, G)
        cons = HT.constraints(P, K, G, n, EN.lang())
        w = K // 2
        best = 0
        for j in range(n // w):                                   # intervals are independent
            cols = range(j * w, (j + 1) * w)
            best += min(sum(cost[c, h] for c, h in zip(cols, Hs))
                        for Hs in itertools.product(range(G, n + 1), repeat=w)
                        if HT.feasible(Hs, tuple(a[j:j + 1] for a in cons), w))
        x, st = HT.project(ts, X, P, K, G)
        assert st["texture_cost"] == best
        check_exact(ts, x, P, K, G)


# ------------------------------------------------------ exactness end to end
CASES = [(64, 16, 4), (64, 16, 8), (64, 8, 4), (128, 16, 8), (128, 8, 4)]


@pytest.mark.parametrize("n,K,G", CASES)
@pytest.mark.parametrize("tex", ["random", "noisy", "contradicting"])
def test_exact_after_project_and_every_sweep(ts, n, K, G, tex):
    rng = np.random.default_rng(hash((n, K, G, tex)) % 2**32)
    ref = heightmap(ts, random_H(rng, n, G, rng.choice(["walk", "steps", "iid"])), n, rng)
    P = level_grid(ts, ref, K, G)
    X = texture(ts, tex, ref, rng)
    x0, _ = HT.project(ts, X, P, K, G, seed=5)
    check_exact(ts, x0, P, K, G)
    seen = []

    def cb(s, x):
        check_exact(ts, x, P, K, G)
        seen.append(s)

    x, st = HT.sample_tiles(ts, X, P, K, G, seed=5, sweeps=4, callback=cb)
    assert seen == list(range(4))
    check_exact(ts, x, P, K, G)
    assert st["smooth"]["height"]["accepted"] > 0


def test_height_chain_mixes_at_w_tex_0(ts):
    rng = np.random.default_rng(6)
    n, K, G = 64, 16, 4
    ref = heightmap(ts, random_H(rng, n, G, "steps"), n, rng)
    P = level_grid(ts, ref, K, G)
    H0 = HT.heights_of(ts, ref, G)[0]
    x, st = HT.sample_tiles(ts, ref, P, K, G, seed=1, w_tex=0.0, sweeps=60)
    check_exact(ts, x, P, K, G)
    H1 = HT.heights_of(ts, x, G)[0]
    assert np.abs(H1 - H0).sum() > n                    # not frozen
    sw = st["smooth"]["swap"]
    assert sw["accepted"] > 0 and sw["witness_moves"] > 0


def test_deterministic(ts):
    rng = np.random.default_rng(7)
    n, K, G = 64, 8, 4
    ref = heightmap(ts, random_H(rng, n, G, "walk"), n, rng)
    P = level_grid(ts, ref, K, G)
    X = texture(ts, "random", ref, rng)
    a, _ = HT.sample_tiles(ts, X, P, K, G, seed=3, sweeps=3)
    b, _ = HT.sample_tiles(ts, X, P, K, G, seed=3, sweeps=3)
    c, _ = HT.sample_tiles(ts, X, P, K, G, seed=4, sweeps=3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# -------------------------------------------------------- chain correctness
def _moves(Hs, cons, w, swaps=True):
    out = []
    for c in range(w):
        for d in (-1, 1):
            H2 = list(Hs); H2[c] += d
            if HT.feasible(H2, cons, w):
                out.append(tuple(H2))
    if swaps:
        for c, c2 in itertools.combinations(range(w), 2):
            H2 = list(Hs); H2[c], H2[c2] = H2[c2], H2[c]
            out.append(tuple(H2))
    return out


def _connected(cons, w, top, swaps=True):
    states = [Hs for Hs in itertools.product(range(top + 1), repeat=w) if HT.feasible(Hs, cons, w)]
    seen, q = {states[0]}, deque([states[0]])
    while q:
        for s in _moves(q.popleft(), cons, w, swaps):
            if s not in seen:
                seen.add(s); q.append(s)
    return len(seen) == len(states)


def test_height_moves_are_irreducible_with_swaps():
    for w, top in ((2, 7), (3, 6), (4, 4)):
        for a0, a1, b0, b1 in itertools.product(range(top + 1), repeat=4):
            if not (a0 <= a1 and b0 <= b1 and a0 <= b1 and a1 <= b1 and b0 >= a0):
                continue
            if w == 1 and max(a0, b0) > min(a1, b1):
                continue
            cons = tuple(np.array([v]) for v in (a0, a1, b0, b1))
            assert _connected(cons, w, top), (w, a0, a1, b0, b1)
    # without swaps a pinned two-column interval is frozen
    cons = tuple(np.array([v]) for v in (2, 2, 6, 6))
    assert not _connected(cons, 2, 7, swaps=False)


def test_local_deltas_match_total_log_weight(ts):
    rng = np.random.default_rng(8)
    n, K, G = 32, 8, 4
    ref = heightmap(ts, random_H(rng, n, G, "iid"), n, rng)
    P = level_grid(ts, ref, K, G)
    X = texture(ts, "noisy", ref, rng)
    for torus in (True, False):
        ch = HT.Chain(ts, ref, P, K, G, X, seed=0, T=0.7, w_tex=0.8, torus=torus)
        lw = lambda: HT.log_weight(ts, ch.x, X, G, 0.7, 0.8, torus)
        for _ in range(30):
            c, c2 = rng.choice(n, 2, replace=False)
            before, tot0 = ch._cols_logw((c, c2)), lw()
            ch.x[:, [c, c2]] = ch.x[:, [c2, c]]
            assert np.isclose(ch._cols_logw((c, c2)) - before, lw() - tot0, atol=1e-3)  # float32 tables in pair_energy
            y, c = rng.integers(0, n - G), rng.integers(n)
            L, tot0, old = ch._cell_logits(y, c), lw(), ch.x[y, c]
            new = rng.integers(ts.n_sig)
            ch.x[y, c] = new
            assert np.isclose(L[new] - L[old], lw() - tot0, atol=1e-3)


TOY = {
    "name": "toy",
    "sockets": ["wall", "face"],
    "connects": [["face", "face"]],
    "wall": {"unary": 0.5, "tags": ["solid"]},
    "gate": {"socket": "face"},
    "kinds": [
        {"name": "air", "tags": ["sky"], "sides": {s: "face" for s in "NESW"}},
        {"name": "rock", "tags": ["solid"], "solid": True, "unary": 0.2, "sides": {s: "face" for s in "NESW"}},
    ],
    "rules": [
        {"a": "sky", "b": "sky", "when": "contact", "energy": -0.5},
        {"a": "solid", "b": "sky", "when": "beside", "energy": 0.4},
        {"a": "rock", "b": "rock", "when": "above", "energy": -0.6},
        {"a": "sky", "b": "rock", "when": "above", "energy": 0.3},
    ],
}


def test_stationary_distribution_matches_brute_force():
    """4x4 map, K = 4, I = 2 (two intervals of 2 columns), G = 1: the exact
    marginal of H (materials summed out, texture term on) vs the chain."""
    ts = tileset.compile_spec(TOY)
    n, K, G, T, w_tex = 4, 4, 1, 0.8, 0.5
    rng = np.random.default_rng(9)
    lg = EN.lang()
    rows = np.arange(n)[:, None]
    ref = np.where(rows >= n - np.array([1, 4, 2, 3])[None, :], 1, 0).astype(np.int32)
    P = level_grid(ts, ref, K, G)
    cons = HT.constraints(P, K, G, n, lg)
    X = rng.integers(0, 2, (n, n)); X[rng.random((n, n)) < 0.2] = ts.WALL
    solid_cls = np.flatnonzero(ts.solid & (np.arange(ts.n_sig) != ts.GATE))
    air = int(np.flatnonzero(~ts.solid)[0])
    stone = ex.sig_by_name(ts, "rock")
    exact = {}
    for Hs in itertools.product(range(G, n + 1), repeat=n):
        if not HT.feasible(Hs, cons, 2):
            continue
        cells = [(y, c) for c in range(n) for y in range(n - Hs[c], n - G)]
        tot = []
        for mats in itertools.product(solid_cls, repeat=len(cells)):
            x = np.full((n, n), air); x[-G:] = stone
            for (y, c), s in zip(cells, mats):
                x[y, c] = s
            eh, ev = ex.pair_energy(ts, x, True)
            tau = w_tex * ((x != X).astype(float) + (ts.solid[x] != ts.solid[X]))[:-G].sum()
            tot.append(-(eh.sum() + ev.sum()) / T + ts.np_tables["logz"][x].sum() - tau)
        exact[Hs] = np.logaddexp.reduce(tot)
    keys = list(exact)
    z = np.logaddexp.reduce(list(exact.values()))
    p = np.array([np.exp(exact[k] - z) for k in keys])
    assert len(keys) >= 6
    x0, _ = HT.project(ts, X, P, K, G, T=T, ground_tile="rock")
    ch = HT.Chain(ts, x0, P, K, G, X, seed=0, T=T, w_tex=w_tex)
    cnt = Counter()
    sweeps = 6000
    for s in range(sweeps):
        ch.sweep(hmoves=1, swaps=1)
        cnt[tuple(ch.H.tolist())] += 1
    q = np.array([cnt[k] / sweeps for k in keys])
    assert sum(cnt.values()) == sweeps and set(cnt) <= set(keys)
    tv = 0.5 * np.abs(p - q).sum()
    assert tv < 0.05, (tv, sorted(zip(keys, p.round(3), q.round(3))))
    assert ch.stats["swap"]["witness_moves"] > 0


# ------------------------------------------------------------------- speed
def test_speed_128(ts):
    rng = np.random.default_rng(10)
    n, K, G = 128, 16, 4
    ref = heightmap(ts, random_H(rng, n, G, "walk"), n, rng)
    P = level_grid(ts, ref, K, G)
    X = texture(ts, "random", ref, rng)
    t = time.perf_counter()
    x, st = HT.sample_tiles(ts, X, P, K, G, sweeps=10)
    assert time.perf_counter() - t < 5.0
    check_exact(ts, x, P, K, G)
