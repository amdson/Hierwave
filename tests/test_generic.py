import numpy as np
import pytest

from castlegen import generic as GN
from castlegen import refheights as RH
from castlegen import tileset


@pytest.fixture(scope="module")
def setup():
    ts = tileset.load("cliffs")
    E = np.load("castlegen/exemplars/cliffs_big.npy")[::4, ::4]             # 64 x 64, sky on top, ground below
    return ts, E, GN.AverageFill(ts, 16)


def _random_P(con, R, C, seed):
    tab = {h: dict(u=np.zeros(con.V), pair=np.zeros((2, con.V, con.V)), parent=np.zeros((4, con.V, con.V)))
           for h in (16, 32)}
    tgt = [np.zeros((R, C), np.int64), np.zeros((R // 2, C // 2), np.int64)]
    full = con.encode(np.array([con.w, con.w]))
    ps = GN.PromiseSampler(con, tab, [16, 32], tgt, np.zeros(C, np.int64), np.full(C, full), lam=0.0, seed=seed)
    return ps.run(5)


def test_witness_realises_promises(setup):
    ts, E, con = setup
    rule = GN.support_rule(ts)
    for seed in range(5):
        P = _random_P(con, 4, 4, seed)
        x = con.witness(P, np.ones(64, np.int32), np.zeros(64, np.int32))
        leaf = con.leaf(x.reshape(4, 16, 4, 16).transpose(0, 2, 1, 3))
        assert (leaf == P).all()
        col = np.concatenate([np.zeros((1, 64), np.int32), x, np.ones((1, 64), np.int32)])
        assert rule[1][col[:-1], col[1:]].all()


def test_guided_witness_is_feasible_and_follows_the_hint(setup):
    ts, E, con = setup
    rule = GN.support_rule(ts)
    fill, empty = E[-1], E[0]
    ch = lambda x, P: GN.TileChain(ts, x, E, [con], [P], rule, {8: 1.0}, seed=0)      # asserts validity
    for seed in range(3):
        rng = np.random.default_rng(seed)
        P = _random_P(con, 4, 4, seed)
        hint = rng.integers(0, ts.n_sig, (64, 64)).astype(np.int32)           # any map, even an invalid one
        ch(GN.guided_witness(con, P, hint, rule, fill, empty, rng, sweeps=1), P)
    H = np.clip(32 + np.cumsum(np.random.default_rng(5).integers(-3, 4, 64)), 1, 64)   # a feasible terrain hint
    hint = RH.from_heights(ts, H, 64, 1, "stone", "stone")
    P = con.leaf(hint.reshape(4, 16, 4, 16).transpose(0, 2, 1, 3))
    x = GN.guided_witness(con, P, hint, rule, fill, empty, np.random.default_rng(0), tau=0.1, sweeps=6)
    ch(x, P)
    assert (x == hint).mean() > 0.99


def test_compat_is_necessary(setup):
    """The leaf grid of every supported map (a heightmap) is compatible."""
    ts, E, con = setup
    rng = np.random.default_rng(0)
    for _ in range(20):
        H = rng.integers(0, 65, 64)
        x = RH.from_heights(ts, np.maximum(H, 1), 64, 1, "stone", "stone")
        P = con.leaf(x.reshape(4, 16, 4, 16).transpose(0, 2, 1, 3))
        assert con.compat[1][P[:-1], P[1:]].all()


def test_tile_move_energy_exact(setup):
    ts, E, con = setup
    P = _random_P(con, 4, 4, 1)
    x = con.witness(P, E[-1], E[0])
    ch = GN.TileChain(ts, x, E, [con], [P], GN.support_rule(ts), {1: 0.3, 2: 1.0, 8: 2.0}, seed=0,
                      local=GN.fit_local(E, ts.n_sig))
    for _ in range(2):
        ch.sweep()
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        y, xx = rng.integers(0, 64, 2)
        ok = ch._allowed(np.array([y]), np.array([xx]))[0]
        for s in np.flatnonzero(ok):
            if s == ch.t[y, xx]:
                continue
            pred = ch.dE(np.array([y]), np.array([xx]))[0, s]
            e0 = ch.energy_ex() + ch.energy_local()
            old = ch.t[y, xx]
            ch.t[y, xx] = s
            ch._refresh()
            assert ch.energy_ex() + ch.energy_local() - e0 == pytest.approx(pred, rel=1e-8, abs=1e-8)
            assert ch.valid()
            ch.t[y, xx] = old
            ch._refresh()
            checked += 1
    assert checked > 0


def test_chain_keeps_constraints_and_bookkeeping(setup):
    ts, E, con = setup
    P = _random_P(con, 4, 4, 2)
    ch = GN.TileChain(ts, con.witness(P, E[-1], E[0]), E, [con], [P], GN.support_rule(ts), {4: 1.0, 8: 2.0},
                      s_every=2, seed=0)
    x0 = ch.t.copy()
    for _ in range(4):
        ch.sweep()
        assert ch.valid()
    assert (ch.t != x0).any()
    B = {h: c["B"].copy() for h, c in ch.sc.items()}
    ch._refresh()
    for h, c in ch.sc.items():
        assert np.abs(c["B"] - B[h]).max() < 1e-9


def test_mh_S_matches_exact_conditional(setup):
    """With x fixed, the MH update of S_h has the exact heat-bath's stationary
    law: compare the mean patch energy per sample over many steps."""
    ts, E, con = setup
    P = _random_P(con, 4, 4, 3)
    E2 = E[::2, ::2]                                                        # small bank
    kw = dict(seed=0, T=1.0)
    x = con.witness(P, E2[-1, np.arange(64) % 32], E2[0, np.arange(64) % 32])
    ex = GN.TileChain(ts, x, E2, [con], [P], GN.support_rule(ts), {4: 0.05}, **kw)
    mh = GN.TileChain(ts, x, E2, [con], [P], GN.support_rule(ts), {4: 0.05}, exact_budget=0, mh_steps=1, eps=0.5, **kw)
    assert ex.sc[4]["exact"] and not mh.sc[4]["exact"]
    c = ex.sc[4]
    N = ex._patches(4).reshape(-1, c["NE"].shape[1])
    e_all = c["w"] * ((N[:, None] - c["NE"][None]) ** 2).sum(-1)            # (R C, bank)
    p = np.exp(-(e_all - e_all.min(1, keepdims=True)))
    p /= p.sum(1, keepdims=True)
    want = (p * e_all).sum(1)
    got = np.zeros(len(N))
    steps = 3000
    for s in range(steps + 200):
        mh.resample_S(4)
        if s >= 200:
            got += e_all[np.arange(len(N)), mh.sc[4]["S"].ravel()]
    got /= steps
    assert np.abs(got - want).mean() < 0.05 * np.abs(want).mean() + 1e-3


def test_fit_local_reproduces_exemplar_conditionals(setup):
    """The fitted local conditionals put most mass on the exemplar's own tile,
    and almost none on a tile the exemplar never uses."""
    ts, E, con = setup
    loc = GN.fit_local(E, ts.n_sig)
    col = np.concatenate([E[:1], E, E[-1:]], 0)
    lg = -(loc["u"][None, None] + loc["Eh"][np.roll(E, 1, 1)] + np.moveaxis(loc["Eh"][:, np.roll(E, -1, 1)], 0, -1)
           + loc["Ev"][col[:-2]] + np.moveaxis(loc["Ev"][:, col[2:]], 0, -1))
    assert (lg.argmax(-1) == E).mean() > 0.9
    unused = np.setdiff1d(np.arange(ts.n_sig), np.unique(E))
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    assert len(unused) and p[..., unused].sum(-1).mean() < 1e-3 and p[..., unused].sum(-1).max() < 0.02
