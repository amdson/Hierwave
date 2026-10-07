"""The Potts test's exact pieces: the collapsed mid move (column transfer
matrix) and top move against brute force, the induced mid-mid coupling at
large kappa, the forward model and oracle running, and the generic
kernel's energy against numpy."""
import itertools

import numpy as np

from castlegen.channels.potts import Forward, Oracle, Potts, dist


def _logsumexp(a):
    a = np.asarray(a)
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def _tile_energy(P, tiles, mid):
    """J d over all tile 4-neighbour pairs + kappa mismatches against mid."""
    e = P.J * (dist(tiles[:, :-1], tiles[:, 1:], P.q).sum() + dist(tiles[:-1], tiles[1:], P.q).sum())
    mup = np.repeat(np.repeat(mid, P.BM, 0), P.BM, 1)
    return e + P.kappa * (tiles != mup).sum()


def test_mid_logZ_brute_force():
    rng = np.random.default_rng(1)
    P = Potts(2, 2, BM=2, BT=2, J=0.7, kappa=1.3, lam=2.0)
    O = Oracle(P, seed=3)
    BM = P.BM
    for i, j in [(1, 1), (0, 0), (3, 2), (0, 3)]:
        O.tile.grid[:] = rng.integers(P.q, size=O.tile.grid.shape)
        y0, x0 = i * BM, j * BM
        lz = O.mid_logZ(i, j)
        for m in range(P.q):
            mid = O.mid.grid.copy()
            mid[i, j] = m
            t = O.tile.grid.copy()
            es = []
            for cfg in itertools.product(range(P.q), repeat=BM * BM):
                t[y0:y0 + BM, x0:x0 + BM] = np.array(cfg).reshape(BM, BM)
                es.append(_tile_energy(P, t, mid))
            const = _const_part(P, t, mid, y0, x0)          # terms not touching the block
            assert np.isclose(lz[m], _logsumexp(-(np.array(es) - const))), (i, j, m)
        p = O.mid_probs(i, j)
        tpar = O.top.grid[i // P.BT, j // P.BT]
        ref = np.exp(-P.lam * ~P.PAL[tpar] + lz)
        assert np.allclose(p, ref / ref.sum())


def _const_part(P, t, mid, y0, x0):
    """Energy of the terms that do not touch the block (tile pairs with no
    end in it, mismatches outside it)."""
    BM = P.BM
    inb = np.zeros(t.shape, bool)
    inb[y0:y0 + BM, x0:x0 + BM] = True
    dh = P.J * dist(t[:, :-1], t[:, 1:], P.q) * ~(inb[:, :-1] | inb[:, 1:])
    dv = P.J * dist(t[:-1], t[1:], P.q) * ~(inb[:-1] | inb[1:])
    mup = np.repeat(np.repeat(mid, BM, 0), BM, 1)
    return dh.sum() + dv.sum() + P.kappa * ((t != mup) & ~inb).sum()


def test_mid_move_keeps_state_valid():
    P = Potts(1, 1, BM=2)
    O = Oracle(P, seed=0)
    for _ in range(5):
        O.mid_move(1, 0)
    assert O.tile.grid.min() >= 0 and O.tile.grid.max() < P.q


def test_top_probs_brute_force():
    rng = np.random.default_rng(2)
    P = Potts(2, 2, BM=2, BT=2, J=1.0, kappa=0.9, lam=1.7)
    O = Oracle(P, seed=4)
    O.tile.grid[:] = rng.integers(P.q, size=O.tile.grid.shape)
    BM, BT = P.BM, P.BT
    for i, j in [(0, 0), (1, 0), (1, 1)]:
        logw = np.zeros(P.P)
        for T in range(P.P):
            ws = []
            for ms in itertools.product(range(P.q), repeat=BT * BT):
                e = 0.0
                for k, m in enumerate(ms):
                    a, b = divmod(k, BT)
                    yy, xx = (i * BT + a) * BM, (j * BT + b) * BM
                    e += P.lam * (not P.PAL[T, m]) + P.kappa * (O.tile.grid[yy:yy + BM, xx:xx + BM] != m).sum()
                ws.append(-e)
            logw[T] = _logsumexp(ws)
        ref = np.exp(logw - logw.max())
        assert np.allclose(O.top_probs(i, j), ref / ref.sum())


def test_induced_mid_coupling():
    """At kappa -> inf the left block's log Z differs across m by BM J d(m, c')
    with c' the colour of the right block's tiles."""
    P = Potts(1, 2, BT=1, J=1.0, kappa=50.0, lam=0.0)
    O = Oracle(P, seed=0)
    BM = P.BM
    for cp in range(P.q):
        O.tile.grid[:, BM:] = cp
        lz = O.mid_logZ(0, 0)
        got = lz[cp] - lz
        want = BM * P.J * dist(np.arange(P.q), cp, P.q)
        assert np.allclose(got, want, atol=1e-2), (cp, got, want)


def test_forward_and_oracle_run():
    P = Potts(2, 2)
    F = Forward(P, P.theta0(), seed=0)
    O = Oracle(P, seed=0)
    O.sweep(2)
    for s in (F.run(S_T=3, S_M=3, S_F=3), P.stats(O.top, O.mid, O.tile)):
        for k, shape in dict(mid_h=(4, 4), mid_v=(4, 4), mid_u=(4,), top_h=(4, 4), top_v=(4, 4), top_u=(4,),
                             tile_d=(3,)).items():
            assert s[k].shape == shape, k
            assert np.all(s[k] >= 0) and np.isclose(s[k].sum(), 1), k
        for k in ("mismatch", "pal_viol", "edge_same_mid", "edge_same_top"):
            assert 0 <= s[k] <= 1, k
    mo = O.moments(1, 2)
    assert set(mo) == set(s)
    img = P.render(O.top, O.mid, O.tile)
    assert img.shape == (P.H, P.W, 3) and img.dtype == np.uint8


def test_tile_energy_matches_numpy():
    P = Potts(2, 2, kappa=1.0)
    O = Oracle(P, seed=5)
    O.sweep(1)
    tot, nviol = O.model.energy("tile")
    assert nviol == 0
    assert np.isclose(tot, _tile_energy(P, O.tile.grid, O.mid.grid))


def test_block_backward_sample():
    """The block's tiles redrawn by backward sampling follow the exact
    conditional (BM = 2, q^4 states)."""
    rng = np.random.default_rng(7)
    P = Potts(2, 2, BM=2, J=1.0, kappa=0.8)
    O = Oracle(P, seed=8)
    O.tile.grid[:] = rng.integers(P.q, size=O.tile.grid.shape)
    i, j, m = 1, 2, 3
    y0, x0 = 2 * i, 2 * j
    O.mid.grid[i, j] = m
    t = O.tile.grid.copy()
    cfgs = list(itertools.product(range(P.q), repeat=4))
    es = []
    for cfg in cfgs:
        t[y0:y0 + 2, x0:x0 + 2] = np.array(cfg).reshape(2, 2)
        es.append(_tile_energy(P, t, O.mid.grid))
    ref = np.exp(-(np.array(es) - min(es)))
    ref /= ref.sum()
    idx = {c: k for k, c in enumerate(cfgs)}
    cnt = np.zeros(len(cfgs))
    n = 40000
    for _ in range(n):
        O._filter(i, j, m, True)
        cnt[idx[tuple(O.tile.grid[y0:y0 + 2, x0:x0 + 2].ravel())]] += 1
    assert 0.5 * np.abs(cnt / n - ref).sum() < 0.03
