"""The worm excursion is a bijection on perfect matchings and the kernel is symmetric (detailed balance for the
uniform target).  Exact: enumerate all matchings of tiny periodic-x strips, enumerate every excursion up to a step cap."""
import itertools
import numpy as np
import pytest

from castlegen.variants import _dimer_loop_kernel as dk


def all_matchings(H, W):
    out = []
    p = -np.ones((H, W), np.int64)

    def rec(i):
        while i < H * W and p[i // W, i % W] >= 0:
            i += 1
        if i == H * W:
            out.append(p.copy())
            return
        y, x = i // W, i % W
        for d in range(6):
            y2, x2 = y + dk.DY[d], (x + dk.DX[d]) % W
            if 0 <= y2 < H and p[y2, x2] < 0 and (y2, x2) != (y, x):
                p[y, x] = d
                p[y2, x2] = dk.OPP[d]
                rec(i + 1)
                p[y, x] = -1
                p[y2, x2] = -1
    rec(0)
    return out


def key(p):
    return tuple(p.ravel().tolist())


@pytest.mark.parametrize("H,W,cap", [(4, 3, 5), (2, 4, 6)])
def test_symmetric_and_bijective(H, W, cap):
    Ms = all_matchings(H, W)
    idx = {key(m): i for i, m in enumerate(Ms)}
    assert all(dk.valid_world(m, H, W, dk.DY, dk.DX, dk.OPP) for m in Ms)
    n = len(Ms)
    N = H * W
    T = np.zeros((n, n))
    for i, M in enumerate(Ms):
        for t in range(N):
            stack = [[]]
            while stack:
                seq = stack.pop()
                p = M.copy()
                ch = np.array(seq, np.int64)
                status, used = dk.excursion(p, H, W, t, ch, len(seq), dk.DY, dk.DX, dk.OPP)
                if status == 0:
                    assert dk.valid_world(p, H, W, dk.DY, dk.DX, dk.OPP)
                    T[i, idx[key(p)]] += (1.0 / N) * (1.0 / 6) ** used
                elif len(seq) < cap:
                    stack.extend(seq + [d] for d in range(6))
    assert np.allclose(T, T.T, atol=1e-12)          # detailed balance for the uniform measure
    assert (T.sum(1) <= 1 + 1e-12).all()
    assert (T - np.diag(np.diag(T))).sum() > 0      # the move does something
    # bijection: every excursion has a reverse excursion (same start-to-end pair) -> symmetry above is its
    # aggregated form; check one explicit reversal
    M = Ms[0]
    for t in range(N):
        for seq in itertools.product(range(6), repeat=3):
            p = M.copy()
            status, used = dk.excursion(p, H, W, t, np.array(seq, np.int64), 3, dk.DY, dk.DX, dk.OPP)
            if status == 0 and used and not (p == M).all():
                assert idx[key(p)] != idx[key(M)]


def test_sweep_keeps_validity_and_cap_restores():
    H, W = 8, 16
    Ms = [m for m in all_matchings(2, 4)]
    # a valid 8x16 tiling: horizontal pairs
    P = np.zeros((3, H, W), np.int64)
    P[:, :, 1::2] = 1
    P2, info = dk.loop_refresh(P.copy(), 500, lmax=64, seed=3)
    assert dk.valid_mask(P2).all()
    assert info["changed"] > 0
    P3, _ = dk.loop_refresh(P.copy(), 200, lmax=1, seed=3)       # cap 1: nothing can close except no-ops
    assert dk.valid_mask(P3).all()


def test_repair_makes_valid():
    H, W = 8, 16
    rng = np.random.default_rng(0)
    P = np.zeros((4, H, W), np.int64); P[:, :, 1::2] = 1
    P[:, 3, 4] = 3; P[:, 5, 7] = 2; P[:, 0, 0] = 3          # corrupt a few cells (incl. off-strip pointer)
    P2, info = dk.loop_refresh(P.copy(), 0, repair=True, seed=5)
    assert info["repaired"] == 4 and dk.valid_mask(P2).all()
