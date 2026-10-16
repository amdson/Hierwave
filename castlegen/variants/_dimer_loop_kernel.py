"""Sector-changing, p*-invariant refresh kernel for the dimers bench problem: a worm (alternating-loop) excursion.

Extended chain on perfect matchings M and two-monomer states (M; t, h).  Open: pick a cell t uniformly, delete its
dimer (t, h).  Step: the head h picks one of its six lattice directions uniformly (1/6); if the neighbour w is
off the strip the step is a null step; if w == t the loop closes (add dimer (h, t)); otherwise w's dimer (w, z) is
deleted, (h, w) is added and z becomes the head.  Every step is reversible with the same 1/6, open has probability
1/N, close 1/6, so the excursion M -> M' has the same probability as its reversed excursion M' -> M (same length,
same weight (1/N) (1/6)^(k+1)).  Excursions longer than `lmax` steps are rejected (M stays): the reverse has the
same length, so the cap keeps the kernel symmetric.  Hence the kernel is symmetric on the set of perfect matchings
and leaves the uniform target (the dimers p*, no soft terms) invariant.  Generic in the sense of the amendment: it
uses only the matching constraint of the declared factors (one pair per cell, six neighbours); no problem-
specific knowledge of orientations.  Long loops that wind around the periodic x direction change the Z2 sector
and the orientation mix, which local blocks cannot.

Operates on the 'p' channel (n, H, W) with direction codes 0 E, 1 W, 2 S, 3 N, 4 SE, 5 NW (dy, dx).
Worlds that are not perfect matchings are left untouched.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

DY = np.array([0, 0, 1, -1, 1, -1], np.int64)
DX = np.array([1, -1, 0, 0, 1, -1], np.int64)
OPP = np.array([1, 0, 3, 2, 5, 4], np.int64)


@njit(cache=True)
def valid_world(p, H, W, DY, DX, OPP):
    for y in range(H):
        for x in range(W):
            d = p[y, x]
            if d < 0 or d > 5:
                return False
            y2 = y + DY[d]
            if y2 < 0 or y2 >= H:
                return False
            x2 = (x + DX[d]) % W
            if p[y2, x2] != OPP[d]:
                return False
    return True


@njit(cache=True)
def _repair_world(p, H, W, cap, DY, DX, OPP):
    """Projection of an invalid draw onto perfect matchings (NOT part of the invariant kernel): cells whose pair is
    not reciprocated become monomers; each monomer walks as a worm head (random direction, shift along the matched
    neighbour's dimer) until it lands next to another monomer and pairs with it.  Returns the number of monomers
    left (0 on success)."""
    for y in range(H):
        for x in range(W):
            d = p[y, x]
            ok = d >= 0 and d <= 5
            if ok:
                y2 = y + DY[d]
                ok = y2 >= 0 and y2 < H and p[y2, (x + DX[d]) % W] == OPP[d]
            if not ok:
                p[y, x] = -2
    # -2: marked for clearing (must not unmark partners mid-scan, so clear in a second pass)
    for y in range(H):
        for x in range(W):
            if p[y, x] == -2:
                p[y, x] = -1
    # a cell that survived but whose partner was cleared cannot occur: reciprocity was tested before clearing, and
    # clearing cell c only happens if c is unreciprocated, so its pointee (if any) pointed elsewhere.  Re-test.
    for y in range(H):
        for x in range(W):
            d = p[y, x]
            if d >= 0:
                y2 = y + DY[d]
                if p[y2, (x + DX[d]) % W] != OPP[d]:
                    p[y, x] = -1
    left = 0
    for m in range(H * W):
        my, mx = m // W, m % W
        if p[my, mx] != -1:
            continue
        hy, hx = my, mx
        done = False
        for k in range(cap):
            d = np.random.randint(6)
            wy = hy + DY[d]
            if wy < 0 or wy >= H:
                continue
            wx = (hx + DX[d]) % W
            e = p[wy, wx]
            if e == -1:
                p[hy, hx] = d
                p[wy, wx] = OPP[d]
                done = True
                break
            zy, zx = wy + DY[e], (wx + DX[e]) % W
            p[hy, hx] = d
            p[wy, wx] = OPP[d]
            p[zy, zx] = -1
            hy, hx = zy, zx
        if not done:
            left += 1
    return left


@njit(parallel=True, cache=True)
def _repair(P, cap, DY, DX, OPP, seed):
    n, H, W = P.shape
    left = np.zeros(n, np.int64)
    for b in prange(n):
        np.random.seed(seed + 7919 * b)
        left[b] = _repair_world(P[b], H, W, cap, DY, DX, OPP)
    return left


@njit(cache=True)
def excursion(p, H, W, t, choices, nch, DY, DX, OPP):
    """Deterministic excursion from start cell t (flat index) driven by `choices[:nch]`.  Mutates p.
    Returns (status, used): status 0 closed (p is the new perfect matching), 1 ran out of choices (p is dirty,
    caller restores)."""
    ty, tx = t // W, t % W
    d0 = p[ty, tx]
    hy, hx = ty + DY[d0], (tx + DX[d0]) % W
    p[ty, tx] = -1
    p[hy, hx] = -1
    for k in range(nch):
        d = choices[k]
        wy = hy + DY[d]
        if wy < 0 or wy >= H:
            continue
        wx = (hx + DX[d]) % W
        if wy == ty and wx == tx:
            p[hy, hx] = d
            p[ty, tx] = OPP[d]
            return 0, k + 1
        e = p[wy, wx]
        zy, zx = wy + DY[e], (wx + DX[e]) % W
        p[hy, hx] = d
        p[wy, wx] = OPP[d]
        p[zy, zx] = -1
        hy, hx = zy, zx
    return 1, nch


@njit(cache=True)
def _sweep_world(p, H, W, nattempt, lmax, DY, DX, OPP, jy, jx, jv):
    nacc = 0
    nchg = 0
    for _ in range(nattempt):
        t = np.random.randint(H * W)
        ty, tx = t // W, t % W
        d0 = p[ty, tx]
        hy, hx = ty + DY[d0], (tx + DX[d0]) % W
        nj = 0
        jy[nj] = ty; jx[nj] = tx; jv[nj] = p[ty, tx]; nj += 1
        jy[nj] = hy; jx[nj] = hx; jv[nj] = p[hy, hx]; nj += 1
        p[ty, tx] = -1
        p[hy, hx] = -1
        closed = False
        for k in range(lmax):
            d = np.random.randint(6)
            wy = hy + DY[d]
            if wy < 0 or wy >= H:
                continue
            wx = (hx + DX[d]) % W
            if wy == ty and wx == tx:
                p[hy, hx] = d
                p[ty, tx] = OPP[d]
                closed = True
                break
            e = p[wy, wx]
            zy, zx = wy + DY[e], (wx + DX[e]) % W
            jy[nj] = hy; jx[nj] = hx; jv[nj] = p[hy, hx]; nj += 1
            jy[nj] = wy; jx[nj] = wx; jv[nj] = p[wy, wx]; nj += 1
            jy[nj] = zy; jx[nj] = zx; jv[nj] = p[zy, zx]; nj += 1
            p[hy, hx] = d
            p[wy, wx] = OPP[d]
            p[zy, zx] = -1
            hy, hx = zy, zx
        if closed:
            nacc += 1
            if nj > 2:
                nchg += 1
        else:
            for i in range(nj - 1, -1, -1):
                p[jy[i], jx[i]] = jv[i]
    return nacc, nchg


@njit(parallel=True, cache=True)
def _sweep(P, nattempt, lmax, DY, DX, OPP, ok, seed):
    n, H, W = P.shape
    acc = np.zeros(n, np.int64)
    chg = np.zeros(n, np.int64)
    for b in prange(n):
        if ok[b]:
            np.random.seed(seed + b)
            jy = np.empty(3 * lmax + 8, np.int64)
            jx = np.empty(3 * lmax + 8, np.int64)
            jv = np.empty(3 * lmax + 8, np.int64)
            a, c = _sweep_world(P[b], H, W, nattempt, lmax, DY, DX, OPP, jy, jx, jv)
            acc[b] = a
            chg[b] = c
    return acc, chg


def valid_mask(P):
    P = np.asarray(P)
    return np.array([valid_world(P[b], P.shape[1], P.shape[2], DY, DX, OPP) for b in range(P.shape[0])])


def loop_refresh(P, nattempt, lmax=1024, seed=0, repair=False):
    """Apply `nattempt` worm excursions per world (a sweep = H*W attempts) to the valid worlds of P (n, H, W)
    in place.  Returns (P, info)."""
    P = np.ascontiguousarray(np.asarray(P), dtype=np.int64)
    ok = valid_mask(P)
    n_rep = 0
    if repair and not ok.all():
        bad = np.where(~ok)[0]
        sub = np.ascontiguousarray(P[bad])
        _repair(sub, 20000, DY, DX, OPP, int(seed) + 13)
        P[bad] = sub
        n_rep = len(bad)
        ok = valid_mask(P)
    acc, chg = _sweep(P, int(nattempt), int(lmax), DY, DX, OPP, ok, int(seed))
    return P, {"valid_frac": float(ok.mean()), "repaired": n_rep, "acc": float(acc[ok].sum() / max(ok.sum() * nattempt, 1)),
               "changed": float(chg[ok].sum() / max(ok.sum() * nattempt, 1))}


def hvd(P):
    o = np.asarray(P) // 2
    return [round(float((o == i).mean()), 4) for i in range(3)]
