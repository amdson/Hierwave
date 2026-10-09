"""Bit-sliced tile Gibbs (castlegen/csrc/bitgibbs.c): a site's conditional
over up to 256 tiles drawn with word-wide bit operations.

Model: every term the site sees is a FACTOR, a table row over tiles of
integer levels 0 .. 2^FB - 1 or forbidden, picked by a small input (a
neighbour's tile and side, the site's parent context, its field value, ...).
Its row is stored as FB level planes plus a forbid plane, each NW 64-bit
words, bit t for tile t.  A draw adds the site's F rows with a bit-sliced
adder (AB = 5 accumulator bits, a carry past them forbids the tile), splits
the sum into its 32 level masks, and picks exactly from weights cw[level]
(any integers; cw = 2^(31 - l) is the dyadic case, cw = round(2^40
exp(-unit l)) a finer energy grid).  Integer draws only: one level by
Z = sum_l popcount(mask_l) cw[l], one tile within it.

Energies become levels by quantize(): each row shifted to its minimum (a
row's constant does not change the conditional), divided by unit, rounded;
>= 2^FB or infinite is forbidden.

bg_sweep_pairs: a pair MRF on a torus.  bg_sweep_field: one k x k block with
blockfield.py's (g, d) fields, alternating the tile and (g, d) per site with
validity hard (see the C source)."""
from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "csrc", "bitgibbs.c")
LIB = os.path.join(os.path.dirname(__file__), "..", "cache", "bitgibbs.dylib")
AB, NL = 5, 32


def _lib():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.check_call(["cc", "-O3", "-mcpu=native", "-shared", "-fPIC", SRC, "-o", LIB])
    lib = ctypes.CDLL(LIB)
    P = ctypes.c_void_p
    I, U, Dd = ctypes.c_int, ctypes.c_uint64, ctypes.c_double
    lib.bg_sample.argtypes = [I, I, I, P, P, P, U]
    lib.bg_sample.restype = I
    lib.bg_sample_many.argtypes = [I, I, I, P, P, P, U, I, P]
    lib.bg_sweep_pairs.argtypes = [I, I, I, I, I, P, P, P, P, P, U, I, P, P, Dd, P]
    lib.bg_sweep_field.argtypes = [I, I, I, I, I, I, P, P, P, P, P, P, P, P, P, P, P, P, P, I, I, P, I, I, I,
                                   Dd, Dd, Dd, P, U, I]
    return lib


_L = None


def lib():
    global _L
    if _L is None:
        _L = _lib()
    return _L


def _p(a):
    assert a.flags.c_contiguous
    return a.ctypes.data_as(ctypes.c_void_p)


def nwords(S):
    return (S + 63) // 64


def masks(B):
    """(..., S) bool -> (..., NW) uint64, bit t of word t // 64."""
    B = np.asarray(B, bool)
    S = B.shape[-1]
    NW = nwords(S)
    pad = np.zeros(B.shape[:-1] + (NW * 64,), bool)
    pad[..., :S] = B
    bits = pad.reshape(B.shape[:-1] + (NW, 64)).astype(np.uint64)
    return np.ascontiguousarray((bits << np.arange(64, dtype=np.uint64)).sum(-1, dtype=np.uint64))


def stacks(L, FB):
    """Levels (..., S) int (-1: forbidden) -> stacks (..., FB + 1, NW)."""
    L = np.asarray(L)
    planes = [((L >> b) & 1).astype(bool) & (L >= 0) for b in range(FB)] + [L < 0]
    return np.ascontiguousarray(np.stack([masks(p) for p in planes], -2))


def quantize(E, unit, FB):
    """Energies (..., S) -> levels (-1 forbidden), each row shifted to its minimum."""
    E = np.asarray(E, np.float64)
    fin = np.isfinite(E)
    mn = np.where(fin, E, np.inf).min(-1, keepdims=True)
    L = np.round((E - np.where(np.isfinite(mn), mn, 0)) / unit)
    return np.where(fin & (L < 2 ** FB), L, -1).astype(np.int64)


def weights(unit=np.log(2.0)):
    """cw (32,) uint64: exp(-unit l) scaled to integers (exactly 2^(31 - l) when unit = ln 2)."""
    if np.isclose(unit, np.log(2.0)):
        return np.array([1 << (31 - l) for l in range(NL)], np.uint64)
    return np.array([max(1, round(2.0 ** 40 * np.exp(-unit * l))) for l in range(NL)], np.uint64)


def exact(L_rows, legal, cw):
    """Reference: the distribution bg_sample draws from (rows of levels added,
    forbidden or sums >= 32 or not legal -> 0)."""
    L = np.asarray(L_rows)
    bad = (L < 0).any(0)
    lev = np.where(L < 0, 0, L).sum(0)
    ok = ~bad & (lev < NL) & np.asarray(legal, bool)
    w = np.where(ok, cw[np.minimum(lev, NL - 1)].astype(np.float64), 0.0)
    return w / w.sum()


def sample_many(stk, legal, cw, seed, n):
    """n draws from the factors stk (F, FB + 1, NW) with legal (NW,)."""
    F, FB1, NW = stk.shape
    out = np.empty(n, np.int32)
    lib().bg_sample_many(NW, FB1 - 1, F, _p(np.ascontiguousarray(stk)), _p(legal), _p(cw), seed, n, _p(out))
    return out


def pair_tables(Eh, Ev, unit, FB):
    """(4, S, FB + 1, NW) stacks: tile t at p with neighbour n on side d of p."""
    E = np.stack([Ev, Eh.T, Ev.T, Eh])                                     # [d][n][t]
    return stacks(quantize(E, unit, FB), FB)


def sweep_pairs(T, pair, unary, legal, cw, seed, sweeps, exact=None, unit=0.0):
    """exact: (Ex (4, S, S) [d][n][t], ux (S,)) float energies: Metropolis-
    Hastings-correct the quantized draws to them (cw must be exp(-unit l)).
    Returns the acceptance rate (1.0 without exact)."""
    H, W = T.shape
    _, S, FB1, NW = pair.shape
    acc = np.zeros(2, np.int64)
    Ex, ux = (None, None) if exact is None else (np.ascontiguousarray(exact[0], np.float64),
                                                np.ascontiguousarray(exact[1], np.float64))
    lib().bg_sweep_pairs(H, W, NW, FB1 - 1, S, _p(T), _p(pair), None if unary is None else _p(unary), _p(legal),
                         _p(cw), seed, sweeps, None if Ex is None else _p(Ex), None if ux is None else _p(ux), unit,
                         _p(acc))
    return acc[0] / acc[1] if acc[1] else 1.0


def sweep_field(T, Gf, Df, k, pair, unary, gcost, J, legal, exits, openout, all_, seambad, labs, lport, lseam,
                node, wall, G, D, mu, eps, delta, cw, seed, sweeps):
    """A map T (H, W) of k x k blocks with patterns labs (H/k, W/k, 9) (the C
    source documents the energy); every per-site table is (H W, ...)."""
    H, W = T.shape
    _, S, FB1, NW = pair.shape
    labs = np.ascontiguousarray(labs, np.int32)
    lib().bg_sweep_field(H, W, k, NW, FB1 - 1, S, _p(T), _p(Gf), _p(Df), _p(pair), _p(unary), _p(gcost), _p(J),
                         _p(legal), _p(exits), _p(openout), _p(all_), _p(seambad), _p(labs), lport, lseam, _p(node),
                         wall, G, D, mu, eps, delta, _p(cw), seed, sweeps)


def field_tables(ts, tabs):
    """(J, openout, all, seambad, node words) for sweep_field from blockconn.tables."""
    node, Dh, Dv, sock = tabs
    S = len(node)
    Jb = np.stack([Dv, Dh.T, Dv.T, Dh])                                     # [d][n][t]: t joined to n on t's side d
    op = np.stack([node & sock[d] for d in range(4)])                      # opening of t on its side d
    bad = (op[:, None, :] | op[[2, 3, 0, 1]][:, :, None]) & ~Jb             # the seam rule (blockconn._seam_bad)
    return masks(Jb), masks(op), masks(np.ones(S, bool)), masks(bad), masks(node)


def site_exits(lab9s, tabs, k):
    """(H W, NW) exits per site of a map of blocks with labs (R, C, 9):
    openings on class ports, and nodes at a sink block's root cell."""
    from castlegen.blockconn import MLAB
    node, _, _, sock = tabs
    R, C, _ = lab9s.shape
    S = len(node)
    E = np.zeros((R * k, C * k, S), bool)
    hh = k // 2
    for by in range(R):
        for bx in range(C):
            lab = lab9s[by, bx]
            blk = E[by * k:(by + 1) * k, bx * k:(bx + 1) * k]
            for q in range(8):
                if not 0 < lab[q] <= MLAB:
                    continue
                d, h = divmod(q, 2)
                i = np.arange(h * hh, (h + 1) * hh)
                cells = ((0 * i, i), (i, 0 * i + k - 1), (0 * i + k - 1, i), (i, 0 * i))[d]
                blk[cells[0], cells[1]] |= node & sock[d]
            if lab[8] >= 0:
                blk[lab[8] // k, lab[8] % k] |= node
    return masks(E.reshape(-1, S))
