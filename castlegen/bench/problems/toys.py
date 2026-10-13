"""Enumerable toy problems: toy_support and toy_modes (notes/bench.md).

FROZEN once approved.  Both live on T = 2 (4 x 4 fine blocks), H = 8, with
every factor confined to its 4 x 4 block (pairs: within = 2; counts are
block-local by construction), so the target factorises into iid blocks.  The
coarse channels rho1 (level 1, D 5) and rho2 (level 2, D 17) are hard count
equalities (sum of the 4 children), hence determined by the fine block, and
exact answers come from enumerating the 2^16 fine states of one block.

Fine cell (y, x) of a block is bit 4 y + x of the block state; y grows down.
"""
from __future__ import annotations

import numpy as np

from ..problem import INF, Chan, Problem, count, pair, unary

T, H, W = 2, 8, 64
B = 4                                  # block side in fine cells


# ------------------------------------------------------------------ shared
def _chans():
    return [Chan("rho2", 2, 17), Chan("rho1", 1, 5), Chan("t", 0, 2)]


def _count_factors():
    eq5 = np.full((5, 5), INF)
    np.fill_diagonal(eq5, 0.0)
    eq17 = np.full((17, 17), INF)
    np.fill_diagonal(eq17, 0.0)
    return [count(("t", "id"), ("rho1", "id"), eq5, name="rho1=sum t"),
            count(("rho1", "id"), ("rho2", "id"), eq17, name="rho2=sum rho1")]


def lift(t):
    """Fine t (h, w) -> {t, rho1, rho2} with the coarse channels as exact counts."""
    t = np.asarray(t, np.int64)
    r1 = t[0::2, 0::2] + t[0::2, 1::2] + t[1::2, 0::2] + t[1::2, 1::2]
    r2 = r1[0::2, 0::2] + r1[0::2, 1::2] + r1[1::2, 0::2] + r1[1::2, 1::2]
    return {"rho2": r2, "rho1": r1, "t": t}


def all_blocks():
    """(65536, 4, 4) int array: every fine block state, index = bit pattern."""
    s = np.arange(1 << 16)
    bits = (s[:, None] >> np.arange(16)[None, :]) & 1
    return bits.reshape(-1, B, B).astype(np.int64)


def blocks_of(t):
    """Fine t (h, w), h, w multiples of 4 -> (nblocks, 4, 4)."""
    h, w = t.shape
    return np.asarray(t).reshape(h // B, B, w // B, B).transpose(0, 2, 1, 3).reshape(-1, B, B)


def tile_blocks(blocks, h, w):
    """(nb, 4, 4) -> fine (h, w), the inverse of blocks_of."""
    return blocks.reshape(h // B, w // B, B, B).transpose(0, 2, 1, 3).reshape(h, w)


def _probs(E):
    """Boltzmann probabilities from block energies (inf -> 0)."""
    ok = np.isfinite(E)
    lw = np.where(ok, -E, -np.inf)
    p = np.exp(lw - lw[ok].max())
    return p / p.sum()


def sample_t(rng, p, h, w):
    """Exact sample of the fine level: iid blocks drawn from p over 2^16 states."""
    nb = (h // B) * (w // B)
    idx = rng.choice(p.size, size=nb, p=p)
    return tile_blocks(all_blocks()[idx], h, w)


def _render(t, colours, scale=4):
    img = np.asarray(colours, np.uint8)[np.asarray(t)]
    return np.repeat(np.repeat(img, scale, 0), scale, 1)


def _make_probes(p, n=4):
    def probes(seed, region):
        rng = np.random.default_rng(seed)
        return [lift(sample_t(rng, p, H, region.W)) for _ in range(n)]
    return probes


def _exact(stat_fn, p, keys):
    """Expected per-block statistics under p, over all 2^16 blocks."""
    vals = stat_fn(all_blocks())
    return {k: float(np.dot(p, vals[k])) for k in keys}


# ------------------------------------------------------------- toy_support
SUP_SMOOTH, SUP_DIRT = 0.5, -0.3


def support_block_energy(blocks):
    """Energy of each (n, 4, 4) block under toy_support's factors (inf = hard)."""
    blocks = np.asarray(blocks)
    hard = ((blocks[:, :-1, :] == 1) & (blocks[:, 1:, :] == 0)).any(axis=(1, 2))
    soft = SUP_SMOOTH * (blocks[:, :, 1:] != blocks[:, :, :-1]).sum(axis=(1, 2)) \
        + SUP_DIRT * blocks.sum(axis=(1, 2))
    return np.where(hard, INF, soft)


def support_block_stats(blocks):
    """Per-block statistics (n,) arrays; per-world stats are their means."""
    blocks = np.asarray(blocks)
    hcol = blocks.sum(axis=1)                       # (n, 4) dirt per fine column
    return {
        "dirt_frac": blocks.mean(axis=(1, 2)),
        "col_height": hcol.mean(axis=1),
        "empty_block_frac": (hcol.sum(axis=1) == 0).astype(float),
        "flat_block_frac": (hcol == hcol[:, :1]).all(axis=1).astype(float),
    }


SUPPORT_KEYS = ("dirt_frac", "col_height", "empty_block_frac", "flat_block_frac")


def build_toy_support():
    chans = _chans()
    supp = np.zeros((2, 2))
    supp[1, 0] = INF                                # dirt above air in the block
    smooth = SUP_SMOOTH * (1.0 - np.eye(2))
    factors = _count_factors() + [
        pair(("t", "id"), ("t", "id"), (1, 0), supp, within=2, name="support"),
        pair(("t", "id"), ("t", "id"), (0, 1), smooth, within=2, name="smooth"),
        unary(("t", "id"), [0.0, SUP_DIRT], name="dirt"),
    ]
    p = _probs(support_block_energy(all_blocks()))

    def stats(x):
        st = support_block_stats(blocks_of(np.asarray(x["t"])))
        return {k: float(st[k].mean()) for k in SUPPORT_KEYS}

    return Problem(
        name="toy_support", T=T, H=H, chans=chans, factors=factors,
        W_train=W, W_eval=W, stats=stats,
        render=lambda x: _render(x["t"], [(150, 200, 240), (120, 80, 40)]),
        probes=_make_probes(p),
        exact=_exact(support_block_stats, p, SUPPORT_KEYS),
        tol={"dirt_frac": 0.01, "col_height": 0.04, "empty_block_frac": 0.01,
             "flat_block_frac": 0.01},
        notes="iid 4x4 blocks; dirt (t=1) only on dirt within a block (bottom block row free), "
              "horizontal smoothness 0.5, dirt bonus 0.3; rho1/rho2 hard counts. Exact by "
              "enumerating the 2^16 block states (625 valid: a height 0..4 per column).",
    )


# --------------------------------------------------------------- toy_modes
MODES_J, MODES_N = 1.5, 8
_yy, _xx = np.mgrid[0:B, 0:B]
MODE_MASKS = {"L": (_xx < 2).astype(np.int64), "R": (_xx >= 2).astype(np.int64),
              "T": (_yy < 2).astype(np.int64), "B": (_yy >= 2).astype(np.int64)}
MODES_KEYS = ("mode_L", "mode_R", "mode_T", "mode_B", "mode_other", "broken_bonds")


def modes_block_energy(blocks):
    blocks = np.asarray(blocks)
    hard = blocks.sum(axis=(1, 2)) != MODES_N
    soft = MODES_J * ((blocks[:, :, 1:] != blocks[:, :, :-1]).sum(axis=(1, 2))
                      + (blocks[:, 1:, :] != blocks[:, :-1, :]).sum(axis=(1, 2)))
    return np.where(hard, INF, soft)


def modes_block_stats(blocks):
    blocks = np.asarray(blocks)
    out, any_mode = {}, np.zeros(len(blocks), bool)
    for m, mask in MODE_MASKS.items():
        hit = (blocks == mask).sum(axis=(1, 2)) >= 14     # masks differ on 8 cells: disjoint
        out["mode_" + m] = hit.astype(float)
        any_mode |= hit
    out["mode_other"] = (~any_mode).astype(float)
    out["broken_bonds"] = ((blocks[:, :, 1:] != blocks[:, :, :-1]).sum(axis=(1, 2))
                           + (blocks[:, 1:, :] != blocks[:, :-1, :]).sum(axis=(1, 2))).astype(float)
    return out


def build_toy_modes():
    chans = _chans()
    r2 = np.full(17, INF)
    r2[MODES_N] = 0.0
    ferro = MODES_J * (1.0 - np.eye(2))
    factors = _count_factors() + [
        unary(("rho2", "id"), r2, name="rho2=8"),
        pair(("t", "id"), ("t", "id"), (0, 1), ferro, within=2, name="ferro_h"),
        pair(("t", "id"), ("t", "id"), (1, 0), ferro, within=2, name="ferro_v"),
    ]
    p = _probs(modes_block_energy(all_blocks()))

    def stats(x):
        st = modes_block_stats(blocks_of(np.asarray(x["t"])))
        return {k: float(st[k].mean()) for k in MODES_KEYS}

    return Problem(
        name="toy_modes", T=T, H=H, chans=chans, factors=factors,
        W_train=W, W_eval=W, stats=stats,
        render=lambda x: _render(x["t"], [(235, 235, 235), (40, 60, 160)]),
        probes=_make_probes(p),
        exact=_exact(modes_block_stats, p, MODES_KEYS),
        tol={"mode_L": 0.02, "mode_R": 0.02, "mode_T": 0.02, "mode_B": 0.02,
             "mode_other": 0.02, "broken_bonds": 0.1},
        notes="iid 4x4 blocks with exactly 8 dirt cells (hard rho2 = 8), ferro coupling 1.5 "
              "within the block; four equal-mass half-block modes L/R/T/B (a block is in a "
              "mode if it agrees with the half mask on >= 14 of 16 cells). Exact by "
              "enumerating the 2^16 block states.",
    )
