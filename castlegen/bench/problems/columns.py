"""columns: tall thin columns, x-independent, exactly solvable per fine column
(notes/bench.md).  FROZEN once approved.

Fine channel k (level 0, D 13): 0 air, 1..12 the position in a column of
length 12 counted from the bottom (1 bottom cell, 12 top cell).  Rows grow
downward, so a column reads 12, 11, ..., 1 going down, with air above its top
and below its bottom.  Rows off the strip read air (pads 0), so a column must
fit inside the strip.  Soft: -mu per column cell.  Coarse channels n1..n4
(level l, D 4^l + 1) are hard count equalities on the column-cell count, so
they are determined by k and the fine marginal is exactly the 1-D chain of
each fine column (x columns independent): forward-backward over the H rows.
"""
from __future__ import annotations

import numpy as np

from ..problem import INF, Chan, Problem, count, pair, unary

T, H, W = 4, 64, 128
L, MU = 12, 0.6
D = L + 1


def allowed_table():
    """(13, 13) bool: allowed (k at row y, k at row y + 1)."""
    A = np.zeros((D, D), bool)
    for j in range(2, L + 1):
        A[j, j - 1] = True
    A[1, 0] = True
    A[0, 0] = True
    A[0, L] = True
    return A


def chain_parts(table, unary_tab):
    """Transfer matrix, start and end masks of one fine column, read off the
    declared tables (pads 0: the row above the strip and below it are air)."""
    M = np.isfinite(table).astype(float)
    w = np.exp(-unary_tab)
    start = M[0] * w              # row 0 must be allowed below a pad-air row
    end = M[:, 0]                 # the last row must be allowed above pad air
    return M, w, start, end


def forward_backward(M, w, start, end, h=H):
    """Normalised forward/backward messages; returns per-row marginals (h, D)
    and log Z."""
    alpha = np.zeros((h, D))
    logz = 0.0
    a = start.copy()
    for y in range(h):
        if y > 0:
            a = (alpha[y - 1] @ M) * w
        s = a.sum()
        logz += np.log(s)
        alpha[y] = a / s
    beta = np.ones((h, D))
    beta[-1] = end
    for y in range(h - 2, -1, -1):
        b = M @ (w * beta[y + 1])
        beta[y] = b / b.sum()
    marg = alpha * beta
    marg /= marg.sum(axis=1, keepdims=True)
    logz += np.log((alpha[-1] * end).sum())
    return marg, alpha, logz


def sample_chains(rng, n, parts, h=H):
    """n exact iid samples of one fine column, (h, n), by backward sampling:
    p(k_y | k_{y+1}) proportional to alpha_y(k_y) M[k_y, k_{y+1}]."""
    M, w, start, end = parts
    _, alpha, _ = forward_backward(M, w, start, end, h)
    cond = alpha[:-1, None, :] * M.T[None, :, :]          # (h - 1, k_{y+1}, k_y)
    cond /= np.maximum(cond.sum(-1, keepdims=True), 1e-300)
    cdf = np.cumsum(cond, -1)
    out = np.zeros((h, n), np.int64)
    pl = alpha[-1] * end
    out[-1] = (rng.random((n, 1)) > np.cumsum(pl / pl.sum())[None, :]).sum(1)
    for y in range(h - 2, -1, -1):
        out[y] = (rng.random((n, 1)) > cdf[y][out[y + 1]]).sum(1)
    return np.minimum(out, D - 1)


def lift(k):
    k = np.asarray(k, np.int64)
    x = {"k": k}
    c = (k > 0).astype(np.int64)
    for l in range(1, T + 1):
        c = c[0::2, 0::2] + c[0::2, 1::2] + c[1::2, 0::2] + c[1::2, 1::2]
        x[f"n{l}"] = c
    return x


def column_stats(x):
    """Per-world stats of a sample (or a fine k array (h, w))."""
    k = np.asarray(x["k"] if isinstance(x, dict) else x)
    col = k > 0
    # gaps: air runs between consecutive columns of an x-column = the air between
    # its first top and its last bottom; (number of columns - 1) gaps
    ncol = (k == 1).sum(axis=0)
    h = k.shape[0]
    rows = np.arange(h)[:, None]
    first_top = np.where(k == L, rows, h).min(axis=0)
    last_bot = np.where(k == 1, rows, -1).max(axis=0)
    span = np.where(ncol > 0, last_bot - first_top + 1, 0)
    gap_cells = (span - L * ncol).sum()
    ngaps = np.maximum(ncol - 1, 0).sum()
    return {
        "density": float(col.mean()),
        "bottom_density": float((k == 1).mean()),
        "gap_len": float(gap_cells / ngaps) if ngaps else float("nan"),
        "empty_xcol_frac": float((~col.any(axis=0)).mean()),
    }


KEYS = ("density", "bottom_density", "gap_len", "empty_xcol_frac")


def build_columns():
    chans = [Chan(f"n{l}", l, 4 ** l + 1) for l in range(T, 0, -1)]
    chans.append(Chan("k", 0, D, views={"col": (np.arange(D) > 0).astype(np.int64)}))
    table = np.where(allowed_table(), 0.0, INF)
    utab = np.where(np.arange(D) > 0, -MU, 0.0)
    factors = [
        pair(("k", "id"), ("k", "id"), (1, 0), table, pad_a=0, pad_b=0, name="column"),
        unary(("k", "id"), utab, name="mu"),
    ]
    prev = ("k", "col")
    for l in range(1, T + 1):
        n = 4 ** l + 1
        eq = np.full((n, n), INF)
        np.fill_diagonal(eq, 0.0)
        factors.append(count(prev, (f"n{l}", "id"), eq, name=f"n{l}=sum"))
        prev = (f"n{l}", "id")

    parts = chain_parts(table, utab)
    marg, _, logz = forward_backward(*parts)
    exact = {
        "density": float(marg[:, 1:].sum(1).mean()),
        "bottom_density": float(marg[:, 1].mean()),
        "empty_xcol_frac": float(np.exp(-logz)),     # the all-air column has weight 1
    }
    # gap_len is a per-world pooled ratio: exact-sampling Monte Carlo (fixed seed)
    rng = np.random.default_rng(12345)
    g = []
    for _ in range(4):
        ks = sample_chains(rng, W * 500, parts)
        g += [column_stats(ks[:, i * W:(i + 1) * W])["gap_len"] for i in range(500)]
    exact["gap_len"] = float(np.nanmean(g))

    def probes(seed, region):
        rng = np.random.default_rng(seed)
        return [lift(sample_chains(rng, region.W, parts)) for _ in range(4)]

    def render(x):
        k = np.asarray(x["k"])
        img = np.zeros(k.shape + (3,), np.uint8)
        img[:] = (235, 240, 250)
        shade = (60 + 12 * k).clip(0, 255).astype(np.uint8)
        img[k > 0] = np.stack([shade, shade // 2 + 40, 30 + 0 * shade], -1)[k > 0]
        img[k == 1] = (200, 40, 40)
        return np.repeat(np.repeat(img, 2, 0), 2, 1)

    return Problem(
        name="columns", T=T, H=H, chans=chans, factors=factors,
        W_train=W, W_eval=W, stats=column_stats, render=render, probes=probes,
        exact=exact,
        tol={"density": 0.01, "bottom_density": 0.002, "gap_len": 0.1,
             "empty_xcol_frac": 0.01},
        notes="Columns of exactly 12 cells (k counts up from the bottom), air pads above and "
              "below the strip, -0.6 per column cell, hard count channels n1..n4. Fine "
              "columns are iid 1-D chains: density, bottom density and P(empty x-column) "
              "exact by forward-backward; gap_len (mean air run between consecutive columns "
              "in an x-column, pooled per world) by exact-sampling MC (2000 worlds, s.e. 0.003).",
    )
