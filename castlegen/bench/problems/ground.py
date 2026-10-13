"""ground: the certland target (worktree-certland castlegen/channels/certland.py)
on the bench strip (notes/bench.md).

T = 4, H = 64.  Level l has side h = 2^l fine cells.  At every level l >= 1:
  rho{l}   0..h^2        solid tiles in the block
  cert{l}  (A, B)        0 <= A <= B <= h: A promised full columns, B allowed
                         occupied columns; values in certland's cert_domain
                         order, views "A" and "B"
At level 0 the tile t in {0, 1}; its certificate is the tile (views "A" and
"B" both = t), its density is the tile.

Designed terms (certland's only hand-written ones, kappa = 1, lam = 0.5,
beta = 4 as certland's World defaults):
  count prior  kappa (sum of the 4 children's rho - parent rho)^2 / h_c^2,
               h_c the CHILDREN's side: certland's _designed_rho evaluates it
               with the side of the level being sampled (the child level)
  smoothness   lam |rho_p - rho_q| / h, horizontal neighbours, every level
  top noise    beta (rho4 - 256 f)^2 / 256, f = fill / 16 (designed channel)
  support      B(p) <= A(p below), below the strip is full (hard)
               (tiles: dirt only on dirt)
  row honour   per row of a parent's 2 x 2 children: sum A >= A_parent,
               sum B <= B_parent (hard)
x is periodic (as in certland) or a window of the infinite strip; the top row
has no constraint from above (certland: y > 0 only).

The fill target: a smooth ground line g(x) = H (0.35 + 0.3 n(x)) fine rows
from the top, n = value_noise_1d(cell 32); a fine cell (y, x) is below the
line iff y >= floor(g(x) + 0.5); f of a top block is the fraction of its fine
cells below the line, quantised to 17 levels (fill = round(16 f)).
"""
from __future__ import annotations

import numpy as np

from .. import noise
from ..problem import INF, Chan, Problem, count, pair, value_noise_1d

T, H, W = 4, 64, 128
KAPPA, LAM, BETA = 1.0, 0.5, 4.0
CELL = 32
TAG_LINE = noise.tag_of("ground/line")
TAG_PROBE = noise.tag_of("ground/probe")


# ------------------------------------------------------------------ domains
def cert_domain(h):
    """(A, B) arrays over the certificate values of side h (certland order)."""
    if h == 1:
        return np.array([0, 1], np.int64), np.array([0, 1], np.int64)
    ab = [(a, b) for b in range(h + 1) for a in range(b + 1)]
    return np.array([a for a, _ in ab], np.int64), np.array([b for _, b in ab], np.int64)


def cert_index(h):
    A, B = cert_domain(h)
    idx = np.full((h + 1, h + 1), -1, np.int64)
    idx[A, B] = np.arange(A.size)
    return idx


# ------------------------------------------------------------------ the designed line
def ground_line(seed, Xf, period_f=None):
    """g(x) in fine rows from the top at fine columns Xf (absolute)."""
    return H * (0.35 + 0.3 * value_noise_1d(seed, TAG_LINE, Xf, CELL, period_f))


def surface_row(seed, Xf, period_f=None):
    """First dirt row of the designed line: dirt iff y >= surface_row."""
    return np.floor(ground_line(seed, Xf, period_f) + 0.5).astype(np.int64)


def fill_designed(seed, Y, X, period):
    """Designed fill of top blocks (level-4 coordinates) in 0..16."""
    Y = np.asarray(Y, np.int64)
    X = np.asarray(X, np.int64)
    s = 1 << T
    Xf = X[..., None] * s + np.arange(s)
    pf = None if period is None else period * s
    sr = surface_row(seed, Xf, pf)
    below = np.clip((Y[..., None] + 1) * s - sr, 0, s)          # dirt rows of each fine column in the block
    f = below.sum(-1) / (s * s)
    return np.rint(16 * f).astype(np.int64)


# ------------------------------------------------------------------ build
def _chans():
    ch = [Chan("fill", T, 17, designed=fill_designed)]
    for l in range(T, 0, -1):
        h = 1 << l
        A, B = cert_domain(h)
        ch.append(Chan(f"rho{l}", l, h * h + 1))
        ch.append(Chan(f"cert{l}", l, A.size, views={"A": A, "B": B}))
    ch.append(Chan("t", 0, 2, views={"A": [0, 1], "B": [0, 1]}))
    return ch


def _rho(l):
    return "t" if l == 0 else f"rho{l}"


def _cert(l):
    return "t" if l == 0 else f"cert{l}"


def _factors():
    fs = []
    # top noise
    r = np.arange(257)[:, None]
    f = np.arange(17)[None, :] / 16
    fs.append(pair((_rho(T), "id"), ("fill", "id"), (0, 0), BETA * (r - 256 * f) ** 2 / 256,
                   name="noise4"))
    for l in range(T, -1, -1):
        h = 1 << l
        D = h * h + 1
        v = np.arange(D)
        # smoothness
        fs.append(pair((_rho(l), "id"), (_rho(l), "id"), (0, 1),
                       LAM * np.abs(v[:, None] - v[None, :]) / h, name=f"smooth{l}"))
        # support: B(p) <= A(p below); below the strip is full
        Bv, Av = np.arange(h + 1)[:, None], np.arange(h + 1)[None, :]
        fs.append(pair((_cert(l), "B"), (_cert(l), "A"), (1, 0), np.where(Bv > Av, INF, 0.0),
                       pad_b=h, name=f"support{l}"))
        if l == 0:
            continue
        hc = h >> 1
        # count prior between the children's rho and this level's rho
        s = np.arange(4 * hc * hc + 1)[None, :]
        rr = v[:, None]
        fs.append(count((_rho(l - 1), "id"), (_rho(l), "id"), KAPPA * (s - rr) ** 2 / (hc * hc),
                        name=f"prior{l}"))
        # row honour, per child row
        P = np.arange(h + 1)[:, None]
        s = np.arange(2 * hc + 1)[None, :]
        for row, sub in (("top", (0, 1)), ("bot", (2, 3))):
            fs.append(count((_cert(l - 1), "A"), (_cert(l), "A"), np.where(s < P, INF, 0.0),
                            subset=sub, name=f"honourA{l}{row}"))
            fs.append(count((_cert(l - 1), "B"), (_cert(l), "B"), np.where(s > P, INF, 0.0),
                            subset=sub, name=f"honourB{l}{row}"))
    return fs


# ------------------------------------------------------------------ lifting, stats, render
def lift(t):
    """Fine tiles (support-valid) -> every sampled channel: rho exact counts,
    cert A = full columns of the block, B = occupied columns."""
    t = np.asarray(t, np.int64)
    out = {"t": t}
    for l in range(1, T + 1):
        h = 1 << l
        hh, ww = t.shape[0] // h, t.shape[1] // h
        b = t.reshape(hh, h, ww, h)
        col = b.sum(1)
        out[f"rho{l}"] = b.sum((1, 3))
        out[f"cert{l}"] = cert_index(h)[(col == h).sum(2), (col > 0).sum(2)]
    return out


def _line_from_fill(fill, w):
    """Ground line reconstructed from the fill channel: each top column's
    fine dirt rows per fine column is sum_Y fill (exact up to quantisation),
    placed at the column's centre and interpolated periodically."""
    s = 1 << T
    gX = H - fill.sum(0).astype(np.float64)
    cx = np.arange(gX.size) * s + (s - 1) / 2
    x = np.arange(w)
    return np.interp(x, cx, gX, period=w)


def stats(x):
    t = np.asarray(x["t"], np.int64)
    h, w = t.shape
    dirt = t.astype(bool)
    has = dirt.any(0)
    top = np.where(has, dirt.argmax(0), h)
    both = has[:-1] & has[1:]
    rough = float(np.abs(np.diff(top))[both].mean()) if both.any() else 0.0
    # dirt cells with air somewhere below in their column
    below_air = np.flip(np.maximum.accumulate(np.flip(~dirt, 0), 0), 0)
    below_air = np.concatenate([below_air[1:], np.zeros((1, w), bool)], 0)
    floating = float((dirt & below_air).sum())
    fill = np.asarray(x["fill"], np.int64)
    g = _line_from_fill(fill, w)
    yc = np.arange(h)[:, None] + 0.5
    near = np.abs(yc - g[None, :]) <= 2
    match = float((dirt == (yc > g[None, :]))[near].mean()) if near.any() else 1.0
    f = fill / 16
    fill_err = float(np.abs(np.asarray(x["rho4"]) - 256 * f).mean() / 256)
    df = float(dirt.mean())
    return {"dirt_frac": df, "dirt_excess": df - float(f.mean()), "roughness": rough,
            "line_match": match, "floating": floating, "fill_err": fill_err}


def render(x, scale=2):
    t = np.asarray(x["t"], np.int64)
    col = np.array([[169, 212, 240], [138, 98, 66]], np.uint8)[t]
    return np.repeat(np.repeat(col, scale, 0), scale, 1)


def probes(seed, region, n=4):
    """Valid states by construction: dirt below a jittered designed line
    (monotone columns), every coarser channel lifted exactly (lift)."""
    w = region.W
    Xf = np.arange(w) + region.x0
    pf = w if region.periodic else None
    Xk = np.mod(Xf, w) if region.periodic else Xf
    s = 1 << T
    Y4, X4 = np.meshgrid(np.arange(H // s), np.arange(w // s) + (region.x0 // s), indexing="ij")
    fill = fill_designed(seed, Y4, X4, (w // s) if region.periodic else None)
    out = []
    for k in range(n):
        sr = surface_row(seed, Xf, pf)
        amp = k                                                  # 0, 1, 2, 3 rows of jitter
        u = noise.uniform(np, seed, TAG_PROBE, k, np.zeros_like(Xk), Xk)
        sr = np.clip(sr + np.rint((2 * u - 1) * amp).astype(np.int64), 0, H)
        t = (np.arange(H)[:, None] >= sr[None, :]).astype(np.int64)
        x = lift(t)
        x["fill"] = fill.copy()
        out.append(x)
    return out


def build_ground():
    return Problem(
        name="ground", T=T, H=H, chans=_chans(), factors=_factors(), W_train=W, W_eval=W,
        stats=stats, render=render, probes=probes, exact={},
        tol={"dirt_frac": 0.03, "dirt_excess": 0.02, "roughness": 0.15, "line_match": 0.03,
             "floating": 0.5, "fill_err": 0.01},
        notes="certland terrain: rho and (A, B) column certificates at h = 16..2, tiles at h = 1; "
              "designed count prior, smoothness, top fill noise (soft) and support, row honour (hard); "
              "fill from a smooth designed ground line (value noise, cell 32).")
