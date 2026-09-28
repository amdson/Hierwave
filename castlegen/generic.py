"""Generic constrained texture sampler: exemplar texture + hard local rules +
hierarchical promises, with nothing specific to one constraint or one view.

Target over the tile map x (n x W, torus horizontally; above row 0 and below
row n-1 the exemplar's own top / bottom rows continue forever, the boundary
of texsyn bounds="edge"):

    p(x, S | P) ∝ exp(-(E_ex(x, S) + E_loc(x)) / T) * prod_rules 1[rule] * prod_c 1[leaf_c(x) = P_c]

E_loc is a unary + nearest-neighbour pair energy on the tiles, learned from
the exemplar by pseudo-likelihood (fit_local): the detail finer than the
patch scales.

E_ex is the multi-scale patch energy of castlegen.exchain on one-hot tile
features (the image itself, as in texture optimisation), with latent
exemplar coordinates S_h.  Rules are hard pair relations between
neighbouring tiles (e.g. support: a solid tile needs a solid tile below).

A constraint c supplies only
    K, V                 leaf block side, number of values;
    leaf(blocks)         (..., K, K) tiles -> (...) values;
    merge(children)      (..., 2, 2) child values [dy, dx] -> (...) parent values (exact);
    compat               (2, V, V) bool hard seam rule between level-K blocks,
                         [0] left|right, [1] upper/lower, such that a grid of
                         level-K values is realisable iff every seam is compatible;
    witness(P)           a tile map realising a compatible level-K grid;
and, for a start near the texture (guided_witness), a sequential realisation:
    build_order          the order blocks are realised in;
    block_start          a realisation of one block valid given those before it;
    extendable(blk)      whether a block keeps the unrealised blocks realisable;
    dist(a, b)           distance between values (the texture coupling).

Stages (all sample; nothing projects or takes an argmin):
  1. coordinates  texsyn (bounded, Gumbel-max corrections at coord_T) -> x_tex;
  2. promises     P_K ~ exp(-sum_h [tables_h(merge^h P_K) + lam dist(merge^h P_K,
                  merge^h leaf(x_tex))] / T) * 1[compatible], coarse values being
                  exact functions of P_K; exact single-block heat-bath,
                  annealed from T_hot to T, then at T;
  3. tiles        start from guided_witness(P_K, hint=x_tex), exact single-site heat-bath over every
                  tile, restricted to the tiles that keep every rule and every
                  block's leaf value, weighted by exp(-dE_ex / T); S_h resampled
                  exactly every s_every sweeps.
"""
from __future__ import annotations

import numpy as np

from castlegen import hier, texsyn
from castlegen.exchain import OFF, VPAD, _cols_kernel, _gauss, _rows_kernel


def _gumbel_pick(logits, rng):
    """Gumbel-max per row of (k, S) logits (-inf = forbidden)."""
    assert np.isfinite(logits.max(1)).all(), "a site has no allowed value"
    return np.argmax(logits + rng.gumbel(size=logits.shape), 1)


def _children(g):
    """(N, R, C) -> (N, R/2, C/2, 2, 2) child values [dy, dx] of each parent."""
    N, R, C = g.shape
    return g.reshape(N, R // 2, 2, C // 2, 2).transpose(0, 1, 3, 2, 4)


# ---------------------------------------------------------------- constraints
class AverageFill:
    """Average fill: every block has 2 sub-columns of w = K/2 columns; a
    level-K value is b_j = floor(solid cells in sub-column j / K) in 0..w
    (full-column equivalents), a parent's b_j = floor(sum of the 4 child
    sub-column values under it / 4).  Under the support rule matter in the
    upper block of a seam needs at least b_u full columns below it, and the
    lower block holds at most b_l, so compat[1] = (b_u == 0 or b_u <= b_l)
    per sub-column; the witness (b_j full columns at the left of each
    sub-column) shows that is also sufficient."""

    def __init__(self, ts, K=16):
        self.solid = np.asarray(ts.solid, bool)
        self.K, self.w = K, K // 2
        self.P = self.w + 1
        self.V = self.P ** 2
        v = np.arange(self.V)
        self.SUB = np.stack([v // self.P, v % self.P], 1)                   # (V, 2)
        bu, bl = self.SUB[:, None], self.SUB[None, :]
        self.compat = np.stack([np.ones((self.V, self.V), bool), ((bu == 0) | (bu <= bl)).all(-1)])

    def encode(self, b):
        return b[..., 0] * self.P + b[..., 1]

    def leaf(self, blocks):
        s = self.solid[blocks]                                             # (..., K, K)
        cnt = s.reshape(s.shape[:-1] + (2, self.w)).sum((-3, -1))          # (..., 2)
        return self.encode(cnt // self.K)

    def merge(self, c):
        sub = self.SUB[c]                                                  # (..., 2, 2, 2) [dy, dx, j]
        return self.encode(sub.sum((-3, -1)) // 4)

    def dist(self, a, b):
        return np.abs(self.SUB[a] - self.SUB[b]).sum(-1)

    def witness(self, P, fill, empty):
        """fill / empty: (W,) solid / non-solid tile per column."""
        R, C = P.shape
        K, w = self.K, self.w
        b = self.SUB[P]                                                    # (R, C, 2)
        col = np.arange(K)
        full = (col % w)[None, None, :] < b[:, :, col // w]                # (R, C, K) full columns
        s = np.repeat(full, K, 0).reshape(R * K, C * K)
        return np.where(s, fill[None], empty[None]).astype(np.int32)

    # ---- sequential realisation (guided_witness)
    def build_order(self, R, C):
        """Bottom-up: a block's support comes from the block below."""
        return [(y, x) for y in range(R - 1, -1, -1) for x in range(C)]

    def extendable(self, blk, by, bx, P):
        """(...) bool: blk (..., K, K) leaves the block above realisable, i.e.
        has at least b_u full columns in each sub-column."""
        if by == 0:
            return np.ones(blk.shape[:-2], bool)
        full = self.solid[blk].all(-2).reshape(blk.shape[:-2] + (2, self.w)).sum(-1)     # (..., 2)
        return (full >= self.SUB[P[by - 1, bx]]).all(-1)

    def block_start(self, P, by, bx, x, fill, empty, rng, hint=None, tau=1.0):
        """(K, K) realisation of block (by, bx) given the realised block below
        (a column there is full iff the cell under this block is solid).  Per
        sub-column, a total s in the feasible part of the bin and, under a
        non-empty block above, b_j full columns (the widest support the
        block's own mass allows, >= the b_u that block needs) as one run of
        consecutive supported columns (none under an empty one); the rest of
        s spread evenly (bottom-aligned) over the other supported columns.  So
        a partly filled block starts as a plateau, and a block above can
        spread its matter instead of standing on a few thin columns.
        (s, run position) are drawn uniformly, or with a hint map with
        probability ∝ exp(-#cells whose solidity differs from the hint / tau)
        (the spare cells of the even spread going to the columns where the
        hint is tallest)."""
        K, w = self.K, self.w
        y0, x0 = by * K, bx * K
        below = self.solid[x[y0 + K, x0:x0 + K]] if y0 + K < x.shape[0] else np.ones(K, bool)
        need = self.SUB[P[by - 1, bx]] if by > 0 else np.zeros(2, np.int64)
        hs = None if hint is None else self.solid[hint[y0:y0 + K, x0:x0 + K]]    # (K, K)
        rows = np.arange(K)[:, None]
        hgt = np.zeros(K, np.int64)
        for j, b in enumerate(self.SUB[P[by, bx]]):
            elig = np.flatnonzero(below[j * w:(j + 1) * w]) + j * w
            f = min(int(b), len(elig)) if need[j] else 0          # widest support the block's own mass allows
            lo, hi = max(b * K, f * K), min((b + 1) * K - 1 if b < w else w * K, len(elig) * K)
            assert lo <= hi, "the block below is not extendable"
            opts = []
            for tot in range(lo, hi + 1):
                for a in range(len(elig) - f + 1):
                    full, rest = elig[a:a + f], np.concatenate([elig[:a], elig[a + f:]])
                    hj = np.zeros(K, np.int64)
                    hj[full] = K
                    if len(rest):
                        q, extra = divmod(tot - f * K, len(rest))
                        hj[rest] = q
                        if extra:
                            pick = (rest[np.argsort(-hs[:, rest].sum(0), kind="stable")[:extra]] if hs is not None
                                    else rng.choice(rest, extra, replace=False))
                            hj[pick] += 1
                    opts.append(hj[j * w:(j + 1) * w])
            if hs is None:
                k = int(rng.integers(len(opts)))
            else:
                sl = slice(j * w, (j + 1) * w)
                mis = np.array([((rows >= K - o[None, :]) != hs[:, sl]).sum() for o in opts], float)
                k = int(_gumbel_pick(-mis[None] / tau, rng)[0])
            hgt[j * w:(j + 1) * w] = opts[k]
        s = rows >= K - hgt[None, :]                                           # bottom-aligned
        return np.where(s, fill[None, x0:x0 + K], empty[None, x0:x0 + K]).astype(np.int32)


class SurfaceClass:
    """Surface class: every block has 2 sub-columns of w = K/2 columns, each
    E (no solid), S (solid but no full column), P (a full column, not all) or
    F (all solid); V = 16.  Under the support rule matter reaches the bottom
    of its sub-column, so a non-empty upper sub-column needs a full column
    under it and a full one needs all: compat[1] = (u == E or l == F or
    (u in {S, P} and l == P)) per sub-column, sufficient as the upper matter
    can stand on the lower full columns."""

    def __init__(self, ts, K=16):
        self.solid = np.asarray(ts.solid, bool)
        self.K, self.w = K, K // 2
        self.V = 16
        v = np.arange(self.V)
        self.SUB = np.stack([v // 4, v % 4], 1)                             # 0 E, 1 S, 2 P, 3 F
        u, l = self.SUB[:, None], self.SUB[None, :]
        ok = (u == 0) | (l == 3) | (((u == 1) | (u == 2)) & (l == 2))
        self.compat = np.stack([np.ones((self.V, self.V), bool), ok.all(-1)])

    def leaf(self, blocks):
        s = self.solid[blocks]                                             # (..., K, K)
        s = s.reshape(s.shape[:-1] + (2, self.w))                          # (..., K, 2, w)
        cnt, full = s.sum((-3, -1)), s.all(-3).sum(-1)                     # (..., 2)
        c = np.where(cnt == 0, 0, np.where(full == self.w, 3, np.where(full > 0, 2, 1)))
        return c[..., 0] * 4 + c[..., 1]


def support_rule(ts):
    """(2, S, S) allowed pairs [0] left|right, [1] upper/lower: a solid tile
    needs a solid tile below."""
    solid = np.asarray(ts.solid, bool)
    S = len(solid)
    return np.stack([np.ones((S, S), bool), ~solid[:, None] | solid[None, :]])


def fit_local(E, n_sig, l2=0.1):
    """Local energy of the finest level, learned from the exemplar by
    pseudo-likelihood: E_loc(x) = sum_i u[x_i] + sum Eh[left, right] +
    sum Ev[upper, lower], every cell's conditional given its 4 neighbours
    fitted to E (torus horizontally, E's top / bottom rows continued
    vertically, as the sampler's boundary).  l2: ridge on all parameters.
    -> dict(u (S,), Eh (S, S), Ev (S, S))."""
    from scipy.optimize import minimize
    E = np.asarray(E)
    S = n_sig
    X = E.ravel()
    col = np.concatenate([E[:1], E, E[-1:]], 0)
    Lf, Rf = np.roll(E, 1, 1).ravel(), np.roll(E, -1, 1).ravel()
    Uf, Df = col[:-2].ravel(), col[2:].ravel()
    oh = lambda v: np.eye(S)[v]
    OX, OL, OR, OU, OD = oh(X), oh(Lf), oh(Rf), oh(Uf), oh(Df)

    def unpack(th):
        return th[:S], th[S:S + S * S].reshape(S, S), th[S + S * S:].reshape(S, S)

    def f(th):
        u, Eh, Ev = unpack(th)
        lg = -(u[None] + Eh[Lf] + Eh[:, Rf].T + Ev[Uf] + Ev[:, Df].T)            # (N, S)
        mx = lg.max(1, keepdims=True)
        z = np.exp(lg - mx)
        lse = mx[:, 0] + np.log(z.sum(1))
        nll = float((lse - lg[np.arange(len(X)), X]).sum()) + 0.5 * l2 * float(th @ th)
        G = z / z.sum(1, keepdims=True) - OX                                    # d nll / d logits
        gu = -G.sum(0)
        gh = -(OL.T @ G) - (G.T @ OR)
        gv = -(OU.T @ G) - (G.T @ OD)
        return nll, np.concatenate([gu, gh.ravel(), gv.ravel()]) + l2 * th

    th = minimize(f, np.zeros(S + 2 * S * S), jac=True, method="L-BFGS-B").x
    u, Eh, Ev = unpack(th)
    return dict(u=u, Eh=Eh, Ev=Ev)


def guided_witness(con, P, hint, rule, fill, empty, rng, tau=1.0, sweeps=4, local=None):
    """A random realisation of the level-K grid P near the hint map.  Blocks
    are realised one at a time in con.build_order; each starts from
    con.block_start (valid given the blocks realised so far) and then takes
    `sweeps` single-site heat-bath sweeps, raster scans alternating bottom-up
    and top-down (so a run of cells can grow or shrink in one sweep), of

        q(block) ∝ exp(-E_loc(block | realised cells) - #(cells differing from the hint) / tau)

(E_loc: fit_local's energy counting only pairs with realised cells and the
boundary, omitted when local is None; the hint term omitted when hint is None)

    restricted to blocks with leaf = P_b, every rule holding against the
    realised cells and the map boundary (fill below, empty above), and
    con.extendable (the blocks not realised yet stay realisable).  Every
    draw is feasible; the result is only the tile chain's start, so q need
    not be sampled exactly.  Needs: con.build_order, block_start, extendable."""
    n, W = P.shape[0] * con.K, P.shape[1] * con.K
    hint = None if hint is None else np.asarray(hint, np.int32)
    K, S = con.K, rule.shape[1]
    R, C = P.shape
    x = np.zeros((n, W), np.int32)                                           # unrealised cells are never read
    done = np.zeros((R, C), bool)
    for by, bx in con.build_order(R, C):
        known = lambda y, xx: done[y // K, xx // K] or (y // K == by and xx // K == bx)
        y0, x0 = by * K, bx * K
        x[y0:y0 + K, x0:x0 + K] = con.block_start(P, by, bx, x, fill, empty, rng, hint=hint, tau=tau)
        for sw in range(sweeps):
            for k in (range(K * K) if sw % 2 else range(K * K - 1, -1, -1)):     # raster up, then down
                y, xx = y0 + k // K, x0 + k % K
                ok = np.ones(S, bool)
                e = np.zeros(S)
                up = empty[xx] if y == 0 else (x[y - 1, xx] if known(y - 1, xx) else None)
                dn = fill[xx] if y == n - 1 else (x[y + 1, xx] if known(y + 1, xx) else None)
                lf = x[y, (xx - 1) % W] if known(y, (xx - 1) % W) else None
                rt = x[y, (xx + 1) % W] if known(y, (xx + 1) % W) else None
                if up is not None:
                    ok &= rule[1][up]
                if dn is not None:
                    ok &= rule[1][:, dn]
                if lf is not None:
                    ok &= rule[0][lf]
                if rt is not None:
                    ok &= rule[0][:, rt]
                if local is not None:
                    e += local["u"]
                    e += 0 if up is None else local["Ev"][up]
                    e += 0 if dn is None else local["Ev"][:, dn]
                    e += 0 if lf is None else local["Eh"][lf]
                    e += 0 if rt is None else local["Eh"][:, rt]
                if hint is not None:
                    e += (np.arange(S) != hint[y, xx]) / tau
                blk = np.repeat(x[None, y0:y0 + K, x0:x0 + K], S, 0)
                blk[:, y - y0, xx - x0] = np.arange(S)
                ok &= (con.leaf(blk) == P[by, bx]) & con.extendable(blk, by, bx, P)
                logits = np.where(ok, -e, -np.inf)
                x[y, xx] = _gumbel_pick(logits[None], rng)[0]
        done[by, bx] = True
    return x


def pyramid(con, P, levels):
    """[P, merge(P), ...] for len(levels) levels.  P: (R, C) or (N, R, C)."""
    g = P[None] if P.ndim == 2 else P
    out = [g]
    for _ in levels[1:]:
        g = con.merge(_children(g))
        out.append(g)
    return [o[0] for o in out] if P.ndim == 2 else out


def boundary_values(con, edge_row):
    """(C,) leaf value of a block of K copies of the boundary row (W,)."""
    K = con.K
    blocks = np.repeat(edge_row.reshape(-1, 1, K), K, 1)                  # (C, K, K)
    return con.leaf(blocks)


# ------------------------------------------------------------------- tables
def fit_tables(con, maps, levels, alpha=0.5):
    """Count tables per level h from maps (N, n, W): unary u (V,), pair (2, V,
    V) [orientation, a, b] (negative PMI; horizontal on the torus, vertical
    without wrap), parent (4, V, V) [2 dy + dx, child, parent] (negative PMI)."""
    V = con.V
    grids = {h: [] for h in levels}
    for x in maps:
        R, C = x.shape[0] // con.K, x.shape[1] // con.K
        P = con.leaf(x.reshape(R, con.K, C, con.K).transpose(0, 2, 1, 3))
        for h, g in zip(levels, pyramid(con, P, levels)):
            grids[h].append(g)
    tab = {}
    for i, h in enumerate(levels):
        cu, cp, cc = np.zeros(V), np.zeros((2, V, V)), np.zeros((4, V, V))
        for k, g in enumerate(grids[h]):
            np.add.at(cu, g.ravel(), 1)
            np.add.at(cp[0], (g.ravel(), np.roll(g, -1, 1).ravel()), 1)
            np.add.at(cp[1], (g[:-1].ravel(), g[1:].ravel()), 1)
            if i + 1 < len(levels):
                Gp = grids[levels[i + 1]][k]
                for dy in (0, 1):
                    for dx in (0, 1):
                        np.add.at(cc[2 * dy + dx], (g[dy::2, dx::2].ravel(), Gp.ravel()), 1)
        pu = (cu + alpha) / (cu.sum() + V * alpha)
        pp = (cp + alpha) / (cp.sum((1, 2), keepdims=True) + V * V * alpha)
        pair = -np.log(pp) + np.log(pp.sum(2, keepdims=True)) + np.log(pp.sum(1, keepdims=True))
        par = np.zeros((4, V, V))
        if i + 1 < len(levels):
            pc = (cc + alpha) / (cc.sum(1, keepdims=True) + V * alpha)
            par = -np.log(pc) + np.log(pu)[None, :, None]
        tab[h] = dict(u=-np.log(pu), pair=pair, parent=par)
    return tab


def _table_layout(V, levels):
    """Offsets of every level's tables in one flat parameter vector."""
    off, lay = 0, {}
    for i, h in enumerate(levels):
        lay[h] = dict(u=off, ph=off + V, pv=off + V + V * V)
        off += V + 2 * V * V
        if i + 1 < len(levels):
            lay[h]["par"] = off
            off += 4 * V * V
    return lay, off


def _energy_index(con, grids, levels, lay):
    """(N, n_terms) indices into the flat parameters whose sum is each grid's
    promise energy (PromiseSampler.energy without the texture term).  grids:
    (N, R, C) level-K grids."""
    V = con.V
    cols, g = [], grids
    for i, h in enumerate(levels):
        N = g.shape[0]
        L = lay[h]
        cols += [L["u"] + g.reshape(N, -1),
                 L["ph"] + (g * V + np.roll(g, -1, 2)).reshape(N, -1),
                 L["pv"] + (g[:, :-1] * V + g[:, 1:]).reshape(N, -1)]
        if i + 1 < len(levels):
            Gp = con.merge(_children(g))
            for dy in (0, 1):
                for dx in (0, 1):
                    cols.append(L["par"] + (2 * dy + dx) * V * V + (g[:, dy::2, dx::2] * V + Gp).reshape(N, -1))
            g = Gp
    return np.concatenate(cols, 1)


def fit_tables_pl(con, maps, levels, top, bottom, l2=1.0, maxiter=300, init=None, log=None):
    """Promise tables (fit_tables' format) fitted by maximum pseudo-likelihood:
    every level-K block's conditional given the rest of its grid - exactly the
    distribution PromiseSampler's heat-bath draws from (coarse values merged,
    candidates restricted by compat and the boundary values top / bottom) -
    fitted to the corpus maps' own grids, with a ridge l2 on all parameters.
    init: tables to start from (e.g. fit_tables')."""
    import jax
    with jax.enable_x64(True):
        return _fit_tables_pl(con, maps, levels, top, bottom, l2, maxiter, init, log)


def _fit_tables_pl(con, maps, levels, top, bottom, l2, maxiter, init, log):
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    from scipy.optimize import minimize
    V, K, cp = con.V, con.K, con.compat
    lay, D = _table_layout(V, levels)
    IDX, OK, TRUE = [], [], []
    for x in maps:
        R, C = x.shape[0] // K, x.shape[1] // K
        P = con.leaf(x.reshape(R, K, C, K).transpose(0, 2, 1, 3))
        for y in range(R):
            for xx in range(C):
                up = P[y - 1, xx] if y else top[xx]
                dn = P[y + 1, xx] if y + 1 < R else bottom[xx]
                cand = np.arange(V)
                ok = cp[1][up, cand] & cp[1][cand, dn] & cp[0][P[y, (xx - 1) % C], cand] & cp[0][cand, P[y, (xx + 1) % C]]
                Pb = np.repeat(P[None], V, 0)
                Pb[:, y, xx] = cand
                IDX.append(_energy_index(con, Pb, levels, lay))
                OK.append(ok)
                TRUE.append(P[y, xx])
    IDX, OK, TRUE = jnp.asarray(np.stack(IDX)), jnp.asarray(np.stack(OK)), jnp.asarray(np.array(TRUE))

    def obj(th):
        lg = -th[IDX].sum(-1)                                                  # (blocks, V)
        lg = jnp.where(OK, lg, -jnp.inf)
        nll = (logsumexp(lg, 1) - jnp.take_along_axis(lg, TRUE[:, None], 1)[:, 0]).sum()
        return nll + 0.5 * l2 * jnp.sum(th ** 2)

    vg = jax.jit(jax.value_and_grad(obj))
    th0 = np.zeros(D)
    if init is not None:
        for i, h in enumerate(levels):
            L, t = lay[h], init[h]
            th0[L["u"]:L["u"] + V] = t["u"]
            th0[L["ph"]:L["ph"] + V * V] = t["pair"][0].ravel()
            th0[L["pv"]:L["pv"] + V * V] = t["pair"][1].ravel()
            if "par" in L:
                th0[L["par"]:L["par"] + 4 * V * V] = t["parent"].ravel()
    res = minimize(lambda th: tuple(np.asarray(t, float) for t in vg(jnp.asarray(th))), th0, jac=True,
                   method="L-BFGS-B", options=dict(maxiter=maxiter))
    if log:
        log(f"fit_tables_pl: {len(TRUE)} blocks, nll/block {res.fun / len(TRUE):.3f} ({res.nit} it)")
    th = res.x
    tab = {}
    for i, h in enumerate(levels):
        L = lay[h]
        tab[h] = dict(u=th[L["u"]:L["u"] + V], pair=np.stack([th[L["ph"]:L["ph"] + V * V].reshape(V, V),
                                                                th[L["pv"]:L["pv"] + V * V].reshape(V, V)]),
                      parent=th[L["par"]:L["par"] + 4 * V * V].reshape(4, V, V) if "par" in L else np.zeros((4, V, V)))
    return tab


def pl_nll(con, tab, maps, levels, top, bottom):
    """Mean negative log pseudo-likelihood per block of maps under tables tab."""
    lay, D = _table_layout(con.V, levels)
    th = np.zeros(D)
    V = con.V
    for h in levels:
        L, t = lay[h], tab[h]
        th[L["u"]:L["u"] + V] = t["u"]
        th[L["ph"]:L["ph"] + V * V] = t["pair"][0].ravel()
        th[L["pv"]:L["pv"] + V * V] = t["pair"][1].ravel()
        if "par" in L:
            th[L["par"]:L["par"] + 4 * V * V] = t["parent"].ravel()
    tot, n, K, cp = 0.0, 0, con.K, con.compat
    for x in maps:
        R, C = x.shape[0] // K, x.shape[1] // K
        P = con.leaf(x.reshape(R, K, C, K).transpose(0, 2, 1, 3))
        for y in range(R):
            for xx in range(C):
                up = P[y - 1, xx] if y else top[xx]
                dn = P[y + 1, xx] if y + 1 < R else bottom[xx]
                cand = np.arange(V)
                ok = cp[1][up, cand] & cp[1][cand, dn] & cp[0][P[y, (xx - 1) % C], cand] & cp[0][cand, P[y, (xx + 1) % C]]
                Pb = np.repeat(P[None], V, 0)
                Pb[:, y, xx] = cand
                lg = np.where(ok, -th[_energy_index(con, Pb, levels, lay)].sum(-1), -np.inf)
                m = lg.max()
                tot += m + np.log(np.exp(lg - m).sum()) - lg[P[y, xx]]
                n += 1
    return tot / n


# ----------------------------------------------------------------- promises
class PromiseSampler:
    """Level-K promise grid with soft coarse levels (module docstring, stage 2).
    targets: [(R_h, C_h) value grid per level] the texture's own values;
    top, bottom: (C,) boundary values above and below the grid."""

    def __init__(self, con, tables, levels, targets, top, bottom, lam=1.0, T=1.0, seed=0):
        self.con, self.tab, self.levels = con, tables, levels
        self.targets, self.top, self.bottom = targets, top, bottom
        self.lam, self.T = lam, T
        self.rng = np.random.default_rng(seed)
        R, C = targets[0].shape
        self.P = np.repeat(top[None], R, 0)                                # all sky: compatible by assumption
        assert self._compatible(self.P), "the top boundary value stacked on itself must be compatible"

    def energy(self, Pb):
        """(N,) energy of a batch of level-K grids (N, R, C)."""
        con, tot, g = self.con, 0.0, Pb
        for i, h in enumerate(self.levels):
            t = self.tab[h]
            tot = tot + t["u"][g].sum((1, 2)) + t["pair"][0][g, np.roll(g, -1, 2)].sum((1, 2)) \
                + t["pair"][1][g[:, :-1], g[:, 1:]].sum((1, 2)) + self.lam * con.dist(g, self.targets[i][None]).sum((1, 2))
            if i + 1 < len(self.levels):
                Gp = con.merge(_children(g))
                for dy in (0, 1):
                    for dx in (0, 1):
                        tot = tot + t["parent"][2 * dy + dx][g[:, dy::2, dx::2], Gp].sum((1, 2))
                g = Gp
        return tot

    def _compatible(self, P):
        cp = self.con.compat
        col = np.concatenate([self.top[None], P, self.bottom[None]], 0)
        return bool(cp[1][col[:-1], col[1:]].all() and cp[0][P, np.roll(P, -1, 1)].all())

    def sweep(self, T=None):
        T = self.T if T is None else T
        P, cp, V = self.P, self.con.compat, self.con.V
        R, C = P.shape
        cand = np.arange(V)
        for k in self.rng.permutation(R * C):
            y, x = divmod(int(k), C)
            up = P[y - 1, x] if y else self.top[x]
            dn = P[y + 1, x] if y + 1 < R else self.bottom[x]
            ok = cp[1][up, cand] & cp[1][cand, dn] & cp[0][P[y, (x - 1) % C], cand] & cp[0][cand, P[y, (x + 1) % C]]
            Pb = np.repeat(P[None], V, 0)
            Pb[:, y, x] = cand
            logits = np.where(ok, -self.energy(Pb) / T, -np.inf)
            P[y, x] = _gumbel_pick(logits[None], self.rng)[0]

    def run(self, sweeps, T_hot=10.0):
        """Anneal geometrically from T_hot to T over the first half of the
        sweeps (the all-sky start is a metastable state of single-block
        moves), then sample at T."""
        hot = sweeps // 2
        for s in range(sweeps):
            self.sweep(self.T * (T_hot / self.T) ** (1 - s / hot) if s < hot else self.T)
        assert self._compatible(self.P)
        return self.P


# -------------------------------------------------------------------- tiles
def _patches_bank(Phi, E, h):
    """(P, 25 D) blurred exemplar neighbourhoods at spacing h for every column
    and every centre row from 4h above to 4h below the exemplar, which
    continues its top / bottom rows (per column) beyond its edges."""
    E = np.asarray(E)
    s = Phi[E]
    m, W = s.shape[:2]
    rows = np.arange(-4 * h - 2 * h, m + 4 * h + 2 * h + 1)
    Ry, below, above = _rows_kernel(rows, m, h, above=True)
    Kx = np.array([np.roll(_gauss((np.arange(W) + W / 2) % W - W / 2, h), i) for i in range(W)])
    Kx = Kx / Kx.sum(1, keepdims=True)
    B = (np.einsum("ry,yxd,cx->rcd", Ry, s, Kx, optimize=True) + np.einsum("r,cx,xd->rcd", below, Kx, Phi[E[-1]])
         + np.einsum("r,cx,xd->rcd", above, Kx, Phi[E[0]]))
    off = 2 * h
    cen = np.arange(off, len(rows) - off)
    cols = [np.roll(B[cen + dy * h], -dx * h, 1).reshape(-1, B.shape[-1]) for dy, dx in OFF]
    return np.stack(cols, 1).reshape(len(cols[0]), -1)


class TileChain:
    """Exact MCMC for p(x, S | P) (module docstring, stage 3).
    cons / Ps: constraints and their level-K grids; rule: (2, S, S) allowed
    pairs; E: exemplar; w: {h: weight} of E_ex."""

    def __init__(self, ts, x, E, cons, Ps, rule, w, T=0.9, s_every=5, seed=0, exact_budget=1e8, n_pca=16, knn=8,
                 eps=0.05, mh_steps=4, local=None):
        self.t = np.array(x, np.int32)
        self.n, self.W = self.t.shape
        self.E = np.asarray(E)
        m = self.E.shape[1]
        self.top, self.bot = self.E[0, np.arange(self.W) % m], self.E[-1, np.arange(self.W) % m]
        self.cons, self.Ps, self.rule = list(cons), [np.asarray(P) for P in Ps], rule
        self.T, self.s_every = float(T), int(s_every)
        self.knn, self.eps, self.mh_steps = int(knn), float(eps), int(mh_steps)
        self.local = local                                                       # fit_local's dict, or None
        self.rng = np.random.default_rng(seed)
        self.Phi = np.eye(ts.n_sig) / np.sqrt(2.0)
        self.D = self.Phi.shape[1]
        self.sc = {}
        for h, wh in w.items():
            assert self.n % h == 0 and self.W % h == 0
            R, C = self.n // h, self.W // h
            Ry, below, above = _rows_kernel(np.arange(-VPAD, R + VPAD) * h + (h - 1) / 2, self.n, h, above=True)
            Cx = _cols_kernel(C, self.W, h, (h - 1) / 2)
            NE = _patches_bank(self.Phi, self.E, h)
            cv = np.array([sum(0 <= r - VPAD - dy < R for dy in range(-2, 3)) for r in range(R + 2 * VPAD)])
            c0 = np.einsum("r,cx,xd->rcd", below, Cx, self.Phi[self.bot]) + np.einsum("r,cx,xd->rcd", above, Cx, self.Phi[self.top])
            self.sc[h] = dict(w=wh, R=R, Ry=Ry, Cx=Cx, NE=NE, NE2=(NE ** 2).sum(1), c0=c0,
                              cnt=(5.0 * cv)[:, None, None], qy=(5.0 * cv)[:, None] * Ry ** 2, qx=Cx ** 2,
                              exact=R * C * len(NE) <= exact_budget, rows=len(NE) // m, m=m)
            if not self.sc[h]["exact"]:                                              # MH proposals: kNN in PCA space
                from scipy.spatial import cKDTree
                mu = NE.mean(0)
                pc = texsyn._pca((NE - mu).astype(np.float32), n_pca, self.rng)
                self.sc[h].update(mu=mu, pca=pc, tree=cKDTree((NE - mu) @ pc))
        self._refresh()
        for h in self.sc:
            self.resample_S(h)
        self._sweeps = 0
        assert self.valid(), "start map breaks a rule or a promise"

    # ---- checks
    def valid(self, x=None):
        x = self.t if x is None else x
        col = np.concatenate([self.top[None], x, self.bot[None]], 0)
        ok = self.rule[1][col[:-1], col[1:]].all() and self.rule[0][x, np.roll(x, -1, 1)].all()
        for con, P in zip(self.cons, self.Ps):
            K = con.K
            R, C = self.n // K, self.W // K
            ok = ok and (con.leaf(x.reshape(R, K, C, K).transpose(0, 2, 1, 3)) == P).all()
        return bool(ok)

    # ---- E_ex bookkeeping (as castlegen.exchain.ExChain)
    def _refresh(self):
        s = self.Phi[self.t]
        for c in self.sc.values():
            c["B"] = np.einsum("ry,yxd,cx->rcd", c["Ry"], s, c["Cx"], optimize=True) + c["c0"]

    def _patches(self, h):
        c = self.sc[h]
        B, R = c["B"], c["R"]
        P = np.stack([np.roll(B[VPAD + dy:VPAD + dy + R], -dx, 1) for dy, dx in OFF], 2)
        return P.reshape(P.shape[0], P.shape[1], -1)

    def _Q(self, h):
        c = self.sc[h]
        R = c["R"]
        A = c["NE"][c["S"]].reshape(R, -1, len(OFF), self.D)
        Q = np.zeros_like(c["B"])
        for k, (dy, dx) in enumerate(OFF):
            Q[VPAD + dy:VPAD + dy + R] += np.roll(A[:, :, k], dx, 1)
        return Q

    def energy_local(self, x=None):
        """E_loc(x) (fit_local), boundary pairs included."""
        if self.local is None:
            return 0.0
        x = self.t if x is None else x
        u, Eh, Ev = self.local["u"], self.local["Eh"], self.local["Ev"]
        col = np.concatenate([self.top[None], x, self.bot[None]], 0)
        return float(u[x].sum() + Eh[x, np.roll(x, -1, 1)].sum() + Ev[col[:-1], col[1:]].sum())

    def energy_ex(self):
        return sum(c["w"] * float(((self._patches(h) - c["NE"][c["S"]]) ** 2).sum()) for h, c in self.sc.items())

    def resample_S(self, h, chunk=256):
        """Exact heat-bath of S_h given x (the S_h(s) are independent), or
        Metropolis-Hastings (_mh_S) when that costs more than exact_budget."""
        c = self.sc[h]
        if not c["exact"]:
            return self._mh_S(h)
        N = self._patches(h).reshape(-1, len(OFF) * self.D)
        S = np.empty(len(N), np.int64)
        for a in range(0, len(N), chunk):
            Nc = N[a:a + chunk]
            e = c["w"] * ((Nc ** 2).sum(1)[:, None] - 2 * Nc @ c["NE"].T + c["NE2"][None, :])
            S[a:a + chunk] = np.argmin(e / self.T - self.rng.gumbel(size=e.shape), 1)
        c["S"] = S.reshape(c["R"], -1)
        c["Q"] = self._Q(h)

    def _mh_S(self, h):
        """mh_steps Metropolis-Hastings updates of every S_h(s) given x.  The
        proposal for s depends only on x and the other S_h (never on S_h(s)):
        with probability eps uniform over the bank, else uniform over the list
        [k nearest bank patches to N(s) in PCA space] + [the 8 neighbours'
        coordinates shifted back by their offset (rows clipped; the first kNN
        entry for a neighbour off the map)], so
        q(u) = eps / N + (1 - eps) mult(u) / L and the acceptance ratio is
        exact.  Neighbours of s have another 2 x 2 colour, so each colour is
        updated in parallel."""
        c, T, rng = self.sc[h], self.T, self.rng
        R, m, rows = c["R"], c["m"], c["rows"]
        C = self.W // h
        N = self._patches(h).reshape(R * C, -1)
        NE, Nb = c["NE"], len(c["NE"])
        _, knn = c["tree"].query((N - c["mu"]) @ c["pca"], k=self.knn)             # (R C, knn)
        energy = lambda idx, u: c["w"] * ((N[idx] - NE[u]) ** 2).sum(-1)
        if "S" not in c:                                                           # start: a random kNN entry
            c["S"] = knn[np.arange(R * C), rng.integers(self.knn, size=R * C)].reshape(R, C)
        S = c["S"]
        nb = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        L = self.knn + len(nb)
        for _ in range(self.mh_steps):
            for i, j in hier.COLOURS:
                ys, xs = np.meshgrid(np.arange(i, R, 2), np.arange(j, C, 2), indexing="ij")
                ys, xs = ys.ravel(), xs.ravel()
                idx = ys * C + xs
                coh = []
                for dy, dx in nb:                         # a neighbour off the map: a kNN entry instead
                    inside = (ys + dy >= 0) & (ys + dy < R)
                    u = S[np.clip(ys + dy, 0, R - 1), (xs + dx) % C]
                    u = np.clip(u // m - dy * h, 0, rows - 1) * m + (u % m - dx * h) % m
                    coh.append(np.where(inside, u, knn[idx, 0]))
                lst = np.concatenate([knn[idx], np.stack(coh, 1)], 1)                # (k, L)
                cur = S[ys, xs]
                uni = rng.random(len(idx)) < self.eps
                new = np.where(uni, rng.integers(Nb, size=len(idx)), lst[np.arange(len(idx)), rng.integers(L, size=len(idx))])
                q = lambda u: self.eps / Nb + (1 - self.eps) * (lst == u[:, None]).sum(1) / L
                la = -(energy(idx, new) - energy(idx, cur)) / T + np.log(q(cur)) - np.log(q(new))
                acc = np.log(rng.random(len(idx))) < la
                S[ys[acc], xs[acc]] = new[acc]
        c["Q"] = self._Q(h)

    # ---- the move
    def _allowed(self, ys, xs):
        """(k, S) tiles each site may take: every rule and every block's leaf value kept."""
        t, n, W, rl = self.t, self.n, self.W, self.rule
        S = len(self.Phi)
        up = np.where(ys > 0, t[np.maximum(ys - 1, 0), xs], self.top[xs])
        dn = np.where(ys + 1 < n, t[np.minimum(ys + 1, n - 1), xs], self.bot[xs])
        ok = rl[1][up] & rl[1][:, dn].T & rl[0][t[ys, (xs - 1) % W]] & rl[0][:, t[ys, (xs + 1) % W]].T
        for con, P in zip(self.cons, self.Ps):
            K = con.K
            by, bx = ys // K, xs // K
            r = (by * K)[:, None] + np.arange(K)
            cl = (bx * K)[:, None] + np.arange(K)
            blk = np.repeat(t[r[:, :, None], cl[:, None, :]][:, None], S, 1)          # (k, S, K, K)
            blk[np.arange(len(ys))[:, None], np.arange(S)[None], (ys % K)[:, None], (xs % K)[:, None]] = np.arange(S)
            ok &= con.leaf(blk) == P[by, bx][:, None]
        return ok

    def dE(self, ys, xs):
        """(k, S) exact change of E_ex (+ E_loc) if site k alone took each
        tile (sites more than 3 h_max apart add up)."""
        cur = self.t[ys, xs]
        dphi = self.Phi[None] - self.Phi[cur][:, None]                               # (k, S, D)
        out = np.zeros(dphi.shape[:2])
        for c in self.sc.values():
            G2 = 2 * (c["cnt"] * c["B"] - c["Q"])
            lin = np.einsum("rk,rcd,ck->kd", c["Ry"][:, ys], G2, c["Cx"][:, xs])
            quad = c["qy"][:, ys].sum(0) * c["qx"][:, xs].sum(0)
            out += c["w"] * (np.einsum("kd,ksd->ks", lin, dphi) + quad[:, None] * (dphi ** 2).sum(-1))
        if self.local is not None:                                                  # + the change of E_loc
            t, n, W = self.t, self.n, self.W
            u, Eh, Ev = self.local["u"], self.local["Eh"], self.local["Ev"]
            U = np.where(ys > 0, t[np.maximum(ys - 1, 0), xs], self.top[xs])
            D = np.where(ys + 1 < n, t[np.minimum(ys + 1, n - 1), xs], self.bot[xs])
            L, R = t[ys, (xs - 1) % W], t[ys, (xs + 1) % W]
            e = u[None] + Eh[L] + Eh[:, R].T + Ev[U] + Ev[:, D].T                 # (k, S)
            out += e - e[np.arange(len(ys)), cur][:, None]
        return out

    def site_sweep(self):
        """Every cell once; cells more than 3 h_max apart share no E_ex term,
        no rule and no block, so each batch is updated exactly in parallel."""
        t, n, W = self.t, self.n, self.W
        st = 3 * max(self.sc) + 1
        assert st > max((con.K for con in self.cons), default=0)
        for oy, ox in self.rng.permutation([(a, b) for a in range(st) for b in range(st)]):
            ys, xs = np.meshgrid(np.arange(oy, n, st), np.arange(ox, W, st), indexing="ij")
            ys, xs = ys.ravel(), xs.ravel()
            if not len(ys):
                continue
            cur = t[ys, xs]
            logits = -self.dE(ys, xs) / self.T
            new = _gumbel_pick(np.where(self._allowed(ys, xs), logits, -np.inf), self.rng)
            ch = new != cur
            if ch.any():
                for c in self.sc.values():
                    d = self.Phi[new[ch]] - self.Phi[cur[ch]]
                    c["B"] += np.einsum("rk,ck,kd->rcd", c["Ry"][:, ys[ch]], c["Cx"][:, xs[ch]], d)
                t[ys[ch], xs[ch]] = new[ch]

    def sweep(self):
        self.site_sweep()
        self._sweeps += 1
        if self._sweeps % self.s_every == 0:
            for h in self.sc:
                self.resample_S(h)


# ---------------------------------------------------------------- pipeline
class RuleCoordVar(texsyn.CoordVar):
    """Texture synthesis coordinates whose corrections also pay gamma x the
    rule violations a candidate would create on its seams: a cell of side h
    with coordinate u stands for the exemplar window rows / columns
    u - h/2 .. u + h/2 - 1, and each of its four seams compares the window's
    edge row (column) with the facing edge of the neighbour's current window,
    or with the boundary row (E[0] above the map, E[-1] below it) - the
    fraction of the h pairs a rule forbids, summed over the four seams."""

    def __init__(self, an, r, rule, gamma, **kw):
        super().__init__(an, r, **kw)
        self.rule, self.gamma = rule, float(gamma)

    def _edge(self, U, h, side):
        """(..., h) tiles along one edge of the windows at coords U (..., 2);
        side: 'top', 'bottom', 'left', 'right'."""
        an = self.an
        off = np.arange(h) - h // 2
        if side in ("top", "bottom"):
            r = -(h // 2) if side == "top" else h - 1 - h // 2
            return an.E[an.wrap_y(U[..., 0] + r)[..., None], (U[..., 1][..., None] + off) % an.m]
        c = -(h // 2) if side == "left" else h - 1 - h // 2
        return an.E[an.wrap_y(U[..., 0][..., None] + off), ((U[..., 1] + c) % an.m)[..., None]]

    def energy(self, C, level, colour, ctx):
        cost = super().energy(C, level, colour, ctx)
        if not self.gamma:
            return cost
        an, h, rule = self.an, level.h, self.rule
        S = level.vars[self.name]
        R = S.shape[0]
        i, j = colour
        rows = np.arange(i, R, 2)
        nb = lambda dy, dx: self._nb(S, dy, dx)[i::2, j::2]
        m = an.m
        xs = (np.arange(j, S.shape[1], 2)[:, None] * h + np.arange(h) - h // 2) % m     # boundary rows per column
        above = self._edge(nb(-1, 0), h, "bottom")                              # (a, b, h)
        above = np.where((rows == 0)[:, None, None], an.E[0][xs][None], above)
        below = self._edge(nb(1, 0), h, "top")
        below = np.where((rows == R - 1)[:, None, None], an.E[-1][xs][None], below)
        left, right = self._edge(nb(0, -1), h, "right"), self._edge(nb(0, 1), h, "left")
        v = ((~rule[1][above[:, :, None], self._edge(C, h, "top")]).mean(-1)
             + (~rule[1][self._edge(C, h, "bottom"), below[:, :, None]]).mean(-1)
             + (~rule[0][left[:, :, None], self._edge(C, h, "left")]).mean(-1)
             + (~rule[0][self._edge(C, h, "right"), right[:, :, None]]).mean(-1))
        return cost + self.gamma * v



def texture(an, n, seed, coord_T=1.0, kappa=4.0, r=(1, 1), corrections=2, first_corrected=1, h_top=None,
            rule=None, gamma=0.0):
    """Stage 1: bounded texture synthesis -> tile map (n, n).  h_top: start at
    cells of side h_top (default: one cell as large as the map), every cell of
    that level at an independent uniform exemplar coordinate; r then applies
    to the levels below it only.  rule, gamma: corrections also pay gamma x
    the seam rule violations of each candidate (RuleCoordVar)."""
    r = list(r) + [0.0] * (an.L + 1 - len(r))
    jitter = None
    if h_top is not None:
        rng = np.random.default_rng(seed)
        l_top = an.L - hier.level_index(h_top)

        def jitter(l, h, shape):
            if l == l_top:
                return np.stack([rng.integers(0, an.my, shape[:2]), rng.integers(0, an.m, shape[:2])], -1)
            return np.floor(h * r[l] * (2 * rng.random(shape) - 1) + 0.5).astype(np.int64)
    kw = dict(kappa=kappa, corrections=corrections, first_corrected=first_corrected, T=coord_T, jitter=jitter)
    coord = RuleCoordVar(an, r, rule, gamma, **kw) if gamma else texsyn.CoordVar(an, r, **kw)
    coord.h_max = min(coord.h_max, n, h_top or n)
    S = hier.run([coord], [], n, hier.Ctx(seed)).vars[coord.name]
    return an.E[S[..., 0], S[..., 1]].astype(np.int32)


def generate(ts, E, con, tables, levels, an, n=128, seed=0, coord_T=1.0, lam=4.0, T_prom=1.0, prom_sweeps=100,
             w=None, T=0.9, sweeps=20, local=True, tau=0.1, x_tex=None, sub_seed=None, h_top=None, log=None):
    """Stages 1-3 for one constraint.  -> dict(tex, P, start, x).  w: E_ex
    weights, by default lam_ex h^2 / 25 (lam_ex = 8) at scales 4..16; local:
    add the learned finest-level energy (fit_local) for the detail below them;
    tau: temperature of the texture-guided start (guided_witness); x_tex: a
    given texture (stage 1 skipped); sub_seed: seed of stages 2-3 (default seed);
    h_top: the texture's coarsest cell side (texture)."""
    E = np.asarray(E)
    m = E.shape[1]
    top_row, bot_row = E[0, np.arange(n) % m], E[-1, np.arange(n) % m]
    x_tex = texture(an, n, seed, coord_T, h_top=h_top) if x_tex is None else np.asarray(x_tex, np.int32)
    seed = seed if sub_seed is None else sub_seed
    K = con.K
    R = n // K
    tgt = pyramid(con, con.leaf(x_tex.reshape(R, K, R, K).transpose(0, 2, 1, 3)), levels)
    ps = PromiseSampler(con, tables, levels, tgt, boundary_values(con, top_row), boundary_values(con, bot_row),
                        lam, T_prom, seed)
    P = ps.run(prom_sweeps)
    loc = fit_local(E, ts.n_sig) if local else None
    x0 = guided_witness(con, P, x_tex, support_rule(ts), bot_row, top_row, np.random.default_rng(seed), tau, local=loc)
    ch = TileChain(ts, x0, E, [con], [P], support_rule(ts), w or {4: 5.12, 8: 20.48, 16: 81.92}, T, seed=seed,
                   local=loc)
    for s in range(sweeps):
        ch.sweep()
        if log:
            log(f"sweep {s}: E_ex {ch.energy_ex():.1f}")
    assert ch.valid()
    return dict(tex=x_tex, P=P, start=x0, x=ch.t.copy())
