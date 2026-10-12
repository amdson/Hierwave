"""Experimental: 2x2 Metropolis-Hastings moves inside 3D block Gibbs
(castlegen.infinite._block3d), TileModel3D without soft pair energies.

A move clears a 2x2 group of cells on one layer and refills it in a fixed
order (NW, NE, SW, SE), each cell drawn from its single-site conditional given
the cells outside the group and the cells already refilled (cells not yet
refilled are ignored).  The proposal does not depend on the current
interior, so with q(x) computed by the same sequential construction the
acceptance min(1, pi(x') q(x) / (pi(x) q(x'))) gives detailed balance for
the group's conditional (an independence sampler).  At gamma = inf the
target is lexicographic: fewer violations always win, equal violations use
the soft ratio, q(x) = 0 rejects.

Groups: anchors on a 2-lattice offset by (o, o), o = inner sweep & 1, colour
((Y - o) / 2 + (X - o) / 2 + layer) & 1 in world coordinates; groups lie
inside a block (cells of the block's last row/column without a partner stay
put).  A group sweep runs after the tile half-steps of every inner sweep
whose schedule row is >= mh_from.  Draws use keys disjoint from the tile
sweeps', so the result is still a function of (seed, position): exact
across requests (same light cone as _block3d)."""
import numpy as np
from numba import njit

from .infinite import _draw, _init3, _key3, _site3, _u01, M64

MH_KEY = np.int64(1) << np.int64(40)
# cell c = (di, dj): 0 NW (0,0), 1 NE (0,1), 2 SW (1,0), 3 SE (1,1)


@njit(cache=True)
def _vext(grid, k, i0, j0, F, vx):
    """vx[c, t]: forbidden pairs of module t in cell c of the group anchored
    at (k, i0, j0) with the neighbours outside the group."""
    L = grid.shape[0]
    T = vx.shape[1]
    for c in range(4):
        i, j = i0 + (c >> 1), j0 + (c & 1)
        for t in range(T):
            vx[c, t] = 0
        for d in range(6):
            kk = k + (d == 4) - (d == 1)
            if kk < 0 or kk >= L:
                continue
            ii = i + (d == 2) - (d == 5)
            jj = j + (d == 3) - (d == 0)
            if kk == k and i0 <= ii <= i0 + 1 and j0 <= jj <= j0 + 1:
                continue
            Fr = F[d, grid[kk, ii, jj]]
            for t in range(T):
                vx[c, t] += Fr[t]


@njit(cache=True)
def _cond(c, x, vx, F, bs, eg, hard_inf, w):
    """Unnormalised proposal weights w for cell c given refilled cells x[:c]
    (internal neighbours: NE <- NW (d 0), SW <- NW (d 5), SE <- SW (d 0),
    SE <- NE (d 5)); returns their sum."""
    T = w.shape[0]
    vmin = 1 << 30
    for t in range(T):
        n = vx[c, t]
        if c == 1:
            n += F[0, x[0], t]
        elif c == 2:
            n += F[5, x[0], t]
        elif c == 3:
            n += F[0, x[2], t] + F[5, x[1], t]
        w[t] = n                                    # violations, temporarily
        if bs[t] > 0.0 and n < vmin:
            vmin = n
    tot = 0.0
    for t in range(T):
        n = int(w[t])
        if bs[t] == 0.0:
            w[t] = 0.0
        elif hard_inf:
            w[t] = bs[t] if n == vmin else 0.0
        else:
            w[t] = bs[t] * eg[n]
        tot += w[t]
    return tot


@njit(cache=True)
def _group_v(x, vx, F):
    """Total violations of group config x: external + 4 internal pairs."""
    return (vx[0, x[0]] + vx[1, x[1]] + vx[2, x[2]] + vx[3, x[3]]
            + F[0, x[0], x[1]] + F[0, x[2], x[3]] + F[5, x[0], x[2]] + F[5, x[1], x[3]])


@njit(cache=True)
def _mh_group(grid, k, i0, j0, seed, gy0, gx0, key, F, bs, g, hard_inf, eg, vx, w, xn, xo, stats):
    _vext(grid, k, i0, j0, F, vx)
    for c in range(4):
        xo[c] = grid[k, i0 + (c >> 1), j0 + (c & 1)]
    # forward proposal and log q(x')
    lqn = 0.0
    for c in range(4):
        tot = _cond(c, xn, vx, F, bs, eg, hard_inf, w)
        if tot <= 0.0:
            stats[2] += 1
            return
        r = _u01(seed, gy0 + i0 + (c >> 1), gx0 + j0 + (c & 1), _key3(key, k))
        xn[c] = _draw(w, w.shape[0], r)
        lqn += np.log(w[xn[c]] / tot)
    # reverse: log q(x)
    lqo = 0.0
    q0 = True
    for c in range(4):
        tot = _cond(c, xo, vx, F, bs, eg, hard_inf, w)
        if tot <= 0.0 or w[xo[c]] <= 0.0:
            q0 = False
            break
        lqo += np.log(w[xo[c]] / tot)
    vn, vo = _group_v(xn, vx, F), _group_v(xo, vx, F)
    lpn = 0.0
    lpo = 0.0
    for c in range(4):
        lpn += np.log(bs[xn[c]])
        lpo += np.log(bs[xo[c]])
    if hard_inf:
        if vn < vo:
            acc = True
        elif vn > vo or not q0:
            acc = False
        else:
            acc = np.log(_u01(seed, gy0 + i0, gx0 + j0, _key3(key + 1, k)) + 1e-300) < lpn - lpo + lqo - lqn
    else:
        if not q0:
            acc = False
        else:
            la = (lpn - g * vn) - (lpo - g * vo) + lqo - lqn
            acc = np.log(_u01(seed, gy0 + i0, gx0 + j0, _key3(key + 1, k)) + 1e-300) < la
    stats[0 + 3 * hard_inf] += 1
    if acc:
        stats[1 + 3 * hard_inf] += 1
        for c in range(4):
            grid[k, i0 + (c >> 1), j0 + (c & 1)] = xn[c]


@njit(cache=True)
def _block3d_mh(grid, BY0, BX0, B, seed, base, beta, gam, Sb, K, u, F, E, soft, ups, fix, shift, mh_from,
                stats):
    """_block3d plus a 2x2 MH group sweep after each inner sweep whose row
    >= mh_from.  stats: [proposed, accepted, dead] at finite gamma, then
    [proposed, accepted] at gamma = inf (indices 3, 4)."""
    L, H, W = grid.shape
    T = base.shape[2]
    gy0, gx0 = BY0 * B, BX0 * B
    v = np.empty(T, np.int64)
    w = np.empty(T)
    e = np.empty(T)
    eg = np.empty(16)
    vx = np.empty((4, T), np.int64)
    xn = np.empty(4, np.int64)
    xo = np.empty(4, np.int64)
    _init3(grid, gy0, gx0, seed, base[0])
    for hb in range(1, 2 * Sb + 1):
        sb = (hb - 1) >> 1
        bpar = (hb - 1) & 1
        off = (B >> 1) if (shift and (sb & 1)) else 0
        lo = (hb - 1) * B + 1
        for y0 in range(off - B, H, B):
            if y0 < lo or y0 + B > H - lo:
                continue
            BYg = (gy0 + y0 - off) // B
            for x0 in range(off - B, W, B):
                if x0 < lo or x0 + B > W - lo:
                    continue
                if ((BYg + (gx0 + x0 - off) // B) & 1) != bpar:
                    continue
                for kin in range(K):
                    row = sb * K + kin
                    g = gam[row]
                    hard_inf = np.isinf(g)
                    for n in range(16):
                        eg[n] = 0.0 if hard_inf and n > 0 else np.exp(-g * n)
                    b = beta[row]
                    for p in range(2):
                        hkey = 1 + (((hb - 1) * K + kin) << 1) + p
                        for k in range(L):
                            bs = base[row, k]
                            for i in range(y0, y0 + B):
                                j0 = x0 + ((p - (k + gy0 + i + gx0 + x0)) & 1)
                                for j in range(j0, x0 + B, 2):
                                    r = _u01(seed, gy0 + i, gx0 + j, _key3(hkey, k))
                                    grid[k, i, j] = _site3(grid, k, i, j, F, E, soft, u, b, bs, g, hard_inf,
                                                           eg, False, v, w, e, r)
                                    ups[0] += 1
                    if row < mh_from:
                        continue
                    o = kin & 1
                    for col in range(2):
                        key = MH_KEY + ((((hb - 1) * K + kin) << 2) + (col << 1))
                        for k in range(L):
                            bs = base[row, k]
                            for i0 in range(y0 + o, y0 + B - 1, 2):
                                for j0 in range(x0 + o, x0 + B - 1, 2):
                                    gc = ((gy0 + i0 - o) // 2 + (gx0 + j0 - o) // 2 + k) & 1
                                    if gc != col:
                                        continue
                                    _mh_group(grid, k, i0, j0, seed, gy0, gx0, key, F, bs, g, hard_inf, eg,
                                              vx, w, xn, xo, stats)
                                    ups[0] += 8
    if fix:
        m0 = 2 * Sb * B + 1
        hkey = 1 + ((2 * Sb * K) << 1)
        row = base.shape[0] - 1
        b = beta[row]
        for k in range(L):
            bs = base[row, k]
            for i in range(m0, H - m0):
                j0 = m0 + ((0 - (k + gy0 + i + gx0 + m0)) & 1)
                for j in range(j0, W - m0, 2):
                    r = _u01(seed, gy0 + i, gx0 + j, _key3(hkey, k))
                    grid[k, i, j] = _site3(grid, k, i, j, F, E, soft, u, b, bs, np.inf, True, eg, True,
                                           v, w, e, r)
                    ups[0] += 1


def generate_mh(model, bs, seed, y0, x0, H, W, mh_from=0, stats=None):
    """generate() for TileModel3D + BlockSchedule with 2x2 MH group sweeps
    from schedule row mh_from on (rows run 0 .. Sb K - 1)."""
    if model.soft:
        raise NotImplementedError("soft pair energies")
    B = bs.B
    pb = 2 * bs.Sb + model.extra_pad
    BY0, BX0 = y0 // B - pb, x0 // B - pb
    BY1, BX1 = -(-(y0 + H) // B) + pb, -(-(x0 + W) // B) + pb
    grid = np.zeros((model.L, (BY1 - BY0) * B, (BX1 - BX0) * B), np.int32)
    ups = np.zeros(1, np.int64)
    st = np.zeros(5, np.int64)
    beta = bs.beta.ravel()
    _block3d_mh(grid, np.int64(BY0), np.int64(BX0), np.int64(B), np.uint64(seed & M64), model.base(beta), beta,
                bs.gamma.ravel(), bs.Sb, bs.K, model.u, model.F, model.E, model.soft, ups, model.wild, bs.shift,
                np.int64(mh_from), st)
    if stats is not None:
        for kk, vv in zip(("prop", "acc", "dead", "prop_hard", "acc_hard"), st):
            stats[kk] = stats.get(kk, 0) + int(vv)
        stats["updates"] = stats.get("updates", 0) + int(ups[0])
    oy, ox = y0 - BY0 * B, x0 - BX0 * B
    return grid[:, oy:oy + H, ox:ox + W].copy()
