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

Groups: 2 x 2 on two of the axes (layer, row, col), 1 on the third
(shapes 'h', 'vx', 'vy'); anchors on a 2-lattice along the paired axes
offset by o (alternating), colour = sum of group-lattice coordinates & 1 in
world coordinates; groups lie inside a block (cells without a partner stay
put).  A group sweep runs after the tile half-steps of every inner sweep
whose schedule row is >= mh_from.  Draws use keys disjoint from the tile
sweeps', so the result is still a function of (seed, position): exact
across requests (same light cone as _block3d)."""
import numpy as np
from numba import njit

from .infinite import _draw, _init3, _key3, _site3, _u01, M64

MH_KEY = np.int64(1) << np.int64(40)
DOM_KEY = np.int64(1) << np.int64(41)
# cell c = (di, dj): 0 NW (0,0), 1 NE (0,1), 2 SW (1,0), 3 SE (1,1)


# Group shapes: two paired axes (2 cells each) and one single axis; axes
# 0 layer, 1 row, 2 col.  Cell c = (bit 1 along axis a, bit 0 along axis b).
SHAPES = {"h": (1, 2), "vx": (0, 2), "vy": (0, 1)}
_DIRS = ((0, 0, -1), (-1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, -1, 0))   # infinite._D3 order


def shape_tables(shape):
    """offs (4, 3) cell offsets (layer, row, col); nb (4, 6): the group cell
    that is cell c's neighbour in direction d, or -1."""
    a, b = SHAPES[shape]
    offs = np.zeros((4, 3), np.int64)
    for c in range(4):
        offs[c, a] = c >> 1
        offs[c, b] = c & 1
    nb = -np.ones((4, 6), np.int64)
    for c in range(4):
        for d, dl in enumerate(_DIRS):
            q = offs[c] + np.array(dl)
            for c2 in range(4):
                if (offs[c2] == q).all():
                    nb[c, d] = c2
    return offs, nb


@njit(cache=True)
def _vext(grid, k0, i0, j0, offs, nb, F, vx):
    """vx[c, t]: forbidden pairs of module t in group cell c with the
    neighbours outside the group."""
    L = grid.shape[0]
    T = vx.shape[1]
    for c in range(4):
        k, i, j = k0 + offs[c, 0], i0 + offs[c, 1], j0 + offs[c, 2]
        for t in range(T):
            vx[c, t] = 0
        for d in range(6):
            if nb[c, d] >= 0:
                continue
            kk = k + (d == 4) - (d == 1)
            if kk < 0 or kk >= L:
                continue
            ii = i + (d == 2) - (d == 5)
            jj = j + (d == 3) - (d == 0)
            Fr = F[d, grid[kk, ii, jj]]
            for t in range(T):
                vx[c, t] += Fr[t]


@njit(cache=True)
def _cond(c, x, nb, vx, F, bs, eg, hard_inf, w):
    """Unnormalised proposal weights for group cell c given the refilled
    cells x[:c] (group cells after c are ignored); returns their sum."""
    T = w.shape[0]
    vmin = 1 << 30
    for t in range(T):
        n = vx[c, t]
        for d in range(6):
            c2 = nb[c, d]
            if c2 >= 0 and c2 < c:
                n += F[d, x[c2], t]
        w[t] = n
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
def _group_v(x, nb, vx, F):
    """Total violations of group config x: external + internal pairs once."""
    n = 0
    for c in range(4):
        n += vx[c, x[c]]
        for d in range(6):
            c2 = nb[c, d]
            if c2 >= 0 and c2 < c:
                n += F[d, x[c2], x[c]]
    return n


@njit(cache=True)
def _mh_group(grid, k0, i0, j0, offs, nb, seed, gy0, gx0, key, F, base_row, g, hard_inf, eg, vx, w, xn, xo,
              stats):
    _vext(grid, k0, i0, j0, offs, nb, F, vx)
    for c in range(4):
        xo[c] = grid[k0 + offs[c, 0], i0 + offs[c, 1], j0 + offs[c, 2]]
    lqn = 0.0
    for c in range(4):
        k = k0 + offs[c, 0]
        tot = _cond(c, xn, nb, vx, F, base_row[k], eg, hard_inf, w)
        if tot <= 0.0:
            stats[2] += 1
            return
        r = _u01(seed, gy0 + i0 + offs[c, 1], gx0 + j0 + offs[c, 2], _key3(key, k))
        xn[c] = _draw(w, w.shape[0], r)
        lqn += np.log(w[xn[c]] / tot)
    lqo = 0.0
    q0 = True
    for c in range(4):
        tot = _cond(c, xo, nb, vx, F, base_row[k0 + offs[c, 0]], eg, hard_inf, w)
        if tot <= 0.0 or w[xo[c]] <= 0.0:
            q0 = False
            break
        lqo += np.log(w[xo[c]] / tot)
    vn, vo = _group_v(xn, nb, vx, F), _group_v(xo, nb, vx, F)
    lpn = 0.0
    lpo = 0.0
    for c in range(4):
        k = k0 + offs[c, 0]
        lpn += np.log(base_row[k, xn[c]])
        lpo += np.log(base_row[k, xo[c]])
    ra = np.log(_u01(seed, gy0 + i0, gx0 + j0, _key3(key + 1, k0)) + 1e-300)
    if hard_inf:
        if vn < vo:
            acc = True
        elif vn > vo or not q0:
            acc = False
        else:
            acc = ra < lpn - lpo + lqo - lqn
    else:
        acc = q0 and ra < (lpn - g * vn) - (lpo - g * vo) + lqo - lqn
    stats[0 + 3 * hard_inf] += 1
    if acc:
        stats[1 + 3 * hard_inf] += 1
        for c in range(4):
            grid[k0 + offs[c, 0], i0 + offs[c, 1], j0 + offs[c, 2]] = xn[c]


@njit(cache=True)
def _group_sweep(grid, y0, x0, B, gy0, gx0, offs, nb, o, keybase, seed, F, base_row, g, hard_inf, eg, vx, w,
                 xn, xo, stats, ups):
    """All groups of one shape inside the block [y0, y0 + B) x [x0, x0 + B),
    lattice offset o along the paired axes, two colours in turn."""
    L = grid.shape[0]
    sk = 2 if offs[3, 0] else 1          # step along each axis: 2 if paired
    si = 2 if offs[3, 1] else 1
    sj = 2 if offs[3, 2] else 1
    ok = o if sk == 2 else 0
    oi = o if si == 2 else 0
    oj = o if sj == 2 else 0
    for col in range(2):
        key = keybase + (col << 1)
        for k0 in range(ok, L - (sk - 1), sk):
            for i0 in range(y0 + oi, y0 + B - (si - 1), si):
                for j0 in range(x0 + oj, x0 + B - (sj - 1), sj):
                    gc = ((k0 - ok) // sk + (gy0 + i0 - oi) // si + (gx0 + j0 - oj) // sj) & 1
                    if gc != col:
                        continue
                    _mh_group(grid, k0, i0, j0, offs, nb, seed, gy0, gx0, key, F, base_row, g, hard_inf, eg,
                              vx, w, xn, xo, stats)
                    ups[0] += 8



# ------------------------------------------------------------ exact dominoes
# A domino is two neighbouring cells along axis ax (0 layer, 1 row, 2 col).
# At gamma = inf the target is lexicographic: fewest violations of the pair
# (external + between them), then the soft weight.  A pair with
# vA[a] > min vA + 1 has V >= min vA + min vB + 2 > the best V (<= min vA +
# min vB + 1), so enumerating vA <= min + 1 and vB <= min + 1 is exact Gibbs
# on the domino.
_AXD = (4, 2, 3)          # direction from cell A to cell B along each axis (up, row+1, col+1)


@njit(cache=True)
def _vext2(grid, kA, iA, jA, ax, F, vA, vB):
    L = grid.shape[0]
    T = vA.shape[0]
    dk, di, dj = (1, 0, 0) if ax == 0 else ((0, 1, 0) if ax == 1 else (0, 0, 1))
    for side in range(2):
        k, i, j = kA + side * dk, iA + side * di, jA + side * dj
        v = vA if side == 0 else vB
        for t in range(T):
            v[t] = 0
        for d in range(6):
            kk = k + (d == 4) - (d == 1)
            if kk < 0 or kk >= L:
                continue
            ii = i + (d == 2) - (d == 5)
            jj = j + (d == 3) - (d == 0)
            if kk == kA + (1 - side) * dk and ii == iA + (1 - side) * di and jj == jA + (1 - side) * dj:
                continue
            Fr = F[d, grid[kk, ii, jj]]
            for t in range(T):
                v[t] += Fr[t]


@njit(cache=True)
def _domino(grid, kA, iA, jA, ax, seed, gy0, gx0, key, F, base_row, vA, vB, ca, cb, wl, pa, pb, stats):
    T = vA.shape[0]
    _vext2(grid, kA, iA, jA, ax, F, vA, vB)
    kB = kA + (ax == 0)
    iB = iA + (ax == 1)
    jB = jA + (ax == 2)
    bA, bB = base_row[kA], base_row[kB]
    mA = 1 << 30
    mB = 1 << 30
    for t in range(T):
        if bA[t] > 0.0 and vA[t] < mA:
            mA = vA[t]
        if bB[t] > 0.0 and vB[t] < mB:
            mB = vB[t]
    na = 0
    nb_ = 0
    for t in range(T):
        if bA[t] > 0.0 and vA[t] <= mA + 1:
            ca[na] = t
            na += 1
        if bB[t] > 0.0 and vB[t] <= mB + 1:
            cb[nb_] = t
            nb_ += 1
    dAB = _AXD[ax]
    best = 1 << 30
    for x in range(na):
        a = ca[x]
        for y in range(nb_):
            b = cb[y]
            V = vA[a] + vB[b] + F[dAB, b, a]
            if V < best:
                best = V
    n = 0
    for x in range(na):
        a = ca[x]
        for y in range(nb_):
            b = cb[y]
            if vA[a] + vB[b] + F[dAB, b, a] == best:
                if n >= wl.shape[0]:
                    stats[6] += 1          # truncated (capacity)
                    break
                wl[n] = bA[a] * bB[b]
                pa[n] = a
                pb[n] = b
                n += 1
    stats[5] += na * nb_
    r = _u01(seed, gy0 + iA, gx0 + jA, _key3(key, kA))
    z = _draw(wl, n, r)
    old = vA[grid[kA, iA, jA]] + vB[grid[kB, iB, jB]] + F[dAB, grid[kB, iB, jB], grid[kA, iA, jA]]
    stats[7] += old - best
    grid[kA, iA, jA] = pa[z]
    grid[kB, iB, jB] = pb[z]


@njit(cache=True)
def _domino_sweep(grid, y0, x0, B, gy0, gx0, ax, o, keybase, seed, F, base_row, vA, vB, ca, cb, wl, pa, pb,
                  stats, ups):
    L = grid.shape[0]
    sk = 2 if ax == 0 else 1
    si = 2 if ax == 1 else 1
    sj = 2 if ax == 2 else 1
    ok = o if ax == 0 else 0
    oi = o if ax == 1 else 0
    oj = o if ax == 2 else 0
    for col in range(2):
        key = keybase + (col << 1)
        for k0 in range(ok, L - (sk - 1), sk):
            for i0 in range(y0 + oi, y0 + B - (si - 1), si):
                for j0 in range(x0 + oj, x0 + B - (sj - 1), sj):
                    if (((k0 - ok) // sk + (gy0 + i0 - oi) // si + (gx0 + j0 - oj) // sj) & 1) != col:
                        continue
                    _domino(grid, k0, i0, j0, ax, seed, gy0, gx0, key, F, base_row, vA, vB, ca, cb, wl, pa, pb,
                            stats)
                    ups[0] += 2


@njit(cache=True)
def _block3d_mh(grid, BY0, BX0, B, seed, base, beta, gam, Sb, K, u, F, E, soft, ups, fix, shift, mh_from,
                stats, offs_all, nb_all, active, domino):
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
    vA = np.empty(T, np.int64)
    vB = np.empty(T, np.int64)
    ca = np.empty(T, np.int64)
    cb = np.empty(T, np.int64)
    wl = np.empty(1 << 16)
    pa = np.empty(1 << 16, np.int64)
    pb = np.empty(1 << 16, np.int64)
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
                                    grid[k, i, j] = _site3(grid, k, i, j, F, E, soft, u[k], b, bs, g, hard_inf,
                                                           eg, False, v, w, e, r)
                                    ups[0] += 1
                    if domino and hard_inf:
                        for ax in range(3):
                            _domino_sweep(grid, y0, x0, B, gy0, gx0, ax, kin & 1,
                                          DOM_KEY + ((((hb - 1) * K + kin) << 4) + (ax << 2)), seed, F,
                                          base[row], vA, vB, ca, cb, wl, pa, pb, stats, ups)
                    if row < mh_from:
                        continue
                    ns = offs_all.shape[0]
                    sh = kin % ns
                    if not active[sh]:
                        continue
                    _group_sweep(grid, y0, x0, B, gy0, gx0, offs_all[sh], nb_all[sh], (kin // ns) & 1,
                                 MH_KEY + ((((hb - 1) * K + kin) << 2)), seed, F, base[row], g, hard_inf, eg,
                                 vx, w, xn, xo, stats, ups)
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
                    grid[k, i, j] = _site3(grid, k, i, j, F, E, soft, u[k], b, bs, np.inf, True, eg, True,
                                           v, w, e, r)
                    ups[0] += 1


def _shape_arrays(shapes):
    """None in shapes = no group sweep after that inner sweep."""
    t = [shape_tables(sh or "h") for sh in shapes]
    act = np.array([sh is not None for sh in shapes], np.bool_)
    return np.stack([a for a, _ in t]), np.stack([b for _, b in t]), act


def generate_mh(model, bs, seed, y0, x0, H, W, mh_from=0, stats=None, shapes=("h",), domino=False):
    """generate() for TileModel3D + BlockSchedule with MH group sweeps from
    schedule row mh_from on (rows run 0 .. Sb K - 1); inner sweep kin uses
    shapes[kin % len(shapes)] ('h' 2x2 in a layer, 'vx' 2 layers x 2 cols,
    'vy' 2 layers x 2 rows), lattice offset alternating every cycle."""
    if model.soft:
        raise NotImplementedError("soft pair energies")
    B = bs.B
    pb = 2 * bs.Sb + model.extra_pad
    BY0, BX0 = y0 // B - pb, x0 // B - pb
    BY1, BX1 = -(-(y0 + H) // B) + pb, -(-(x0 + W) // B) + pb
    grid = np.zeros((model.L, (BY1 - BY0) * B, (BX1 - BX0) * B), np.int32)
    ups = np.zeros(1, np.int64)
    st = np.zeros(8, np.int64)
    beta = bs.beta.ravel()
    _block3d_mh(grid, np.int64(BY0), np.int64(BX0), np.int64(B), np.uint64(seed & M64), model.base(beta), beta,
                bs.gamma.ravel(), bs.Sb, bs.K, model.ul, model.F, model.E, model.soft, ups, model.wild, bs.shift,
                np.int64(mh_from), st, *_shape_arrays(shapes), domino)
    if stats is not None:
        for kk, vv in zip(("prop", "acc", "dead", "prop_hard", "acc_hard", "dom_pairs", "dom_trunc", "dom_gain"),
                          st):
            stats[kk] = stats.get(kk, 0) + int(vv)
        stats["updates"] = stats.get("updates", 0) + int(ups[0])
    oy, ox = y0 - BY0 * B, x0 - BX0 * B
    return grid[:, oy:oy + H, ox:ox + W].copy()
