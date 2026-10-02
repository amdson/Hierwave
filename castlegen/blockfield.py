"""Local auxiliary fields that guide a k x k tile block toward connectivity
(the quick test of notes/experiments/block_field.py; blockconn's port
semantics, its score only as the final check).

Each cell p carries, beside its tile t_p:
  g_p in {0..G, INF}   gaps on its route to an exit (INF: no route claimed)
  d_p in {0..D}        steps along that route, strictly decreasing (no cycles)
Step cost s(p, q) = 0 if p and q are nodes joined across their seam, else 1.
Exit: a node on the block edge with an opening on a class port (1..MLAB).
Valid p: g_p = INF, or p an exit, or some in-block neighbour q has
g_q < INF, d_q < d_p and g_q + s(p, q) <= g_p.
Energy (lam ramps; mu = lam MU, eps = mu EPS; LV fixed, not ramped):
  tile energy as blockconn.sweep (pair energies, closed ports facing WALL,
  -logz, parent term)
  + lam per opening on a closed port (per cell)
  + lam per open port with no opening (reads the port's other cells as they are)
  + LV per invalid cell (large: a hard rule.  Ramped with lam it was 0 early,
    and one invalid cell then let a whole orphan component claim g = 0)
  + mu g~_p for nodes, eps g~_p for the rest (g~ = g, INF -> G + 1), + DELTA d_p
All g = 0 on the nodes <=> every node joined to an exit inside the block.
A sink block (no class port) gets lab[8] = a root cell, an exit when walkable
(map_sweep charges lam while it is not).
Every term but the open one spans a cell and its 4 neighbours, so cells 3
apart never share one: the 5 classes of (x + 2 y) mod 5 update in parallel
exactly (here in a loop: the same result).  Each site draws (t, g, d) from
its exact conditional: for a fixed tile, the energy over the (g, d) grid is
mu g + DELTA d plus violation counts that switch at the neighbours' bounds, so
the grid splits into <= 81 rectangles of constant count, each summed as two
geometric series (_gd_rects); the result depends on the tile only through
its key (gap bits, exit, node), so it is computed once per distinct key.
Then t, a rectangle, and g, d within it (site_logw_enum: the enumeration it
equals).
"""
import heapq

import numpy as np
from numba import njit

from castlegen.blockconn import MLAB, OPEN

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@njit(cache=True, inline="always")
def _joined(tp, tq, d, Dh, Dv):
    """Tile tp joined to tile tq on its side d (conn_tables' Dh, Dv need nodes)."""
    if d == 0:
        return Dv[tq, tp]
    if d == 1:
        return Dh[tp, tq]
    if d == 2:
        return Dv[tp, tq]
    return Dh[tq, tp]


@njit(cache=True, inline="always")
def _port(y, x, d, k):
    """Port index of edge cell (y, x) on side d, or -1 if not on that side."""
    hh = k // 2
    if d == 0 and y == 0:
        return x // hh
    if d == 1 and x == k - 1:
        return 2 + y // hh
    if d == 2 and y == k - 1:
        return 4 + x // hh
    if d == 3 and x == 0:
        return 6 + y // hh
    return -1


@njit(cache=True, inline="always")
def _exit(y, x, t, lab, node, sock, k):
    """t at (y, x) is an exit: an opening on a class port, or the block's root
    cell (lab[8] = y k + x when lab has 9 entries: a sink block's designated
    exit, so its nodes form one component)."""
    if not node[t]:
        return False
    if lab.shape[0] > 8 and lab[8] == y * k + x:
        return True
    for d in range(4):
        q = _port(y, x, d, k)
        if q >= 0 and 0 < lab[q] < OPEN and sock[d, t]:
            return True
    return False


@njit(cache=True)
def _valid(y, x, T, Gf, Df, lab, node, Dh, Dv, sock, k, INF, skip):
    """Cell (y, x) valid under the current fields, ignoring neighbour side skip
    as a witness (-1: none)."""
    g, dd, t = Gf[y, x], Df[y, x], T[y, x]
    if g >= INF or _exit(y, x, t, lab, node, sock, k):
        return True
    for d in range(4):
        if d == skip:
            continue
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if ny < 0 or ny >= k or nx < 0 or nx >= k:
            continue
        gq = Gf[ny, nx]
        if gq >= INF or Df[ny, nx] >= dd:
            continue
        s = 0 if (node[t] and node[T[ny, nx]] and _joined(t, T[ny, nx], d, Dh, Dv)) else 1
        if gq + s <= g:
            return True
    return False


@njit(cache=True)
def _prep(y, x, T, Gf, Df, lab, node, Dh, Dv, sock, k, INF, base, cnt):
    """Per site: base[d] = neighbour d valid without p as its witness (True off
    the block); cnt[q] = openings on port q from the port's other cells."""
    hh = k // 2
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        base[d] = True
        if 0 <= ny < k and 0 <= nx < k:
            base[d] = _valid(ny, nx, T, Gf, Df, lab, node, Dh, Dv, sock, k, INF, (d + 2) % 4)
    for q in range(8):
        cnt[q] = 0
    for d in range(4):
        q = _port(y, x, d, k)
        if q < 0:
            continue
        i = q % 2
        for j in range(hh):
            s = i * hh + j
            cy, cx = ((0, s), (s, k - 1), (k - 1, s), (s, 0))[d]
            if (cy != y or cx != x) and node[T[cy, cx]] and sock[d, T[cy, cx]]:
                cnt[q] += 1


@njit(cache=True)
def _tile_e(y, x, t, T, lab, node, sock, Eh, Ev, logz, pc, wall, lam, cnt, k):
    """The terms of t at (y, x) that do not involve the fields."""
    e = -logz[t] + pc[y, x, t]
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        q = _port(y, x, d, k)
        if 0 <= ny < k and 0 <= nx < k:
            n = T[ny, nx]
            e += (Ev[n, t], Eh[t, n], Ev[t, n], Eh[n, t])[d]
        elif lab[q] == 0:
            e += (Ev[wall, t], Eh[t, wall], Ev[t, wall], Eh[wall, t])[d]
        if q >= 0:
            op = node[t] and sock[d, t]
            if lab[q] == 0 and op:
                e += lam
            if lab[q] > 0 and cnt[q] == 0 and not op:
                e += lam
    return e


@njit(cache=True, inline="always")
def _lgeom(n, delta):
    """log sum_{j < n} exp(-delta j)."""
    if delta == 0.0:
        return np.log(n)
    return np.log(-np.expm1(-delta * n)) - np.log(-np.expm1(-delta))


NR = 82                                                                    # <= 9 x 9 rectangles + the INF entry


@njit(cache=True, inline="always")
def _key(y, x, t, T, lab, node, Dh, Dv, sock, k):
    """What the (g, d) conditional of tile t at (y, x) depends on: the step
    cost to each neighbour (bit d: a gap), exit, node."""
    key = 0
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if 0 <= ny < k and 0 <= nx < k:
            n = T[ny, nx]
            if not (node[t] and node[n] and _joined(t, n, d, Dh, Dv)):
                key |= 1 << d
    if _exit(y, x, t, lab, node, sock, k):
        key |= 16
    if node[t]:
        key |= 32
    return key


@njit(cache=True)
def _gd_rects(y, x, key, Gf, Df, base, k, G, D, mut, delta, L, rg, rng_, rd, rnd, rw):
    """The (g, d) grid of a candidate with key at (y, x), cut into rectangles
    [rg, rg + rng_) x [rd, rd + rnd) of constant violation count, plus the
    INF entry (rng_ = 0).  rw: log sum over each of exp(-(L viol + mut g +
    delta d)).  Returns the entry count.
      own   valid where g >= g_q + s_q and d >= d_q + 1 for some witness q
            (any g < INF neighbour), everywhere if an exit
      dep   q with base false is satisfied where g <= g_q - s_q, d <= d_q - 1
    At INF: own valid, every dependent unsatisfied."""
    ex = (key >> 4) & 1
    oa = np.empty(4, np.int64)
    ob = np.empty(4, np.int64)
    dc = np.empty(4, np.int64)
    de = np.empty(4, np.int64)
    gc = np.empty(10, np.int64)
    dcut = np.empty(10, np.int64)
    no, nd, ngc, ndc = 0, 0, 0, 0
    INF = G + 1
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if ny < 0 or ny >= k or nx < 0 or nx >= k:
            continue
        s = (key >> d) & 1
        gq, dq = Gf[ny, nx], Df[ny, nx]
        if not ex and gq < INF:
            oa[no], ob[no] = gq + s, dq + 1
            gc[ngc], dcut[ndc] = gq + s, dq + 1
            no += 1
            ngc += 1
            ndc += 1
        if not base[d]:
            dc[nd], de[nd] = gq - s, dq - 1
            gc[ngc], dcut[ndc] = gq - s + 1, dq
            nd += 1
            ngc += 1
            ndc += 1
    gc[ngc], dcut[ndc] = G + 1, D + 1
    ngc += 1
    ndc += 1
    gc[:ngc].sort()
    dcut[:ndc].sort()
    nr = 0
    g0 = 0
    for i in range(ngc):
        g1 = min(gc[i], G + 1)
        if g1 <= g0:
            continue
        d0 = 0
        for j in range(ndc):
            d1 = min(dcut[j], D + 1)
            if d1 <= d0:
                continue
            v = 0
            if not ex:
                ok = False
                for w in range(no):
                    ok |= g0 >= oa[w] and d0 >= ob[w]
                v += 0 if ok else 1
            for w in range(nd):
                if not (g0 <= dc[w] and d0 <= de[w]):
                    v += 1
            rg[nr], rng_[nr], rd[nr], rnd[nr] = g0, g1 - g0, d0, d1 - d0
            rw[nr] = -L * v - mut * g0 - delta * d0 + _lgeom(g1 - g0, mut) + _lgeom(d1 - d0, delta)
            nr += 1
            d0 = d1
        g0 = g1
    rg[nr], rng_[nr], rd[nr], rnd[nr] = INF, 0, 0, 1
    rw[nr] = -L * nd - mut * INF
    return nr + 1


@njit(cache=True, inline="always")
def _lse(w, n):
    mx = w[:n].max()
    return mx + np.log(np.exp(w[:n] - mx).sum())


@njit(cache=True, inline="always")
def _tgeom(n, rate):
    """A draw from p(j) ~ exp(-rate j), j < n."""
    u = np.random.random()
    if rate == 0.0:
        j = int(u * n)
    else:
        j = int(np.floor(-np.log1p(u * np.expm1(-rate * n)) / rate))
    return min(max(j, 0), n - 1)


@njit(cache=True)
def sweep(T, Gf, Df, seed, Eh, Ev, logz, allowed, pc, lab, node, Dh, Dv, sock, lam, MU, LV, EPS, DELTA, G, D, wall):
    """One sweep: the 5 classes in turn, each site drawing (t, g, d) from its
    exact conditional: t over e(t) and the (g, d) sum of its key (_gd_rects,
    cached per site: few distinct keys), then a rectangle, then g and d
    within it (tests/test_blockfield.py checks it against enumeration)."""
    np.random.seed(seed)
    k = T.shape[0]
    S = logz.shape[0]
    INF = G + 1
    mu = lam * MU
    eps = mu * EPS
    base = np.zeros(4, np.bool_)
    cnt = np.zeros(8, np.int64)
    rg = np.empty(NR, np.int64)
    rn = np.empty(NR, np.int64)
    rd = np.empty(NR, np.int64)
    rdn = np.empty(NR, np.int64)
    rw = np.empty(NR)
    memo = np.empty(64)
    seen = np.zeros(64, np.bool_)
    for c in range(5):
        for y in range(k):
            for x in range(k):
                if (x + 2 * y) % 5 != c:
                    continue
                _prep(y, x, T, Gf, Df, lab, node, Dh, Dv, sock, k, INF, base, cnt)
                seen[:] = False
                best, bt, bk = -np.inf, T[y, x], 0
                for t in range(S):
                    if not allowed[t]:
                        continue
                    key = _key(y, x, t, T, lab, node, Dh, Dv, sock, k)
                    if not seen[key]:
                        n = _gd_rects(y, x, key, Gf, Df, base, k, G, D, mu if key & 32 else eps, DELTA, LV,
                                      rg, rn, rd, rdn, rw)
                        memo[key] = _lse(rw, n)
                        seen[key] = True
                    r = memo[key] - _tile_e(y, x, t, T, lab, node, sock, Eh, Ev, logz, pc, wall, lam, cnt, k) \
                        - np.log(-np.log(np.random.random()))
                    if r > best:
                        best, bt, bk = r, t, key
                mut = mu if bk & 32 else eps
                n = _gd_rects(y, x, bk, Gf, Df, base, k, G, D, mut, DELTA, LV, rg, rn, rd, rdn, rw)
                bi, br = 0, -np.inf
                for i in range(n):
                    r = rw[i] - np.log(-np.log(np.random.random()))
                    if r > br:
                        bi, br = i, r
                if rn[bi] == 0:
                    g, dd = INF, 0
                else:
                    g = rg[bi] + _tgeom(rn[bi], mut)
                    dd = rd[bi] + _tgeom(rdn[bi], DELTA)
                T[y, x], Gf[y, x], Df[y, x] = bt, g, dd


@njit(cache=True)
def site_logw_enum(y, x, T, Gf, Df, Eh, Ev, logz, allowed, pc, lab, node, Dh, Dv, sock, lam, MU, LV, EPS, DELTA,
                   G, D, wall):
    """Reference: (S, G + 2, D + 1) log weights of every (t, g, d) at (y, x) by
    enumeration (-inf: not drawn; at INF only d = 0)."""
    k = T.shape[0]
    S = logz.shape[0]
    INF = G + 1
    mu = lam * MU
    eps = mu * EPS
    base = np.zeros(4, np.bool_)
    cnt = np.zeros(8, np.int64)
    _prep(y, x, T, Gf, Df, lab, node, Dh, Dv, sock, k, INF, base, cnt)
    out = np.full((S, INF + 1, D + 1), -np.inf)
    for t in range(S):
        if not allowed[t]:
            continue
        et = _tile_e(y, x, t, T, lab, node, sock, Eh, Ev, logz, pc, wall, lam, cnt, k)
        ex = _exit(y, x, t, lab, node, sock, k)
        for g in range(INF + 1):
            for dd in range(D + 1):
                if g == INF and dd > 0:
                    break
                v = et + (mu if node[t] else eps) * g + (DELTA * dd if g < INF else 0.0)
                ok = g >= INF or ex
                for d in range(4):
                    ny, nx = y + DIRS[d][0], x + DIRS[d][1]
                    if ny < 0 or ny >= k or nx < 0 or nx >= k:
                        continue
                    n = T[ny, nx]
                    jn = node[t] and node[n] and _joined(t, n, d, Dh, Dv)
                    gq, dq = Gf[ny, nx], Df[ny, nx]
                    if not ok and gq < INF and dq < dd and gq + (0 if jn else 1) <= g:
                        ok = True
                    if not base[d]:
                        if not (g < INF and dd < dq and g + (0 if jn else 1) <= gq):
                            v += LV
                if not ok:
                    v += LV
                out[t, g, dd] = -v
    return out


@njit(cache=True)
def site_logw_closed(y, x, T, Gf, Df, Eh, Ev, logz, allowed, pc, lab, node, Dh, Dv, sock, lam, MU, LV, EPS, DELTA,
                     G, D, wall):
    """(S,) log weight of each tile, (g, d) summed in closed form (as sweep)."""
    k = T.shape[0]
    S = logz.shape[0]
    mu = lam * MU
    eps = mu * EPS
    base = np.zeros(4, np.bool_)
    cnt = np.zeros(8, np.int64)
    rg = np.empty(NR, np.int64)
    rn = np.empty(NR, np.int64)
    rd = np.empty(NR, np.int64)
    rdn = np.empty(NR, np.int64)
    rw = np.empty(NR)
    _prep(y, x, T, Gf, Df, lab, node, Dh, Dv, sock, k, G + 1, base, cnt)
    out = np.full(S, -np.inf)
    for t in range(S):
        if allowed[t]:
            key = _key(y, x, t, T, lab, node, Dh, Dv, sock, k)
            n = _gd_rects(y, x, key, Gf, Df, base, k, G, D, mu if key & 32 else eps, DELTA, LV, rg, rn, rd, rdn, rw)
            out[t] = _lse(rw, n) - _tile_e(y, x, t, T, lab, node, sock, Eh, Ev, logz, pc, wall, lam, cnt, k)
    return out


@njit(cache=True)
def gd_probs_closed(y, x, t, T, Gf, Df, lab, node, Dh, Dv, sock, lam, MU, LV, EPS, DELTA, G, D):
    """(G + 2, D + 1) the closed-form conditional of (g, d) given tile t (test helper)."""
    k = T.shape[0]
    mu = lam * MU
    base = np.zeros(4, np.bool_)
    cnt = np.zeros(8, np.int64)
    _prep(y, x, T, Gf, Df, lab, node, Dh, Dv, sock, k, G + 1, base, cnt)
    rg = np.empty(NR, np.int64)
    rn = np.empty(NR, np.int64)
    rd = np.empty(NR, np.int64)
    rdn = np.empty(NR, np.int64)
    rw = np.empty(NR)
    key = _key(y, x, t, T, lab, node, Dh, Dv, sock, k)
    mut = mu if key & 32 else mu * EPS
    n = _gd_rects(y, x, key, Gf, Df, base, k, G, D, mut, DELTA, LV, rg, rn, rd, rdn, rw)
    Z = _lse(rw, n)
    out = np.zeros((G + 2, D + 1))
    for i in range(n):
        if rn[i] == 0:
            out[G + 1, 0] = np.exp(rw[i] - Z)
            continue
        for a in range(rn[i]):
            for b in range(rdn[i]):
                out[rg[i] + a, rd[i] + b] = np.exp(rw[i] - Z - _lgeom(rn[i], mut) - _lgeom(rdn[i], DELTA)
                                                   - mut * a - DELTA * b)
    return out


@njit(cache=True)
def _map_tile_e(Y, X, t, T, k, lab, node, Dh, Dv, sock, Eh, Ev, logz, pc, wall, lam, ws, cnt):
    """_tile_e on a map T (H, W) of k-blocks, at global (Y, X): pairs with the
    real neighbours (off the map: WALL), lam ws per seam rule broken (across a
    block seam, either tile an opening and the two unjoined), the port terms
    of the cell's block (lab, cnt), lam if the block's root cell is not a node."""
    H, W = T.shape
    y, x = Y % k, X % k
    e = -logz[t] + pc[Y, X, t]
    for d in range(4):
        NY, NX = Y + DIRS[d][0], X + DIRS[d][1]
        if 0 <= NY < H and 0 <= NX < W:
            n = T[NY, NX]
            e += (Ev[n, t], Eh[t, n], Ev[t, n], Eh[n, t])[d]
            if NY // k != Y // k or NX // k != X // k:
                o = (d + 2) % 4
                if ((node[t] and sock[d, t]) or (node[n] and sock[o, n])) and not _joined(t, n, d, Dh, Dv):
                    e += lam * ws
        else:
            e += (Ev[wall, t], Eh[t, wall], Ev[t, wall], Eh[wall, t])[d]
        q = _port(y, x, d, k)
        if q >= 0:
            op = node[t] and sock[d, t]
            if lab[q] == 0 and op:
                e += lam
            if lab[q] > 0 and cnt[q] == 0 and not op:
                e += lam
    if lab[8] == y * k + x and not node[t]:
        e += lam
    return e


@njit(cache=True)
def map_sweep(T, Gf, Df, k, labs, seed, Eh, Ev, logz, allowed, pc, node, Dh, Dv, sock, lam, MU, LV, EPS, DELTA, G, D,
              wall, ws):
    """sweep on a map T (H, W) of k-blocks with patterns labs (H/k, W/k, 9):
    fields and witnesses within each block, tile terms as _map_tile_e.  The 5
    classes of (X + 2 Y) mod 5 on global coordinates: every term is still
    within distance 2, so each class updates in parallel exactly."""
    np.random.seed(seed)
    H, W = T.shape
    S = logz.shape[0]
    INF = G + 1
    mu = lam * MU
    eps = mu * EPS
    base = np.zeros(4, np.bool_)
    cnt = np.zeros(8, np.int64)
    rg = np.empty(NR, np.int64)
    rn = np.empty(NR, np.int64)
    rd = np.empty(NR, np.int64)
    rdn = np.empty(NR, np.int64)
    rw = np.empty(NR)
    memo = np.empty(64)
    seen = np.zeros(64, np.bool_)
    for c in range(5):
        for Y in range(H):
            for X in range(W):
                if (X + 2 * Y) % 5 != c:
                    continue
                by, bx = Y // k, X // k
                y, x = Y - by * k, X - bx * k
                B = T[by * k:(by + 1) * k, bx * k:(bx + 1) * k]
                BG = Gf[by * k:(by + 1) * k, bx * k:(bx + 1) * k]
                BD = Df[by * k:(by + 1) * k, bx * k:(bx + 1) * k]
                lab = labs[by, bx]
                _prep(y, x, B, BG, BD, lab, node, Dh, Dv, sock, k, INF, base, cnt)
                seen[:] = False
                best, bt, bk = -np.inf, T[Y, X], 0
                for t in range(S):
                    if not allowed[t]:
                        continue
                    key = _key(y, x, t, B, lab, node, Dh, Dv, sock, k)
                    if not seen[key]:
                        n = _gd_rects(y, x, key, BG, BD, base, k, G, D, mu if key & 32 else eps, DELTA, LV,
                                      rg, rn, rd, rdn, rw)
                        memo[key] = _lse(rw, n)
                        seen[key] = True
                    r = memo[key] - _map_tile_e(Y, X, t, T, k, lab, node, Dh, Dv, sock, Eh, Ev, logz, pc, wall, lam,
                                                ws, cnt) - np.log(-np.log(np.random.random()))
                    if r > best:
                        best, bt, bk = r, t, key
                mut = mu if bk & 32 else eps
                n = _gd_rects(y, x, bk, BG, BD, base, k, G, D, mut, DELTA, LV, rg, rn, rd, rdn, rw)
                bi, br = 0, -np.inf
                for i in range(n):
                    r = rw[i] - np.log(-np.log(np.random.random()))
                    if r > br:
                        bi, br = i, r
                if rn[bi] == 0:
                    g, dd = INF, 0
                else:
                    g = rg[bi] + _tgeom(rn[bi], mut)
                    dd = rd[bi] + _tgeom(rdn[bi], DELTA)
                T[Y, X], Gf[Y, X], Df[Y, X] = bt, g, dd


def init_fields(T, lab, tabs, G, D):
    """Lexicographic Dijkstra on (g, d) from the exits: the least valid fields
    (g > G or d > D: INF)."""
    node, Dh, Dv, sock = tabs
    k = T.shape[0]
    INF = G + 1
    Gf = np.full((k, k), INF, np.int64)
    Df = np.zeros((k, k), np.int64)
    h = []
    for y in range(k):
        for x in range(k):
            if _exit(y, x, T[y, x], lab, node, sock, k):
                Gf[y, x] = 0
                heapq.heappush(h, (0, 0, y, x))
    best = {}
    while h:
        g, d, y, x = heapq.heappop(h)
        if (y, x) in best:
            continue
        best[(y, x)] = (g, d)
        for e, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < k and 0 <= nx < k) or (ny, nx) in best:
                continue
            o = (e + 2) % 4                                              # side of (ny, nx) facing (y, x)
            s = 0 if (node[T[ny, nx]] and node[T[y, x]] and _joined(T[ny, nx], T[y, x], o, Dh, Dv)) else 1
            if g + s <= G and d + 1 <= D:
                heapq.heappush(h, (g + s, d + 1, ny, nx))
    for (y, x), (g, d) in best.items():
        Gf[y, x], Df[y, x] = g, d
    return Gf, Df


def field_energy_ok(T, Gf, Df, lab, tabs, G):
    """(invalid cells, nodes with g > 0)."""
    node, Dh, Dv, sock = tabs
    k = T.shape[0]
    bad = sum(not _valid(y, x, T, Gf, Df, lab, node, Dh, Dv, sock, k, G + 1, -1) for y in range(k) for x in range(k))
    return bad, int((node[T] & (Gf > 0)).sum())
