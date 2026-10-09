"""The generic level-1 Gibbs kernel (numba).  It reads core.Packed: a tuple
of channel grids, a tuple of views, factor descriptor rows and a tuple of
tables, and knows nothing about which channel set wrote them.

Per site, colour by colour: the energy of every candidate value is the sum
of its factor rows (pair / count / unary, and with below packing the
below-pair / below-count rows of finer channels that read it), then one Gumbel-max draw over the
finite candidates.

Certificate (notes/history/channels.tex, 14): the home channel's cells carry a mass, a
trunk flag and join bits (joins[d, t, t']: t at p is joined to t' across
side d), and a d channel holds a certificate per cell.  A cell is valid iff
  mass = 0 and d = INF, or a trunk with d = 0, or
  d < INF and some joined neighbour q has mass_q >= mass_p and d_q < d_p.
The site draws (t, d) jointly: for each t the valid d form an interval
(above the smallest witness d, below the smallest d of a dependant that p
must keep valid), whose geometric sum in exp(-delta d) is closed form; t is
drawn by Gumbel-max over the summed weights, then d within its interval.
Neighbours' d are read as they are and dependants read their other
neighbours, so the colouring needs radius 2."""
import numpy as np
from numba import njit

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@njit(cache=True)
def seed(s):
    np.random.seed(s)


# ------------------------------------------------------------- convpot (kind 5)
# A convpot row is (5, home, aview, -1, k, m, pad, ci); convs[ci] is one flat
# float64 vector [a (k) | A (9 k k) | W (9 k m) | b (m) | v (m) | E (V k)]
# (core.Packed).  Offsets d row-major over (dy, dx) in (-1, 0, 1)^2, 4 = centre.

@njit(cache=True, inline="always")
def _softplus(x):
    if x > 0.0:
        return x + np.log1p(np.exp(-x))
    return np.log1p(np.exp(x))


@njit(cache=True)
def _conv_unpack(cv, k, m):
    o = 0
    a = cv[o:o + k]
    o += k
    A = cv[o:o + 9 * k * k].reshape((9, k, k))
    o += 9 * k * k
    W = cv[o:o + 9 * k * m].reshape((9, k, m))
    o += 9 * k * m
    b = cv[o:o + m]
    o += m
    v = cv[o:o + m]
    o += m
    E = cv[o:].reshape(((cv.shape[0] - o) // k, k))
    return a, A, W, b, v, E


@njit(cache=True)
def _conv_energies(y, x, g, av, cv, k, m, pad, out):
    """out[t] += the convpot terms that involve site p = (y, x) with z_p = t."""
    rows, cols = g.shape
    a, A, W, b, v, E = _conv_unpack(cv, k, m)
    V = E.shape[0]
    # bilinear: e_t . gv, gv = sum_d [A_d e(p+d) + (p+d on grid) A_{-d}^T e(p+d)]
    gv = a.copy()
    for d in range(9):
        if d == 4:
            continue
        ny, nx = y + d // 3 - 1, x + d % 3 - 1
        on = 0 <= ny < rows and 0 <= nx < cols
        if on:
            en = E[av[g[ny, nx]]]
        elif pad >= 0:
            en = E[pad]
        else:
            continue
        Ad = A[d]
        Ar = A[8 - d]
        for i in range(k):
            s = 0.0
            for j in range(k):
                s += Ad[i, j] * en[j]
            gv[i] += s
        if on:
            for j in range(k):
                ej = en[j]
                for i in range(k):
                    gv[i] += Ar[j, i] * ej
    ev = np.zeros(V)
    for u in range(V):
        s = 0.0
        for i in range(k):
            s += E[u, i] * gv[i]
        ev[u] = s
    # head: every q = p - d on the grid, h_q = rest_q + W_d^T e_t
    if m > 0:
        rest = np.empty(m)
        hu = np.empty(m)
        for d in range(9):
            qy, qx = y - (d // 3 - 1), x - (d % 3 - 1)
            if qy < 0 or qy >= rows or qx < 0 or qx >= cols:
                continue
            for j in range(m):
                rest[j] = b[j]
            for d2 in range(9):
                if d2 == d:
                    continue
                ny, nx = qy + d2 // 3 - 1, qx + d2 % 3 - 1
                if 0 <= ny < rows and 0 <= nx < cols:
                    en = E[av[g[ny, nx]]]
                elif pad >= 0:
                    en = E[pad]
                else:
                    continue
                Wd = W[d2]
                for i in range(k):
                    ei = en[i]
                    for j in range(m):
                        rest[j] += Wd[i, j] * ei
            Wd = W[d]
            for u in range(V):
                for j in range(m):
                    hu[j] = rest[j]
                for i in range(k):
                    ei = E[u, i]
                    for j in range(m):
                        hu[j] += Wd[i, j] * ei
                s = 0.0
                for j in range(m):
                    s += v[j] * _softplus(hu[j])
                ev[u] += s
    for t in range(out.shape[0]):
        out[t] += ev[av[t]]


@njit(cache=True)
def _conv_total(g, av, cv, k, m, pad):
    """sum_p [U + B + H] of one convpot over the grid g (the definition)."""
    rows, cols = g.shape
    a, A, W, b, v, E = _conv_unpack(cv, k, m)
    total = 0.0
    h = np.empty(m)
    for y in range(rows):
        for x in range(cols):
            ep = E[av[g[y, x]]]
            for i in range(k):
                total += a[i] * ep[i]
            for j in range(m):
                h[j] = b[j]
            for d in range(9):
                ny, nx = y + d // 3 - 1, x + d % 3 - 1
                if 0 <= ny < rows and 0 <= nx < cols:
                    en = E[av[g[ny, nx]]]
                elif pad >= 0:
                    en = E[pad]
                else:
                    continue
                if d != 4:
                    Ad = A[d]
                    for i in range(k):
                        s = 0.0
                        for j in range(k):
                            s += Ad[i, j] * en[j]
                        total += ep[i] * s
                if m > 0:
                    Wd = W[d]
                    for i in range(k):
                        ei = en[i]
                        for j in range(m):
                            h[j] += Wd[i, j] * ei
            for j in range(m):
                total += v[j] * _softplus(h[j])
    return total


@njit(cache=True, inline="always")
def _energies(y, x, home, grids, hs, views, fac, tabs, out, convs=None):
    D = out.shape[0]
    for t in range(D):
        out[t] = 0.0
    hc = hs[home]
    g_home = grids[home]
    for f in range(fac.shape[0]):
        kind = fac[f, 0]
        av = views[fac[f, 2]]
        if kind == 5:                                           # convpot (skipped without convs)
            if convs is not None:
                _conv_energies(y, x, g_home, av, convs[fac[f, 7]], fac[f, 4], fac[f, 5], fac[f, 6], out)
            continue
        tab = tabs[fac[f, 7]]
        if kind == 2:
            for t in range(D):
                out[t] += tab[av[t], 0]
            continue
        b = fac[f, 1]
        hb = hs[b]
        g = grids[b]
        if kind == 0:
            qy = (y * hc) // hb + fac[f, 4]
            qx = (x * hc) // hb + fac[f, 5]
            if 0 <= qy < g.shape[0] and 0 <= qx < g.shape[1]:
                vb = views[fac[f, 3]][g[qy, qx]]
            elif fac[f, 6] >= 0:
                vb = fac[f, 6]
            else:
                continue
            for t in range(D):
                out[t] += tab[av[t], vb]
        elif kind == 3:                                         # below-pair: b is home, a = the fine cells
            r = hc // hb
            vh = views[fac[f, 3]]
            nu = tab.shape[0]
            hist = np.zeros(nu, np.int64)
            for yy in range(y * r, y * r + r):
                for xx in range(x * r, x * r + r):
                    hist[av[g[yy, xx]]] += 1
            for u in range(nu):
                if hist[u] > 0:
                    for t in range(D):
                        out[t] += hist[u] * tab[u, vh[t]]
        elif kind == 4:                                         # below-count
            r = hc // hb
            vh = views[fac[f, 3]]
            s = 0
            for yy in range(y * r, y * r + r):
                for xx in range(x * r, x * r + r):
                    s += av[g[yy, xx]]
            for t in range(D):
                out[t] += tab[vh[t], s]
        else:                                                   # count
            r = hb // hc
            qy = (y * hc) // hb
            qx = (x * hc) // hb
            vb = views[fac[f, 3]][g[qy, qx]]
            s = 0
            for yy in range(qy * r, qy * r + r):
                for xx in range(qx * r, qx * r + r):
                    if yy != y or xx != x:
                        s += av[g_home[yy, xx]]
            for t in range(D):
                out[t] += tab[vb, s + av[t]]


@njit(cache=True)
def site_energies(y, x, home, grids, hs, views, fac, tabs, D, convs=None):
    out = np.empty(D)
    _energies(y, x, home, grids, hs, views, fac, tabs, out, convs)
    return out


@njit(cache=True, inline="always")
def _witnessed(y, x, t, dd, g_t, g_d, mass, joins, rows, cols, skip):
    """A joined neighbour q (side != skip) with mass_q >= mass_t > 0 and d_q < dd."""
    m = mass[t]
    for d in range(4):
        if d == skip:
            continue
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
            continue
        tq = g_t[ny, nx]
        if mass[tq] >= m and g_d[ny, nx] < dd and joins[d, t, tq]:
            return True
    return False


@njit(cache=True, inline="always")
def _cell_valid(y, x, g_t, g_d, mass, trunk, joins, rows, cols, INF, tree):
    t = g_t[y, x]
    m, dd = mass[t], g_d[y, x]
    if m == 0:
        return dd == INF
    if not tree:
        if trunk[t]:
            return dd == 0
        return dd < INF and _witnessed(y, x, t, dd, g_t, g_d, mass, joins, rows, cols, -1)
    # tree: exactly one parent (joined, mass >= m, d_q < dd); every other join a child (mass <= m, d_q > dd)
    if trunk[t] and dd != 0:
        return False
    if dd >= INF:
        return False
    parents = 0
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
            continue
        tq = g_t[ny, nx]
        if not joins[d, t, tq]:
            continue
        dq = g_d[ny, nx]
        if dq < dd:
            if mass[tq] < m:
                return False
            parents += 1
        elif dq == dd or mass[tq] > m:
            return False
    return parents == (0 if trunk[t] else 1)


@njit(cache=True, inline="always")
def _lgeom(n, rate):
    """log sum_{j < n} exp(-rate j), n >= 1."""
    if rate == 0.0:
        return np.log(n)
    return np.log(-np.expm1(-rate * n)) - np.log(-np.expm1(-rate))


@njit(cache=True, inline="always")
def _tgeom(n, rate):
    """j < n with p(j) ~ exp(-rate j)."""
    u = np.random.random()
    if rate == 0.0:
        j = int(u * n)
    else:
        j = int(np.floor(-np.log1p(u * np.expm1(-rate * n)) / rate))
    return min(max(j, 0), n - 1)


@njit(cache=True)
def _draw(e, n):
    """Gumbel-max over the finite entries of e[:n]; -1 if none."""
    best, arg = -np.inf, -1
    for i in range(n):
        if e[i] < np.inf:
            u = np.random.random()
            v = -e[i] - np.log(-np.log(u))
            if v > best:
                best, arg = v, i
    return arg


@njit(cache=True, inline="always")
def _tree_interval(y, x, t, m, g, g_d, mass, trunk, joins, need, rows, cols, INF, Dmax, delta, T, e, w, lo, hi):
    if m == 0:
        for d in range(4):
            if need[d]:
                return
        lo[t], hi[t] = INF, INF + 1
        w[t] = e[t] / T
        return
    d1, d2, i1, nj = INF, INF, -1, 0                   # smallest joined d (the parent), the next, its side
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
            continue
        tq = g[ny, nx]
        if joins[d, t, tq]:
            nj += 1
            dq = g_d[ny, nx]
            if dq < d1:
                d2, d1, i1 = d1, dq, d
            elif dq < d2:
                d2 = dq
        elif need[d]:
            return                                      # a root that had only p as parent, now unjoined
    if trunk[t]:
        for d in range(4):                              # every join is to a child
            ny, nx = y + DIRS[d][0], x + DIRS[d][1]
            if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                continue
            tq = g[ny, nx]
            if joins[d, t, tq] and (g_d[ny, nx] <= 0 or not need[d]):
                return
        lo[t], hi[t] = 0, 1
        w[t] = e[t] / T
        return
    if nj == 0 or d1 == d2 or d1 >= INF:
        return
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
            continue
        tq = g[ny, nx]
        if not joins[d, t, tq]:
            continue
        if d == i1:
            if mass[tq] < m or need[d]:                 # the parent: mass >= m, and it keeps a parent of its own
                return
        elif mass[tq] > m or not need[d]:               # a child: mass <= m, no parent but p
            return
    a, b = d1 + 1, min(d2, Dmax + 1)
    if a >= b:
        return
    lo[t], hi[t] = a, b
    w[t] = (e[t] + delta * a) / T - _lgeom(b - a, delta / T)


@njit(cache=True)
def site_weights(y, x, e, g, g_d, mass, trunk, joins, Dmax, delta, T, w, lo, hi, need, tree):
    """The certificate at one site: from tile energies e[t], the log-weight
    w[t] (inf = no valid d) summed over the valid d interval [lo[t], hi[t]),
    with the dependants' needs in `need` (a root neighbour with no parent
    but p).  Shared by the tile kernel and any joint kernel that draws the
    tile with something else.  tree: the parent is the joined neighbour of
    smallest d and is unique; the other joined neighbours are p's children
    (mass <= m, d above p's, and no parent other than p); a non-joined root
    neighbour must keep a parent of its own."""
    rows, cols = g.shape
    D = e.shape[0]
    INF = Dmax + 1
    for d in range(4):
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        need[d] = False
        if 0 <= ny < rows and 0 <= nx < cols:
            tq = g[ny, nx]
            if mass[tq] > 0 and not trunk[tq]:
                dq = g_d[ny, nx]
                need[d] = not (dq < INF and _witnessed(ny, nx, tq, dq, g, g_d, mass, joins, rows, cols, (d + 2) % 4))
    for t in range(D):
        w[t] = np.inf
        if e[t] == np.inf:
            continue
        m = mass[t]
        if tree:
            _tree_interval(y, x, t, m, g, g_d, mass, trunk, joins, need, rows, cols, INF, Dmax, delta, T, e, w, lo, hi)
            continue
        l, h = -1, INF                                  # valid d: l < d < h, d <= Dmax
        ok = True
        for d in range(4):
            ny, nx = y + DIRS[d][0], x + DIRS[d][1]
            if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                continue
            tq = g[ny, nx]
            dq = g_d[ny, nx]
            if need[d]:
                if m >= mass[tq] and joins[d, t, tq]:
                    h = min(h, dq)
                else:
                    ok = False
                    break
            if m > 0 and mass[tq] >= m and joins[d, t, tq] and dq < INF:
                if l < 0 or dq < l:
                    l = dq                              # smallest witness d
        if not ok:
            continue
        if m == 0:
            if h < INF:                                 # some dependant needs p
                continue
            lo[t], hi[t] = INF, INF + 1
            w[t] = e[t] / T
        elif trunk[t]:
            if h <= 0:
                continue
            lo[t], hi[t] = 0, 1
            w[t] = e[t] / T
        else:
            if l < 0:
                continue
            a, b = l + 1, min(h, Dmax + 1)              # d in [a, b)
            if a >= b:
                continue
            lo[t], hi[t] = a, b
            w[t] = (e[t] + delta * a) / T - _lgeom(b - a, delta / T)


@njit(cache=True)
def sweep(home, grids, hs, views, fac, tabs, fixed, colours, ncol, cert, joins, delta, T, convs=None):
    g = grids[home]
    rows, cols = g.shape
    D = 0
    if fac.shape[0] > 0:
        D = views[fac[0, 3 if fac[0, 0] == 3 or fac[0, 0] == 4 else 2]].shape[0]   # the home view's domain
    has_cert = cert[4] == 1
    if has_cert:
        D = views[cert[0]].shape[0]
    e = np.empty(D)
    w = np.empty(D)
    lo = np.empty(D, np.int64)
    hi = np.empty(D, np.int64)
    Dmax = cert[3]
    need = np.empty(4, np.bool_)
    bad = 0
    for col in range(ncol):
        for y in range(rows):
            for x in range(cols):
                if colours[y, x] != col or fixed[y, x]:
                    continue
                _energies(y, x, home, grids, hs, views, fac, tabs, e, convs)
                if not has_cert:
                    for t in range(D):
                        e[t] /= T
                    pick = _draw(e, D)
                    if pick < 0:
                        bad += 1
                    else:
                        g[y, x] = pick
                    continue
                mass, trunk, g_d = views[cert[0]], views[cert[1]], grids[cert[2]]
                site_weights(y, x, e, g, g_d, mass, trunk, joins, Dmax, delta, T, w, lo, hi, need, cert[5] == 1)
                pick = _draw(w, D)
                if pick < 0:
                    bad += 1
                else:
                    g[y, x] = pick
                    g_d[y, x] = lo[pick] + _tgeom(hi[pick] - lo[pick], delta / T)
    return bad


@njit(cache=True)
def total_energy(home, grids, hs, views, fac, tabs, cert, joins, delta, convs=None):
    """(finite energy, inf count): the home-side rows at the current state
    (below rows skipped: they belong to the finer channel), each pair once (reflected rows, those with a negative offset in the home
    channel, skipped), counts once per block, convpots as defined (each
    site's U + 8 bilinear terms + head; skipped without convs), plus the
    certificate."""
    g = grids[home]
    rows, cols = g.shape
    total = 0.0
    nviol = 0
    hc = hs[home]
    if convs is not None:
        for f in range(fac.shape[0]):
            if fac[f, 0] == 5:
                total += _conv_total(g, views[fac[f, 2]], convs[fac[f, 7]], fac[f, 4], fac[f, 5], fac[f, 6])
    for y in range(rows):
        for x in range(cols):
            t = g[y, x]
            for f in range(fac.shape[0]):
                kind = fac[f, 0]
                if kind >= 3:                                   # counted on the fine side / convpot
                    continue
                av = views[fac[f, 2]]
                tab = tabs[fac[f, 7]]
                if kind == 2:
                    total += tab[av[t], 0]
                    continue
                b = fac[f, 1]
                hb = hs[b]
                gb = grids[b]
                if kind == 0:
                    if b == home and (fac[f, 4] < 0 or (fac[f, 4] == 0 and fac[f, 5] < 0)):
                        continue
                    qy = (y * hc) // hb + fac[f, 4]
                    qx = (x * hc) // hb + fac[f, 5]
                    if 0 <= qy < gb.shape[0] and 0 <= qx < gb.shape[1]:
                        vb = views[fac[f, 3]][gb[qy, qx]]
                    elif fac[f, 6] >= 0:
                        vb = fac[f, 6]
                    else:
                        continue
                    v = tab[av[t], vb]
                else:
                    r = hb // hc
                    if y % r != 0 or x % r != 0:
                        continue
                    qy = (y * hc) // hb
                    qx = (x * hc) // hb
                    s = 0
                    for yy in range(qy * r, qy * r + r):
                        for xx in range(qx * r, qx * r + r):
                            s += av[g[yy, xx]]
                    v = tab[views[fac[f, 3]][gb[qy, qx]], s]
                if v == np.inf:
                    nviol += 1
                else:
                    total += v
    if cert[4]:
        mass, trunk, g_d = views[cert[0]], views[cert[1]], grids[cert[2]]
        INF = cert[3] + 1
        for y in range(rows):
            for x in range(cols):
                if not _cell_valid(y, x, g, g_d, mass, trunk, joins, rows, cols, INF, cert[5] == 1):
                    nviol += 1
                elif mass[g[y, x]] > 0:
                    total += delta * g_d[y, x]
    return total, nviol


# ------------------------------------------------- hard rows first, candidate cap, active blocks
# Packed from Model.compile(home, hard_first=True): rows [0, nhard) hold a +inf
# entry, rows [nhard, F) are soft.  A candidate is admissible iff its hard
# energy is finite; soft rows are evaluated on admissible candidates only.

@njit(cache=True, inline="always")
def _rows(y, x, home, grids, hs, views, fac, tabs, convs, f0, f1, cand, nc, out, scratch, skip):
    """out[i] += rows f0..f1-1 at z_p = cand[i], for i < nc (skip: only where out[i] < inf)."""
    hc = hs[home]
    g_home = grids[home]
    for f in range(f0, f1):
        kind = fac[f, 0]
        av = views[fac[f, 2]]
        if kind == 5:
            for t in range(scratch.shape[0]):
                scratch[t] = 0.0
            _conv_energies(y, x, g_home, av, convs[fac[f, 7]], fac[f, 4], fac[f, 5], fac[f, 6], scratch)
            for i in range(nc):
                if not skip or out[i] < np.inf:
                    out[i] += scratch[cand[i]]
            continue
        tab = tabs[fac[f, 7]]
        if kind == 2:
            for i in range(nc):
                if not skip or out[i] < np.inf:
                    out[i] += tab[av[cand[i]], 0]
            continue
        b = fac[f, 1]
        hb = hs[b]
        g = grids[b]
        if kind == 0:
            qy = (y * hc) // hb + fac[f, 4]
            qx = (x * hc) // hb + fac[f, 5]
            if 0 <= qy < g.shape[0] and 0 <= qx < g.shape[1]:
                vb = views[fac[f, 3]][g[qy, qx]]
            elif fac[f, 6] >= 0:
                vb = fac[f, 6]
            else:
                continue
            for i in range(nc):
                if not skip or out[i] < np.inf:
                    out[i] += tab[av[cand[i]], vb]
        elif kind == 3:
            r = hc // hb
            vh = views[fac[f, 3]]
            nu = tab.shape[0]
            hist = np.zeros(nu, np.int64)
            for yy in range(y * r, y * r + r):
                for xx in range(x * r, x * r + r):
                    hist[av[g[yy, xx]]] += 1
            for u in range(nu):
                if hist[u] > 0:
                    for i in range(nc):
                        if not skip or out[i] < np.inf:
                            out[i] += hist[u] * tab[u, vh[cand[i]]]
        elif kind == 4:
            r = hc // hb
            vh = views[fac[f, 3]]
            s = 0
            for yy in range(y * r, y * r + r):
                for xx in range(x * r, x * r + r):
                    s += av[g[yy, xx]]
            for i in range(nc):
                if not skip or out[i] < np.inf:
                    out[i] += tab[vh[cand[i]], s]
        else:
            r = hb // hc
            qy = (y * hc) // hb
            qx = (x * hc) // hb
            vb = views[fac[f, 3]][g[qy, qx]]
            s = 0
            for yy in range(qy * r, qy * r + r):
                for xx in range(qx * r, qx * r + r):
                    if yy != y or xx != x:
                        s += av[g_home[yy, xx]]
            for i in range(nc):
                if not skip or out[i] < np.inf:
                    out[i] += tab[vb, s + av[cand[i]]]


@njit(cache=True, inline="always")
def _site_cap(y, x, home, grids, hs, views, fac, tabs, convs, nhard, K, cand, eh, out, scratch):
    """Candidate set and energies at (y, x); returns nc.  Hard rows on all D
    first (eh); K <= 0: cand = 0..D-1, out = eh plus the soft rows.  K > 0:
    cand[0] = z_p, cand[1:nc] = min(K - 1, n) values drawn uniformly without
    replacement from the n admissible values != z_p (z_p read only to
    include it), out[i] = energy of cand[i]."""
    D = eh.shape[0]
    for t in range(D):
        cand[t] = t
        eh[t] = 0.0
    _rows(y, x, home, grids, hs, views, fac, tabs, convs, 0, nhard, cand, D, eh, scratch, True)
    if K <= 0:
        for t in range(D):
            out[t] = eh[t]
        _rows(y, x, home, grids, hs, views, fac, tabs, convs, nhard, fac.shape[0], cand, D, out, scratch, nhard > 0)
        return D
    z = grids[home][y, x]
    n = 0
    for t in range(D):
        if t != z and eh[t] < np.inf:
            n += 1
            cand[n] = t
    cand[0] = z
    k = min(K - 1, n)
    for j in range(k):
        r = j + np.random.randint(0, n - j)
        c = cand[1 + j]
        cand[1 + j] = cand[1 + r]
        cand[1 + r] = c
    nc = 1 + k
    for i in range(nc):
        out[i] = eh[cand[i]]
    _rows(y, x, home, grids, hs, views, fac, tabs, convs, nhard, fac.shape[0], cand, nc, out, scratch, nhard > 0)
    return nc


@njit(cache=True)
def site_candidates(y, x, home, grids, hs, views, fac, tabs, convs, nhard, K, D):
    """(cand, e): the set the capped kernel draws over at (y, x) and its
    energies (K <= 0: every value, inf where a hard row forbids it)."""
    cand = np.empty(D, np.int64)
    eh = np.empty(D)
    out = np.empty(D)
    nc = _site_cap(y, x, home, grids, hs, views, fac, tabs, convs, nhard, K, cand, eh, out, np.empty(D))
    return cand[:nc].copy(), out[:nc].copy()


@njit(cache=True)
def admissible(home, grids, hs, views, fac, tabs, convs, rows_sel, D):
    """(rows, cols, D) bool: finite energy under the packed rows listed in
    rows_sel (the hard rows a caller treats as parents)."""
    g = grids[home]
    R, C = g.shape
    ok = np.empty((R, C, D), np.bool_)
    cand = np.arange(D)
    e = np.empty(D)
    scratch = np.empty(D)
    for y in range(R):
        for x in range(C):
            for t in range(D):
                e[t] = 0.0
            for i in range(rows_sel.shape[0]):
                f = rows_sel[i]
                _rows(y, x, home, grids, hs, views, fac, tabs, convs, f, f + 1, cand, D, e, scratch, True)
            for t in range(D):
                ok[y, x, t] = e[t] < np.inf
    return ok


@njit(cache=True)
def sweep_cap(home, grids, hs, views, fac, tabs, fixed, dormant, active, hb, colours, ncol, cert, joins, delta,
              T, convs, nhard, K, D):
    """One sweep of `sweep` with hard rows first, the candidate cap K (K <= 0:
    full enumeration, the same draws as `sweep`) and block skipping: sites
    are visited colour by colour, block by block (hb x hb home cells,
    active[by, bx] tested first), raster inside a block; fixed or dormant
    sites are skipped.  hb = 1 is `sweep`'s raster order.  The certificate
    path needs K <= 0."""
    g = grids[home]
    rows, cols = g.shape
    has_cert = cert[4] == 1
    cand = np.empty(D, np.int64)
    eh = np.empty(D)
    e = np.empty(D)
    scratch = np.empty(D)
    w = np.empty(D)
    lo = np.empty(D, np.int64)
    hi = np.empty(D, np.int64)
    Dmax = cert[3]
    need = np.empty(4, np.bool_)
    nby = (rows + hb - 1) // hb
    nbx = (cols + hb - 1) // hb
    bad = 0
    for col in range(ncol):
        for by in range(nby):
            for bx in range(nbx):
                if not active[by, bx]:
                    continue
                for y in range(by * hb, min(rows, by * hb + hb)):
                    for x in range(bx * hb, min(cols, bx * hb + hb)):
                        if colours[y, x] != col or fixed[y, x] or dormant[y, x]:
                            continue
                        nc = _site_cap(y, x, home, grids, hs, views, fac, tabs, convs, nhard, K, cand, eh, e,
                                       scratch)
                        if not has_cert:
                            for i in range(nc):
                                e[i] /= T
                            pick = _draw(e, nc)
                            if pick < 0:
                                bad += 1
                            else:
                                g[y, x] = cand[pick]
                            continue
                        mass, trunk, g_d = views[cert[0]], views[cert[1]], grids[cert[2]]
                        site_weights(y, x, e, g, g_d, mass, trunk, joins, Dmax, delta, T, w, lo, hi, need,
                                     cert[5] == 1)
                        pick = _draw(w, D)
                        if pick < 0:
                            bad += 1
                        else:
                            g[y, x] = pick
                            g_d[y, x] = lo[pick] + _tgeom(hi[pick] - lo[pick], delta / T)
    return bad


# ------------------------------------------- admitted lists (dsl_updates C1, A_p)
# The parent hard rows (unaries, pairs to other channels) are constant while
# the parents are, so the values they admit at a site, and their energies
# there, are computed once (adm_lists) and stored as CSR lists deduplicated by
# content: aid[y, x] -> list a = aidx[aptr[a]:aptr[a + 1]] (ascending values)
# with parent energies ae[...].  sweep_adm starts every site from its list:
# the remaining hard rows (sib) on the list, the cap from what they admit,
# soft rows on the capped set.  Valid until the parents change.

@njit(cache=True)
def adm_lists(home, grids, hs, views, fac, tabs, convs, rows_sel, D):
    """(aid (rows, cols) int32, aptr, aidx int64, ae float64): per-site index
    into the distinct (value, parent energy) lists of finite energy under the
    rows in rows_sel.  Dedup by a 64-bit hash, confirmed by comparison."""
    g = grids[home]
    R, C = g.shape
    aid = np.empty((R, C), np.int32)
    cap = 4 * D + 4
    aidx = np.empty(cap, np.int64)
    ae = np.empty(cap)
    aptr = np.zeros(R * C + 1, np.int64)
    hsh = np.empty(R * C, np.uint64)
    cand = np.arange(D)
    e = np.empty(D)
    eb = e.view(np.uint64)
    scratch = np.empty(D)
    nl = 0
    P1, P2 = np.uint64(1099511628211), np.uint64(0x9E3779B97F4A7C15)
    for y in range(R):
        for x in range(C):
            for t in range(D):
                e[t] = 0.0
            for i in range(rows_sel.shape[0]):
                f = rows_sel[i]
                _rows(y, x, home, grids, hs, views, fac, tabs, convs, f, f + 1, cand, D, e, scratch, True)
            h = np.uint64(14695981039346656037)
            m = 0
            for t in range(D):
                if e[t] < np.inf:
                    h = (h ^ np.uint64(t)) * P1
                    h = (h ^ eb[t]) * P2
                    m += 1
            found = -1
            for l in range(nl):
                if hsh[l] == h and aptr[l + 1] - aptr[l] == m:
                    j, same = aptr[l], True
                    for t in range(D):
                        if e[t] < np.inf:
                            if aidx[j] != t or ae[j] != e[t]:
                                same = False
                                break
                            j += 1
                    if same:
                        found = l
                        break
            if found < 0:
                p = aptr[nl]
                if p + m > cap:
                    cap = 2 * (p + m)
                    a2, e2 = np.empty(cap, np.int64), np.empty(cap)
                    a2[:p] = aidx[:p]
                    e2[:p] = ae[:p]
                    aidx, ae = a2, e2
                for t in range(D):
                    if e[t] < np.inf:
                        aidx[p] = t
                        ae[p] = e[t]
                        p += 1
                aptr[nl + 1] = p
                hsh[nl] = h
                found = nl
                nl += 1
            aid[y, x] = found
    return aid, aptr[:nl + 1].copy(), aidx[:aptr[nl]].copy(), ae[:aptr[nl]].copy()


@njit(cache=True, inline="always")
def _site_adm(y, x, home, grids, hs, views, fac, tabs, convs, nhard, sib, aid, aptr, aidx, ae, K, lst, le, cand, out,
              scratch):
    """_site_cap over the admitted list of (y, x): the sib hard rows (runs
    [sib[j, 0], sib[j, 1]) of packed rows) on the list (parent energies ae
    first), then as _site_cap: K <= 0 every listed
    value; K > 0 cand[0] = z_p (energy inf if unlisted) and min(K - 1, n) of
    the n listed values != z_p that sib admits, the same uniform draws as
    _site_cap over the same ascending order; soft rows on the result."""
    a = aid[y, x]
    p0 = aptr[a]
    m = aptr[a + 1] - p0
    for i in range(m):
        lst[i] = aidx[p0 + i]
        le[i] = ae[p0 + i]
    for j in range(sib.shape[0]):
        _rows(y, x, home, grids, hs, views, fac, tabs, convs, sib[j, 0], sib[j, 1], lst, m, le, scratch, True)
    if K <= 0:
        for i in range(m):
            cand[i] = lst[i]
            out[i] = le[i]
        _rows(y, x, home, grids, hs, views, fac, tabs, convs, nhard, fac.shape[0], cand, m, out, scratch, nhard > 0)
        return m
    z = grids[home][y, x]
    ez = np.inf
    n = 0
    for i in range(m):
        t = lst[i]
        if t == z:
            ez = le[i]
        elif le[i] < np.inf:
            n += 1
            cand[n] = t
            out[n] = le[i]
    cand[0] = z
    out[0] = ez
    k = min(K - 1, n)
    for j in range(k):
        r = j + np.random.randint(0, n - j)
        c = cand[1 + j]
        cand[1 + j] = cand[1 + r]
        cand[1 + r] = c
        v = out[1 + j]
        out[1 + j] = out[1 + r]
        out[1 + r] = v
    nc = 1 + k
    _rows(y, x, home, grids, hs, views, fac, tabs, convs, nhard, fac.shape[0], cand, nc, out, scratch, nhard > 0)
    return nc


@njit(cache=True)
def site_candidates_adm(y, x, home, grids, hs, views, fac, tabs, convs, nhard, sib, aid, aptr, aidx, ae, K, D):
    """(cand, e): the set sweep_adm draws over at (y, x) and its energies."""
    cand = np.empty(D, np.int64)
    out = np.empty(D)
    nc = _site_adm(y, x, home, grids, hs, views, fac, tabs, convs, nhard, sib, aid, aptr, aidx, ae, K,
                   np.empty(D, np.int64), np.empty(D), cand, out, np.empty(D))
    return cand[:nc].copy(), out[:nc].copy()


@njit(cache=True)
def sweep_adm(home, grids, hs, views, fac, tabs, fixed, dormant, active, hb, colours, ncol, T, convs, nhard, sib,
              aid, aptr, aidx, ae, K, D):
    """sweep_cap (no certificates) with every site started from its admitted
    list: the draws of sweep_cap for the same seed (identical when the parent
    rows' finite entries are 0 or no hard sibling row precedes a parent row;
    else equal up to the order of the floating-point sum)."""
    g = grids[home]
    rows, cols = g.shape
    lst = np.empty(D, np.int64)
    le = np.empty(D)
    cand = np.empty(D, np.int64)
    e = np.empty(D)
    scratch = np.empty(D)
    nby = (rows + hb - 1) // hb
    nbx = (cols + hb - 1) // hb
    bad = 0
    for col in range(ncol):
        for by in range(nby):
            for bx in range(nbx):
                if not active[by, bx]:
                    continue
                for y in range(by * hb, min(rows, by * hb + hb)):
                    for x in range(bx * hb, min(cols, bx * hb + hb)):
                        if colours[y, x] != col or fixed[y, x] or dormant[y, x]:
                            continue
                        nc = _site_adm(y, x, home, grids, hs, views, fac, tabs, convs, nhard, sib, aid, aptr, aidx,
                                       ae, K, lst, le, cand, e, scratch)
                        for i in range(nc):
                            e[i] /= T
                        pick = _draw(e, nc)
                        if pick < 0:
                            bad += 1
                        else:
                            g[y, x] = cand[pick]
    return bad
