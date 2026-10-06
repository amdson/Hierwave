"""The generic level-1 Gibbs kernel (numba).  It reads core.Packed: a tuple
of channel grids, a tuple of views, factor descriptor rows and a tuple of
tables, and knows nothing about which channel set wrote them.

Per site, colour by colour: the energy of every candidate value is the sum
of its factor rows (pair / count / unary), then one Gumbel-max draw over the
finite candidates.

Certificate (channels.tex, 14): the home channel's cells carry a mass, a
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


@njit(cache=True, inline="always")
def _energies(y, x, home, grids, hs, views, fac, tabs, out):
    D = out.shape[0]
    for t in range(D):
        out[t] = 0.0
    hc = hs[home]
    g_home = grids[home]
    for f in range(fac.shape[0]):
        kind = fac[f, 0]
        av = views[fac[f, 2]]
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
def site_energies(y, x, home, grids, hs, views, fac, tabs, D):
    out = np.empty(D)
    _energies(y, x, home, grids, hs, views, fac, tabs, out)
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
def sweep(home, grids, hs, views, fac, tabs, fixed, colours, ncol, cert, joins, delta, T):
    g = grids[home]
    rows, cols = g.shape
    D = views[fac[0, 2]].shape[0] if fac.shape[0] > 0 else 0
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
                _energies(y, x, home, grids, hs, views, fac, tabs, e)
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
def total_energy(home, grids, hs, views, fac, tabs, cert, joins, delta):
    """(finite energy, inf count): the home-side rows at the current state,
    each pair once (reflected rows, those with a negative offset in the home
    channel, skipped), counts once per block, plus the certificate."""
    g = grids[home]
    rows, cols = g.shape
    total = 0.0
    nviol = 0
    hc = hs[home]
    for y in range(rows):
        for x in range(cols):
            t = g[y, x]
            for f in range(fac.shape[0]):
                kind = fac[f, 0]
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
