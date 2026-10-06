"""The generic level-1 Gibbs kernel (numba).  It reads core.Packed: a tuple
of channel grids, a tuple of views, factor descriptor rows and a tuple of
tables, and knows nothing about which channel set wrote them.

Per site, colour by colour: the energy of every candidate value is the sum
of its factor rows (pair / count / unary), then one Gumbel-max draw over the
finite candidates.  With a certificate the candidates are (t, d) and the
validity of the site and of its four neighbours (the dependants, whose
other witnesses are precomputed) is added as inf per violation; the
neighbours' d are read as they are, so the colouring needs radius 2."""
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
def _witnessed(y, x, m, dd, g_t, g_d, mass, rows, cols, skip):
    """A neighbour q (side != skip) with mass_q >= m > 0 and d_q < dd."""
    for d in range(4):
        if d == skip:
            continue
        ny, nx = y + DIRS[d][0], x + DIRS[d][1]
        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
            continue
        if mass[g_t[ny, nx]] >= m and g_d[ny, nx] < dd:
            return True
    return False


@njit(cache=True, inline="always")
def _own_valid(m, is_trunk, dd, INF):
    """Validity of a cell by itself: mass 0 holds d = INF; a trunk holds d = 0;
    anything else needs a witness (checked by the caller) and d < INF."""
    if m == 0:
        return dd == INF
    if is_trunk:
        return dd == 0
    return dd < INF


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


@njit(cache=True)
def sweep(home, grids, hs, views, fac, tabs, fixed, colours, ncol, cert, delta, T):
    g = grids[home]
    rows, cols = g.shape
    D = views[fac[0, 2]].shape[0] if fac.shape[0] > 0 else 0
    if cert[4]:
        D = views[cert[0]].shape[0]
    e = np.empty(D)
    has_cert = cert[4] == 1
    Dmax = cert[3]
    INF = Dmax + 1
    ND = Dmax + 2
    ej = np.empty(D * ND)
    base = np.empty(4, np.bool_)
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
                # dependants: neighbour q with mass > 0, not a trunk, valid only through p
                for d in range(4):
                    ny, nx = y + DIRS[d][0], x + DIRS[d][1]
                    base[d] = True
                    if 0 <= ny < rows and 0 <= nx < cols:
                        tq = g[ny, nx]
                        mq = mass[tq]
                        if mq > 0 and not trunk[tq]:
                            dq = g_d[ny, nx]
                            base[d] = dq < INF and _witnessed(ny, nx, mq, dq, g, g_d, mass, rows, cols, (d + 2) % 4)
                for t in range(D):
                    m = mass[t]
                    for dd in range(ND):
                        v = e[t] + delta * dd
                        if not _own_valid(m, trunk[t] == 1, dd, INF):
                            v = np.inf
                        elif m > 0 and not trunk[t] and not _witnessed(y, x, m, dd, g, g_d, mass, rows, cols, -1):
                            v = np.inf
                        else:
                            for d in range(4):
                                if base[d]:
                                    continue
                                ny, nx = y + DIRS[d][0], x + DIRS[d][1]
                                if not (m >= mass[g[ny, nx]] and dd < g_d[ny, nx]):
                                    v = np.inf
                                    break
                        ej[t * ND + dd] = v / T
                pick = _draw(ej, D * ND)
                if pick < 0:
                    bad += 1
                else:
                    g[y, x] = pick // ND
                    g_d[y, x] = pick % ND
    return bad


@njit(cache=True)
def total_energy(home, grids, hs, views, fac, tabs, cert, delta):
    """Sum over sites of the home-side rows at the current value, reflected
    rows excluded (each pair once) by taking only rows with positive offset
    or a different channel; counts and unaries once per site."""
    g = grids[home]
    rows, cols = g.shape
    D = views[fac[0, 2]].shape[0] if fac.shape[0] > 0 else 0
    e = np.empty(D)
    total = 0.0
    nviol = 0
    for y in range(rows):
        for x in range(cols):
            hc = hs[home]
            for f in range(fac.shape[0]):
                kind = fac[f, 0]
                av = views[fac[f, 2]]
                tab = tabs[fac[f, 7]]
                t = g[y, x]
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
                    qy = (y * hc) // hb
                    qx = (x * hc) // hb
                    if y % r != 0 or x % r != 0:
                        continue
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
                t = g[y, x]
                m = mass[t]
                dd = g_d[y, x]
                ok = _own_valid(m, trunk[t] == 1, dd, INF)
                if ok and m > 0 and not trunk[t]:
                    ok = _witnessed(y, x, m, dd, g, g_d, mass, rows, cols, -1)
                if not ok:
                    nviol += 1
                elif m > 0:
                    total += delta * dd
    return total, nviol
