"""Labelled-port connectivity of one k x k tile block: can a block be sampled
to a given port pattern?  (notes/experiments/block_ports.py)

Ports: the block's 8 half-sides q = 2 d + i (side d = N, E, S, W; half i = 0
the left / top half, as castlegen/ports.py), k/2 edge tiles each.  A pattern
lab (8,) gives each port 0 (closed), a class 1..MLAB or OPEN.  Ports of one
class are joined outside the block (a bridge elsewhere), so every opening of a
class joins that class's virtual node.  An OPEN port must have openings but
assumes nothing outside: no virtual node, not an exit.  grp (MLAB + 1,) puts the classes in groups:
the classes of one group must be joined inside the block (all in one group:
the block plus its virtual nodes is one component).

Opening: an edge tile on side d that is a node whose side-d socket can join
some node (conn_tables' sock: the block across is taken to accommodate it,
as a port promise would guarantee).  Joins inside the block: conn_tables' Dh, Dv.

Constraints, on the graph of the block's nodes plus the virtual nodes:
  closed  no opening on a port of class 0
  open    at least one opening on every open port (a class or OPEN)
  merge   the virtual nodes of a group's classes (those with an opening) lie
          in one component
  exit    every node lies in a component holding a virtual node; a block
          with open ports but no class (a sink): its nodes form one component
Scores (0 iff every constraint holds), parts [closed, open, merge, exit]:
  mode 0 (binary)  closed ports with an opening, empty open ports, groups
                   split over > 1 component, 1 if any node is orphaned (a
                   sink: 1 if > 1 component)
  mode 1 (comps)   openings on closed ports, empty open ports,
                   sum over groups (components - 1), orphan components (a
                   sink: components - 1)
  mode 2 (nodes)   as mode 1, but the exit part counts orphan nodes (a sink:
                   still components - 1)
  mode 3 (dist)    as mode 1, but each orphan component costs its distance
                   to the nearest class port's edge cells (Manhattan, its
                   nearest tile; geometry only) plus half its size: a path
                   grown toward an exit pays its way, which nodes does not
Everything reads the block alone: one union-find over k^2 + MLAB + 1 nodes
(_parts); site_scores scores every tile at one site from one union-find.
"""
import numpy as np
from numba import njit, prange

from castlegen import generic as GN

MLAB = 4
OPEN = MLAB + 1
MODES = {"binary": 0, "comps": 1, "nodes": 2, "dist": 3}


def tables(ts):
    """(node (S,), Dh (S, S), Dv (S, S), sock (4, S)) as contiguous bool arrays."""
    return tuple(np.ascontiguousarray(a, np.bool_) for a in GN.conn_tables(ts))


@njit(cache=True, inline="always")
def _find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


@njit(cache=True, inline="always")
def _exitdist(y, x, lab, k):
    """Manhattan distance from (y, x) to the nearest edge cell of a class port."""
    hh = k // 2
    best = 1e9
    for q in range(8):
        if not 0 < lab[q] < OPEN:
            continue
        d, i = q // 2, q % 2
        lo, hi = i * hh, (i + 1) * hh - 1
        if d == 0:
            v = y + max(0, lo - x, x - hi)
        elif d == 1:
            v = k - 1 - x + max(0, lo - y, y - hi)
        elif d == 2:
            v = k - 1 - y + max(0, lo - x, x - hi)
        else:
            v = x + max(0, lo - y, y - hi)
        best = min(best, float(v))
    return best


@njit(cache=True)
def _parts(T, lab, grp, node, Dh, Dv, sock, mode, par, out):
    """Score parts of block T (k, k) into out (4,); par: (k k + MLAB + 1,) scratch."""
    k = T.shape[0]
    n = k * k
    for a in range(n + MLAB + 1):
        par[a] = a
    for y in range(k):
        for x in range(k):
            t = T[y, x]
            if not node[t]:
                continue
            if x + 1 < k and Dh[t, T[y, x + 1]]:
                ra, rb = _find(par, y * k + x), _find(par, y * k + x + 1)
                if ra != rb:
                    par[ra] = rb
            if y + 1 < k and Dv[t, T[y + 1, x]]:
                ra, rb = _find(par, y * k + x), _find(par, (y + 1) * k + x)
                if ra != rb:
                    par[ra] = rb
    for j in range(4):
        out[j] = 0.0
    used = np.zeros(MLAB + 1, np.bool_)                                    # class has an opening
    hh = k // 2
    for q in range(8):
        d, i = q // 2, q % 2
        l = lab[q]
        nopen = 0
        for j in range(hh):
            s = i * hh + j
            if d == 0:
                y, x = 0, s
            elif d == 1:
                y, x = s, k - 1
            elif d == 2:
                y, x = k - 1, s
            else:
                y, x = s, 0
            t = T[y, x]
            if not (node[t] and sock[d, t]):
                continue
            nopen += 1
            if l == 0 or l == OPEN:
                continue
            used[l] = True
            ra, rb = _find(par, y * k + x), _find(par, n + l)
            if ra != rb:
                par[ra] = rb
        if l == 0:
            if nopen > 0:
                out[0] += 1.0 if mode == 0 else nopen
        elif nopen == 0:
            out[1] += 1.0
    good = np.zeros(n + MLAB + 1, np.bool_)                                # root holds a virtual node
    for l in range(1, MLAB + 1):
        if used[l]:
            good[_find(par, n + l)] = True
    for g in range(1, MLAB + 1):
        nc = 0
        for l in range(1, MLAB + 1):
            if used[l] and grp[l] == g:
                r = _find(par, n + l)
                first = True
                for l2 in range(1, l):
                    if used[l2] and grp[l2] == g and _find(par, n + l2) == r:
                        first = False
                if first:
                    nc += 1
        if nc > 1:
            out[2] += 1.0 if mode == 0 else nc - 1
    sink = True
    anyopen = False
    for q in range(8):
        sink &= lab[q] == 0 or lab[q] == OPEN
        anyopen |= lab[q] > 0
    if sink and anyopen:
        nc = 0
        for a in range(n):
            if node[T[a // k, a % k]] and _find(par, a) == a:
                nc += 1
        if nc > 1:
            out[3] = 1.0 if mode == 0 else nc - 1
        return
    orphan = np.zeros(n, np.bool_)
    dmin = np.full(n, 1e9)
    for a in range(n):
        if node[T[a // k, a % k]]:
            r = _find(par, a)
            if not good[r]:
                if mode == 2:
                    out[3] += 1.0
                elif mode == 3:
                    out[3] += 0.5
                    dmin[r] = min(dmin[r], _exitdist(a // k, a % k, lab, k))
                elif not orphan[r]:
                    orphan[r] = True
                    out[3] += 1.0
    if mode == 3:
        for a in range(n):
            if dmin[a] < 1e9:
                out[3] += dmin[a]
    if mode == 0 and out[3] > 0:
        out[3] = 1.0


@njit(cache=True)
def parts(T, lab, grp, node, Dh, Dv, sock, mode):
    out = np.zeros(4)
    _parts(T, lab, grp, node, Dh, Dv, sock, mode, np.empty(T.size + MLAB + 1, np.int64), out)
    return out


@njit(cache=True, parallel=True)
def cand_scores(T, py, px, cand, lab, grp, node, Dh, Dv, sock, mode, w):
    """(len(py), M) weighted score of block T with tile (py[i], px[i]) set to
    each of cand[i] (tile ids), one cell at a time (the others as in T)."""
    k, M = len(py), cand.shape[1]
    S = node.shape[0]
    res = np.zeros((k, M))
    for i in prange(k):
        B = T.copy()
        par = np.empty(T.size + MLAB + 1, np.int64)
        out = np.zeros(4)
        memo = np.full(S, -1.0)
        for j in range(M):
            t = cand[i, j]
            if memo[t] < 0:
                B[py[i], px[i]] = t
                _parts(B, lab, grp, node, Dh, Dv, sock, mode, par, out)
                memo[t] = w[0] * out[0] + w[1] * out[1] + w[2] * out[2] + w[3] * out[3]
            res[i, j] = memo[t]
    return res


def pattern(T, tabs, m=MLAB):
    """(8,) the pattern block T realises: each port takes the component of its
    largest-component opening; more than m components merge the smallest into
    the largest (window_labels' rule, on block-internal joins)."""
    node, Dh, Dv, sock = tabs
    k = T.shape[0]
    par = np.arange(k * k)
    for y in range(k):
        for x in range(k):
            if node[T[y, x]]:
                if x + 1 < k and Dh[T[y, x], T[y, x + 1]]:
                    par[_find(par, y * k + x)] = _find(par, y * k + x + 1)
                if y + 1 < k and Dv[T[y, x], T[y + 1, x]]:
                    par[_find(par, y * k + x)] = _find(par, (y + 1) * k + x)
    comp = np.array([_find(par, a) for a in range(k * k)])
    size = np.bincount(comp, minlength=k * k)
    lab, hh = np.full(8, -1), k // 2
    for q in range(8):
        d, i = divmod(q, 2)
        span = np.arange(i * hh, (i + 1) * hh)
        yy, xx = np.broadcast_arrays(*((0, span), (span, k - 1), (k - 1, span), (span, 0))[d])
        on = node[T[yy, xx]] & sock[d, T[yy, xx]]
        if on.any():
            cs = comp[yy[on] * k + xx[on]]
            lab[q] = cs[np.argmax(size[cs])]
    used = list(dict.fromkeys(lab[lab >= 0]))
    while len(used) > m:
        used.sort(key=lambda c: size[c])
        lab[lab == used[0]] = used[-1]
        size[used[-1]] += size[used[0]]
        used = used[1:]
    canon, res = {}, np.zeros(8, np.int64)
    for q in range(8):
        if lab[q] >= 0:
            res[q] = canon.setdefault(lab[q], len(canon) + 1)
    return res


def fmt(lab):
    """'A.B.|....' style: one letter per port (. closed), sides N E S W."""
    return " ".join("".join(".ABCD+"[lab[2 * d + i]] for i in range(2)) for d in range(4))


# ------------------------------------------------------------ one site
@njit(cache=True, inline="always")
def _cunion(cp, m):
    """Union the classes of bit mask m in the class union-find cp."""
    f = -1
    for l in range(1, MLAB + 1):
        if (m >> l) & 1:
            r = _find(cp, l)
            if f < 0:
                f = r
            elif r != f:
                cp[r] = f


@njit(cache=True)
def _site_scores(T, y0, x0, lab, grp, node, Dh, Dv, sock, mode, w, allowed, par, res):
    """res (S,): the weighted score of T with tile (y0, x0) set to each
    allowed tile, equal to _parts' (tests/test_blockconn.py).  One union-find
    without the site; per tile, merge the <= 4 neighbour components it joins
    (plus its own opening on an edge) into the summaries of the rest."""
    k = T.shape[0]
    n = k * k
    p = y0 * k + x0
    hh = k // 2
    for a in range(n):
        par[a] = a
    for y in range(k):
        for x in range(k):
            a = y * k + x
            if a == p:
                continue
            t = T[y, x]
            if x + 1 < k and a + 1 != p and Dh[t, T[y, x + 1]]:
                ra, rb = _find(par, a), _find(par, a + 1)
                if ra != rb:
                    par[ra] = rb
            if y + 1 < k and a + k != p and Dv[t, T[y + 1, x]]:
                ra, rb = _find(par, a), _find(par, a + k)
                if ra != rb:
                    par[ra] = rb
    mask = np.zeros(n, np.int64)                                           # class bits of each root
    size = np.zeros(n, np.int64)
    cnt = np.zeros(8, np.int64)                                            # openings per port, the site aside
    for q in range(8):
        d, i = q // 2, q % 2
        for j in range(hh):
            s = i * hh + j
            if d == 0:
                y, x = 0, s
            elif d == 1:
                y, x = s, k - 1
            elif d == 2:
                y, x = k - 1, s
            else:
                y, x = s, 0
            a = y * k + x
            t = T[y, x]
            if a != p and node[t] and sock[d, t]:
                cnt[q] += 1
                if 0 < lab[q] < OPEN:
                    r = _find(par, a)
                    mask[r] |= 1 << lab[q]
    dmin = np.full(n, 1e9)                                                 # nearest exit distance per root
    for a in range(n):
        if a != p and node[T[a // k, a % k]]:
            r = _find(par, a)
            size[r] += 1
            if mode == 3:
                dmin[r] = min(dmin[r], _exitdist(a // k, a % k, lab, k))
    od = 0.0                                                               # mode 3 orphan cost, the site aside
    cpar = np.arange(MLAB + 1)
    used, oc, on, ncb = 0, 0, 0, 0                                         # ncb: components, the site aside
    sink, anyopen = True, False
    for q in range(8):
        sink &= lab[q] == 0 or lab[q] == OPEN
        anyopen |= lab[q] > 0
    sink &= anyopen
    for a in range(n):
        if a != p and node[T[a // k, a % k]] and par[a] == a:
            ncb += 1
            used |= mask[a]
            if mask[a] == 0:
                oc += 1
                on += size[a]
                od += dmin[a] + 0.5 * size[a]
            else:
                _cunion(cpar, mask[a])
    qs = np.full(2, -1)                                                    # the site's ports (a corner: two)
    ds = np.full(2, -1)
    ns = 0
    for d in range(4):
        on_side = (y0 == 0, x0 == k - 1, y0 == k - 1, x0 == 0)[d]
        if on_side:
            qs[ns] = 2 * d + ((x0 if d % 2 == 0 else y0) // hh)
            ds[ns] = d
            ns += 1
    cp = np.empty(MLAB + 1, np.int64)
    roots = np.empty(4, np.int64)
    S = node.shape[0]
    for t in range(S):
        if not allowed[t]:
            continue
        pm = 0
        add0, add1 = 0, 0
        if node[t]:
            for j in range(ns):
                if sock[ds[j], t]:
                    if j == 0:
                        add0 = 1
                    else:
                        add1 = 1
                    if 0 < lab[qs[j]] < OPEN:
                        pm |= 1 << lab[qs[j]]
        closed, empty = 0.0, 0.0
        for q in range(8):
            c = cnt[q] + (add0 if q == qs[0] else 0) + (add1 if q == qs[1] else 0)
            if lab[q] == 0:
                if c > 0:
                    closed += 1.0 if mode == 0 else c
            elif c == 0:
                empty += 1.0
        moc, mon, mod, M, nct = oc, on, od, pm, ncb
        if node[t]:
            nr = 0
            for d in range(4):
                b = -1
                if d == 0 and y0 > 0 and Dv[T[y0 - 1, x0], t]:
                    b = p - k
                elif d == 1 and x0 + 1 < k and Dh[t, T[y0, x0 + 1]]:
                    b = p + 1
                elif d == 2 and y0 + 1 < k and Dv[t, T[y0 + 1, x0]]:
                    b = p + k
                elif d == 3 and x0 > 0 and Dh[T[y0, x0 - 1], t]:
                    b = p - 1
                if b >= 0:
                    r = _find(par, b)
                    dup = False
                    for j in range(nr):
                        dup |= roots[j] == r
                    if not dup:
                        roots[nr] = r
                        nr += 1
            tot = 1
            nct += 1 - nr
            dm = _exitdist(y0, x0, lab, k) if mode == 3 else 0.0
            for j in range(nr):
                r = roots[j]
                M |= mask[r]
                tot += size[r]
                if mask[r] == 0:
                    moc -= 1
                    mon -= size[r]
                    mod -= dmin[r] + 0.5 * size[r]
                    dm = min(dm, dmin[r])
            if M == 0:
                moc += 1
                mon += tot
                mod += dm + 0.5 * tot
        for l in range(MLAB + 1):
            cp[l] = cpar[l]
        _cunion(cp, M)
        u = used | pm
        merge = 0.0
        for g in range(1, MLAB + 1):
            nc, rs = 0, 0                                                  # distinct roots, as bits
            for l in range(1, MLAB + 1):
                if (u >> l) & 1 and grp[l] == g:
                    r = _find(cp, l)
                    if not (rs >> r) & 1:
                        rs |= 1 << r
                        nc += 1
            if nc > 1:
                merge += 1.0 if mode == 0 else nc - 1
        if sink:
            ex = 0.0 if nct <= 1 else (1.0 if mode == 0 else nct - 1.0)
        else:
            ex = (1.0 if moc > 0 else 0.0) if mode == 0 else (moc if mode == 1 else (mon if mode == 2 else mod))
        res[t] = w[0] * closed + w[1] * empty + w[2] * merge + w[3] * ex


@njit(cache=True)
def site_scores(T, y0, x0, lab, grp, node, Dh, Dv, sock, mode, w, allowed):
    res = np.full(node.shape[0], np.inf)
    _site_scores(T, y0, x0, lab, grp, node, Dh, Dv, sock, mode, w, allowed, np.empty(T.size, np.int64), res)
    return res


# ------------------------------------------------------------ tile chain
# Heat-bath on the block's tiles, one site at a time in a random order:
#   e(t) = E_pair(t, 4 neighbours) - logz[t] + pc[y, x, t] + lam score(block with t)
# E_pair from the tile set (p at T = 1).  Off the block: a closed port's
# edge tiles face WALL, an open port's face nothing (no pair term).
# pc: the parent's term (k, k, S), beta times -log p(t | exemplar tiles within
# r of the site's coordinate in the parent window) (parent_cost).  Sequential, so exact for the
# non-local score at any lam.
@njit(cache=True)
def sweep(T, order, gum, Eh, Ev, logz, allowed, pc, lab, grp, node, Dh, Dv, sock, mode, w, lam, wall):
    k = T.shape[0]
    S = logz.shape[0]
    hh = k // 2
    par = np.empty(T.size, np.int64)
    sc = np.zeros(S)
    e = np.empty(S)
    for o in range(order.shape[0]):
        y, x = order[o] // k, order[o] % k
        if lam > 0.0:
            _site_scores(T, y, x, lab, grp, node, Dh, Dv, sock, mode, w, allowed, par, sc)
        for t in range(S):
            if not allowed[t]:
                e[t] = np.inf
                continue
            v = -logz[t] + pc[y, x, t]
            if x > 0:
                v += Eh[T[y, x - 1], t]
            elif lab[6 + y // hh] == 0:
                v += Eh[wall, t]
            if x + 1 < k:
                v += Eh[t, T[y, x + 1]]
            elif lab[2 + y // hh] == 0:
                v += Eh[t, wall]
            if y > 0:
                v += Ev[T[y - 1, x], t]
            elif lab[x // hh] == 0:
                v += Ev[wall, t]
            if y + 1 < k:
                v += Ev[t, T[y + 1, x]]
            elif lab[4 + x // hh] == 0:
                v += Ev[t, wall]
            if lam > 0.0:
                v += lam * sc[t]
            e[t] = v
        best, bt = -np.inf, 0
        for t in range(S):
            g = -e[t] + gum[o, t]
            if g > best:
                best, bt = g, t
        T[y, x] = bt


def energy(T, lab, Eh, Ev, logz, pc, wall):
    """(E_pair - sum logz, sum pc) of block T under the sweep's boundary."""
    k, hh = T.shape[0], T.shape[0] // 2
    ep = Eh[T[:, :-1], T[:, 1:]].sum() + Ev[T[:-1], T[1:]].sum() - logz[T].sum()
    s = np.arange(k)
    ep += (Ev[wall, T[0]] * (lab[s // hh] == 0)).sum() + (Eh[T[:, -1], wall] * (lab[2 + s // hh] == 0)).sum()
    ep += (Ev[T[-1], wall] * (lab[4 + s // hh] == 0)).sum() + (Eh[wall, T[:, 0]] * (lab[6 + s // hh] == 0)).sum()
    yy, xx = np.mgrid[:k, :k]
    return float(ep), float(pc[yy, xx, T].sum())


def parent_cost(E, S, r, alpha=1.0, floor=0.0):
    """(my m, S) -log p(t | tiles of the torus exemplar E within r of the
    coordinate), smoothed toward E's marginal with weight alpha, then mixed
    with the uniform over S with weight floor (floor 0: inf for a tile E
    never holds)."""
    from castlegen import ports
    with np.errstate(divide="ignore"):
        c = ports.cost_table(E.ravel(), S, E.shape[0], E.shape[1], r, alpha)
        return c if floor == 0 else -np.log((1 - floor) * np.exp(-c) + floor / S)


# ------------------------------------------------------------ a map of blocks
# The h = k level of generic's connectivity promise hands each k x k block
# its open sides O and keys; a side is downhill when open into a lower key.
# Block pattern: an open side's two halves are class 1 (downhill: everything
# below already reaches the root, so its openings may be joined outside) or
# OPEN (uphill); the root, with no downhill side, is a sink.  Seam rule
# between blocks: a tile pair across a block seam where either tile is an
# opening must join.  Then a map scoring 0 everywhere is one component:
# by induction along the descent, every node reaches a downhill opening
# inside its block, which crosses into a block already joined to the root.
def block_labels(O, low):
    """(R, C, 8) patterns from open sides O (R, C) and downhill bits low (R, C, 4)."""
    lab = np.zeros(O.shape + (8,), np.int64)
    for d in range(4):
        v = np.where(low[..., d], 1, OPEN) * ((O >> d) & 1)
        lab[..., 2 * d] = lab[..., 2 * d + 1] = v
    return lab


@njit(cache=True, inline="always")
def _seam_bad(a, b, d, node, Dh, Dv, sock):
    """Tile a with b on its side d: either faces it with an opening, unjoined."""
    o = (d + 2) % 4
    op = (node[a] and sock[d, a]) or (node[b] and sock[o, b])
    if d == 0:
        j = Dv[b, a]
    elif d == 1:
        j = Dh[a, b]
    elif d == 2:
        j = Dv[a, b]
    else:
        j = Dh[b, a]
    return op and not j


@njit(cache=True, inline="always")
def _obj_broken(O, nbr, y, x, o):
    """Object rule checks broken across site (y, x) with o[y, x] = o (objects.py;
    off the map: o = -1).  nbr's last row is -1 (the row for o = -1)."""
    H, W = O.shape
    n = 0
    for d in range(4):
        ny, nx = y + (-1, 0, 1, 0)[d], x + (0, 1, 0, -1)[d]
        q = O[ny, nx] if 0 <= ny < H and 0 <= nx < W else -1
        w = nbr[o, d]
        if w >= 0 and q != w:
            n += 1
        w = nbr[q, (d + 2) % 4]
        if w >= 0 and o != w:
            n += 1
    return n


@njit(cache=True, inline="always")
def _map_tile_e(T, y, x, t, k, Eh, Ev, logz, pc, node, Dh, Dv, sock, sc, ws, lam, wall):
    H, W = T.shape
    v = -logz[t] + pc[y, x, t]
    v += Eh[T[y, x - 1], t] if x > 0 else Eh[wall, t]
    v += Eh[t, T[y, x + 1]] if x + 1 < W else Eh[t, wall]
    v += Ev[T[y - 1, x], t] if y > 0 else Ev[wall, t]
    v += Ev[t, T[y + 1, x]] if y + 1 < H else Ev[t, wall]
    if lam > 0.0:
        v += lam * sc[t]
        if y % k == 0 and y > 0 and _seam_bad(t, T[y - 1, x], 0, node, Dh, Dv, sock):
            v += lam * ws
        if x % k == k - 1 and x + 1 < W and _seam_bad(t, T[y, x + 1], 1, node, Dh, Dv, sock):
            v += lam * ws
        if y % k == k - 1 and y + 1 < H and _seam_bad(t, T[y + 1, x], 2, node, Dh, Dv, sock):
            v += lam * ws
        if x % k == 0 and x > 0 and _seam_bad(t, T[y, x - 1], 3, node, Dh, Dv, sock):
            v += lam * ws
    return v


@njit(cache=True)
def map_sweep(T, k, order, gum, Eh, Ev, logz, allowed, pc, labs, grp, node, Dh, Dv, sock, mode, w, ws, lam, wall,
              O, tmpl, nbr, soft, lo):
    """Heat-bath on map T (H, W) of k x k blocks with patterns labs (H/k,
    W/k, 8): sweep's energy (off the map: WALL), lam times the site's block
    score (site_scores) and lam ws per seam rule broken by the site (ws below
    an orphan node's weight, so a sliver orphaned across a seam can retreat
    before its neighbour closes the seam).

    Objects (castlegen/objects.py): the site's (tile, object code) pair is
    drawn jointly, O (H, W) the codes, tmpl / nbr the tables, plus lo per
    object rule check broken.  Candidates: (t, -1) for each allowed tile t
    with soft[t], and (tmpl[c], c) for c the site's own code and each code its
    neighbours imply (at most 5; gum (n, S + 5): their Gumbel noise after the
    tiles').  No objects: O all -1, nbr = [[-1] * 4], soft = allowed."""
    H, W = T.shape
    S = logz.shape[0]
    par = np.empty(k * k, np.int64)
    sc = np.zeros(S)
    cs = np.empty(5, np.int64)
    for o in range(order.shape[0]):
        y, x = order[o] // W, order[o] % W
        by, bx = y // k, x // k
        if lam > 0.0:
            B = T[by * k:(by + 1) * k, bx * k:(bx + 1) * k]
            _site_scores(B, y - by * k, x - bx * k, labs[by, bx], grp, node, Dh, Dv, sock, mode, w, allowed, par, sc)
        best, bt, bc = -np.inf, 0, -1
        nb0 = lo * _obj_broken(O, nbr, y, x, -1)
        for t in range(S):
            if not (allowed[t] and soft[t]):
                continue
            g = -(_map_tile_e(T, y, x, t, k, Eh, Ev, logz, pc, node, Dh, Dv, sock, sc, ws, lam, wall) + nb0) + gum[o, t]
            if g > best:
                best, bt, bc = g, t, -1
        nc = 0                                                     # the codes on offer
        if O[y, x] >= 0:
            cs[0] = O[y, x]
            nc = 1
        for d in range(4):
            ny, nx = y + (-1, 0, 1, 0)[d], x + (0, 1, 0, -1)[d]
            if 0 <= ny < H and 0 <= nx < W and O[ny, nx] >= 0:
                c = nbr[O[ny, nx], (d + 2) % 4]
                new = c >= 0
                for j in range(nc):
                    if cs[j] == c:
                        new = False
                if new:
                    cs[nc] = c
                    nc += 1
        for j in range(nc):
            t = tmpl[cs[j]]
            g = -(_map_tile_e(T, y, x, t, k, Eh, Ev, logz, pc, node, Dh, Dv, sock, sc, ws, lam, wall)
                  + lo * _obj_broken(O, nbr, y, x, cs[j])) + gum[o, S + j]
            if g > best:
                best, bt, bc = g, t, cs[j]
        T[y, x] = bt
        O[y, x] = bc


def map_report(T, k, labs, grp, tabs):
    """{blocks: blocks breaking a rule, parts: (4,) per-part block counts,
    seams: broken seam pairs} (binary scores)."""
    node, Dh, Dv, sock = tabs
    R, C = labs.shape[:2]
    P = np.array([[parts(np.ascontiguousarray(T[by * k:(by + 1) * k, bx * k:(bx + 1) * k]), labs[by, bx], grp,
                         *tabs, 0) for bx in range(C)] for by in range(R)])
    op = lambda t, d: node[t] & sock[d, t]
    a, b = T[:, k - 1:-1:k], T[:, k::k]                                    # across vertical seams
    sh = ((op(a, 1) | op(b, 3)) & ~Dh[a, b]).sum()
    a, b = T[k - 1:-1:k], T[k::k]
    sv = ((op(a, 2) | op(b, 0)) & ~Dv[a, b]).sum()
    return dict(blocks=int((P.sum(-1) > 0).sum()), parts=(P > 0).sum((0, 1)), seams=int(sh + sv))
