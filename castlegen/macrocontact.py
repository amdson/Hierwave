"""Macro objects with CONTACT connectivity: rooms are shapes, doors are left
to rendering.

Rooms are the FULL variants of a macroobj tile set (every rotation, every
door kept; macroobj.load): the door letters only say which sides may hold
doors.  A SLOT is a straight wall cell (outside beyond it, floor inside) on
such a side.  Two rooms are in CONTACT when at least minc of their slot cells
face each other across a flush wall (any door could go there); a room is in
contact with a region exit (macroobj.State.region) when minc of its slots
face the open half-side.  The macro level samples rooms only; the contact
graph is its connectivity graph; rendering (doors) chooses a connected
subgraph of it and cuts one door per chosen contact.

Energy: E = sum_rooms mu[fam(k)] - lam * #contacts (+ nu per block off a
parent labelling) (+ the connectivity heuristic).  Footprints never overlap.

Block update: an exact heat-bath over {empty} + every (k, centre phase) of
the block (all FULL variants x BS^2 phases), b's room lifted.

Connectivity heuristic (cn = conn_state(...)), as macroobj's on the contact
graph: per sweep the components and a random spanning forest (random-order
Kruskal); each forest edge PROTECTS the contact: while the partner block is
occupied (a virtual partner: always), a candidate for b must keep >= minc
contact with it, and b may not go empty while it holds two such
protections (a protection to an exit counts two).  Candidates pay -beta per
extra component they join and per first contact with an exit half-side,
+gamma when they join none, +gsmall when all they join have fewer than smin
rooms (an exit opening excepted).  Pure component count: gamma = beta,
gsmall = 0.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from numba import njit

from castlegen import macroobj as MO
from castlegen.macroobj import BS, TM, DY, DX, VOID, WALL, FLOOR, PORT, VBIG

Ct = namedtuple("Ct", "KH KW CY CX KC SLOT SN SY SX SD FULLK")
St = MO.St


def tables(C):
    """Contact tables for tile set C (macroobj.load)."""
    K = C.K
    SLOT = np.full((K, TM, TM), -1, np.int64)
    S = []
    for k in range(K):
        h, w = C.KH[k], C.KW[k]
        kc = C.KC[k]
        sides = set(C.PD[k, :C.PN[k]].tolist())
        sl = []
        for y in range(h):
            for x in range(w):
                if kc[y, x] not in (WALL, PORT):
                    continue
                for d in sides:
                    oy, ox, iy, ix = y + DY[d], x + DX[d], y - DY[d], x - DX[d]
                    out = not (0 <= oy < h and 0 <= ox < w) or kc[oy, ox] == VOID
                    if out and 0 <= iy < h and 0 <= ix < w and kc[iy, ix] == FLOOR:
                        SLOT[k, y, x] = d
                        sl.append((y, x, d))
        S.append(sl)
    MS = max(len(s) for s in S)
    SN = np.array([len(s) for s in S], np.int64)
    SY, SX, SD = (np.zeros((K, MS), np.int64) for _ in range(3))
    for k, sl in enumerate(S):
        for i, (y, x, d) in enumerate(sl):
            SY[k, i], SX[k, i], SD[k, i] = y, x, d
    return Ct(C.KH, C.KW, C.CY, C.CX, C.KC, SLOT, SN, SY, SX, SD, np.nonzero(C.FULL)[0].astype(np.int64))


def fullof(C):
    """(K,) the FULL variant with the footprint of each variant (its doors
    reopened)."""
    out = np.empty(C.K, np.int64)
    for k in range(C.K):
        for k2 in np.nonzero((C.fam == C.fam[k]) & C.FULL)[0]:
            if C.KH[k2] == C.KH[k] and C.KW[k2] == C.KW[k]:
                a2, a1 = C.KC[k2], C.KC[k]
                if ((a2 == a1) | ((a2 == PORT) & (a1 == WALL))).all():
                    out[k] = k2
                    break
    return out


def to_full(S, C, fo=None):
    """Replace every label of S by its FULL variant (same footprint)."""
    fo = fullof(C) if fo is None else fo
    on = S.lab >= 0
    S.lab[on] = fo[S.lab[on]]


# ---------------------------------------------------------------- primitives
@njit(cache=True)
def _anchor(o, ct, st):
    k = st.lab[o]
    return (o // st.BX) * BS + st.phy[o] - ct.CY[k], (o % st.BX) * BS + st.phx[o] - ct.CX[k]


@njit(cache=True)
def _fits(k, ay, ax, ct, st):
    H, W = st.BY * BS, st.BX * BS
    for y in range(ct.KH[k]):
        row = ((ay + y) % H) * W
        for x in range(ct.KW[k]):
            if ct.KC[k, y, x] >= 0 and st.occ[row + (ax + x) % W] >= 0:
                return False
    return True


@njit(cache=True)
def _paint(k, ay, ax, val, ct, st):
    H, W = st.BY * BS, st.BX * BS
    for y in range(ct.KH[k]):
        row = ((ay + y) % H) * W
        for x in range(ct.KW[k]):
            if ct.KC[k, y, x] >= 0:
                st.occ[row + (ax + x) % W] = val


@njit(cache=True)
def _set(b, k, py, px, ct, st):
    st.lab[b], st.phy[b], st.phx[b] = k, py, px
    if k >= 0:
        ay, ax = _anchor(b, ct, st)
        _paint(k, ay, ax, b, ct, st)


@njit(cache=True)
def _lift(b, ct, st):
    k = st.lab[b]
    if k >= 0:
        ay, ax = _anchor(b, ct, st)
        _paint(k, ay, ax, -1, ct, st)
        st.lab[b] = -1
    return k


@njit(cache=True)
def _contacts(k, ay, ax, ct, st, nb, cnt):
    """Contacts of k at (ay, ax) (not painted, or painted as its own block):
    -> n; nb[:n] the partner nodes (a block, or NB + v for exit half-side
    v - 1), cnt[:n] the facing slot cells."""
    H, W = st.BY * BS, st.BX * BS
    NB = st.lab.shape[0]
    n = 0
    for i in range(ct.SN[k]):
        d = ct.SD[k, i]
        gy, gx = (ay + ct.SY[k, i] + DY[d]) % H, (ax + ct.SX[k, i] + DX[d]) % W
        o = st.occ[gy * W + gx]
        if o < 0:
            continue
        if o == NB:
            v = st.ext[gy * W + gx]
            if v <= 0 or (v - 1) // 2 != d:
                continue
            node = NB + v
        else:
            k2 = st.lab[o]
            if k2 < 0:
                continue
            oy, ox = _anchor(o, ct, st)
            ly, lx = (gy - oy) % H, (gx - ox) % W
            if ly >= ct.KH[k2] or lx >= ct.KW[k2] or ct.SLOT[k2, ly, lx] != (d + 2) % 4:
                continue
            node = o
        j = 0
        while j < n and nb[j] != node:
            j += 1
        if j == n:
            nb[n], cnt[n] = node, 0
            n += 1
        cnt[j] += 1
    return n


# ---------------------------------------------------------------- connectivity heuristic
Cn = namedtuple("Cn", "on beta gamma gsmall smin minc comp cu csz nid npr prt cb vcls vroot hasopen")


def conn_state(NB, beta=0.0, gamma=0.0, gsmall=0.0, smin=0, minc=3, on=True, vcls=None, root=None, maxp=32):
    """vcls (8,): the class of each exit half-side (0: closed), exits of one
    class joined outside; root: the class whose exits are roots (default
    every class); smin = VBIG puts gsmall on joining no rooted structure."""
    i = lambda *sh: np.zeros(sh, np.int64)
    vc = i(8) if vcls is None else np.asarray(vcls, np.int64)
    vr = (vc > 0) if root is None else (vc == root)
    T = NB + 9 + NB + 1
    return Cn(on, float(beta), float(gamma), float(gsmall), np.int64(smin), np.int64(minc), i(NB),
              np.arange(T, dtype=np.int64), i(T), i(1), i(NB), i(NB, maxp), i(maxp), vc, vr.astype(np.bool_), i(8))


@njit(cache=True)
def _find(cu, a):
    while cu[a] != a:
        cu[a] = cu[cu[a]]
        a = cu[a]
    return a


@njit(cache=True)
def _snapshot(ct, st, cn):
    NB = st.lab.shape[0]
    E = NB * 32
    eo = np.empty(E, np.int64)
    eq = np.empty(E, np.int64)
    nb = np.empty(64, np.int64)
    cnt = np.empty(64, np.int64)
    ne = 0
    for o in range(cn.cu.shape[0]):
        cn.cu[o] = o
    for v in range(8):
        cn.hasopen[v] = 0
    for o in range(NB):
        cn.npr[o] = 0
        k = st.lab[o]
        if k < 0:
            continue
        ay, ax = _anchor(o, ct, st)
        n = _contacts(k, ay, ax, ct, st, nb, cnt)
        for i in range(n):
            if cnt[i] < cn.minc:
                continue
            v = nb[i]
            if v > NB:
                cn.hasopen[v - NB - 1] = 1
            if v > o and ne < E:                          # a block pair once, an exit from its room
                eo[ne], eq[ne] = o, v
                ne += 1
    for i in range(ne):
        ra, rb = _find(cn.cu, eo[i]), _find(cn.cu, eq[i])
        if ra != rb:
            cn.cu[ra] = rb
    for q in range(8):
        for q2 in range(q):
            if cn.vcls[q] > 0 and cn.vcls[q] == cn.vcls[q2]:
                ra, rb = _find(cn.cu, NB + 1 + q), _find(cn.cu, NB + 1 + q2)
                if ra != rb:
                    cn.cu[ra] = rb
    for o in range(cn.csz.shape[0]):
        cn.csz[o] = 0
    for q in range(8):
        if cn.vroot[q]:
            cn.csz[_find(cn.cu, NB + 1 + q)] = VBIG
    for o in range(NB):
        cn.comp[o] = _find(cn.cu, o)
        if st.lab[o] >= 0:
            cn.csz[cn.comp[o]] += 1
    tree = np.arange(NB + 9)
    for i in np.random.permutation(ne):
        ra, rb = _find(tree, eo[i]), _find(tree, eq[i])
        if ra == rb:
            continue
        tree[ra] = rb
        o, q = eo[i], eq[i]
        if cn.npr[o] < cn.prt.shape[1]:
            cn.prt[o, cn.npr[o]] = q
            cn.npr[o] += 1
        if q < NB and cn.npr[q] < cn.prt.shape[1]:
            cn.prt[q, cn.npr[q]] = o
            cn.npr[q] += 1
    cn.nid[0] = NB + 9


@njit(cache=True)
def _nlive(b, st, cn):
    """Live protections of b (an exit counts 2: pins the room)."""
    NB = st.lab.shape[0]
    n = 0
    for i in range(cn.npr[b]):
        p = cn.prt[b, i]
        if p > NB:
            n += 2
        elif st.lab[p] >= 0:
            n += 1
    return n


@njit(cache=True)
def _cand_e(b, k, py, px, lam, mu, fam, pa, nu, cn, ct, st, nb, cnt):
    """Energy of k at centre phase (py, px) in block b (lifted)."""
    ay = (b // st.BX) * BS + py - ct.CY[k]
    ax = (b % st.BX) * BS + px - ct.CX[k]
    if not _fits(k, ay, ax, ct, st):
        return np.inf
    NB = st.lab.shape[0]
    n = _contacts(k, ay, ax, ct, st, nb, cnt)
    if cn.on:
        for i in range(cn.npr[b]):
            p = cn.prt[b, i]
            if p < NB and st.lab[p] < 0:
                continue
            ok = False
            for j in range(n):
                if nb[j] == p and cnt[j] >= cn.minc:
                    ok = True
            if not ok:
                return np.inf
    m, nc, big, nopen = 0, 0, 0, 0
    for j in range(n):
        if cnt[j] < cn.minc:
            continue
        m += 1
        if cn.on:
            v = nb[j]
            if v > NB and cn.hasopen[v - NB - 1] == 0:
                nopen += 1
            c = _find(cn.cu, v if v > NB else cn.comp[v])
            new = True
            for i in range(nc):
                if cn.cb[i] == c:
                    new = False
            if new:
                cn.cb[nc] = c
                nc += 1
                big += cn.csz[c]
    e = mu[fam[k]] - lam * m
    if nu != 0 and not (pa[0][b] == k and pa[1][b] == py and pa[2][b] == px):
        e += nu
    if cn.on:
        e += (cn.gamma if nc == 0 else -cn.beta * (nc - 1)) - cn.beta * nopen
        if big < cn.smin and nopen == 0:
            e += cn.gsmall
    return e


@njit(cache=True)
def _join(b, ct, st, cn, nb, cnt):
    k = st.lab[b]
    if k < 0:
        return
    ay, ax = _anchor(b, ct, st)
    NB = st.lab.shape[0]
    n = _contacts(k, ay, ax, ct, st, nb, cnt)
    r = -1
    for j in range(n):
        if cnt[j] < cn.minc:
            continue
        v = nb[j]
        if v > NB:
            cn.hasopen[v - NB - 1] = 1
        c = _find(cn.cu, v if v > NB else cn.comp[v])
        if r < 0:
            r = c
        elif c != r:
            cn.cu[c] = r
            cn.csz[r] += cn.csz[c]
    if r < 0:
        r = cn.nid[0]
        cn.nid[0] = min(cn.nid[0] + 1, cn.cu.shape[0] - 1)
        cn.cu[r] = r
        cn.csz[r] = 1
    cn.comp[b] = r


@njit(cache=True)
def _site(b, lam, T, mu, fam, pa, nu, cn, ct, st, ce, nb, cnt):
    """Exact heat-bath of block b over {empty} + FULLK x BS^2 phases."""
    xk, xy, xx = st.lab[b], st.phy[b], st.phx[b]
    _lift(b, ct, st)
    e0 = nu if nu != 0 and pa[0][b] >= 0 else 0.0
    if cn.on and _nlive(b, st, cn) >= 2:
        e0 = np.inf
    NK = ct.FULLK.shape[0]
    emin = e0
    for i in range(NK):
        k = ct.FULLK[i]
        for p in range(BS * BS):
            e = _cand_e(b, k, p // BS, p % BS, lam, mu, fam, pa, nu, cn, ct, st, nb, cnt)
            ce[i * BS * BS + p] = e
            if e < emin:
                emin = e
    if emin == np.inf:                                    # nothing allowed: keep the current state
        _set(b, xk, xy, xx, ct, st)
        return
    tot = np.exp(-(e0 - emin) / T) if e0 < np.inf else 0.0
    for i in range(NK * BS * BS):
        if ce[i] < np.inf:
            tot += np.exp(-(ce[i] - emin) / T)
    r = np.random.random() * tot
    acc = np.exp(-(e0 - emin) / T) if e0 < np.inf else 0.0
    if r < acc:
        if cn.on:
            cn.comp[b] = -1
        return
    for i in range(NK * BS * BS):
        if ce[i] < np.inf:
            acc += np.exp(-(ce[i] - emin) / T)
            if r < acc or i == NK * BS * BS - 1:
                break
    # (the last finite candidate if rounding ran past)
    while ce[i] == np.inf:
        i -= 1
    _set(b, ct.FULLK[i // (BS * BS)], (i % (BS * BS)) // BS, i % BS, ct, st)
    if cn.on:
        _join(b, ct, st, cn, nb, cnt)


@njit(cache=True)
def _sweeps(lams, Ts, mu, fam, pa, nu, cn, ct, st, seed):
    np.random.seed(seed)
    NB = st.lab.shape[0]
    ce = np.empty(ct.FULLK.shape[0] * BS * BS, np.float64)
    nb = np.empty(64, np.int64)
    cnt = np.empty(64, np.int64)
    for s in range(lams.shape[0]):
        if cn.on:
            _snapshot(ct, st, cn)
        for b in np.random.permutation(NB):
            if st.occ.shape[0] > 0 and st.ext.shape[0] > 1 and _outside(b, st):
                continue
            _site(b, lams[s], Ts[s], mu, fam, pa, nu, cn, ct, st, ce, nb, cnt)


@njit(cache=True)
def _outside(b, st):
    """Block b's cells are all outside the region (nothing can go there)."""
    NB = st.lab.shape[0]
    W = st.BX * BS
    y0, x0 = (b // st.BX) * BS, (b % st.BX) * BS
    for y in range(BS):
        for x in range(BS):
            if st.occ[(y0 + y) * W + x0 + x] != NB:
                return False
    return True


def sweeps(lams, Ts, mu, fam, ct, st, seed, pa=None, nu=0.0, cn=None):
    NB = st.lab.shape[0]
    if pa is None:
        pa = (np.full(NB, -1, np.int64), np.zeros(NB, np.int64), np.zeros(NB, np.int64))
    if cn is None:
        cn = conn_state(1, on=False)
    _sweeps(np.asarray(lams, np.float64), np.asarray(Ts, np.float64), mu, fam, pa, float(nu), cn, ct, st, seed)


# ---------------------------------------------------------------- analysis and rendering
def contact_graph(ct, S, minc=3):
    """-> list of (o, node, cells) contacts with >= minc facing slot cells
    (node: a block, or NB + v for exit half-side v - 1)."""
    NB = S.lab.size
    nb, cnt = np.empty(64, np.int64), np.empty(64, np.int64)
    out = []
    for o in np.nonzero(S.lab >= 0)[0]:
        k = S.lab[o]
        ay, ax = _anchor(o, ct, S.st)
        n = _contacts(k, ay, ax, ct, S.st, nb, cnt)
        for i in range(n):
            if cnt[i] >= minc and (nb[i] > o):
                out.append((int(o), int(nb[i]), int(cnt[i])))
    return out


def contact_cells(ct, S, o, node):
    """Slot cells of o facing node: [(y, x, d)]."""
    H, W = S.BY * BS, S.BX * BS
    NB = S.lab.size
    k = S.lab[o]
    ay, ax = _anchor(o, ct, S.st)
    out = []
    for i in range(ct.SN[k]):
        d = ct.SD[k, i]
        y, x = (ay + ct.SY[k, i]) % H, (ax + ct.SX[k, i]) % W
        gy, gx = (y + DY[d]) % H, (x + DX[d]) % W
        q = S.occ[gy * W + gx]
        if node > NB:
            ok = q == NB and S.ext[gy * W + gx] == node - NB and (node - NB - 1) // 2 == d
        else:
            if q != node:
                continue
            k2 = S.lab[q]
            oy, ox = _anchor(q, ct, S.st)
            ly, lx = (gy - oy) % H, (gx - ox) % W
            ok = ly < ct.KH[k2] and lx < ct.KW[k2] and ct.SLOT[k2, ly, lx] == (d + 2) % 4
        if ok:
            out.append((y, x, d))
    return out


def doors(ct, S, rng, vcls=None, minc=3, pextra=0.15):
    """Render-time doors: a random spanning forest of the contact graph (exit
    classes joined outside) plus each other contact w.p. pextra; one door per
    chosen contact at a random facing cell pair -> [(y, x, d)] door cells on
    the room side (the partner's cell is the one beyond)."""
    NB = S.lab.size
    E = contact_graph(ct, S, minc)
    par = list(range(NB + 9))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    if vcls is not None:
        for q in range(8):
            for q2 in range(q):
                if vcls[q] > 0 and vcls[q] == vcls[q2]:
                    par[find(NB + 1 + q)] = find(NB + 1 + q2)
    out = []
    for i in rng.permutation(len(E)):
        o, v, _ = E[i]
        ra, rb = find(o), find(v)
        tree = ra != rb
        if tree:
            par[ra] = rb
        if tree or rng.random() < pextra:
            cells = contact_cells(ct, S, o, v)
            out.append((cells[rng.integers(len(cells))], v > NB))
    return out


def check(ct, S, vcls, minc=3):
    """-> (ok, components of rooms + exit classes, open half-sides without a
    contact) under the contact graph."""
    NB = S.lab.size
    par = list(range(NB + 9))
    def find(a):
        while par[a] != a:
            a = par[a]
        return a
    for q in range(8):
        for q2 in range(q):
            if vcls[q] > 0 and vcls[q] == vcls[q2]:
                par[find(NB + 1 + q)] = find(NB + 1 + q2)
    opened = np.zeros(8, bool)
    for o, v, _ in contact_graph(ct, S, minc):
        if v > NB:
            opened[v - NB - 1] = True
        ra, rb = find(o), find(v)
        if ra != rb:
            par[ra] = rb
    rooms = np.nonzero(S.lab >= 0)[0]
    nodes = list(rooms) + [NB + 1 + q for q in range(8) if vcls[q] > 0]
    comps = len({find(a) for a in nodes})
    missing = int(sum(vcls[q] > 0 and not opened[q] for q in range(8)))
    return comps == 1 and missing == 0 and len(rooms) > 0, comps, missing


def render(C, ct, S, doorlist, y0=0, x0=0, h=None, w=None, px=1):
    """Rooms (walls, floors coloured by family) with the given doors cut."""
    H, W = S.BY * BS, S.BX * BS
    h, w = h or H, w or W
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = MO.PAL["bg"]
    for o in np.nonzero(S.lab >= 0)[0]:
        k = S.lab[o]
        ay, ax = _anchor(o, ct, S.st)
        kc = C.KC[k, :C.KH[k], :C.KW[k]]
        yy, xx = np.nonzero(kc != VOID)
        fc = MO.ROOMC[C.fam[k] % len(MO.ROOMC)]
        col = np.where((kc[yy, xx] == FLOOR)[:, None], np.array(fc, np.uint8), np.array(MO.PAL["wall"], np.uint8))
        img[(ay + yy) % H, (ax + xx) % W] = col
    for (y, x, d), ext in doorlist:
        img[y, x] = MO.PAL["door"]
        if not ext:
            img[(y + DY[d]) % H, (x + DX[d]) % W] = MO.PAL["door"]
    img = np.roll(img, (-y0, -x0), (0, 1))[:h, :w]
    return np.repeat(np.repeat(img, px, 0), px, 1) if px > 1 else img
