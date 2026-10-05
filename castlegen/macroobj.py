"""Macro objects on a block lattice: rigid components (rooms, corridors) with
lettered ports, sampled at the block level with hard attachment rules.

Components (tilesets/macro_rooms.json): a wall/floor/void template of at
most TM x TM cells; a port is a doorway cell on the footprint's edge facing
out on its one open side d.  Variants: every rotation, and (unless "seal":
false) every non-empty subset of the ports kept, the rest walled up (seal[k,
j]: k with port j walled up).  Port P of object a and port Q of object b are
ATTACHED when Q is the cell P faces, Q faces back, and {letter P, letter Q}
is in compat.  The relation is rigid and sparse: a port of a placed object
admits exactly one placement per compatible (component, port) pair, listed
in the ATTACHMENT TABLE att[(k2, j2)] = [(k, j), ...].

State: a lattice of BY x BX blocks of BS x BS cells (torus).  Block b holds
lab[b] = -1 (empty) or a component k whose centre cell sits at phase
(phy[b], phx[b]) in [0, BS)^2 inside the block; the footprint (template cells
other than void) runs into neighbouring blocks.  occ (H x W cells) holds the
owning block or -1.  Footprints never overlap (hard, always).  BS = 8 is at
most the separation of attached centres (every component spans >= 8 cells
along the axis of each of its ports), so attached objects never compete for
one block; touching objects' centres are within RAD = 3 blocks.

Energy: E = lam * #unmatched ports + sum_objects mu[grp(k)] (+ nu per block
off a parent labelling).  With every other block fixed, placing k in block b
with m of its P_k ports attached changes E by lam (P_k - 2 m) + mu (each
attachment also matches a neighbour port that was unmatched); a neighbour
port facing one of our wall cells is unmatched either way.  grp is the
caller's grouping of components (the experiment: family x sealed ports).

Block update (each kernel leaves exp(-E / T) invariant):
  A  heat-bath over the set {empty} + every state attached to a neighbour by
     the attachment table (built from neighbours only, b's object lifted);
     skipped when the current state is not in the set.
  B  independence Metropolis-Hastings: propose empty (1/2) or a uniform
     (k, phase), accept min(1, pi(y) q(x) / (pi(x) q(y))).
Cost: tens of candidates, each a footprint overlap scan and P_k port checks.

Connectivity heuristic (cn = conn_state(...), not exact): before each
sweep, the components of the attachment graph and a random spanning forest of
each (random-order Kruskal).  Each forest edge protects its two door cells
for the sweep: a candidate for block b must keep a door at each of b's
protected cells whose partner still faces back, and b may not go empty while
two such doors hold (a leaf may retract).  So no move splits a structure.
Candidates also pay -beta per extra structure they join (union-find merged
during the sweep), +gamma when they join none, +gsmall when all they join
have fewer than smin rooms.

Regions (State.region): objects confined to a rectangle; the cells around it
are OUTSIDE (occ = NB).  Each of its 8 half-sides is closed or a class; a door
facing out across an open half-side q is attached to the virtual node
NB + 1 + q (an exit), across a closed one it is unmatched.  In the heuristic,
exits of one class are joined (outside), the exits of the root class are
roots (size VBIG: with smin = VBIG, gsmall falls on joining no rooted
structure), the first opening of a half-side earns beta and never counts as
an orphan, and a room holding a protected exit door is pinned (counts as two
protected doors).

Coarse level: a coordinate U per R x R window of blocks into an exemplar
block lattice (exemplar windows hold whole exemplar labels and phases).
paste(U) copies the window's labels in raster order and drops an object that
overlaps one already there.  The coarse update is a heat-bath over
candidates (the neighbours' coherent shifts, random coordinates, the current
one) on the energy of the pasted state, measured on the window plus a halo
of RAD blocks (an object anchored further away cannot reach the window).
Not exact: the candidate set includes the current coordinate.

Exemplar: kinetic growth.  From one hall, repeatedly pick a random placed
object and port, a random compatible (component, port) by family weight
(full variants only), and place it if its centre's block is free, it fits,
and it blocks no port (every port touching it is attached); then close: a
dangling port gets a one-port component where one fits, else is walled up
(an object whose last port dangles is removed), repeated.
"""
from __future__ import annotations

import json
import os
from collections import namedtuple

import numpy as np
from numba import njit

BS = 8        # block (lattice) size in cells
TM = 16       # largest template side
RAD = 3       # block radius of interaction: centres of touching objects differ by <= TM + 1 cells
DY = np.array([-1, 0, 1, 0], np.int64)
DX = np.array([0, 1, 0, -1], np.int64)
VOID, WALL, FLOOR, PORT = -1, 0, 1, 2
Tabs = namedtuple("Tabs", "KH KW CY CX KC PN PY PX PD PL PCELL compat ATTP ATTK ATTJ MAXP NF FY FX FD DLP DLK DLJ")
St = namedtuple("St", "lab phy phx occ BY BX ext exy exx exd")


class Comps:
    pass


def load(name="macro_rooms"):
    path = os.path.join(os.path.dirname(__file__), "tilesets", name + ".json")
    spec = json.load(open(path))
    letters = sorted({ch for c in spec["components"].values() for r in c["rows"] for ch in r if ch.isalpha()})
    L = len(letters)
    compat = np.zeros((L, L), np.bool_)
    for a, b in spec["compat"]:
        compat[letters.index(a), letters.index(b)] = compat[letters.index(b), letters.index(a)] = True
    grids, pids, fam, rot, mask, full = [], [], [], [], [], []
    fams = list(spec["components"])
    index = {}                                     # (family, rotation, port subset) -> k
    for f, (fname, c) in enumerate(spec["components"].items()):
        g0 = np.array([list(r) for r in c["rows"]])
        ports = list(zip(*np.nonzero(np.char.isalpha(g0))))
        pid0 = np.full(g0.shape, -1, np.int64)
        for p, (y, x) in enumerate(ports):
            pid0[y, x] = p
        allm = (1 << len(ports)) - 1
        for m in (range(1, allm + 1) if c.get("seal", True) else [allm]):
            g = g0.copy()
            for p, (y, x) in enumerate(ports):
                if not m >> p & 1:
                    g[y, x] = "#"
            for r in range(4):
                gr = np.rot90(g, -r)
                same = [k for k in range(len(grids)) if fam[k] == f and grids[k].shape == gr.shape
                        and (grids[k] == gr).all()]
                if same:
                    index[f, r, m] = same[0]
                    continue
                index[f, r, m] = len(grids)
                grids.append(gr)
                pids.append(np.rot90(pid0, -r))
                fam.append(f)
                rot.append(r)
                mask.append(m)
                full.append(m == allm)
    K = len(grids)
    MAXP = max(int(np.char.isalpha(g).sum()) for g in grids)
    C = Comps()
    C.letters, C.fams, C.K, C.MAXP = letters, fams, K, MAXP
    C.kind = [spec["components"][n]["kind"] for n in fams]
    C.weight = np.array([spec["components"][n].get("weight", 1.0) for n in fams])
    C.fam, C.rot, C.grids = np.array(fam, np.int64), np.array(rot, np.int64), grids
    KH = np.array([g.shape[0] for g in grids], np.int64)
    KW = np.array([g.shape[1] for g in grids], np.int64)
    assert KH.max() <= TM and KW.max() <= TM
    KC = np.full((K, TM, TM), VOID, np.int8)
    PCELL = np.full((K, TM, TM), -1, np.int64)
    PN = np.zeros(K, np.int64)
    PY, PX, PD, PL = (np.zeros((K, MAXP), np.int64) for _ in range(4))
    faces = []
    for k, g in enumerate(grids):
        h, w = g.shape
        inside = lambda y, x: 0 <= y < h and 0 <= x < w and g[y, x] != " "
        fk = []
        for y in range(h):
            for x in range(w):
                ch = g[y, x]
                if ch == " ":
                    continue
                out = [d for d in range(4) if not inside(y + DY[d], x + DX[d])]
                fk += [(y, x, d) for d in out]
                if ch.isalpha():
                    assert len(out) == 1, (fams[fam[k]], y, x)
                    j = PN[k]
                    PY[k, j], PX[k, j], PD[k, j], PL[k, j] = y, x, out[0], letters.index(ch)
                    PCELL[k, y, x] = j
                    PN[k] += 1
                    KC[k, y, x] = PORT
                else:
                    KC[k, y, x] = FLOOR if ch == "." else WALL
        faces.append(fk)
    SEAL = np.full((K, MAXP), -1, np.int64)        # k with its port j walled up (-1: it was the last)
    for k in range(K):
        for j in range(PN[k]):
            m = mask[k] & ~(1 << int(pids[k][PY[k, j], PX[k, j]]))
            if m:
                SEAL[k, j] = index[fam[k], rot[k], m]
    C.SEAL, C.FULL = SEAL, np.array(full, np.bool_)
    MAXF = max(len(f) for f in faces)
    NF = np.array([len(f) for f in faces], np.int64)
    FY, FX, FD = (np.zeros((K, MAXF), np.int64) for _ in range(3))
    for k, fk in enumerate(faces):
        for i, (y, x, d) in enumerate(fk):
            FY[k, i], FX[k, i], FD[k, i] = y, x, d
    ptr, ak, aj = [0], [], []
    for k2 in range(K):
        for j2 in range(MAXP):
            if j2 < PN[k2]:
                for k in range(K):
                    for j in range(PN[k]):
                        if PD[k, j] == (PD[k2, j2] + 2) % 4 and compat[PL[k, j], PL[k2, j2]]:
                            ak.append(k)
                            aj.append(j)
            ptr.append(len(ak))
    dptr, dk, dj = [0], [], []                    # ports facing side d (exits: any letter)
    for d in range(4):
        for k in range(K):
            for j in range(PN[k]):
                if PD[k, j] == d:
                    dk.append(k)
                    dj.append(j)
        dptr.append(len(dk))
    C.tb = Tabs(KH, KW, KH // 2, KW // 2, KC, PN, PY, PX, PD, PL, PCELL, compat, np.array(ptr, np.int64),
                np.array(ak, np.int64), np.array(aj, np.int64), np.int64(MAXP), NF, FY, FX, FD,
                np.array(dptr, np.int64), np.array(dk, np.int64), np.array(dj, np.int64))
    for f in Tabs._fields:
        setattr(C, f, getattr(C.tb, f))
    return C


class State:
    def __init__(self, BY, BX):
        self.BY, self.BX = BY, BX
        self.lab = np.full(BY * BX, -1, np.int64)
        self.phy = np.zeros(BY * BX, np.int64)
        self.phx = np.zeros(BY * BX, np.int64)
        self.occ = np.full(BY * BS * BX * BS, -1, np.int64)
        self.ext = np.zeros(1, np.int64)
        self.exy = self.exx = self.exd = np.zeros(0, np.int64)

    @property
    def st(self):
        return St(self.lab, self.phy, self.phx, self.occ, np.int64(self.BY), np.int64(self.BX), self.ext,
                  self.exy, self.exx, self.exd)

    def copy(self):
        s = State(self.BY, self.BX)
        s.lab[:], s.phy[:], s.phx[:], s.occ[:] = self.lab, self.phy, self.phx, self.occ
        s.ext, s.exy, s.exx, s.exd = self.ext.copy(), self.exy, self.exx, self.exd
        return s

    def region(self, tb, y0, x0, h, w, lab8):
        """Confine objects to the h x w cells at (y0, x0): every other cell is
        OUTSIDE (occ = NB), objects that do not fit are removed.  lab8: the 8
        half-side ports q = 2 d + i (side d = N, E, S, W; half i = 0 the
        top / left half); lab8[q] > 0 makes the outside cells along that half
        EXITS (a door facing one is attached to virtual node NB + 1 + q),
        0 leaves them walls."""
        NB, H, W = self.BY * self.BX, self.BY * BS, self.BX * BS
        inside = np.zeros((H, W), bool)
        inside[y0:y0 + h, x0:x0 + w] = True
        for o in np.nonzero(self.lab >= 0)[0]:
            _lift(o, tb, self.st)
        occ = self.occ.reshape(H, W)
        occ[~inside] = NB
        ext = np.zeros((H, W), np.int64)
        ext[~inside] = -1
        ys, xs, ds = [], [], []
        for q in range(8):
            if lab8[q] <= 0:
                continue
            d, i = divmod(q, 2)
            n = w if d % 2 == 0 else h
            t = np.arange(i * (n // 2), (i + 1) * (n // 2) if i == 0 else n)
            cy, cx = ((y0 - 1 + 0 * t, x0 + t), (y0 + t, x0 + w + 0 * t), (y0 + h + 0 * t, x0 + t),
                      (y0 + t, x0 - 1 + 0 * t))[d]
            ext[cy, cx] = q + 1
            ys += list(cy)
            xs += list(cx)
            ds += [(d + 2) % 4] * len(t)                 # the exit faces into the region
        self.ext = ext.ravel()
        self.exy, self.exx, self.exd = (np.array(a, np.int64) for a in (ys, xs, ds))


# ---------------------------------------------------------------- primitives
# Positions: a block stores its object's CENTRE cell (CY[k], CX[k] of the
# template) as a phase inside the block; the functions below take the
# absolute top-left (ay, ax), any integer (wrapped on the torus).

@njit(cache=True)
def _anchor(o, tb, st):
    k = st.lab[o]
    return (o // st.BX) * BS + st.phy[o] - tb.CY[k], (o % st.BX) * BS + st.phx[o] - tb.CX[k]


@njit(cache=True)
def _fits(k, ay, ax, tb, st):
    H, W = st.BY * BS, st.BX * BS
    for y in range(tb.KH[k]):
        row = ((ay + y) % H) * W
        for x in range(tb.KW[k]):
            if tb.KC[k, y, x] >= 0 and st.occ[row + (ax + x) % W] >= 0:
                return False
    return True


@njit(cache=True)
def _paint(k, ay, ax, val, tb, st):
    H, W = st.BY * BS, st.BX * BS
    for y in range(tb.KH[k]):
        row = ((ay + y) % H) * W
        for x in range(tb.KW[k]):
            if tb.KC[k, y, x] >= 0:
                st.occ[row + (ax + x) % W] = val


@njit(cache=True)
def _set(b, k, py, px, tb, st):
    st.lab[b], st.phy[b], st.phx[b] = k, py, px
    if k >= 0:
        ay, ax = _anchor(b, tb, st)
        _paint(k, ay, ax, b, tb, st)


@njit(cache=True)
def _lift(b, tb, st):
    k = st.lab[b]
    if k >= 0:
        ay, ax = _anchor(b, tb, st)
        _paint(k, ay, ax, -1, tb, st)
        st.lab[b] = -1
    return k


@njit(cache=True)
def _port_at(gy, gx, d, tb, st):
    """(block, port) of a port at cell (gy, gx) facing side d, else (-1, -1);
    an exit facing d: (NB + 1 + q, -1) for its half-side q."""
    H, W = st.BY * BS, st.BX * BS
    NB = st.lab.shape[0]
    o = st.occ[gy * W + gx]
    if o < 0:
        return -1, -1
    if o == NB:
        v = st.ext[gy * W + gx]
        if v > 0 and d == ((v - 1) // 2 + 2) % 4:
            return NB + v, -1
        return -1, -1
    k2 = st.lab[o]
    oy, ox = _anchor(o, tb, st)
    j2 = tb.PCELL[k2, (gy - oy) % H, (gx - ox) % W]
    if j2 >= 0 and tb.PD[k2, j2] == d:
        return o, j2
    return -1, -1


@njit(cache=True)
def _match(k, ay, ax, j, tb, st):
    """Block whose port is attached to port j of k at (ay, ax), or -1."""
    H, W = st.BY * BS, st.BX * BS
    d = tb.PD[k, j]
    o, j2 = _port_at((ay + tb.PY[k, j] + DY[d]) % H, (ax + tb.PX[k, j] + DX[d]) % W, (d + 2) % 4, tb, st)
    if o > st.lab.shape[0] or (o >= 0 and tb.compat[tb.PL[k, j], tb.PL[st.lab[o], j2]]):
        return o
    return -1


@njit(cache=True)
def _nmatch(k, ay, ax, tb, st):
    m = 0
    for j in range(tb.PN[k]):
        if _match(k, ay, ax, j, tb, st) >= 0:
            m += 1
    return m


@njit(cache=True)
def _blocked(k, ay, ax, tb, st):
    """Ports touching k at (ay, ax) that are not attached: k's own ports facing
    an occupied cell, and neighbour ports facing k's cells."""
    H, W = st.BY * BS, st.BX * BS
    n = 0
    for i in range(tb.NF[k]):
        cy, cx, d = tb.FY[k, i], tb.FX[k, i], tb.FD[k, i]
        gy, gx = (ay + cy + DY[d]) % H, (ax + cx + DX[d]) % W
        if st.occ[gy * W + gx] < 0:
            continue
        j = tb.PCELL[k, cy, cx]
        mine = j >= 0 and tb.PD[k, j] == d
        o, j2 = _port_at(gy, gx, (d + 2) % 4, tb, st)
        theirs = o >= 0
        if mine and theirs and (o > st.lab.shape[0] or tb.compat[tb.PL[k, j], tb.PL[st.lab[o], j2]]):
            continue
        if mine:
            n += 1
        if theirs:
            n += 1
    return n


@njit(cache=True)
def _unmatched(o, tb, st):
    k = st.lab[o]
    ay, ax = _anchor(o, tb, st)
    return tb.PN[k] - _nmatch(k, ay, ax, tb, st)


# ---------------------------------------------------------------- block Gibbs

Cn = namedtuple("Cn", "on beta gamma gsmall smin comp cu csz nid npr pry prx prd cb vcls vroot hasopen")
VBIG = 1 << 40                                   # size of a virtual (exit) node: rooted


def conn_state(NB, MAXP, beta=0.0, gamma=0.0, gsmall=0.0, smin=0, on=True, vcls=None, root=None):
    """Connectivity heuristic state (see _snapshot and _cand_e); on=False
    disables it.  vcls (8,): the external class of each half-side exit (0:
    none); exits of one class are joined outside.  root: the class whose
    exits are ROOTS (default: every class); smin = VBIG puts gsmall on
    joining no structure that reaches a root."""
    i = lambda *sh: np.zeros(sh, np.int64)
    vc = i(8) if vcls is None else np.asarray(vcls, np.int64)
    vr = (vc > 0) if root is None else (vc == root)
    return Cn(on, float(beta), float(gamma), float(gsmall), np.int64(smin), i(NB),
              np.arange(2 * NB + 16, dtype=np.int64), i(2 * NB + 16), i(1), i(NB), i(NB, MAXP), i(NB, MAXP),
              i(NB, MAXP), i(MAXP), vc, vr.astype(np.bool_), i(8))


@njit(cache=True)
def _find(cu, a):
    while cu[a] != a:
        cu[a] = cu[cu[a]]
        a = cu[a]
    return a


@njit(cache=True)
def _snapshot(tb, st, cn):
    """Per sweep: components of the attachment graph (comp, union-find cu)
    and a random spanning forest of each (random-order Kruskal).  Each forest
    edge protects its two door cells: block b must keep a door at (pry, prx)
    facing prd while the partner still has one facing back."""
    NB = st.lab.shape[0]
    H, W = st.BY * BS, st.BX * BS
    E = NB * tb.MAXP
    eo = np.empty(E, np.int64)
    eq = np.empty(E, np.int64)
    ey = np.empty(E, np.int64)
    ex = np.empty(E, np.int64)
    ed = np.empty(E, np.int64)
    ne = 0
    for o in range(cn.cu.shape[0]):
        cn.cu[o] = o
    for q in range(8):
        cn.hasopen[q] = 0
    for o in range(NB):
        cn.npr[o] = 0
        k = st.lab[o]
        if k < 0:
            continue
        ay, ax = _anchor(o, tb, st)
        for j in range(tb.PN[k]):
            q = _match(k, ay, ax, j, tb, st)
            if q > NB:
                cn.hasopen[q - NB - 1] = 1
            if q > o:
                eo[ne], eq[ne], ed[ne] = o, q, tb.PD[k, j]
                ey[ne], ex[ne] = (ay + tb.PY[k, j]) % H, (ax + tb.PX[k, j]) % W
                ne += 1
    for i in range(ne):                                   # components
        ra, rb = _find(cn.cu, eo[i]), _find(cn.cu, eq[i])
        if ra != rb:
            cn.cu[ra] = rb
    for q in range(8):                                    # exits of one class: joined outside
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
    for i in np.random.permutation(ne):                   # random spanning forest
        ra, rb = _find(tree, eo[i]), _find(tree, eq[i])
        if ra == rb:
            continue
        tree[ra] = rb
        o, q, d = eo[i], eq[i], ed[i]
        if cn.npr[o] < tb.MAXP:
            n = cn.npr[o]
            cn.pry[o, n], cn.prx[o, n], cn.prd[o, n] = ey[i], ex[i], d
            cn.npr[o] += 1
        if q < NB and cn.npr[q] < tb.MAXP:
            n = cn.npr[q]
            cn.pry[q, n], cn.prx[q, n], cn.prd[q, n] = (ey[i] + DY[d]) % H, (ex[i] + DX[d]) % W, (d + 2) % 4
            cn.npr[q] += 1
    cn.nid[0] = NB + 9


@njit(cache=True)
def _live(b, i, tb, st, cn):
    """Protection i of block b still holds on the partner's side: -> the
    partner's door letter, else -1."""
    H, W = st.BY * BS, st.BX * BS
    d = cn.prd[b, i]
    o, j2 = _port_at((cn.pry[b, i] + DY[d]) % H, (cn.prx[b, i] + DX[d]) % W, (d + 2) % 4, tb, st)
    if o < 0 or o == b:
        return -1
    if o > st.lab.shape[0]:
        return tb.compat.shape[0]                         # an exit: any letter
    return tb.PL[st.lab[o], j2]


@njit(cache=True)
def _nlive(b, tb, st, cn):
    """Live protected doors of b; a door to an exit counts 2 (pins the room:
    the half-side keeps its opening)."""
    n = 0
    for i in range(cn.npr[b]):
        lp = _live(b, i, tb, st, cn)
        if lp >= 0:
            n += 2 if lp == tb.compat.shape[0] else 1
    return n


@njit(cache=True)
def _cand_e(b, k, py, px, lam, mu, fam, pa, nu, cn, tb, st):
    """Energy change of k at centre phase (py, px) in block b (lifted), with
    nu if it differs from the parent's label pa (lab, phy, phx); with cn.on,
    inf unless it keeps b's live protected doors, -beta per extra component
    it joins, +gamma if it joins none, +gsmall if all it joins have fewer
    than smin rooms."""
    H, W = st.BY * BS, st.BX * BS
    ay = (b // st.BX) * BS + py - tb.CY[k]
    ax = (b % st.BX) * BS + px - tb.CX[k]
    if cn.on:
        for i in range(cn.npr[b]):
            lp = _live(b, i, tb, st, cn)
            if lp < 0:
                continue
            ly, lx = (cn.pry[b, i] - ay) % H, (cn.prx[b, i] - ax) % W
            if ly >= tb.KH[k] or lx >= tb.KW[k]:
                return np.inf
            j = tb.PCELL[k, ly, lx]
            if j < 0 or tb.PD[k, j] != cn.prd[b, i] or (lp < tb.compat.shape[0] and not tb.compat[tb.PL[k, j], lp]):
                return np.inf
    if not _fits(k, ay, ax, tb, st):
        return np.inf
    NB = st.lab.shape[0]
    m, nc, big, nopen = 0, 0, 0, 0
    for j in range(tb.PN[k]):
        o = _match(k, ay, ax, j, tb, st)
        if o < 0:
            continue
        m += 1
        if cn.on:
            if o > NB and cn.hasopen[o - NB - 1] == 0:
                nopen += 1                                # the first opening of a half-side
            c = _find(cn.cu, o if o > NB else cn.comp[o])
            new = True
            for i in range(nc):
                if cn.cb[i] == c:
                    new = False
            if new:
                cn.cb[nc] = c
                nc += 1
                big += cn.csz[c]
    e = lam * (tb.PN[k] - 2 * m) + mu[fam[k]]
    if nu != 0 and not (pa[0][b] == k and pa[1][b] == py and pa[2][b] == px):
        e += nu
    if cn.on:
        e += (cn.gamma if nc == 0 else -cn.beta * (nc - 1)) - cn.beta * nopen
        if big < cn.smin and nopen == 0:                  # a first opening is never an orphan
            e += cn.gsmall
    return e


@njit(cache=True)
def _join(b, tb, st, cn):
    """After block b is set: its component id, merging those it touches."""
    k = st.lab[b]
    if k < 0:
        return
    ay, ax = _anchor(b, tb, st)
    NB = st.lab.shape[0]
    r = -1
    for j in range(tb.PN[k]):
        o = _match(k, ay, ax, j, tb, st)
        if o < 0:
            continue
        if o > NB:
            cn.hasopen[o - NB - 1] = 1
        c = _find(cn.cu, o if o > NB else cn.comp[o])
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
def _enumerate(b, tb, st, ck, cy, cx):
    """Attached states of block b (its object lifted), deduplicated: -> count
    n in ck / cy / cx (centre phases)."""
    BY, BX = st.BY, st.BX
    H, W = BY * BS, BX * BS
    by, bx = b // BX, b % BX
    n = 0
    for dby in range(-RAD, RAD + 1):
        for dbx in range(-RAD, RAD + 1):
            o = ((by + dby) % BY) * BX + (bx + dbx) % BX
            if o == b or st.lab[o] < 0:
                continue
            k2 = st.lab[o]
            oy, ox = _anchor(o, tb, st)
            for j2 in range(tb.PN[k2]):
                d2 = tb.PD[k2, j2]
                fy, fx = (oy + tb.PY[k2, j2] + DY[d2]) % H, (ox + tb.PX[k2, j2] + DX[d2]) % W
                if st.occ[fy * W + fx] >= 0:
                    continue
                s = k2 * tb.MAXP + j2
                for a in range(tb.ATTP[s], tb.ATTP[s + 1]):
                    k, j = tb.ATTK[a], tb.ATTJ[a]
                    ry = (fy - tb.PY[k, j] + tb.CY[k] - by * BS) % H
                    rx = (fx - tb.PX[k, j] + tb.CX[k] - bx * BS) % W
                    if ry >= BS or rx >= BS:
                        continue
                    dup = False
                    for i in range(n):
                        if ck[i] == k and cy[i] == ry and cx[i] == rx:
                            dup = True
                            break
                    if dup or n >= ck.shape[0]:
                        continue
                    ck[n], cy[n], cx[n] = k, ry, rx
                    n += 1
    for e in range(st.exy.shape[0]):                      # exits: any port facing out at them
        din = st.exd[e]
        fy, fx = (st.exy[e] + DY[din]) % H, (st.exx[e] + DX[din]) % W
        if abs(fy // BS - by) > RAD + 1 or abs(fx // BS - bx) > RAD + 1 or st.occ[fy * W + fx] >= 0:
            continue
        dout = (din + 2) % 4
        for a in range(tb.DLP[dout], tb.DLP[dout + 1]):
            k, j = tb.DLK[a], tb.DLJ[a]
            ry = (fy - tb.PY[k, j] + tb.CY[k] - by * BS) % H
            rx = (fx - tb.PX[k, j] + tb.CX[k] - bx * BS) % W
            if ry >= BS or rx >= BS:
                continue
            dup = False
            for i in range(n):
                if ck[i] == k and cy[i] == ry and cx[i] == rx:
                    dup = True
                    break
            if dup or n >= ck.shape[0]:
                continue
            ck[n], cy[n], cx[n] = k, ry, rx
            n += 1
    return n


@njit(cache=True)
def _site(b, lam, T, mu, fam, pa, nu, cn, tb, st, ck, cy, cx, ce):
    """Kernels A then B at block b."""
    K = tb.KH.shape[0]
    xk, xy, xx = st.lab[b], st.phy[b], st.phx[b]
    _lift(b, tb, st)
    # empty: forbidden for a block holding two live protected doors
    e0 = nu if nu != 0 and pa[0][b] >= 0 else 0.0
    if cn.on and _nlive(b, tb, st, cn) >= 2:
        e0 = np.inf
    # A: heat-bath over {empty} + attached states, if the current state is among them
    n = _enumerate(b, tb, st, ck, cy, cx)
    inset = xk < 0
    emin = e0
    for i in range(n):
        e = _cand_e(b, ck[i], cy[i], cx[i], lam, mu, fam, pa, nu, cn, tb, st)
        ce[i] = e
        if e < emin:
            emin = e
        if ck[i] == xk and cy[i] == xy and cx[i] == xx:
            inset = True
    if inset and np.isfinite(emin):
        z = np.exp(-(e0 - emin) / T)
        for i in range(n):
            z += np.exp(-(ce[i] - emin) / T)
        r = np.random.random() * z - np.exp(-(e0 - emin) / T)
        if r < 0:
            xk = -1
        else:
            for i in range(n):
                r -= np.exp(-(ce[i] - emin) / T)
                if r < 0 or i == n - 1:
                    xk, xy, xx = ck[i], cy[i], cx[i]
                    break
    # B: independence MH, q(empty) = 1/2, q(k, phase) = 1 / (2 K BS^2)
    ex = e0 if xk < 0 else _cand_e(b, xk, xy, xx, lam, mu, fam, pa, nu, cn, tb, st)
    if np.random.random() < 0.5:
        yk, yy, yx, ey = -1, 0, 0, e0
    else:
        yk = np.random.randint(K)
        yy, yx = np.random.randint(BS), np.random.randint(BS)
        ey = _cand_e(b, yk, yy, yx, lam, mu, fam, pa, nu, cn, tb, st)
    if np.isfinite(ey) and not (yk == xk and (yk < 0 or (yy == xy and yx == xx))):
        qx = 0.5 if xk < 0 else 0.5 / (K * BS * BS)
        qy = 0.5 if yk < 0 else 0.5 / (K * BS * BS)
        if not np.isfinite(ex) or np.random.random() < np.exp(-(ey - ex) / T) * qx / qy:
            xk, xy, xx = yk, yy, yx
    _set(b, xk, xy, xx, tb, st)
    if cn.on:
        _join(b, tb, st, cn)


def sweeps(lams, Ts, mu, fam, tb, st, seed, pa=None, nu=0.0, cn=None):
    """One sweep per (lam, T), blocks in random order; pa (lab, phy, phx) a
    parent labelling, nu the cost of differing from it; cn (conn_state) the
    spanning-forest connectivity heuristic, snapshotted before each sweep."""
    if pa is None:
        pa = (st.lab, st.phy, st.phx)
    if cn is None:
        cn = conn_state(1, 1, on=False)
    _sweeps(np.asarray(lams, np.float64), np.asarray(Ts, np.float64), mu, fam, pa, float(nu), cn, tb, st, seed)


@njit(cache=True)
def _sweeps(lams, Ts, mu, fam, pa, nu, cn, tb, st, seed):
    np.random.seed(seed)
    NB = st.lab.shape[0]
    ck = np.empty(1024, np.int64)
    cy = np.empty(1024, np.int64)
    cx = np.empty(1024, np.int64)
    ce = np.empty(1024, np.float64)
    for s in range(lams.shape[0]):
        if cn.on:
            _snapshot(tb, st, cn)
        for b in np.random.permutation(NB):
            _site(b, lams[s], Ts[s], mu, fam, pa, nu, cn, tb, st, ck, cy, cx, ce)


# ---------------------------------------------------------------- coarse level

@njit(cache=True)
def _region_e(by0, bx0, n, lam, mu, fam, tb, st):
    e = 0.0
    for i in range(n):
        for j in range(n):
            o = ((by0 + i) % st.BY) * st.BX + (bx0 + j) % st.BX
            if st.lab[o] >= 0:
                e += lam * _unmatched(o, tb, st) + mu[fam[st.lab[o]]]
    return e


@njit(cache=True)
def _paste(cy0, cx0, R, uy, ux, xs, EB, tb, st):
    """Window of blocks at (cy0, cx0) := exemplar xs's window at block (uy, ux)
    (window objects lifted first); an object that overlaps one already there
    is dropped."""
    BY, BX = st.BY, st.BX
    for i in range(R):
        for j in range(R):
            _lift(((cy0 + i) % BY) * BX + (cx0 + j) % BX, tb, st)
    for i in range(R):
        for j in range(R):
            b = ((cy0 + i) % BY) * BX + (cx0 + j) % BX
            e = ((uy + i) % EB) * EB + (ux + j) % EB
            k = xs.lab[e]
            if k >= 0:
                ay = (b // BX) * BS + xs.phy[e] - tb.CY[k]
                ax = (b % BX) * BS + xs.phx[e] - tb.CX[k]
                if _fits(k, ay, ax, tb, st):
                    _set(b, k, xs.phy[e], xs.phx[e], tb, st)


@njit(cache=True)
def coarse_sweeps(U, R, lams, Ts, nrand, mu, fam, xs, EB, tb, st, seed):
    """U (CY, CX, 2) exemplar block coordinates of R x R windows; the state
    must hold paste(U).  Heat-bath per window over the current coordinate,
    the 4 neighbours' coherent shifts and nrand random ones."""
    np.random.seed(seed)
    CY, CX = U.shape[0], U.shape[1]
    NC = 5 + nrand
    cu = np.empty((NC, 2), np.int64)
    ce = np.empty(NC, np.float64)
    for s in range(lams.shape[0]):
        lam, T = lams[s], Ts[s]
        for c in np.random.permutation(CY * CX):
            py, px = c // CX, c % CX
            n = 0
            cu[n, 0], cu[n, 1] = U[py, px, 0], U[py, px, 1]
            n += 1
            for d in range(4):
                qy, qx = (py + DY[d]) % CY, (px + DX[d]) % CX
                cu[n, 0] = (U[qy, qx, 0] - R * DY[d]) % EB
                cu[n, 1] = (U[qy, qx, 1] - R * DX[d]) % EB
                n += 1
            for i in range(nrand):
                cu[n, 0], cu[n, 1] = np.random.randint(EB), np.random.randint(EB)
                n += 1
            emin = np.inf
            for i in range(n):
                _paste(py * R, px * R, R, cu[i, 0], cu[i, 1], xs, EB, tb, st)
                ce[i] = _region_e(py * R - RAD, px * R - RAD, R + 2 * RAD, lam, mu, fam, tb, st)
                emin = min(emin, ce[i])
            z = 0.0
            for i in range(n):
                z += np.exp(-(ce[i] - emin) / T)
            r = np.random.random() * z
            pick = n - 1
            for i in range(n):
                r -= np.exp(-(ce[i] - emin) / T)
                if r < 0:
                    pick = i
                    break
            U[py, px, 0], U[py, px, 1] = cu[pick, 0], cu[pick, 1]
            _paste(py * R, px * R, R, cu[pick, 0], cu[pick, 1], xs, EB, tb, st)


def paste_all(U, R, xs, EB, tb, st):
    for py in range(U.shape[0]):
        for px in range(U.shape[1]):
            _paste(py * R, px * R, R, U[py, px, 0], U[py, px, 1], xs, EB, tb, st)


# ---------------------------------------------------------------- exemplar growth

@njit(cache=True)
def _try_place(k, ay, ax, tb, st):
    """Place k with top-left (ay, ax) if its centre's block is free, it fits
    and it blocks no port.  -> the block or -1."""
    H, W = st.BY * BS, st.BX * BS
    cy, cx = (ay + tb.CY[k]) % H, (ax + tb.CX[k]) % W
    b = (cy // BS) * st.BX + cx // BS
    if st.lab[b] >= 0 or not _fits(k, ay, ax, tb, st) or _blocked(k, ay, ax, tb, st) > 0:
        return -1
    _set(b, k, cy % BS, cx % BS, tb, st)
    return b


@njit(cache=True)
def grow(k0, target, fw, fam, ok, tb, st, seed, max_tries):
    """Kinetic growth from k0 in the middle with the components ok[k];
    fw[family] the pick weights.  -> objects placed."""
    np.random.seed(seed)
    H, W = st.BY * BS, st.BX * BS
    placed = np.empty(st.lab.shape[0], np.int64)
    np_ = 0
    placed[np_] = _try_place(k0, H // 2, W // 2, tb, st)
    np_ += 1
    wmax = fw.max()
    for t in range(max_tries):
        if np_ >= target:
            break
        o = placed[np.random.randint(np_)]
        if st.lab[o] < 0:
            continue
        k2 = st.lab[o]
        j2 = np.random.randint(tb.PN[k2])
        oy, ox = _anchor(o, tb, st)
        if _match(k2, oy, ox, j2, tb, st) >= 0:
            continue
        s = k2 * tb.MAXP + j2
        na = tb.ATTP[s + 1] - tb.ATTP[s]
        if na == 0:
            continue
        a = tb.ATTP[s] + np.random.randint(na)
        k, j = tb.ATTK[a], tb.ATTJ[a]
        if not ok[k] or np.random.random() * wmax > fw[fam[k]]:
            continue
        d2 = tb.PD[k2, j2]
        fy, fx = oy + tb.PY[k2, j2] + DY[d2], ox + tb.PX[k2, j2] + DX[d2]
        nb = _try_place(k, fy - tb.PY[k, j], fx - tb.PX[k, j], tb, st)
        if nb >= 0:
            placed[np_] = nb
            np_ += 1
    return np_


@njit(cache=True)
def close(seal, tb, st, seed, rounds):
    """Cap each dangling port with a one-port component where one fits, else
    wall it up (seal[k, j]; an object whose last port dangles is removed);
    repeat.  -> unmatched ports left."""
    np.random.seed(seed)
    NB = st.lab.shape[0]
    left = 0
    for r in range(rounds):
        left = 0
        for o in range(NB):
            k2 = st.lab[o]
            if k2 < 0:
                continue
            oy, ox = _anchor(o, tb, st)
            for j2 in range(tb.PN[k2]):
                if _match(k2, oy, ox, j2, tb, st) >= 0:
                    continue
                s = k2 * tb.MAXP + j2
                na = tb.ATTP[s + 1] - tb.ATTP[s]
                ok = False
                start = np.random.randint(max(na, 1))
                for t in range(na):
                    a = tb.ATTP[s] + (start + t) % na
                    k, j = tb.ATTK[a], tb.ATTJ[a]
                    if tb.PN[k] != 1:
                        continue
                    d2 = tb.PD[k2, j2]
                    fy, fx = oy + tb.PY[k2, j2] + DY[d2], ox + tb.PX[k2, j2] + DX[d2]
                    if _try_place(k, fy - tb.PY[k, j], fx - tb.PX[k, j], tb, st) >= 0:
                        ok = True
                        break
                if not ok:
                    if seal[k2, j2] >= 0:
                        st.lab[o] = seal[k2, j2]              # same footprint, a wall where the port was
                    else:
                        _lift(o, tb, st)
                    left += 1
                    break
        if left == 0:
            break
    return left


# ---------------------------------------------------------------- measurement

@njit(cache=True)
def stats(tb, st, par):
    """-> (objects, unmatched ports); par: component root per block
    (union-find over attachments)."""
    NB = st.lab.shape[0]
    for o in range(NB):
        par[o] = o
    nobj, bad = 0, 0
    for o in range(NB):
        k = st.lab[o]
        if k < 0:
            continue
        nobj += 1
        ay, ax = _anchor(o, tb, st)
        for j in range(tb.PN[k]):
            q = _match(k, ay, ax, j, tb, st)
            if q < 0:
                bad += 1
                continue
            if q >= NB:
                continue
            a, b = o, q
            while par[a] != a:
                a = par[a]
            while par[b] != b:
                b = par[b]
            if a != b:
                par[a] = b
    for o in range(NB):
        a = o
        while par[a] != a:
            a = par[a]
        par[o] = a
    return nobj, bad


def measure(C, S):
    par = np.empty(S.BY * S.BX, np.int64)
    nobj, bad = stats(C.tb, S.st, par)
    on = S.lab >= 0
    sizes = np.sort(np.unique(par[on], return_counts=True)[1])[::-1]
    famc = np.bincount(C.fam[S.lab[on]], minlength=len(C.fams))
    return dict(objects=nobj, unmatched=bad, density=on.mean(), comps=len(sizes),
                largest=int(sizes[0]) if len(sizes) else 0,
                in_big=float(sizes[sizes >= 20].sum() / nobj) if nobj else 0.0,
                famfreq=famc / max(nobj, 1), sizes=sizes)


PAL = {"bg": (72, 96, 70), "wall": (28, 24, 22), "corr": (160, 168, 178), "door": (210, 120, 40),
       "bad": (255, 40, 40)}
ROOMC = [(214, 196, 150), (200, 170, 120), (226, 210, 170), (180, 150, 110), (210, 182, 140), (190, 175, 135)]


def render(C, S, y0=0, x0=0, h=None, w=None, px=1):
    """(h px, w px, 3) uint8 image of a window of cells; unmatched ports marked
    by a red 5 x 5 square."""
    H, W = S.BY * BS, S.BX * BS
    h, w = h or H, w or W
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = PAL["bg"]
    famcol, ri = {}, 0
    for f in range(len(C.fams)):
        if C.kind[f] == "room":
            famcol[f], ri = ROOMC[ri % len(ROOMC)], ri + 1
        else:
            famcol[f] = PAL["corr"]
    bads = []
    for o in np.nonzero(S.lab >= 0)[0]:
        k = S.lab[o]
        ay, ax = _anchor(o, C.tb, S.st)
        kc = C.KC[k, :C.KH[k], :C.KW[k]]
        yy, xx = np.nonzero(kc != VOID)
        col = np.array([PAL["wall"], famcol[C.fam[k]], PAL["door"]], np.uint8)[kc[yy, xx]]
        img[(ay + yy) % H, (ax + xx) % W] = col
        for j in range(C.PN[k]):
            if _match(k, ay, ax, j, C.tb, S.st) < 0:
                bads.append(((ay + C.PY[k, j]) % H, (ax + C.PX[k, j]) % W))
    for y, x in bads:
        img[np.ix_(np.arange(y - 2, y + 3) % H, np.arange(x - 2, x + 3) % W)] = PAL["bad"]
    img = np.roll(img, (-y0, -x0), (0, 1))[:h, :w]
    return np.repeat(np.repeat(img, px, 0), px, 1) if px > 1 else img
