"""Conditional divide-and-conquer SMC subtree move on certland's designed p*
(Lindsten et al. 2017, arXiv:1406.4993; Kuntz, Crucinio & Johansen 2024,
arXiv:2110.15782).  A bottom-up alternative to certland_chain.subtree_move
with the same contract (move_level(..., move=functools.partial(dc_move, ...))).

Generic: the algorithm reads only the hierarchy (the quadtree of the root's
subtree) and the designed factor graph (which cells each designed term
touches); nothing in it is about certificates, gravity or columns.

The tree
--------
Root (i, ry, rx), S = the root's subtree (cells of levels i..4).  Nodes:
every non-tile cell c of S (levels i..3); T(c) = c and its descendants.
Leaves: the 2 x 2 tile block under each h = 2 cell (T = those 4 tiles).
Children of a node at level l < 3: its four level-(l+1) cells; of a node at
level 3: its tile block (one leaf).

Intermediate targets and the weight accounting
----------------------------------------------
For a node or leaf n, gamma_n(z_T(n)) = exp(-E_n), E_n = the sum of every
designed term tau (soft terms; hard terms as +inf) with
    vars(tau) & T(n) != {}   and   vars(tau) & S  subset of  T(n),
i.e. the terms touching T(n) whose other cells are all OUTSIDE THE WHOLE
ROOT SUBTREE (fixed) or inside T(n).  Terms reaching a sibling's subtree or
a not-yet-built ancestor are left out.  Hence each designed term touching S
is counted at exactly one place: the node n* = the lowest common ancestor in
the DC tree of its cells inside S (for a term inside one tile block, the
leaf).  E_root = every term touching S = certland_chain.subtree_energy, so
gamma_root is p* restricted to the subtree given everything outside.
At a merge at node c with children k: dE_c = E_c(merged) - sum_k E_k
= the terms whose LCA is c: those involving c itself (the count prior and
row honour against c's children; and, only when c is the root, the count
prior and row honour against c's parent and siblings and the top noise;
smoothness / support with same-level neighbours that are outside S) plus
those spanning two children's subtrees (smoothness, support between the
children's subtrees at every finer level).  The terms involving c are the
offset of c's proposal (`_rho_terms`, `_cert_viol`); the rest only enter
the weight.

The proposal and the weights (conditional DC-SMC)
-------------------------------------------------
Every node keeps N + 1 particles; particle 0 is the current state of T(n)
at every node (the reference: its children's particles 0 are the current
children subtrees).
  * leaf: particles 1..N iid from gamma_leaf exactly (16 states); all leaf
    weights are equal (log Z_leaf), so the leaf population is not resampled:
    the level-3 merge takes leaf particle j for its particle j;
  * node c: for each child population, ancestors a_k^0 = 0 and a_k^j ~
    Categorical(w_k) for j >= 1 independently (multinomial, conditional on
    the reference); merged particle j = (child k's particle a_k^j)_k; then
    c's rho ~ softmax(-(off_rho + b_rho)) and c's cert ~ softmax(-(off_cert
    + b_cert)) (the cert context sees the drawn rho); off = the designed
    terms of c with LCA c (hard ones +inf), b the learned bottom-up bias
    (zero at theta = 0: designed-only).  Particle 0 is forced to c's
    current values and its q is computed.
    log w_c^j = -dE_c - log q_c(c^j | children, fixed); -inf if a hard term
    of dE_c fails (a cross-children death) or c has no admissible value (a
    q death).
  * root: one particle selected with probability proportional to w_root
    (i-SIR); the world takes it.
This is the conditional SMC (particle Gibbs) kernel of the DC-SMC on this
tree: the extended-target argument of Andrieu, Doucet & Holenstein 2010
carries over node by node because each node's ancestor draws for different
children are independent, so the move leaves gamma_root (= p* given the
outside) invariant for any N >= 1 and any bottom-up predictor, as long as q
reads only T(c) and cells outside S.  The context of the predictor (see
up_features) reads exactly that: c's children, the fixed (outside-S) cells
of c's 3 x 3 same-level window and of the ring around c's children, c's
parent when it is outside S, all with known flags.

Flat state layout of a node = certland_chain's flat layout of the node taken
as a root (levels l..4: rho block, cert block; tiles once).

Bottom-up predictor: cp.Preds(nh, nf=NFU) (per (level 0..3, channel) MLP
over the NFU features, full-domain logits); trained by weighted CE on
(features, offsets, value) of every node of the chain's states
(record_up / UpCollector / fit_up)."""
import time

import numpy as np
from numba import njit

from . import certland as cl
from . import certland_chain as cc
from . import certland_pred as cp

NL = cl.NL
NCELL = cp.NCELL
# up-predictor features (NFU), see up_features
U_TOP = 0                                   # 1{level 0: no parent level}
U_PAR = 1                                   # parent slot NCELL + known flag
U_QUAD = U_PAR + NCELL + 1                  # y & 1, x & 1 (when the parent is known)
U_WIN = U_QUAD + 2                          # 3 x 3 same-level window, 9 x (NCELL + flag)
U_CH = U_WIN + 9 * (NCELL + 1)              # 2 x 2 children, 4 x NCELL
U_RING = U_CH + 4 * NCELL                   # 12-cell ring around the children, 12 x (NCELL + flag)
NFU = U_RING + 12 * (NCELL + 1)             # 675


# ------------------------------------------------------------------ membership
@njit(cache=True, inline="always")
def _inb(l, y, x, bl, by, bx, bs):
    """(l, y, x) inside the box (by.., bx.., side bs at level bl) or under it."""
    if l < bl:
        return False
    k = l - bl
    return (by << k) <= y < ((by + bs) << k) and (bx << k) <= x < ((bx + bs) << k)


@njit(cache=True, inline="always")
def _ok(l, y, x, tl, ty, tx, ts, i, ry, rx):
    """A cell may appear in a term of gamma_T: inside T or outside S."""
    return _inb(l, y, x, tl, ty, tx, ts) or not _inb(l, y, x, i, ry, rx, 1)


# ------------------------------------------------------------------ intermediate targets
@njit(cache=True)
def _gamma(tl, ty, tx, ts, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta):
    """(soft, violations) of E_T: every designed term touching T = the box
    (ty, tx, side ts) at level tl and all levels below, whose cells are all
    inside T or outside S = the subtree of root (i, ry, rx)."""
    soft = 0.0
    nv = 0
    for l in range(tl, 5):
        h = 16 >> l
        k = l - tl
        Y0, Y1, X0, X1 = ty << k, (ty + ts) << k, tx << k, (tx + ts) << k
        rho, cert, Aof, Bof = rhos[l], certs[l], Aofs[l], Bofs[l]
        rows, cols = rho.shape
        if l == 0:
            for y in range(Y0, Y1):
                for x in range(X0, X1):
                    d = rho[y, x] - h * h * fill[y, x]
                    soft += beta * d * d / (h * h)
        else:
            prho, pcert, pA, pB = rhos[l - 1], certs[l - 1], Aofs[l - 1], Bofs[l - 1]
            for py in range(Y0 // 2, (Y1 - 1) // 2 + 1):
                for px in range(X0 // 2, (X1 - 1) // 2 + 1):
                    yy, xx = 2 * py, 2 * px
                    okP = _ok(l - 1, py, px, tl, ty, tx, ts, i, ry, rx)
                    ok00 = _ok(l, yy, xx, tl, ty, tx, ts, i, ry, rx)
                    ok01 = _ok(l, yy, xx + 1, tl, ty, tx, ts, i, ry, rx)
                    ok10 = _ok(l, yy + 1, xx, tl, ty, tx, ts, i, ry, rx)
                    ok11 = _ok(l, yy + 1, xx + 1, tl, ty, tx, ts, i, ry, rx)
                    if okP and ok00 and ok01 and ok10 and ok11:
                        d = rho[yy, xx] + rho[yy, xx + 1] + rho[yy + 1, xx] + rho[yy + 1, xx + 1] - prho[py, px]
                        soft += kappa * d * d / (h * h)
                    if okP:
                        AP, BP = pA[pcert[py, px]], pB[pcert[py, px]]
                        for y in range(max(yy, Y0), min(yy + 2, Y1)):
                            if (ok00 and ok01) if y == yy else (ok10 and ok11):
                                sa = Aof[cert[y, xx]] + Aof[cert[y, xx + 1]]
                                sb = Bof[cert[y, xx]] + Bof[cert[y, xx + 1]]
                                if sa < AP:
                                    nv += 1
                                if sb > BP:
                                    nv += 1
        for y in range(Y0, Y1):
            for x in range(X0 - 1, X1):
                a, b = x % cols, (x + 1) % cols
                if _ok(l, y, a, tl, ty, tx, ts, i, ry, rx) and _ok(l, y, b, tl, ty, tx, ts, i, ry, rx):
                    soft += lam * abs(rho[y, a] - rho[y, b]) / h
        for x in range(X0, X1):
            for y in range(Y0 - 1, Y1):
                if y >= 0 and y + 1 < rows and _ok(l, y, x, tl, ty, tx, ts, i, ry, rx) \
                        and _ok(l, y + 1, x, tl, ty, tx, ts, i, ry, rx):
                    if Bof[cert[y, x]] > Aof[cert[y + 1, x]]:
                        nv += 1
    return soft, nv


@njit(cache=True)
def _rho_terms(l, y, x, t, i, ry, rx, rhos, certs, fill, kappa, lam, beta):
    """Soft designed terms with LCA = node (l, y, x) that involve its rho,
    at rho = t (the other cells as they stand): the count prior against its
    children; when the parent block is outside S (c the root) the count
    prior against the parent and siblings; smoothness with same-level
    neighbours outside S; the top noise."""
    h = 16 >> l
    rho = rhos[l]
    rows, cols = rho.shape
    e = 0.0
    if l == 0:
        d = t - h * h * fill[y, x]
        e += beta * d * d / (h * h)
    else:
        py, px = y // 2, x // 2
        ok = _ok(l - 1, py, px, l, y, x, 1, i, ry, rx)
        for yy in range(2 * py, 2 * py + 2):
            for xx in range(2 * px, 2 * px + 2):
                if not _ok(l, yy, xx, l, y, x, 1, i, ry, rx):
                    ok = False
        if ok:
            s = t
            for yy in range(2 * py, 2 * py + 2):
                for xx in range(2 * px, 2 * px + 2):
                    if yy != y or xx != x:
                        s += rho[yy, xx]
            d = s - rhos[l - 1][py, px]
            e += kappa * d * d / (h * h)
    if l < 4:
        hc = h // 2
        r = rhos[l + 1]
        d = r[2 * y, 2 * x] + r[2 * y, 2 * x + 1] + r[2 * y + 1, 2 * x] + r[2 * y + 1, 2 * x + 1] - t
        e += kappa * d * d / (hc * hc)
    for dx in (-1, 1):
        xx = (x + dx) % cols
        if _ok(l, y, xx, l, y, x, 1, i, ry, rx):
            e += lam * abs(t - rho[y, xx]) / h
    return e


@njit(cache=True)
def _cert_viol(l, y, x, t, i, ry, rx, rhos, certs, Aofs, Bofs):
    """Hard designed terms with LCA = node (l, y, x) that involve its cert,
    violated at cert = t: row honour against its children's two rows;
    support with same-level neighbours above / below outside S; when the
    parent and sibling are outside S, its own row's honour."""
    cert, Aof, Bof = certs[l], Aofs[l], Bofs[l]
    rows = cert.shape[0]
    A, B = Aof[t], Bof[t]
    nv = 0
    if y > 0 and _ok(l, y - 1, x, l, y, x, 1, i, ry, rx) and Bof[cert[y - 1, x]] > A:
        nv += 1
    if y + 1 < rows and _ok(l, y + 1, x, l, y, x, 1, i, ry, rx) and B > Aof[cert[y + 1, x]]:
        nv += 1
    if l > 0:
        py, px = y // 2, x // 2
        xs = x ^ 1
        if _ok(l - 1, py, px, l, y, x, 1, i, ry, rx) and _ok(l, y, xs, l, y, x, 1, i, ry, rx):
            c = certs[l - 1][py, px]
            if A + Aof[cert[y, xs]] < Aofs[l - 1][c]:
                nv += 1
            if B + Bof[cert[y, xs]] > Bofs[l - 1][c]:
                nv += 1
    if l < 4:
        cc_ = certs[l + 1]
        cA, cB = Aofs[l + 1], Bofs[l + 1]
        for yy in range(2 * y, 2 * y + 2):
            if cA[cc_[yy, 2 * x]] + cA[cc_[yy, 2 * x + 1]] < A:
                nv += 1
            if cB[cc_[yy, 2 * x]] + cB[cc_[yy, 2 * x + 1]] > B:
                nv += 1
    return nv


# ------------------------------------------------------------------ bottom-up predictor context
@njit(cache=True)
def up_features(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, out):
    """Context of node (l, y, x), channel ch, inside root (i, ry, rx), into
    out[:NFU].  Reads only T(c) and cells outside S:
      [U_TOP]   1{l = 0}
      [U_PAR]   the parent (side 2h) when it is outside S, + known flag
      [U_QUAD]  y & 1, x & 1 when the parent is known
      [U_WIN]   3 x 3 same-level window: per slot the cell (cp encoding:
                off-grid flags, rho, cert) and a known flag (off the grid or
                outside S); the centre: its rho (flag 1) when drawing cert,
                nothing (flag 0) when drawing rho
      [U_CH]    the 2 x 2 children (side h/2; tiles as rho = A = B = t)
      [U_RING]  the 12 level-(l+1) cells around the children, + known flags."""
    for j in range(NFU):
        out[j] = 0.0
    h = 16 >> l
    if l == 0:
        out[U_TOP] = 1.0
    else:
        py, px = y // 2, x // 2
        if not _inb(l - 1, py, px, i, ry, rx, 1):
            cp._enc_cell(2 * h, py, px, rhos[l - 1], certs[l - 1], Aofs[l - 1], Bofs[l - 1], False, False,
                         out, U_PAR)
            out[U_PAR + NCELL] = 1.0
            out[U_QUAD] = y & 1
            out[U_QUAD + 1] = x & 1
    rho, cert, Aof, Bof = rhos[l], certs[l], Aofs[l], Bofs[l]
    rows, cols = rho.shape
    for d in range(9):
        yy, xx = y + d // 3 - 1, (x + d % 3 - 1) % cols
        o = U_WIN + d * (NCELL + 1)
        if d == 4:
            if ch == 1:
                cp._enc_cell(h, yy, xx, rho, cert, Aof, Bof, False, True, out, o)
                out[o + NCELL] = 1.0
        elif yy < 0 or yy >= rows or not _inb(l, yy, xx, i, ry, rx, 1):
            cp._enc_cell(h, yy, xx, rho, cert, Aof, Bof, False, False, out, o)
            out[o + NCELL] = 1.0
    hc = h // 2
    r1, c1, A1, B1 = rhos[l + 1], certs[l + 1], Aofs[l + 1], Bofs[l + 1]
    rows1, cols1 = r1.shape
    for d in range(4):
        cp._enc_cell(hc, 2 * y + d // 2, 2 * x + d % 2, r1, c1, A1, B1, False, False, out, U_CH + d * NCELL)
    k = 0
    for d in range(16):
        a, b = d // 4, d % 4
        if 1 <= a <= 2 and 1 <= b <= 2:
            continue
        yy, xx = 2 * y - 1 + a, (2 * x - 1 + b) % cols1
        o = U_RING + k * (NCELL + 1)
        if yy < 0 or yy >= rows1 or not _inb(l + 1, yy, xx, i, ry, rx, 1):
            cp._enc_cell(hc, yy, xx, r1, c1, A1, B1, False, False, out, o)
            out[o + NCELL] = 1.0
        k += 1


@njit(cache=True)
def _offsets(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta, e):
    """e[:D] = designed offsets of node (l, y, x) channel ch (+inf excluded)."""
    if ch == 0:
        D = (16 >> l) * (16 >> l) + 1
        for t in range(D):
            e[t] = _rho_terms(l, y, x, t, i, ry, rx, rhos, certs, fill, kappa, lam, beta)
    else:
        D = Aofs[l].shape[0]
        for t in range(D):
            e[t] = 0.0 if _cert_viol(l, y, x, t, i, ry, rx, rhos, certs, Aofs, Bofs) == 0 else np.inf
    return D


@njit(cache=True)
def _draw_cell(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
               ths, ons, nh, forced, fval, e, bb, f):
    """(value, log q) of node (l, y, x) channel ch from softmax(-(off + b)); (-1, 0) if none."""
    D = _offsets(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta, e)
    if ons[2 * l + ch]:
        up_features(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, f)
        cp._mlp_n(f, ths[2 * l + ch], nh, D, NFU, bb)
        for t in range(D):
            e[t] += bb[t]
    return cc._pick(e, D, forced, fval)


# ------------------------------------------------------------------ the move
@njit(cache=True)
def _flat_size_nb(i):
    """certland_chain.flat_size inside numba."""
    n = 0
    for l in range(i, 5):
        s = 1 << (l - i)
        n += s * s if l == 4 else 2 * s * s
    return n


@njit(cache=True)
def _tiles_get(rhos, y0, x0):
    t = rhos[4]
    return t[y0, x0] + 2 * t[y0, x0 + 1] + 4 * t[y0 + 1, x0] + 8 * t[y0 + 1, x0 + 1]


@njit(cache=True)
def _tiles_set(rhos, y0, x0, c):
    t = rhos[4]
    t[y0, x0] = c & 1
    t[y0, x0 + 1] = (c >> 1) & 1
    t[y0 + 1, x0] = (c >> 2) & 1
    t[y0 + 1, x0 + 1] = (c >> 3) & 1


@njit(cache=True)
def _multinomial(lw, n, out):
    """out[1:n] iid ~ Categorical(exp(lw)); out[0] = 0 (the reference)."""
    mx = lw.max()
    m = lw.shape[0]
    cdf = np.empty(m)
    c = 0.0
    for j in range(m):
        c += np.exp(lw[j] - mx)
        cdf[j] = c
    out[0] = 0
    for j in range(1, n):
        u = np.random.random() * c
        a = np.searchsorted(cdf, u, side="right")
        out[j] = min(a, m - 1)


@njit(cache=True)
def _dc(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta, N1, ths, ons, nh, seed,
        root_states, root_logw, dead_q, dead_x, info):
    """Conditional DC-SMC on the subtree of (i, ry, rx), N1 particles per
    node (particle 0 the reference).  Out: root_states (N1, flat), root_logw,
    dead_q[l] / dead_x[l] (deaths of fresh merged particles at node level l:
    no admissible value / a hard cross-children term), info = [chosen,
    changed, reference ok, ...].  The world ends in the chosen state."""
    np.random.seed(seed)
    nflat = root_states.shape[1]
    orig = np.empty(nflat, np.int64)
    cc._save(i, ry, rx, rhos, certs, orig)
    e = np.empty(257)
    bb = np.empty(257)
    f = np.empty(NFU)
    info[2] = 1
    # ---- leaves: tile blocks under the level-3 cells
    s3 = 1 << (3 - i)
    n3 = s3 * s3
    Y3, X3 = ry * s3, rx * s3
    LP = np.empty((n3, N1, 4), np.int64)
    LE = np.empty((n3, N1))
    es = np.empty(16)
    for a in range(s3):
        for b in range(s3):
            node = a * s3 + b
            y0, x0 = 2 * (Y3 + a), 2 * (X3 + b)
            cur = _tiles_get(rhos, y0, x0)
            for c in range(16):
                _tiles_set(rhos, y0, x0, c)
                sft, nv = _gamma(4, y0, x0, 2, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta)
                es[c] = sft if nv == 0 else np.inf
            _tiles_set(rhos, y0, x0, cur)
            if es[cur] == np.inf:
                info[2] = 0
                return
            for j in range(N1):
                pk = cur
                if j > 0:
                    pk, lp = cc._pick(es, 16, False, 0)
                LP[node, j, 0] = pk & 1
                LP[node, j, 1] = (pk >> 1) & 1
                LP[node, j, 2] = (pk >> 2) & 1
                LP[node, j, 3] = (pk >> 3) & 1
                LE[node, j] = es[pk]
    CP = LP
    CE = LE
    CW = np.zeros((n3, N1))
    leaf = True
    anc = np.empty((4, N1), np.int64)
    for l in range(3, i - 1, -1):
        s = 1 << (l - i)
        n = s * s
        F = _flat_size_nb(l)
        P = np.zeros((n, N1, F), np.int64)
        E = np.zeros((n, N1))
        Wt = np.full((n, N1), -np.inf)
        Y0, X0 = ry * s, rx * s
        o_l = cc._offset(i, l)
        nk = 1 if l == 3 else 4
        for a in range(s):
            for b in range(s):
                node = a * s + b
                y, x = Y0 + a, X0 + b
                ref_r = orig[o_l + node]
                ref_c = orig[o_l + n + node]
                for k in range(nk):
                    ci = node if nk == 1 else (2 * a + k // 2) * (2 * s) + 2 * b + k % 2
                    if leaf:
                        for j in range(N1):
                            anc[k, j] = j
                    else:
                        _multinomial(CW[ci], N1, anc[k])
                for j in range(N1):
                    Esum = 0.0
                    for k in range(nk):
                        aj = anc[k, j]
                        if nk == 1:
                            ci = node
                            yy, xx = 2 * y, 2 * x
                            t = rhos[4]
                            t[yy, xx] = CP[ci, aj, 0]
                            t[yy, xx + 1] = CP[ci, aj, 1]
                            t[yy + 1, xx] = CP[ci, aj, 2]
                            t[yy + 1, xx + 1] = CP[ci, aj, 3]
                        else:
                            ci = (2 * a + k // 2) * (2 * s) + 2 * b + k % 2
                            cc._load(l + 1, 2 * y + k // 2, 2 * x + k % 2, rhos, certs, CP[ci, aj])
                        Esum += CE[ci, aj]
                    fr = j == 0
                    pr, lqr = _draw_cell(l, y, x, 0, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                                         ths, ons, nh, fr, ref_r, e, bb, f)
                    if pr < 0:
                        if fr:
                            info[2] = 0
                            return
                        dead_q[l] += 1
                        continue
                    rhos[l][y, x] = pr
                    pc, lqc = _draw_cell(l, y, x, 1, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                                         ths, ons, nh, fr, ref_c, e, bb, f)
                    if pc < 0:
                        if fr:
                            info[2] = 0
                            return
                        dead_q[l] += 1
                        continue
                    certs[l][y, x] = pc
                    Ec, nv = _gamma(l, y, x, 1, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta)
                    if nv > 0:
                        if fr:
                            info[2] = 0
                            return
                        dead_x[l] += 1
                        continue
                    E[node, j] = Ec
                    Wt[node, j] = -(Ec - Esum) - lqr - lqc
                    cc._save(l, y, x, rhos, certs, P[node, j])
        CP = P
        CE = E
        CW = Wt
        leaf = False
    # ---- root selection
    lw = CW[0]
    mx = lw.max()
    tot = 0.0
    for j in range(N1):
        tot += np.exp(lw[j] - mx)
    u = np.random.random() * tot
    c = 0.0
    chosen = 0
    for j in range(N1):
        w = np.exp(lw[j] - mx)
        if w > 0:
            chosen = j
            c += w
            if c > u:
                break
    for j in range(N1):
        root_logw[j] = lw[j]
        for k in range(nflat):
            root_states[j, k] = CP[0, j, k]
    cc._load(i, ry, rx, rhos, certs, CP[0, chosen])
    ch = 0
    for k in range(nflat):
        if CP[0, chosen, k] != orig[k]:
            ch += 1
    info[0] = chosen
    info[1] = ch


# ------------------------------------------------------------------ python API
_ZB = np.zeros(1)


def UpPreds(nh=32):
    """The bottom-up predictor set: per (level 0..3, channel) MLP over NFU."""
    return cp.Preds(nh, nf=NFU)


def up_args(preds_up):
    """(ths 8-tuple indexed 2 l + ch, ons (8,), nh)."""
    if preds_up is None:
        return (_ZB,) * 8, np.zeros(8, np.int64), 1
    ths, ons, nh = [], np.zeros(8, np.int64), int(preds_up.nh)
    for l in range(NL - 1):
        for ch in range(2):
            th, _, on = preds_up.args(l, ch)
            ths.append(np.ascontiguousarray(th, np.float64))
            ons[2 * l + ch] = int(on)
    return tuple(ths), ons, nh


def dc_move(world, preds_up, i, ry, rx, N, rng, collect=None, _bias=None):
    """Conditional DC-SMC move of the subtree of (i, ry, rx): N fresh
    particles per node plus the reference.  Returns dict(logw (N+1,) root
    weights, chosen, ess, changed (flat entries), n (flat entries), dead
    (deaths per node: fresh merged particles killed, averaged over the
    nodes, so move_level's dead / (n K) is the per-merge death rate),
    dead_q / dead_x (per level l, counts: no admissible value / cross-
    children hard term), merges (per level: fresh merged particles),
    states (N+1, n) the root particles, flat).
    collect(world, i, ry, rx, states, wbar) is called with the world in the
    chosen state (states: the root particles as subtree states)."""
    rhos, certs, Aofs, Bofs = cc._tup(world)
    ths, ons, nh = _bias if _bias is not None else up_args(preds_up)
    n = cc.flat_size(i)
    N1 = int(N) + 1
    states = np.zeros((N1, n), np.int64)
    logw = np.full(N1, -np.inf)
    dq = np.zeros(NL - 1, np.int64)
    dx = np.zeros(NL - 1, np.int64)
    info = np.zeros(4, np.int64)
    _dc(i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam, world.beta, N1,
        ths, ons, nh, int(rng.integers(1 << 31)), states, logw, dq, dx, info)
    assert info[2] == 1, f"dc_move: current subtree of ({i}, {ry}, {rx}) is not valid"
    lw = logw - logw.max()
    wbar = np.exp(lw)
    wbar /= wbar.sum()
    merges = np.zeros(NL - 1, np.int64)
    nodes = 0
    for l in range(i, NL - 1):
        merges[l] = int(N) * (1 << (2 * (l - i)))
        nodes += 1 << (2 * (l - i))
    if collect is not None:
        collect(world, i, ry, rx, [cc.flat_to_state(i, s) for s in states], wbar)
    return dict(logw=logw, chosen=int(info[0]), ess=float(1.0 / (wbar ** 2).sum()), changed=int(info[1]), n=n,
                dead=float((dq.sum() + dx.sum()) / nodes), dead_q=dq, dead_x=dx, merges=merges, states=states)


def gamma(world, tl, ty, tx, ts, i, ry, rx):
    """(soft, violations) of E_T for the box (ty, tx, side ts) at level tl
    inside root (i, ry, rx) (see the module doc)."""
    rhos, certs, Aofs, Bofs = cc._tup(world)
    s, nv = _gamma(tl, ty, tx, ts, i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam,
                   world.beta)
    return float(s), int(nv)


def node_offsets(world, l, y, x, ch, i, ry, rx):
    """Designed offsets (D,) of node (l, y, x) channel ch inside root (i, ry, rx)."""
    rhos, certs, Aofs, Bofs = cc._tup(world)
    e = np.empty(257)
    D = _offsets(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam,
                 world.beta, e)
    return e[:D].copy()


def node_bias(world, preds_up, l, y, x, ch, i, ry, rx):
    """The bottom-up bias (D,) of node (l, y, x) channel ch (numba)."""
    rhos, certs, Aofs, Bofs = cc._tup(world)
    f = features_up(world, l, y, x, ch, i, ry, rx)
    th, nh, on = preds_up.args(l, ch)
    D = cp.domain_size(l, ch)
    out = np.zeros(D)
    if on:
        cp._mlp_n(f, np.ascontiguousarray(th), nh, D, NFU, out)
    return out


def features_up(world, l, y, x, ch, i, ry, rx):
    rhos, certs, Aofs, Bofs = cc._tup(world)
    f = np.zeros(NFU)
    up_features(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, f)
    return f


# ------------------------------------------------------------------ training the bottom-up predictor
@njit(cache=True)
def _record_up(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta, l, ch, feats, offs, vals):
    """Every node of level l of the subtree as it stands: features, offsets, value."""
    s = 1 << (l - i)
    Y0, X0 = ry * s, rx * s
    e = np.empty(257)
    k = 0
    for a in range(s):
        for b in range(s):
            y, x = Y0 + a, X0 + b
            up_features(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, feats[k])
            D = _offsets(l, y, x, ch, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta, e)
            for t in range(D):
                offs[k, t] = e[t]
            vals[k] = rhos[l][y, x] if ch == 0 else certs[l][y, x]
            k += 1


def record_up(world, i, ry, rx):
    """{(l, ch): (feats (n, NFU), offs (n, D), vals (n,))} of every node of
    the subtree of (i, ry, rx) in the world's current state: the bottom-up
    proposal's own context and offsets, the state's value.  (Under the
    proposal the cert context sees the node's rho: here the state's.)"""
    rhos, certs, Aofs, Bofs = cc._tup(world)
    out = {}
    for l in range(i, NL - 1):
        n = 1 << (2 * (l - i))
        for ch in (0, 1):
            D = cp.domain_size(l, ch)
            feats = np.zeros((n, NFU))
            offs = np.zeros((n, D))
            vals = np.zeros(n, np.int64)
            _record_up(i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam, world.beta,
                       l, ch, feats, offs, vals)
            out[(l, ch)] = (feats, offs, vals)
    return out


class UpCollector:
    """collect for dc_move / subtree_move: records record_up of the chain's
    state.  mode "chosen": the world's state when called (dc_move calls it
    after installing the chosen particle: one exact-chain state, weight 1);
    "weighted": every root particle with wbar_j > wmin, weight wbar_j
    (installs them; the world is left as found).  store: a
    certland_train.Examples or Archive (add(i, ch, feats, offs, val, w, upd))."""

    def __init__(self, store, upd, mode="chosen", wmin=0.01, levels=None):
        self.store, self.upd, self.mode, self.wmin, self.levels = store, upd, mode, wmin, levels
        self.time = 0.0

    def _add(self, rec, w):
        for (l, ch), (f, o, v) in rec.items():
            if self.levels is not None and l not in self.levels:
                continue
            assert np.isfinite(o[np.arange(len(v)), v]).all(), "UpCollector: a state value is excluded"
            self.store.add(l, ch, f.astype(np.float32), o.astype(np.float32), v, np.full(len(v), w),
                           np.full(len(v), self.upd, np.int64))

    def __call__(self, world, i, ry, rx, states, wbar):
        t0 = time.time()
        if self.mode == "chosen":
            self._add(record_up(world, i, ry, rx), 1.0)
        else:
            save = cc.get_subtree(world, i, ry, rx)
            for st, w in zip(states, wbar):
                if w > self.wmin:
                    cc.set_subtree(world, i, ry, rx, st)
                    self._add(record_up(world, i, ry, rx), float(w))
            cc.set_subtree(world, i, ry, rx, save)
        self.time += time.time() - t0


def fit_up(preds_up, fresh, archive, steps=50, lr=1e-3, batch=1024, archive_sample=8192, l2=1e-6, rng=None,
           seed=0):
    """Adam steps per (l, ch) on the fresh examples + an archive sample
    (certland_train.fit_steps / weighted_ce: -sum w log softmax(-(offs + b))[val]);
    Preds.set; the fresh examples join the archive.  Returns {(l, ch): dict}."""
    from . import certland_train as ct
    rng = np.random.default_rng(seed) if rng is None else rng
    log = {}
    for (l, ch) in fresh.keys():
        new = fresh.get(l, ch)
        old = archive.sample(l, ch, archive_sample, rng)
        pool = new if old is None else {k: np.concatenate([new[k].astype(old[k].dtype), old[k]]) for k in new}
        pool["feats"] = pool["feats"].astype(np.float64)
        pool["offs"] = pool["offs"].astype(np.float64)
        p0 = preds_up.params(l, ch)
        if p0 is None:
            p0 = cp.init_params(l, ch, preds_up.nh, seed + 101 * l + ch, NFU)
        pre = ct.eval_ce(p0, pool)
        p, st, fit = ct.fit_steps(p0, pool, steps, lr, archive.opt.get((l, ch)), batch, rng, l2)
        archive.opt[(l, ch)] = st
        preds_up.set(l, ch, p)
        log[(l, ch)] = dict(n_new=len(new["val"]), n_arch=0 if old is None else len(old["val"]), loss_pre=pre,
                            loss=ct.eval_ce(p, pool), loss0=ct.eval_ce(None, pool))
        archive.add(l, ch, new["feats"], new["offs"], new["val"], new["w"], new["upd"])
    return log


def move_level_dc(world, preds_up, i, N, rng, collect=None):
    """certland_chain.move_level's sweep (colours (y % 2, x % 2)) with
    dc_move, aggregating the per-level death counts.  Returns dict(ess,
    changed (fraction), dead (per merge), n, dead_q, dead_x, merges (per
    level), time)."""
    rows, cols = world.rho[i].shape
    bias = up_args(preds_up)
    ess = changed = tot = n = 0
    dq = np.zeros(NL - 1, np.int64)
    dx = np.zeros(NL - 1, np.int64)
    mg = np.zeros(NL - 1, np.int64)
    t0 = time.time()
    for cy in range(2):
        for cx in range(2):
            for ry in range(cy, rows, 2):
                for rx in range(cx, cols, 2):
                    r = dc_move(world, preds_up, i, ry, rx, N, rng, collect=collect, _bias=bias)
                    ess += r["ess"]
                    changed += r["changed"]
                    tot += r["n"]
                    dq += r["dead_q"]
                    dx += r["dead_x"]
                    mg += r["merges"]
                    n += 1
    return dict(ess=ess / max(n, 1), changed=changed / max(tot, 1), dead=float((dq.sum() + dx.sum()) / max(mg.sum(), 1)),
                n=n, dead_q=dq, dead_x=dx, merges=mg, time=time.time() - t0)
