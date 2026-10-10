"""Subtree moves on certland's designed p* (notes/certland_chain_spec.md).

A root cell (i, ry, rx), i <= 3, and its subtree (the root and every
descendant down to the tiles) is resampled by i-SIR: particle 0 is the
current subtree, particles 1..K are drawn by one ordered top-down pass
`q`, the weights are p*(z_j | outside) / q(z_j) and one particle is kept.
Exact for any K and any proposal (predictor), as long as q is a fixed
function of the cells outside the subtree.

The pass (`_pass`):
  * root level: the root's rho and cert are set to the placeholder 0 (so
    a bias reading the centre's other channel sees a fixed value), then rho
    then cert are drawn;
  * each lower level l: `_refine` writes deterministic placeholders for the
    whole level-l box from the final level l - 1 (cl.refine's packing, side
    left iff the parent column is even; rho split evenly, the remainder to
    the bottom row first, then top-left), then every cell is drawn in raster
    order (rows reversed for order="bottomup"), rho then cert;
  * tiles: per 2 x 2 block, from the exact 16-state conditional.
Soft designed terms (and the bias) are evaluated against the current values,
placeholders included. Hard rows (support above / below, row honour with the
sibling and the parent) are evaluated only against KNOWN cells: outside the
subtree, at a coarser level of the subtree, or already drawn at this level.
Each hard row is so enforced by whichever endpoint is drawn last and a
completed pass is valid. With L0 = inf three necessary single-cell bounds
from the parent's row honour are also applied to certificates
(B <= B_P, A >= A_P - h, B - A <= B_P - A_P). A cell with no admissible value
kills the pass (dead). Nothing in the pass reads a value of the subtree that
it has not written in this pass, so q does not depend on the current state.

Flat state layout (used internally and by `states_flat`): for l = i..3 the
level-l box's rho (s*s, row-major) then cert (s*s), s = 2^(l-i); then the
tiles (s*s)."""
import numpy as np
from numba import njit

from . import certland as cl
from . import certland_pred as cp

NL = cl.NL


# ------------------------------------------------------------------ geometry
def subtree_slices(i, ry, rx):
    """[(l, y0, y1, x0, x1)] for l = i..4 (rows y0..y1-1, columns x0..x1-1)."""
    out = []
    for l in range(i, NL):
        s = 1 << (l - i)
        out.append((l, ry * s, (ry + 1) * s, rx * s, (rx + 1) * s))
    return out


def flat_size(i):
    n = 0
    for l in range(i, NL):
        s = 1 << (l - i)
        n += s * s if l == NL - 1 else 2 * s * s
    return n


def get_subtree(world, i, ry, rx):
    """[(rho_block, cert_block)] for l = i..4 (copies; at l = 4 one array twice)."""
    out = []
    for l, y0, y1, x0, x1 in subtree_slices(i, ry, rx):
        r = world.rho[l][y0:y1, x0:x1].copy()
        c = r if l == NL - 1 else world.cert[l][y0:y1, x0:x1].copy()
        out.append((r, c))
    return out


def set_subtree(world, i, ry, rx, state):
    for (l, y0, y1, x0, x1), (r, c) in zip(subtree_slices(i, ry, rx), state):
        world.rho[l][y0:y1, x0:x1] = r
        if l < NL - 1:
            world.cert[l][y0:y1, x0:x1] = c


def flat_to_state(i, flat):
    out, o = [], 0
    for l in range(i, NL):
        s = 1 << (l - i)
        r = flat[o:o + s * s].reshape(s, s).copy()
        o += s * s
        if l == NL - 1:
            out.append((r, r))
        else:
            out.append((r, flat[o:o + s * s].reshape(s, s).copy()))
            o += s * s
    return out


def _tup(world):
    return tuple(world.rho), tuple(world.cert), tuple(world.Aof), tuple(world.Bof)


_ZB = np.zeros(1)


def _bias_args(preds):
    """(ths: 8-tuple indexed 2 i + ch, ons (9,), nh); ons[8] = 1 for a
    proposal-only predictor (cp.QPreds: input NFQ with the boundary)."""
    if preds is None:
        return (_ZB,) * 8, np.zeros(9, np.int64), 1
    ths, ons, nh = [], np.zeros(9, np.int64), 1
    ons[8] = int(getattr(preds, "proposal", False))
    for i in range(NL - 1):
        for ch in range(2):
            th, nh_, on = preds.args(i, ch)
            ths.append(np.ascontiguousarray(th, np.float64))
            ons[2 * i + ch] = int(on)
            nh = int(nh_)
    return tuple(ths), ons, nh


def _dargs(world, l):
    """world.args(l) with the learned part off (designed terms only)."""
    a = world.args(l)
    return a[:-3] + (np.zeros(1), 0, 0)


# ------------------------------------------------------------------ numba core
@njit(cache=True)
def _save(i, ry, rx, rhos, certs, out):
    o = 0
    for l in range(i, 5):
        s = 1 << (l - i)
        y0, x0 = ry * s, rx * s
        r = rhos[l]
        for a in range(s):
            for b in range(s):
                out[o] = r[y0 + a, x0 + b]
                o += 1
        if l < 4:
            c = certs[l]
            for a in range(s):
                for b in range(s):
                    out[o] = c[y0 + a, x0 + b]
                    o += 1


@njit(cache=True)
def _load(i, ry, rx, rhos, certs, src):
    o = 0
    for l in range(i, 5):
        s = 1 << (l - i)
        y0, x0 = ry * s, rx * s
        r = rhos[l]
        for a in range(s):
            for b in range(s):
                r[y0 + a, x0 + b] = src[o]
                o += 1
        if l < 4:
            c = certs[l]
            for a in range(s):
                for b in range(s):
                    c[y0 + a, x0 + b] = src[o]
                    o += 1


@njit(cache=True)
def _offset(i, l):
    """Flat offset of level l's block."""
    o = 0
    for k in range(i, l):
        s = 1 << (k - i)
        o += 2 * s * s
    return o


@njit(cache=True)
def _refine(l, i, ry, rx, rhos, certs, Aofs, Bofs):
    """Deterministic placeholders for level l of the subtree from level l - 1."""
    h = 16 >> l
    P = l - 1
    s = 1 << (P - i)
    py0, px0 = ry * s, rx * s
    prho, pcert, pA, pB = rhos[P], certs[P], Aofs[P], Bofs[P]
    rho, cert = rhos[l], certs[l]
    for py in range(py0, py0 + s):
        for px in range(px0, px0 + s):
            c = pcert[py, px]
            cA, cB = pA[c], pB[c]
            a_lo, a_hi = min(cA, h), max(0, cA - h)
            b_lo, b_hi = min(cB, h), max(0, cB - h)
            if px % 2 == 0:
                a0, a1, b0, b1 = a_lo, a_hi, b_lo, b_hi
            else:
                a0, a1, b0, b1 = a_hi, a_lo, b_hi, b_lo
            y, x = 2 * py, 2 * px
            if h == 1:
                rho[y, x] = a0
                rho[y, x + 1] = a1
                rho[y + 1, x] = b0
                rho[y + 1, x + 1] = b1
            else:
                cert[y, x] = b0 * (b0 + 1) // 2 + a0
                cert[y, x + 1] = b1 * (b1 + 1) // 2 + a1
                cert[y + 1, x] = b0 * (b0 + 1) // 2 + b0
                cert[y + 1, x + 1] = b1 * (b1 + 1) // 2 + b1
                q = prho[py, px] // 4
                rem = prho[py, px] - 4 * q
                hh = h * h
                rho[y + 1, x] = min(q + (1 if rem > 0 else 0), hh)
                rho[y + 1, x + 1] = min(q + (1 if rem > 1 else 0), hh)
                rho[y, x] = min(q + (1 if rem > 2 else 0), hh)
                rho[y, x + 1] = min(q, hh)


@njit(cache=True)
def _energy(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta):
    """(soft, violations) of every designed term touching the subtree, once."""
    soft = 0.0
    nv = 0
    for l in range(i, 5):
        h = 16 >> l
        s = 1 << (l - i)
        y0, x0 = ry * s, rx * s
        y1, x1 = y0 + s, x0 + s
        rho, cert, Aof, Bof = rhos[l], certs[l], Aofs[l], Bofs[l]
        rows, cols = rho.shape
        if l == 0:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    d = rho[y, x] - h * h * fill[y, x]
                    soft += beta * d * d / (h * h)
        else:
            prho, pcert, pA, pB = rhos[l - 1], certs[l - 1], Aofs[l - 1], Bofs[l - 1]
            for py in range(y0 // 2, (y1 + 1) // 2):
                for px in range(x0 // 2, (x1 + 1) // 2):
                    yy, xx = 2 * py, 2 * px
                    d = rho[yy, xx] + rho[yy, xx + 1] + rho[yy + 1, xx] + rho[yy + 1, xx + 1] - prho[py, px]
                    soft += kappa * d * d / (h * h)
                    AP, BP = pA[pcert[py, px]], pB[pcert[py, px]]
                    for y in range(max(yy, y0), min(yy + 2, y1)):
                        sa = Aof[cert[y, xx]] + Aof[cert[y, xx + 1]]
                        sb = Bof[cert[y, xx]] + Bof[cert[y, xx + 1]]
                        if sa < AP:
                            nv += 1
                        if sb > BP:
                            nv += 1
        for y in range(y0, y1):
            for x in range(x0 - 1, x1):
                soft += lam * abs(rho[y, x % cols] - rho[y, (x + 1) % cols]) / h
        for x in range(x0, x1):
            for y in range(y0 - 1, y1):
                if y >= 0 and y + 1 < rows and Bof[cert[y, x]] > Aof[cert[y + 1, x]]:
                    nv += 1
    return soft, nv


@njit(cache=True)
def _pick(e, D, forced, fval):
    """(index, log prob) of a draw from softmax(-e[:D]); (-1, 0) if none admissible."""
    mn = np.inf
    for t in range(D):
        if e[t] < mn:
            mn = e[t]
    if mn == np.inf:
        return -1, 0.0
    Z = 0.0
    for t in range(D):
        Z += np.exp(-(e[t] - mn))
    if forced:
        pick = fval
        if e[pick] == np.inf:
            return -1, 0.0
    else:
        u = np.random.random() * Z
        c = 0.0
        pick = -1
        for t in range(D):
            if e[t] < np.inf:
                pick = t
                c += np.exp(-(e[t] - mn))
                if c > u:
                    break
    return pick, -(e[pick] - mn) - np.log(Z)


@njit(cache=True, inline="always")
def _inbox(y, x, y0, y1, x0, x1):
    return y0 <= y < y1 and x0 <= x < x1


@njit(cache=True)
def _known_slots(ch, y, x, y0, y1, x0, x1, rows, cols, known, kn):
    """Known flags of the predictor's same-level context slots during the
    pass (cp.cell_features_k, layout widek): the 4 x 4 window over the
    cell's 2 x 2 block and one ring, row-major.  Known = off the grid,
    outside the subtree box, or drawn already; at the cell itself the slot
    carries its other channel (the cert placeholder while drawing rho, the
    drawn rho while drawing cert).  Other layouts ignore kn."""
    ya, xa = y - (y & 1) - 1, x - (x & 1) - 1
    for d in range(16):
        yy, xx = ya + d // 4, (xa + d % 4) % cols
        if yy == y and xx == x:
            kn[d] = 1 if ch == 1 else 0
        elif yy < 0 or yy >= rows or not _inbox(yy, xx, y0, y1, x0, x1):
            kn[d] = 1
        else:
            kn[d] = 1 if known[yy - y0, xx - x0] else 0


@njit
def _draw_level(l, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                forced, fbuf, L0, bottomup, ths, ons, nh, e, bb, known, rec, RF, RO, RV):
    """One level of the pass. rec: also write, per cell in draw order, the
    known-flag features (RF[2 l + ch]), the bias-free energies e (RO) and
    the drawn value (RV) -- the proposal's own context, for training."""
    h = 16 >> l
    s = 1 << (l - i)
    y0, x0 = ry * s, rx * s
    y1, x1 = y0 + s, x0 + s
    par = l - 1 if l > 0 else 0
    rho, cert, Aof, Bof = rhos[l], certs[l], Aofs[l], Bofs[l]
    prho, pcert, pA, pB = rhos[par], certs[par], Aofs[par], Bofs[par]
    hp = 1 if l > 0 else 0
    rows, cols = rho.shape
    Dr = h * h + 1
    Dc = Aof.shape[0]
    o = _offset(i, l)
    hard = L0 == np.inf
    for a in range(s):
        for b in range(s):
            known[a, b] = False
    kn = np.ones(16, np.int64)
    logq = 0.0
    nrec = 0
    for r in range(s):
        y = y1 - 1 - r if bottomup else y0 + r
        for x in range(x0, x1):
            k = (y - y0) * s + (x - x0)
            # ---- rho
            for t in range(Dr):
                e[t] = cl._designed_rho(l, h, y, x, t, rho, prho, fill, kappa, lam, beta)
            if ons[2 * l] or rec:          # (known flags only matter under layout widek)
                _known_slots(0, y, x, y0, y1, x0, x1, rows, cols, known, kn)
            if rec:
                cp.cell_features_k(h, 0, y, x, rho, cert, Aof, Bof, prho, pcert, pA, pB, hp, kn, RF[2 * l][nrec])
                if ons[8]:
                    cp.boundary_features(i, ry, rx, l, y, x, rhos, certs, Aofs, Bofs, RF[2 * l][nrec], cp.NF)
                for t in range(Dr):
                    RO[2 * l][nrec, t] = e[t]
            if ons[2 * l] and ons[8]:
                cp.cell_bias_q(h, 0, y, x, rho, cert, Aof, Bof, prho, pcert, pA, pB, hp, kn,
                               i, ry, rx, l, rhos, certs, Aofs, Bofs, ths[2 * l], nh, bb[:Dr])
                for t in range(Dr):
                    e[t] += bb[t]
            elif ons[2 * l]:
                cp.cell_bias_k(h, 0, y, x, rho, cert, Aof, Bof, prho, pcert, pA, pB, hp, kn,
                               ths[2 * l], nh, 1, bb[:Dr])
                for t in range(Dr):
                    e[t] += bb[t]
            pk, lp = _pick(e, Dr, forced, fbuf[o + k] if forced else 0)
            if pk < 0:
                return logq, True
            if rec:
                RV[2 * l][nrec] = pk
            rho[y, x] = pk
            logq += lp
            # ---- cert
            if ons[2 * l + 1] or rec:
                _known_slots(1, y, x, y0, y1, x0, x1, rows, cols, known, kn)
            if rec:
                cp.cell_features_k(h, 1, y, x, rho, cert, Aof, Bof, prho, pcert, pA, pB, hp, kn,
                                   RF[2 * l + 1][nrec])
                if ons[8]:
                    cp.boundary_features(i, ry, rx, l, y, x, rhos, certs, Aofs, Bofs, RF[2 * l + 1][nrec], cp.NF)
            if ons[2 * l + 1] and ons[8]:
                cp.cell_bias_q(h, 1, y, x, rho, cert, Aof, Bof, prho, pcert, pA, pB, hp, kn,
                               i, ry, rx, l, rhos, certs, Aofs, Bofs, ths[2 * l + 1], nh, bb[:Dc])
                for t in range(Dc):
                    e[t] = bb[t]
            elif ons[2 * l + 1]:
                cp.cell_bias_k(h, 1, y, x, rho, cert, Aof, Bof, prho, pcert, pA, pB, hp, kn,
                               ths[2 * l + 1], nh, 1, bb[:Dc])
                for t in range(Dc):
                    e[t] = bb[t]
            else:
                for t in range(Dc):
                    e[t] = 0.0
            kn_up = y > 0 and (not _inbox(y - 1, x, y0, y1, x0, x1) or known[y - 1 - y0, x - x0])
            kn_dn = y + 1 < rows and (not _inbox(y + 1, x, y0, y1, x0, x1) or known[y + 1 - y0, x - x0])
            xs = x ^ 1
            kn_sib = l > 0 and (not _inbox(y, xs, y0, y1, x0, x1) or known[y - y0, xs - x0])
            AP, BP = 0, 0
            if l > 0:
                AP, BP = pA[pcert[y // 2, x // 2]], pB[pcert[y // 2, x // 2]]
            for t in range(Dc):
                A, B = Aof[t], Bof[t]
                nv = 0
                if kn_up and Bof[cert[y - 1, x]] > A:
                    nv += 1
                if kn_dn and B > Aof[cert[y + 1, x]]:
                    nv += 1
                if kn_sib:
                    if A + Aof[cert[y, xs]] < AP:
                        nv += 1
                    if B + Bof[cert[y, xs]] > BP:
                        nv += 1
                if hard:
                    if nv > 0 or (l > 0 and (B > BP or A < AP - h or B - A > BP - AP)):
                        e[t] = np.inf
                else:
                    e[t] += L0 * nv
            if rec:
                for t in range(Dc):
                    RO[2 * l + 1][nrec, t] = e[t]       # bias-free when recording (ons off)
            pk, lp = _pick(e, Dc, forced, fbuf[o + s * s + k] if forced else 0)
            if pk < 0:
                return logq, True
            if rec:
                RV[2 * l + 1][nrec] = pk
                nrec += 1
            cert[y, x] = pk
            logq += lp
            known[y - y0, x - x0] = True
    return logq, False


@njit(cache=True)
def _draw_tiles(i, ry, rx, rhos, certs, Aofs, Bofs, kappa, lam,
                forced, fbuf, L0, bottomup, e, known):
    s = 1 << (4 - i)
    y0, x0 = ry * s, rx * s
    y1, x1 = y0 + s, x0 + s
    t = rhos[4]
    prho, pcert, pA, pB = rhos[3], certs[3], Aofs[3], Bofs[3]
    rows, cols = t.shape
    nb = s // 2
    o = _offset(i, 4)
    hard = L0 == np.inf
    for a in range(nb):
        for b in range(nb):
            known[a, b] = False
    logq = 0.0
    for r in range(nb):
        by = (y1 // 2) - 1 - r if bottomup else y0 // 2 + r
        yy = 2 * by
        for bx in range(x0 // 2, x1 // 2):
            xx = 2 * bx
            AP, BP = pA[pcert[by, bx]], pB[pcert[by, bx]]
            kn_up = yy > 0 and (yy - 1 < y0 or known[(yy - 1 - y0) // 2, bx - x0 // 2])
            kn_dn = yy + 2 < rows and (yy + 2 >= y1 or known[(yy + 2 - y0) // 2, bx - x0 // 2])
            lf = t[yy, (xx - 1) % cols], t[yy + 1, (xx - 1) % cols]
            rt = t[yy, (xx + 2) % cols], t[yy + 1, (xx + 2) % cols]
            for c in range(16):
                t00, t01, t10, t11 = c & 1, (c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1
                d = t00 + t01 + t10 + t11 - prho[by, bx]
                en = kappa * d * d
                en += lam * (abs(t00 - t01) + abs(t00 - lf[0]) + abs(t01 - rt[0])
                             + abs(t10 - t11) + abs(t10 - lf[1]) + abs(t11 - rt[1]))
                nv = 0
                if t00 + t01 < AP:
                    nv += 1
                if t00 + t01 > BP:
                    nv += 1
                if t10 + t11 < AP:
                    nv += 1
                if t10 + t11 > BP:
                    nv += 1
                if t00 > t10:
                    nv += 1
                if t01 > t11:
                    nv += 1
                if kn_up:
                    if t[yy - 1, xx] > t00:
                        nv += 1
                    if t[yy - 1, xx + 1] > t01:
                        nv += 1
                if kn_dn:
                    if t10 > t[yy + 2, xx]:
                        nv += 1
                    if t11 > t[yy + 2, xx + 1]:
                        nv += 1
                if nv > 0:
                    en = np.inf if hard else en + L0 * nv
                e[c] = en
            fv = 0
            if forced:
                k = (yy - y0) * s + (xx - x0)
                fv = fbuf[o + k] + 2 * fbuf[o + k + 1] + 4 * fbuf[o + k + s] + 8 * fbuf[o + k + s + 1]
            pk, lp = _pick(e, 16, forced, fv)
            if pk < 0:
                return logq, True
            t[yy, xx] = pk & 1
            t[yy, xx + 1] = (pk >> 1) & 1
            t[yy + 1, xx] = (pk >> 2) & 1
            t[yy + 1, xx + 1] = (pk >> 3) & 1
            logq += lp
            known[by - y0 // 2, bx - x0 // 2] = True
    return logq, False


@njit(cache=True)
def _dummy_rec():
    z2 = np.zeros((1, 1))
    z1 = np.zeros(1, np.int64)
    return (z2, z2, z2, z2, z2, z2, z2, z2), (z2, z2, z2, z2, z2, z2, z2, z2), (z1, z1, z1, z1, z1, z1, z1, z1)


@njit
def _pass(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
          forced, fbuf, L0, bottomup, ths, ons, nh, rec, RF, RO, RV):
    """One ordered top-down pass over the subtree, in place. forced: fbuf
    holds the flat values to force (read them with _save first). rec (with
    forced and ons all 0): record the pass's own contexts into RF / RO / RV."""
    e = np.empty(257)
    bb = np.empty(257)
    smax = 1 << (4 - i)
    known = np.zeros((smax, smax), np.bool_)
    rhos[i][ry, rx] = 0
    certs[i][ry, rx] = 0
    logq, dead = _draw_level(i, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                             forced, fbuf, L0, bottomup, ths, ons, nh, e, bb, known, rec, RF, RO, RV)
    if dead:
        return logq, True
    for l in range(i + 1, 5):
        _refine(l, i, ry, rx, rhos, certs, Aofs, Bofs)
        if l < 4:
            lq, dead = _draw_level(l, i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                                   forced, fbuf, L0, bottomup, ths, ons, nh, e, bb, known, rec, RF, RO, RV)
        else:
            lq, dead = _draw_tiles(i, ry, rx, rhos, certs, Aofs, Bofs, kappa, lam,
                                   forced, fbuf, L0, bottomup, e, known)
        logq += lq
        if dead:
            return logq, True
    return logq, False


@njit
def _pass_entry(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                forced, L0, bottomup, ths, ons, nh, seed, n):
    np.random.seed(seed)
    fbuf = np.zeros(n, np.int64)
    if forced:
        _save(i, ry, rx, rhos, certs, fbuf)
    RF, RO, RV = _dummy_rec()
    return _pass(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                 forced, fbuf, L0, bottomup, ths, ons, nh, False, RF, RO, RV)


@njit
def _move(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
          K, L0, bottomup, ths, ons, nh, seed, states, logw, info):
    """i-SIR move. states (K+1, n), logw (K+1,) out; info out:
    [chosen, dead, changed, violating completed proposals, particle-0 ok]."""
    np.random.seed(seed)
    RF, RO, RV = _dummy_rec()
    _save(i, ry, rx, rhos, certs, states[0])
    lq0, d0 = _pass(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                    True, states[0], L0, bottomup, ths, ons, nh, False, RF, RO, RV)
    s0, nv0 = _energy(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta)
    if d0 or nv0 > 0 or not np.isfinite(lq0):
        _load(i, ry, rx, rhos, certs, states[0])
        info[4] = 0
        return
    info[4] = 1
    logw[0] = -s0 - lq0
    dead = 0
    bad = 0
    for j in range(1, K + 1):
        lq, d = _pass(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                      False, states[0], L0, bottomup, ths, ons, nh, False, RF, RO, RV)
        _save(i, ry, rx, rhos, certs, states[j])
        if d:
            dead += 1
            logw[j] = -np.inf
            continue
        sj, nvj = _energy(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta)
        if nvj > 0:
            bad += 1
            logw[j] = -np.inf
        else:
            logw[j] = -sj - lq
    mx = logw.max()
    tot = 0.0
    for j in range(K + 1):
        tot += np.exp(logw[j] - mx)
    u = np.random.random() * tot
    c = 0.0
    chosen = 0
    for j in range(K + 1):
        w = np.exp(logw[j] - mx)
        if w > 0:
            chosen = j
            c += w
            if c > u:
                break
    _load(i, ry, rx, rhos, certs, states[chosen])
    ch = 0
    for k in range(states.shape[1]):
        if states[chosen, k] != states[0, k]:
            ch += 1
    info[0] = chosen
    info[1] = dead
    info[2] = ch
    info[3] = bad


@njit
def _record_entry(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta, fbuf, L0, bottomup, ths, ons, nh,
                  RF, RO, RV):
    return _pass(i, ry, rx, rhos, certs, Aofs, Bofs, fill, kappa, lam, beta,
                 True, fbuf, L0, bottomup, ths, ons, nh, True, RF, RO, RV)


def state_to_flat(i, state):
    parts = []
    for l, (r, c) in zip(range(i, NL), state):
        parts.append(np.asarray(r, np.int64).ravel())
        if l < NL - 1:
            parts.append(np.asarray(c, np.int64).ravel())
    return np.concatenate(parts)


def record_pass(world, i, ry, rx, state, L0=np.inf, order="raster", proposal=False):
    """The proposal's own training examples for `state` of the subtree of
    (i, ry, rx): a forced pass (bias off) through `state`, recording at
    every non-tile cell and channel, in draw order, the known-flag context
    features (cp.cell_features_k with the pass's known mask: placeholders
    flagged), the pass's bias-free energies (designed soft against the
    placeholders, +inf on values excluded by hard rows against KNOWN cells
    and the single-cell bounds) and the drawn value.  The world ends in
    `state` (callers restore it).  proposal: features of width cp.NFQ
    (cell_features_k then cp.boundary_features), for a proposal-only
    predictor.  Returns {(l, ch): (feats, offs, vals)}."""
    rhos, certs, Aofs, Bofs = _tup(world)
    ths, ons, nh = _bias_args(None)
    ons[8] = int(proposal)
    nf = cp.NFQ if proposal else cp.NF
    RF, RO, RV = [], [], []
    for l in range(NL - 1):
        for ch in range(2):
            if l >= i:
                n = 1 << (2 * (l - i))
                D = cl.HS[l] ** 2 + 1 if ch == cl.RHO else len(world.Aof[l])
                RF.append(np.zeros((n, nf)))
                RO.append(np.zeros((n, D)))
                RV.append(np.zeros(n, np.int64))
            else:
                RF.append(np.zeros((1, 1)))
                RO.append(np.zeros((1, 1)))
                RV.append(np.zeros(1, np.int64))
    lq, dead = _record_entry(i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam, world.beta,
                             state_to_flat(i, state), float(L0), order == "bottomup", ths, ons, nh,
                             tuple(RF), tuple(RO), tuple(RV))
    assert not dead, f"record_pass: state of root ({i}, {ry}, {rx}) has q = 0"
    return {(l, ch): (RF[2 * l + ch], RO[2 * l + ch], RV[2 * l + ch])
            for l in range(i, NL - 1) for ch in range(2)}


# ------------------------------------------------------------------ python API
def refine_block(world, l, i, ry, rx):
    """Deterministic placeholders for level l (> i) of the subtree from its level l - 1."""
    assert i < l < NL
    rhos, certs, Aofs, Bofs = _tup(world)
    _refine(l, i, ry, rx, rhos, certs, Aofs, Bofs)


def subtree_energy(world, i, ry, rx):
    """(soft, nviol) of every designed term touching the subtree, each once."""
    rhos, certs, Aofs, Bofs = _tup(world)
    s, nv = _energy(i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam, world.beta)
    return float(s), int(nv)


def cell_conditional(world, preds, l, ch, y, x, L=np.inf):
    """e (D,): designed soft of every value of channel ch at (y, x) at level l,
    + predictor bias (preds None: none; tiles: none), + L * violations
    against all current cells (L = inf: +inf on excluded values)."""
    h = cl.HS[l]
    tied = h == 1
    D = 2 if tied else (h * h + 1 if ch == cl.RHO else len(world.Aof[l]))
    soft, viol = np.zeros(D), np.zeros(D, np.int64)
    cl.site_terms(*_dargs(world, l), y, x, ch, soft, viol)
    e = soft.copy()
    if preds is not None and not tied:
        th, nh, on = preds.args(l, ch)
        if on:
            a = cp.level_args(world, l)
            b = np.zeros(D)
            cp.cell_bias(a[0], ch, y, x, *a[1:], np.ascontiguousarray(th, np.float64), nh, on, b)
            e += b
    if L == np.inf:
        e[viol > 0] = np.inf
    else:
        e += L * viol
    return e


def pass_subtree(world, preds, i, ry, rx, rng, forced=False, L0=np.inf, order="raster"):
    """One ordered top-down pass over the subtree, in place -> (logq, dead).
    forced: the outcomes are the current values (logq their probability)."""
    assert order in ("raster", "bottomup") and 0 <= i < NL - 1
    rhos, certs, Aofs, Bofs = _tup(world)
    ths, ons, nh = _bias_args(preds)
    lq, dead = _pass_entry(i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam,
                           world.beta, bool(forced), float(L0), order == "bottomup", ths, ons, nh,
                           int(rng.integers(1 << 31)), flat_size(i))
    return float(lq), bool(dead)


def subtree_move(world, preds, i, ry, rx, K, rng, collect=None, L0=np.inf, order="raster", _bias=None):
    """i-SIR on the subtree of (i, ry, rx): particle 0 the current state, K
    passes. Returns dict(logw (K+1,), chosen, dead, ess, changed (flat
    entries: cells x channels, tiles once), viol (completed proposals with a
    violation; 0 when L0 = inf), n (flat entries))."""
    rhos, certs, Aofs, Bofs = _tup(world)
    ths, ons, nh = _bias if _bias is not None else _bias_args(preds)
    n = flat_size(i)
    states = np.zeros((K + 1, n), np.int64)
    logw = np.full(K + 1, -np.inf)
    info = np.zeros(5, np.int64)
    _move(i, ry, rx, rhos, certs, Aofs, Bofs, world.fill, world.kappa, world.lam, world.beta,
          int(K), float(L0), order == "bottomup", ths, ons, nh, int(rng.integers(1 << 31)), states, logw, info)
    assert info[4] == 1, f"current subtree of ({i}, {ry}, {rx}) is not valid / has q = 0"
    lw = logw - logw.max()
    wbar = np.exp(lw)
    wbar /= wbar.sum()
    if collect is not None:
        chosen = flat_to_state(i, states[info[0]])
        collect(world, i, ry, rx, [flat_to_state(i, s) for s in states], wbar)
        set_subtree(world, i, ry, rx, chosen)
    return dict(logw=logw, chosen=int(info[0]), dead=int(info[1]), ess=float(1.0 / (wbar ** 2).sum()),
                changed=int(info[2]), viol=int(info[3]), n=n)


def move_level(world, preds, i, K, rng, collect=None, move=None):
    """Every root of level i, colours (y % 2, x % 2) in a fixed order.
    Returns dict(ess (mean), changed (fraction of subtree flat entries), dead
    (per proposal), n (roots))."""
    rows, cols = world.rho[i].shape
    ess, changed, dead, tot, n = 0.0, 0, 0, 0, 0
    kw = {}
    if move is None:
        move = subtree_move
        kw["_bias"] = _bias_args(preds)
    for cy in range(2):
        for cx in range(2):
            for ry in range(cy, rows, 2):
                for rx in range(cx, cols, 2):
                    r = move(world, preds, i, ry, rx, K, rng, collect=collect, **kw)
                    ess += r["ess"]
                    changed += r["changed"]
                    dead += r["dead"]
                    tot += r.get("n", flat_size(i))
                    n += 1
    return dict(ess=ess / max(n, 1), changed=changed / max(tot, 1), dead=dead / max(n * K, 1), n=n)


def tile_sweep(world, rng):
    """One exact designed Gibbs sweep over the tiles."""
    return int(cl.sweep_level(*_dargs(world, NL - 1), 1, int(rng.integers(1 << 30))))
