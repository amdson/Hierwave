"""Where and why certland_chain's ordered subtree proposals die (theta = 0).

A Python mirror of certland_chain's pass (`_pass`: root placeholder, raster
rho-then-cert draws with hard rows against KNOWN cells, deterministic
refine placeholders, exact 2 x 2 tile blocks) that records every draw and,
on death, the dead cell and the exclusion set of every constraint row.  The
mirror is checked against the module: forced-pass log q on every root of
the test worlds, and the dead rate (binomial z) on sampled passes.

Per death (hard rows only involve certificates and tiles, so feasibility
of a partial pass is an integer program over the subtree's undrawn (A, B)
and tiles with the outside fixed; scipy's HiGHS milp decides it):
  * dead cell: level, position in the subtree box, kind;
  * minimal unsatisfiable sets of the dead cell's constraint rows
    (up / down support, sibling row honour, single-cell bounds; tiles: row
    honour, internal support, up / down support), each tagged in/out
    (other end inside the subtree, drawn this pass, vs outside, fixed);
  * culprit: the first draw after which the partial pass has no valid
    completion (prefix scan); its level and box position;
  * k*: the coarsest level l such that fixing the drawn cells of levels
    <= l alone is infeasible.  k* = root level: COUNT mismatch (the root's
    own (A, B) has no completion).  k* > root: the split at level k* is
    wrong; POSITION if mirroring left/right the children of some parents at
    k* (same counts, different arrangement; binary swap variables) restores
    feasibility, else SPLIT (the magnitudes are wrong too);
  * boundary needed: is the dead partial pass still infeasible with the
    bottom (support with the fixed cells below the box), top, or root
    honour rows dropped / kept alone;
  * coarsest ancestor of the dead cell whose redraw (that one cell freed,
    the rest fixed) restores feasibility;
  * lookahead estimates: (exact-bottom) the culprit prefix is infeasible
    already with only internal rows + the bottom boundary (a draw that saw
    the fixed cells below the box at every finer level, exactly, would have
    rejected the culprit value); (strip) the culprit's own column strip
    (its footprint at every finer level, from its row down to the box
    bottom, plus everything drawn) is infeasible with the culprit value,
    i.e. a local lookahead down the culprit's columns rejects it; (parent
    strip) the same over the culprit's parent's columns;
  * the culprit's place relative to the dead cell (same parent: row
    sibling / above / diagonal; same column; other).
Lookahead simulation: passes whose every draw is filtered by an exact
feasibility oracle (internal rows only / + bottom boundary / parent strip /
all rows), dead rate on the same roots (rejection sampling from the
filtered conditional).
Annealed proposals (certland_anneal, M = 8, L0 = 3): the hard rows still
violated at z_{M-1} of the dead ones, by rule and position.

Usage: NUMBA_NUM_THREADS=1 python notes/experiments/certland_deaths.py [--quick]
Log: /Users/amdson/dev/Hierwave/images/certland_deaths_log.txt"""
import argparse
import itertools
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/amdson/dev/Hierwave/.claude/worktrees/certland")

import numpy as np                                                       # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp                # noqa: E402
from scipy.sparse import coo_matrix                                      # noqa: E402

from castlegen.channels import certland as cl                            # noqa: E402
from castlegen.channels import certland_chain as cc                      # noqa: E402
from castlegen.channels import certland_train as ct                      # noqa: E402

NL = cl.NL
LOGP = "/Users/amdson/dev/Hierwave/images/certland_deaths_log.txt"
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        LOGF.write(s + "\n")
        LOGF.flush()


# ------------------------------------------------------------------ geometry helpers
def box(i, ry, rx, l):
    s = 1 << (l - i)
    return ry * s, (ry + 1) * s, rx * s, (rx + 1) * s


def inbox(i, ry, rx, l, y, x):
    if l < i:
        return False
    y0, y1, x0, x1 = box(i, ry, rx, l)
    return y0 <= y < y1 and x0 <= x < x1


def pos_label(i, ry, rx, l, y, x):
    """Row and column position of (l, y, x) in the subtree box (tiles: of the 2x2 block)."""
    y0, y1, x0, x1 = box(i, ry, rx, l)
    st = 2 if l == NL - 1 else 1
    if y1 - y0 == st:
        r = "root"
    elif y < y0 + st:
        r = "top"
    elif y >= y1 - st:
        r = "bottom"
    else:
        r = "interior"
    if x1 - x0 == st:
        c = "root"
    elif x < x0 + st:
        c = "left"
    elif x >= x1 - st:
        c = "right"
    else:
        c = "interior"
    return r, c


# ------------------------------------------------------------------ the mirrored pass
def rho_energy(w, l, y, x):
    """cl._designed_rho over all t, vectorised (same operation order)."""
    h = cl.HS[l]
    rho = w.rho[l]
    rows, cols = rho.shape
    t = np.arange(h * h + 1, dtype=np.float64)
    e = np.zeros(h * h + 1)
    if l > 0:
        py, px = y // 2, x // 2
        S = int(rho[2 * py:2 * py + 2, 2 * px:2 * px + 2].sum()) - int(rho[y, x])
        d = t + S - w.rho[l - 1][py, px]
        e = e + w.kappa * d * d / (h * h)
    e = e + w.lam * (np.abs(t - rho[y, (x - 1) % cols]) + np.abs(t - rho[y, (x + 1) % cols])) / h
    if l == 0:
        d = t - h * h * w.fill[y, x]
        e = e + w.beta * d * d / (h * h)
    return e


def pick(e, rng, forced, fval):
    """certland_chain._pick: (index, log prob) or (-1, 0)."""
    fin = np.isfinite(e)
    if not fin.any():
        return -1, 0.0
    mn = e[fin].min()
    p = np.where(fin, np.exp(-(e - mn)), 0.0)
    Z = p.sum()
    if forced:
        k = fval
        if not fin[k]:
            return -1, 0.0
    else:
        k = int(rng.choice(len(e), p=p / Z))
    return k, -(e[k] - mn) - np.log(Z)


def refine(w, l, i, ry, rx):
    """certland_chain._refine (placeholders for level l of the box from level l - 1)."""
    h = cl.HS[l]
    P = l - 1
    py0, py1, px0, px1 = box(i, ry, rx, P)
    prho, pcert, pA, pB = w.rho[P], w.cert[P], w.Aof[P], w.Bof[P]
    rho, cert = w.rho[l], w.cert[l]
    for py in range(py0, py1):
        for px in range(px0, px1):
            c = pcert[py, px]
            cA, cB = int(pA[c]), int(pB[c])
            a_lo, a_hi = min(cA, h), max(0, cA - h)
            b_lo, b_hi = min(cB, h), max(0, cB - h)
            if px % 2 == 0:
                a0, a1, b0, b1 = a_lo, a_hi, b_lo, b_hi
            else:
                a0, a1, b0, b1 = a_hi, a_lo, b_hi, b_lo
            y, x = 2 * py, 2 * px
            if h == 1:
                rho[y, x], rho[y, x + 1], rho[y + 1, x], rho[y + 1, x + 1] = a0, a1, b0, b1
            else:
                cert[y, x] = b0 * (b0 + 1) // 2 + a0
                cert[y, x + 1] = b1 * (b1 + 1) // 2 + a1
                cert[y + 1, x] = b0 * (b0 + 1) // 2 + b0
                cert[y + 1, x + 1] = b1 * (b1 + 1) // 2 + b1
                q, rem = divmod(int(prho[py, px]), 4)
                hh = h * h
                rho[y + 1, x] = min(q + (1 if rem > 0 else 0), hh)
                rho[y + 1, x + 1] = min(q + (1 if rem > 1 else 0), hh)
                rho[y, x] = min(q + (1 if rem > 2 else 0), hh)
                rho[y, x + 1] = min(q, hh)


CERT_RULES = ("up", "dn", "sibA", "sibB", "bndB", "bndA", "bndBA")
TILE_RULES = ("honA0", "honB0", "honA1", "honB1", "int0", "int1", "up0", "up1", "dn0", "dn1")
TS = np.arange(16)
T00, T01, T10, T11 = TS & 1, (TS >> 1) & 1, (TS >> 2) & 1, (TS >> 3) & 1


def cert_masks(w, i, ry, rx, l, y, x, known):
    """Per rule, the boolean exclusion mask over the cert domain, plus info."""
    h = cl.HS[l]
    cert, A, B = w.cert[l], w.Aof[l], w.Bof[l]
    rows = cert.shape[0]
    y0, y1, x0, x1 = box(i, ry, rx, l)
    kn = lambda yy, xx: not (y0 <= yy < y1 and x0 <= xx < x1) or known[yy - y0, xx - x0]
    kn_up = y > 0 and kn(y - 1, x)
    kn_dn = y + 1 < rows and kn(y + 1, x)
    xs = x ^ 1
    kn_sib = l > 0 and kn(y, xs)
    AP = BP = 0
    if l > 0:
        pc = w.cert[l - 1][y // 2, x // 2]
        AP, BP = int(w.Aof[l - 1][pc]), int(w.Bof[l - 1][pc])
    z = np.zeros(len(A), bool)
    m = {}
    m["up"] = (w.Bof[l][cert[y - 1, x]] > A) if kn_up else z
    m["dn"] = (B > w.Aof[l][cert[y + 1, x]]) if kn_dn else z
    m["sibA"] = (A + A[cert[y, xs]] < AP) if kn_sib else z
    m["sibB"] = (B + B[cert[y, xs]] > BP) if kn_sib else z
    m["bndB"] = (B > BP) if l > 0 else z
    m["bndA"] = (A < AP - h) if l > 0 else z
    m["bndBA"] = (B - A > BP - AP) if l > 0 else z
    tag = dict(up="out" if (kn_up and not inbox(i, ry, rx, l, y - 1, x)) else "in",
               dn="out" if (kn_dn and not inbox(i, ry, rx, l, y + 1, x)) else "in",
               sib="out" if (kn_sib and not inbox(i, ry, rx, l, y, xs)) else "in")
    info = dict(AP=AP, BP=BP, h=h,
                up=(int(w.Aof[l][cert[y - 1, x]]), int(w.Bof[l][cert[y - 1, x]])) if kn_up else None,
                dn=(int(w.Aof[l][cert[y + 1, x]]), int(w.Bof[l][cert[y + 1, x]])) if kn_dn else None,
                sib=(int(A[cert[y, xs]]), int(B[cert[y, xs]])) if kn_sib else None, tag=tag)
    return m, info


def tile_masks(w, i, ry, rx, by, bx, known):
    t = w.rho[NL - 1]
    rows = t.shape[0]
    y0, y1, x0, x1 = box(i, ry, rx, NL - 1)
    yy, xx = 2 * by, 2 * bx
    pc = w.cert[NL - 2][by, bx]
    AP, BP = int(w.Aof[NL - 2][pc]), int(w.Bof[NL - 2][pc])
    kn_up = yy > 0 and (yy - 1 < y0 or known[(yy - 1 - y0) // 2, bx - x0 // 2])
    kn_dn = yy + 2 < rows and (yy + 2 >= y1 or known[(yy + 2 - y0) // 2, bx - x0 // 2])
    z = np.zeros(16, bool)
    m = dict(honA0=T00 + T01 < AP, honB0=T00 + T01 > BP, honA1=T10 + T11 < AP, honB1=T10 + T11 > BP,
             int0=T00 > T10, int1=T01 > T11,
             up0=(t[yy - 1, xx] > T00) if kn_up else z, up1=(t[yy - 1, xx + 1] > T01) if kn_up else z,
             dn0=(T10 > t[yy + 2, xx]) if kn_dn else z, dn1=(T11 > t[yy + 2, xx + 1]) if kn_dn else z)
    tag = dict(up="out" if (kn_up and yy - 1 < y0) else "in", dn="out" if (kn_dn and yy + 2 >= y1) else "in")
    info = dict(AP=AP, BP=BP, up=(int(t[yy - 1, xx]), int(t[yy - 1, xx + 1])) if kn_up else None,
                dn=(int(t[yy + 2, xx]), int(t[yy + 2, xx + 1])) if kn_dn else None, tag=tag)
    return m, info


def tile_soft(w, by, bx):
    t = w.rho[NL - 1]
    cols = t.shape[1]
    yy, xx = 2 * by, 2 * bx
    prho = w.rho[NL - 2][by, bx]
    lf = t[yy, (xx - 1) % cols], t[yy + 1, (xx - 1) % cols]
    rt = t[yy, (xx + 2) % cols], t[yy + 1, (xx + 2) % cols]
    d = T00 + T01 + T10 + T11 - prho
    en = w.kappa * d * d
    en = en + w.lam * (np.abs(T00 - T01) + np.abs(T00 - lf[0]) + np.abs(T01 - rt[0])
                       + np.abs(T10 - T11) + np.abs(T10 - lf[1]) + np.abs(T11 - rt[1]))
    return en.astype(np.float64)


def opick(e, rng, test):
    """A draw from softmax(-e) restricted to the values passing test (rejection)."""
    e = e.copy()
    while True:
        k, _ = pick(e, rng, False, 0)
        if k < 0 or test(k):
            return k
        e[k] = np.inf


def mirror_pass(w, i, ry, rx, rng, forced=False, oracle=None):
    """The pass in place -> (logq, death or None, draws).  draws: [(l, y, x, value)]
    in order (tiles: (4, by, bx, c)); death: dict of the dead cell.
    oracle(draws) -> bool: if given (sampling only), every certificate / tile
    draw is restricted to values for which oracle(draws + [this draw]) holds
    (the lookahead simulation; logq is then meaningless)."""
    fv = cc.get_subtree(w, i, ry, rx) if forced else None
    draws = []
    w.rho[i][ry, rx] = 0
    w.cert[i][ry, rx] = 0
    logq = 0.0
    for l in range(i, NL):
        if l > i:
            refine(w, l, i, ry, rx)
        y0, y1, x0, x1 = box(i, ry, rx, l)
        s = y1 - y0
        if l < NL - 1:
            known = np.zeros((s, s), bool)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    k, lp = pick(rho_energy(w, l, y, x), rng, forced, fv[l - i][0][y - y0, x - x0] if forced else 0)
                    if k < 0:
                        return logq, dict(l=l, y=y, x=x, kind="rho"), draws
                    w.rho[l][y, x] = k
                    logq += lp
                    m, info = cert_masks(w, i, ry, rx, l, y, x, known)
                    ex = np.zeros(len(w.Aof[l]), bool)
                    for v in m.values():
                        ex |= v
                    if oracle is not None:
                        k, lp = opick(np.where(ex, np.inf, 0.0), rng, lambda v: oracle(draws + [(l, y, x, v)])), 0.0
                    else:
                        k, lp = pick(np.where(ex, np.inf, 0.0), rng, forced, fv[l - i][1][y - y0, x - x0] if forced else 0)
                    if k < 0:
                        return logq, dict(l=l, y=y, x=x, kind="cert", masks=m, info=info), draws
                    w.cert[l][y, x] = k
                    draws.append((l, y, x, k))
                    logq += lp
                    known[y - y0, x - x0] = True
        else:
            nb = s // 2
            known = np.zeros((nb, nb), bool)
            t = w.rho[l]
            for by in range(y0 // 2, y1 // 2):
                for bx in range(x0 // 2, x1 // 2):
                    m, info = tile_masks(w, i, ry, rx, by, bx, known)
                    ex = np.zeros(16, bool)
                    for v in m.values():
                        ex |= v
                    e = np.where(ex, np.inf, tile_soft(w, by, bx))
                    fvv = 0
                    if forced:
                        b = fv[l - i][0][2 * by - y0:2 * by - y0 + 2, 2 * bx - x0:2 * bx - x0 + 2]
                        fvv = int(b[0, 0] + 2 * b[0, 1] + 4 * b[1, 0] + 8 * b[1, 1])
                    if oracle is not None:
                        k, lp = opick(e, rng, lambda v: oracle(draws + [(l, by, bx, v)])), 0.0
                    else:
                        k, lp = pick(e, rng, forced, fvv)
                    if k < 0:
                        return logq, dict(l=l, y=by, x=bx, kind="tile", masks=m, info=info), draws
                    t[2 * by, 2 * bx], t[2 * by, 2 * bx + 1] = k & 1, (k >> 1) & 1
                    t[2 * by + 1, 2 * bx], t[2 * by + 1, 2 * bx + 1] = (k >> 2) & 1, (k >> 3) & 1
                    draws.append((l, by, bx, k))
                    logq += lp
                    known[by - y0 // 2, bx - x0 // 2] = True
    return logq, None, draws


def muses(masks, n):
    """Minimal subsets of the active rules whose exclusions cover all n values."""
    act = [k for k, v in masks.items() if v.any()]
    out = []
    for r in range(1, len(act) + 1):
        for S in itertools.combinations(act, r):
            if any(set(T) <= set(S) for T in out):
                continue
            ex = np.zeros(n, bool)
            for k in S:
                ex |= masks[k]
            if ex.all():
                out.append(S)
    return out


def rule_tag(rule, tag):
    base = rule.rstrip("01")
    if base in ("up", "dn"):
        return f"{base}({tag[base]})"
    if base in ("sibA", "sibB"):
        return f"{base}({tag['sib']})"
    return base


# ------------------------------------------------------------------ feasibility (MILP)
def cellvals(w, l, y, x):
    if l == NL - 1:
        t = int(w.rho[l][y, x])
        return t, t
    c = w.cert[l][y, x]
    return int(w.Aof[l][c]), int(w.Bof[l][c])


def known_from_draws(w, draws, maxlev=NL):
    """{(l, y, x): (A, B)} of the drawn cells (tiles expanded) with level <= maxlev."""
    kn = {}
    for (l, y, x, v) in draws:
        if l > maxlev:
            continue
        if l == NL - 1:
            for a, b, bit in ((0, 0, 1), (0, 1, 2), (1, 0, 4), (1, 1, 8)):
                t = 1 if v & bit else 0
                kn[(l, 2 * y + a, 2 * x + b)] = (t, t)
        else:
            kn[(l, y, x)] = (int(w.Aof[l][v]), int(w.Bof[l][v]))
    return kn


def feasible(w, i, ry, rx, known, free=None, drop=(), swaps=()):
    """Is there an assignment of the free cells (default: every undrawn
    cell of the subtree) satisfying every hard row whose cells are all
    outside / known / free (rows touching an 'absent' cell are dropped; so
    are the boundary groups in drop: 'top', 'bottom', 'roothon')?
    swaps: parents (l, py, px) whose four (known) children may be mirrored
    left/right as a whole (one binary each)."""
    known = dict(known)
    swapped = {}
    for (lp, py, px) in swaps:
        for a in (0, 1):
            for b in (0, 1):
                c = (lp + 1, 2 * py + a, 2 * px + b)
                swapped[c] = (lp, py, px, (lp + 1, 2 * py + a, 2 * px + (1 - b)))
    var = {}
    nv = 0
    lb, ub = [], []
    integ = []
    cells = []
    for l in range(i, NL):
        y0, y1, x0, x1 = box(i, ry, rx, l)
        for y in range(y0, y1):
            for x in range(x0, x1):
                c = (l, y, x)
                if (c in known and c not in swapped) or (free is not None and c not in free and c not in swapped):
                    continue
                h = cl.HS[l]
                if l == NL - 1:
                    var[c] = (nv, nv)
                    nv += 1
                    lb.append(0)
                    ub.append(1)
                else:
                    var[c] = (nv, nv + 1)
                    nv += 2
                    lb += [0, 0]
                    ub += [h, h]
                cells.append(c)
    svar = {}
    for p in swaps:
        svar[p] = nv
        nv += 1
        lb.append(0)
        ub.append(1)
    rows, rlo, rhi = [], [], []

    def term(l, y, x, k):                       # k 0: A, 1: B
        c = (l, y, x)
        if c in var:
            return ("v", var[c][k])
        if not inbox(i, ry, rx, l, y, x):
            return ("c", cellvals(w, l, y, x)[k])
        if c in known:
            return ("c", known[c][k])
        return None

    def add(terms, lo, hi):
        """lo <= sum coef * term <= hi."""
        d = defaultdict(float)
        const = 0.0
        for coef, t in terms:
            if t is None:
                return True
            if t[0] == "c":
                const += coef * t[1]
            else:
                d[t[1]] += coef
        if not d:
            return lo - 1e-9 <= const <= hi + 1e-9
        rows.append(d)
        rlo.append(lo - const)
        rhi.append(hi - const)
        return True

    ok = True
    for c in cells:
        if c[0] < NL - 1:
            ok &= add([(1, ("v", var[c][0])), (-1, ("v", var[c][1]))], -np.inf, 0)
    for c, (lp, py, px, partner) in swapped.items():
        a0, b0 = known[c]
        a1, b1 = known[partner]
        s = svar[(lp, py, px)]
        rows.append({var[c][0]: 1.0, s: -(a1 - a0)})
        rlo.append(a0)
        rhi.append(a0)
        if c[0] < NL - 1:
            rows.append({var[c][1]: 1.0, s: -(b1 - b0)})
            rlo.append(b0)
            rhi.append(b0)
    for l in range(i, NL):
        y0, y1, x0, x1 = box(i, ry, rx, l)
        nrows = w.rho[l].shape[0]
        for x in range(x0, x1):
            for y in range(y0 - 1, y1):
                if y < 0 or y + 1 >= nrows:
                    continue
                g = "top" if y == y0 - 1 else ("bottom" if y == y1 - 1 else "int")
                if g in drop:
                    continue
                ok &= add([(1, term(l, y, x, 1)), (-1, term(l, y + 1, x, 0))], -np.inf, 0)
        if l == 0:
            continue
        g = "roothon" if l == i else "int"
        if g in drop:
            continue
        for py in range(y0 // 2, (y1 + 1) // 2):
            for px in range(x0 // 2, (x1 + 1) // 2):
                PA, PB = term(l - 1, py, px, 0), term(l - 1, py, px, 1)
                for y in range(max(2 * py, y0), min(2 * py + 2, y1)):
                    tA = [(1, term(l, y, 2 * px, 0)), (1, term(l, y, 2 * px + 1, 0)), (-1, PA)]
                    tB = [(1, term(l, y, 2 * px, 1)), (1, term(l, y, 2 * px + 1, 1)), (-1, PB)]
                    ok &= add(tA, 0, np.inf)
                    ok &= add(tB, -np.inf, 0)
    if not ok:
        return False
    if not rows:
        return True
    r, cidx, val = [], [], []
    for k, d in enumerate(rows):
        for j, v in d.items():
            r.append(k)
            cidx.append(j)
            val.append(v)
    M = coo_matrix((val, (r, cidx)), shape=(len(rows), nv)).tocsr()
    res = milp(np.zeros(nv), constraints=LinearConstraint(M, rlo, rhi), integrality=np.ones(nv),
               bounds=Bounds(lb, ub), options=dict(presolve=True))
    if res.status == 0:
        return True
    if res.status == 2:
        return False
    raise RuntimeError(f"milp status {res.status}: {res.message}")


# ------------------------------------------------------------------ death analysis
def parent_strip(i, ry, rx, l, y, x, kn):
    """Undrawn cells of the column strip under the parent of (l, y, x) (a
    tile block's parent is (3, by, bx)): the parent's footprint at every
    level >= l, from the parent's first row down to the box bottom.  None
    for the root (no parent in the subtree)."""
    if l == i:
        return None
    pl = l - 1
    py_, px_ = (y // 2, x // 2) if l < NL - 1 else (y, x)
    out = set()
    for L in range(l, NL):
        f = 1 << (L - pl)
        for yy in range(py_ * f, box(i, ry, rx, L)[1]):
            for xx in range(px_ * f, (px_ + 1) * f):
                if (L, yy, xx) not in kn:
                    out.add((L, yy, xx))
    return out


def relation(c, d):
    """Where the culprit cell c sits relative to the dead cell d (same level)."""
    if c[0] != d[0]:
        return f"coarser (h{cl.HS[c[0]]})"
    l, cy, cx = c
    _, dy, dx = d
    if l < NL - 1 and (cy // 2, cx // 2) == (dy // 2, dx // 2):
        return "same parent: " + ("row sibling" if cy == dy else ("above" if cx == dx else "diagonal"))
    if cx == dx:
        return "same column, other parent"
    return "other"


def analyse_death(w, i, ry, rx, death, draws):
    """w holds the state at death (outside fixed, drawn cells, placeholders)."""
    l, y, x = death["l"], death["y"], death["x"]
    kind = death["kind"]
    n = 16 if kind == "tile" else len(w.Aof[l])
    out = dict(i=i, l=l, y=y, x=x, kind=kind, info=death["info"])
    out["pos"] = pos_label(i, ry, rx, l, 2 * y if kind == "tile" else y, 2 * x if kind == "tile" else x)
    mu = muses(death["masks"], n)
    tag = death["info"]["tag"]
    mus_t = sorted({"+".join(sorted({rule_tag(r, tag) for r in S})) for S in mu}, key=len)
    out["mus"] = mus_t
    out["mus_min"] = mus_t[0] if mus_t else "?"
    # drop-one: which single rules, removed, make a value admissible
    act = [k for k, v in death["masks"].items() if v.any()]
    crit = []
    for k in act:
        ex = np.zeros(n, bool)
        for k2 in act:
            if k2 != k:
                ex |= death["masks"][k2]
        if not ex.all():
            crit.append(rule_tag(k, tag))
    out["critical"] = sorted(set(crit))
    kn_all = known_from_draws(w, draws)
    assert not feasible(w, i, ry, rx, kn_all), "dead partial pass should be infeasible"
    # culprit: first prefix with no completion
    lo, hi = 0, len(draws)                          # prefix lo feasible, hi infeasible
    assert feasible(w, i, ry, rx, {})
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if feasible(w, i, ry, rx, known_from_draws(w, draws[:mid])):
            lo = mid
        else:
            hi = mid
    cl_, cy, cx, cv = draws[hi - 1]
    out["culprit"] = (cl_, cy, cx)
    out["culprit_k"] = hi
    out["n_draws"] = len(draws)
    out["culprit_pos"] = pos_label(i, ry, rx, cl_, 2 * cy if cl_ == NL - 1 else cy, 2 * cx if cl_ == NL - 1 else cx)
    out["culprit_val"] = cv if cl_ == NL - 1 else (int(w.Aof[cl_][cv]), int(w.Bof[cl_][cv]))
    out["culprit_same"] = (cl_, cy, cx) == (l, y, x)
    out["culprit_rel"] = relation((cl_, cy, cx), (l, y, x))
    # k*: coarsest level whose drawn cells (with all coarser) are infeasible
    ks = None
    for L in range(i, NL):
        if not feasible(w, i, ry, rx, known_from_draws(w, draws, L)):
            ks = L
            break
    out["kstar"] = ks
    if ks == i:
        out["mismatch"] = "COUNT"
    else:
        kn = known_from_draws(w, draws, ks)
        pars = []
        P = ks - 1
        py0, py1, px0, px1 = box(i, ry, rx, P)
        for py in range(py0, py1):
            for px in range(px0, px1):
                ch = [(ks, 2 * py + a, 2 * px + b) for a in (0, 1) for b in (0, 1)]
                if all(c in kn for c in ch):
                    pars.append((P, py, px))
        if pars and feasible(w, i, ry, rx, kn, swaps=pars):
            out["mismatch"] = "POSITION-mirror"
        else:
            out["mismatch"] = "POSITION-resplit"
    # boundary rows needed
    b = {}
    for g in ("bottom", "top", "roothon"):
        b["no_" + g] = feasible(w, i, ry, rx, kn_all, drop=(g,))
    b["bottom_only_infeas"] = not feasible(w, i, ry, rx, kn_all, drop=("top", "roothon"))
    b["internal_only_infeas"] = not feasible(w, i, ry, rx, kn_all, drop=("top", "roothon", "bottom"))
    out["bnd"] = b
    # coarsest ancestor of the dead cell whose redraw restores feasibility
    anc = None
    ay, ax = y, x
    chain = []
    for L in range(l - 1, i - 1, -1):
        ay, ax = ay // 2, ax // 2
        chain.append((L, ay, ax))
    for c in reversed(chain):                       # coarsest first
        kn = dict(kn_all)
        kn.pop(c, None)
        if feasible(w, i, ry, rx, kn):
            anc = c[0]
            break
    out["ancestor"] = anc
    # lookahead estimates on the culprit prefix
    knc = known_from_draws(w, draws[:hi])
    out["la_exact_bottom"] = not feasible(w, i, ry, rx, knc, drop=("top", "roothon"))
    strip = set()
    for L in range(cl_, NL):
        f = 1 << (L - cl_)
        cy_, cx_ = (2 * cy if cl_ == NL - 1 else cy), (2 * cx if cl_ == NL - 1 else cx)
        if cl_ == NL - 1:
            ys, xs_ = range(cy_, box(i, ry, rx, L)[1]), range(cx_, cx_ + 2)
        else:
            ys = range(cy_ * f, box(i, ry, rx, L)[1])
            xs_ = range(cx_ * f, (cx_ + 1) * f)
        for yy in ys:
            for xx in xs_:
                if (L, yy, xx) not in knc:
                    strip.add((L, yy, xx))
    out["la_internal"] = not feasible(w, i, ry, rx, knc, drop=("top", "roothon", "bottom"))
    out["la_exact_top"] = not feasible(w, i, ry, rx, knc, drop=("bottom",))
    pstrip = parent_strip(i, ry, rx, cl_, cy, cx, knc)
    if pstrip is not None:
        out["la_pstrip"] = not feasible(w, i, ry, rx, knc, free=pstrip)
        out["la_pstrip_nobottom"] = not feasible(w, i, ry, rx, knc, free=pstrip, drop=("bottom",))
    else:
        out["la_pstrip"] = out["la_pstrip_nobottom"] = False
    out["la_strip"] = not feasible(w, i, ry, rx, knc, free=strip)
    out["la_strip_nobottom"] = not feasible(w, i, ry, rx, knc, free=strip, drop=("bottom",))
    return out


# ------------------------------------------------------------------ annealed survivors
def anneal_violations(w, i, ry, rx):
    """Every violated hard row touching the subtree: (level, rule, row position, col position)."""
    out = []
    for l in range(i, NL):
        y0, y1, x0, x1 = box(i, ry, rx, l)
        cert, Aof, Bof = w.cert[l], w.Aof[l], w.Bof[l]
        nrows = cert.shape[0]
        if l > 0:
            pc, pA, pB = w.cert[l - 1], w.Aof[l - 1], w.Bof[l - 1]
            for py in range(y0 // 2, (y1 + 1) // 2):
                for px in range(x0 // 2, (x1 + 1) // 2):
                    AP, BP = pA[pc[py, px]], pB[pc[py, px]]
                    for y in range(max(2 * py, y0), min(2 * py + 2, y1)):
                        sa = Aof[cert[y, 2 * px]] + Aof[cert[y, 2 * px + 1]]
                        sb = Bof[cert[y, 2 * px]] + Bof[cert[y, 2 * px + 1]]
                        p = pos_label(i, ry, rx, l, y, 2 * px)
                        pre = "roothon" if l == i else "hon"
                        if sa < AP:
                            out.append((l, pre + "A", p[0], p[1]))
                        if sb > BP:
                            out.append((l, pre + "B", p[0], p[1]))
        for x in range(x0, x1):
            for y in range(y0 - 1, y1):
                if y >= 0 and y + 1 < nrows and Bof[cert[y, x]] > Aof[cert[y + 1, x]]:
                    g = "sup-topedge" if y == y0 - 1 else ("sup-bottomedge" if y == y1 - 1 else "sup-int")
                    p = pos_label(i, ry, rx, l, max(y, y0), x)
                    out.append((l, g, p[0], p[1]))
    return out


# ------------------------------------------------------------------ worlds
def make_world(H, W, noise, seed, rounds):
    rng = np.random.default_rng(seed)
    w = cl.World(H, W)
    cl.set_fill(w, cl.density_field(H, W, noise))
    for k in range(20):
        cl.generate(w, sweeps=20, seed=int(rng.integers(1 << 30)))
        v = ct.count_violations(w)
        if sum(v) == 0:
            break
    else:
        raise RuntimeError("no valid generation")
    st = []
    for _ in range(rounds):
        for i in range(NL - 1):
            st.append(cc.move_level(w, None, i, 2, rng)["dead"])
        cc.tile_sweep(w, rng)
    assert sum(ct.count_violations(w)) == 0
    return w, k, st


# ------------------------------------------------------------------ main
def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--P", type=int, default=4, help="proposals per root")
    ap.add_argument("--maxroots", type=int, default=128)
    ap.add_argument("--maxdeaths", type=int, default=250, help="analysed deaths per (world, level)")
    ap.add_argument("--anneal", type=int, default=24, help="annealed proposals per level (h=16, h=8)")
    ap.add_argument("--sim", type=int, default=24, help="lookahead-simulation passes per level and variant")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(LOGP), exist_ok=True)
    LOGF = open(LOGP, "w")
    t00 = time.time()
    sizes = [(64, 128)] if a.quick else [(64, 128), (128, 256)]
    log(f"certland_deaths: theta = 0 (preds None), L0 = inf, order raster; P = {a.P} proposals per root, "
        f"<= {a.maxroots} roots per level, <= {a.maxdeaths} deaths analysed per (world, level)")
    rows = []
    examples = []
    for wi, (H, W) in enumerate(sizes):
        t0 = time.time()
        w, tries, st = make_world(H, W, noise=11 + wi, seed=100 + wi, rounds=3)
        log(f"\nworld {H}x{W} noise {11 + wi}: generate tries {tries + 1}, equilibration dead per level-round "
            f"{np.round(st, 2).tolist()}  ({time.time() - t0:.1f}s)")
        rng = np.random.default_rng(7 + wi)
        # ---- verification of the mirror: forced log q on every root, dead rate
        nchk, maxd = 0, 0.0
        for i in range(NL - 1):
            nr, ncl = w.rho[i].shape
            roots = [(y, x) for y in range(nr) for x in range(ncl)]
            if len(roots) > 64:
                roots = [roots[k] for k in rng.choice(len(roots), 64, replace=False)]
            for (ry, rx) in roots:
                s0 = cc.get_subtree(w, i, ry, rx)
                lq_m, dd = cc.pass_subtree(w, None, i, ry, rx, rng, forced=True)
                cc.set_subtree(w, i, ry, rx, s0)
                lq_p, death, _ = mirror_pass(w, i, ry, rx, rng, forced=True)
                cc.set_subtree(w, i, ry, rx, s0)
                assert not dd and death is None
                maxd = max(maxd, abs(lq_m - lq_p))
                nchk += 1
        log(f"mirror check: forced log q on {nchk} roots (all levels), max |module - mirror| = {maxd:.2e}")
        for i in range(NL - 1):
            nr, ncl = w.rho[i].shape
            roots = [(y, x) for y in range(nr) for x in range(ncl)]
            if len(roots) > 48:
                roots = [roots[k] for k in rng.choice(len(roots), 48, replace=False)]
            dm = dp = n = 0
            for (ry, rx) in roots:
                s0 = cc.get_subtree(w, i, ry, rx)
                for _ in range(6):
                    _, d = cc.pass_subtree(w, None, i, ry, rx, rng)
                    cc.set_subtree(w, i, ry, rx, s0)
                    _, death, _ = mirror_pass(w, i, ry, rx, rng)
                    cc.set_subtree(w, i, ry, rx, s0)
                    dm += d
                    dp += death is not None
                    n += 1
            pm, pp = dm / n, dp / n
            se = np.sqrt(max(pm * (1 - pm) + pp * (1 - pp), 1e-12) / n)
            log(f"mirror check h={cl.HS[i]:2d}: dead module {pm:.3f} mirror {pp:.3f} (n {n} each, z {(pm - pp) / se:+.2f})")
        # ---- deaths
        for i in (0, 1, 2):
            t1 = time.time()
            nr, ncl = w.rho[i].shape
            roots = [(y, x) for y in range(nr) for x in range(ncl)]
            if len(roots) > a.maxroots:
                roots = [roots[k] for k in rng.choice(len(roots), a.maxroots, replace=False)]
            nprop = ndead = 0
            res = []
            for (ry, rx) in roots:
                s0 = cc.get_subtree(w, i, ry, rx)
                for _ in range(a.P):
                    _, death, draws = mirror_pass(w, i, ry, rx, rng)
                    nprop += 1
                    if death is not None:
                        ndead += 1
                        if len(res) < a.maxdeaths:
                            r = analyse_death(w, i, ry, rx, death, draws)
                            r["root"] = (ry, rx)
                            r["world"] = f"{H}x{W}"
                            res.append(r)
                    cc.set_subtree(w, i, ry, rx, s0)
            rows.append(dict(world=f"{H}x{W}", i=i, nprop=nprop, ndead=ndead, res=res))
            log(f"h={cl.HS[i]:2d} roots {len(roots)}: {ndead}/{nprop} dead ({ndead / nprop:.3f}), "
                f"{len(res)} analysed ({time.time() - t1:.1f}s)")
            examples += res[:3]
        anneal_block(w, rng, a.anneal, f"{H}x{W}")
        if a.sim:
            sim_block(w, rng, a.sim, f"{H}x{W}")
    report(rows, examples)
    log(f"\ntotal {time.time() - t00:.0f}s")


ANN = []
SIM = []


def sim_block(w, rng, n, wname, levels=(0, 1)):
    """Lookahead simulation: passes whose every certificate / tile draw is
    restricted to values with a completion under (a) internal rows only,
    (b) internal + bottom boundary (the bottom-aware lookahead), (c) all
    rows (sanity: never dies), (pstrip) the local lookahead: the column
    strip under the drawn cell's parent, down to the box bottom, everything
    known included (root draws: the whole subtree; pstrip-noroot: the root
    draw unfiltered).  Dead rate per variant
    on the same roots."""
    for i in levels:
        t1 = time.time()
        nr, ncl = w.rho[i].shape
        roots = [(y, x) for y in range(nr) for x in range(ncl)]
        sel = [roots[k] for k in rng.choice(len(roots), min(n, len(roots)), replace=False)]
        out = {}
        for name, drop in (("plain", None), ("internal", ("top", "roothon", "bottom")),
                           ("bottom", ("top", "roothon")), ("pstrip", "pstrip"), ("pstrip-noroot", "pstrip0"),
                           ("full", ())):
            nd = 0
            for (ry, rx) in sel:
                s0 = cc.get_subtree(w, i, ry, rx)
                orc = None
                if drop in ("pstrip", "pstrip0"):
                    def orc(dr, ry=ry, rx=rx, drop=drop):
                        kn = known_from_draws(w, dr)
                        l_, y_, x_, _ = dr[-1]
                        if l_ == i and drop == "pstrip0":
                            return True                  # root draw: no lookahead at all
                        return feasible(w, i, ry, rx, kn, free=parent_strip(i, ry, rx, l_, y_, x_, kn))
                elif drop is not None:
                    orc = lambda dr, drop=drop, ry=ry, rx=rx: feasible(w, i, ry, rx, known_from_draws(w, dr), drop=drop)
                _, death, _ = mirror_pass(w, i, ry, rx, rng, oracle=orc)
                nd += death is not None
                cc.set_subtree(w, i, ry, rx, s0)
            out[name] = nd
        SIM.append(dict(world=wname, i=i, n=len(sel), **out))
        log(f"lookahead sim h={cl.HS[i]:2d} {wname}: dead of {len(sel)} passes " +
            " ".join(f"{k} {v}" for k, v in out.items()) + f"  ({time.time() - t1:.1f}s)")


def anneal_block(w, rng, n, wname):
    from castlegen.channels import certland_anneal as ca
    for i in (0, 1):
        t1 = time.time()
        nr, ncl = w.rho[i].shape
        roots = [(y, x) for y in range(nr) for x in range(ncl)]
        sel = [roots[k] for k in rng.choice(len(roots), min(n, len(roots)), replace=False)]
        nd = 0
        viols = []
        for (ry, rx) in sel:
            s0 = cc.get_subtree(w, i, ry, rx)
            info = {}
            lw, dead = ca.annealed_forward(w, None, i, ry, rx, 8, 3.0, rng, info)
            if dead:
                nd += 1
                v = anneal_violations(w, i, ry, rx)
                viols.append(v)
            cc.set_subtree(w, i, ry, rx, s0)
        ANN.append(dict(world=wname, i=i, n=len(sel), dead=nd, viols=viols))
        log(f"anneal h={cl.HS[i]:2d} {wname}: {nd}/{len(sel)} end with violations ({time.time() - t1:.1f}s)")


def pct(c, n):
    return f"{100 * c / max(n, 1):5.1f}%"


def report(rows, examples):
    log("\n================ deaths by root level (both worlds pooled) ================")
    for i in (0, 1, 2):
        rr = [r for r in rows if r["i"] == i]
        nprop = sum(r["nprop"] for r in rr)
        ndead = sum(r["ndead"] for r in rr)
        res = [x for r in rr for x in r["res"]]
        n = len(res)
        log(f"\n--- root h={cl.HS[i]}: dead {ndead}/{nprop} = {ndead / max(nprop, 1):.3f}; {n} deaths analysed")
        if not n:
            continue
        c = Counter((cl.HS[x["l"]], x["kind"]) for x in res)
        log("  dead cell level/kind:   " + "  ".join(f"h{k[0]} {k[1]}: {pct(v, n)}" for k, v in sorted(c.items())))
        c = Counter(x["pos"][0] for x in res)
        log("  dead cell row position: " + "  ".join(f"{k}: {pct(v, n)}" for k, v in c.most_common()))
        c = Counter(x["pos"][1] for x in res)
        log("  dead cell col position: " + "  ".join(f"{k}: {pct(v, n)}" for k, v in c.most_common()))
        c = Counter(x["mus_min"] for x in res)
        log("  smallest minimal unsat set of the dead cell's rows:")
        for k, v in c.most_common():
            log(f"      {k:<28s} {pct(v, n)}")
        c = Counter("+".join(x["critical"]) or "(none single)" for x in res)
        log("  drop-one critical rules (each alone, removed, frees a value):")
        for k, v in c.most_common(8):
            log(f"      {k:<28s} {pct(v, n)}")
        c = Counter(x["mismatch"] for x in res)
        log("  mismatch: " + "  ".join(f"{k}: {pct(v, n)}" for k, v in c.most_common()))
        c = Counter(cl.HS[x["kstar"]] for x in res)
        log("  k* (coarsest infeasible fixed level) h: " + "  ".join(f"h{k}: {pct(v, n)}" for k, v in sorted(c.items(), reverse=True)))
        c = Counter(cl.HS[x["culprit"][0]] for x in res)
        log("  culprit draw level h:    " + "  ".join(f"h{k}: {pct(v, n)}" for k, v in sorted(c.items(), reverse=True)))
        c = Counter(x["culprit_rel"] for x in res)
        log("  culprit relative to the dead cell: " + "  ".join(f"{k}: {pct(v, n)}" for k, v in c.most_common()))
        c = Counter(x["culprit_pos"][0] for x in res)
        log("  culprit row position:    " + "  ".join(f"{k}: {pct(v, n)}" for k, v in c.most_common()))
        log(f"  culprit is the dead cell itself: {pct(sum(x['culprit_same'] for x in res), n)};  "
            f"mean draws before culprit / before death: {np.mean([x['culprit_k'] for x in res]):.1f} / "
            f"{np.mean([x['n_draws'] for x in res]):.1f}")
        c = Counter("none" if x["ancestor"] is None else f"h{cl.HS[x['ancestor']]}" for x in res)
        log("  coarsest ancestor whose redraw restores feasibility: " + "  ".join(f"{k}: {pct(v, n)}" for k, v in c.most_common()))
        b = lambda k: sum(x["bnd"][k] for x in res)
        log(f"  boundary rows: feasible without bottom {pct(b('no_bottom'), n)}, without top {pct(b('no_top'), n)}, "
            f"without root honour {pct(b('no_roothon'), n)}; infeasible with bottom only {pct(b('bottom_only_infeas'), n)}, "
            f"with internal rows only {pct(b('internal_only_infeas'), n)}")
        log(f"  lookahead: culprit rejected by exact bottom-aware check {pct(sum(x['la_exact_bottom'] for x in res), n)}; "
            f"by the culprit's column-strip check {pct(sum(x['la_strip'] for x in res), n)} "
            f"(strip without the bottom rows {pct(sum(x['la_strip_nobottom'] for x in res), n)}); "
            f"by the culprit's PARENT column-strip check {pct(sum(x['la_pstrip'] for x in res), n)} "
            f"(without the bottom rows {pct(sum(x['la_pstrip_nobottom'] for x in res), n)})")
        q = lambda f: pct(sum(1 for x in res if f(x)), n)
        log(f"  culprit prefix infeasible with: internal rows only {q(lambda x: x['la_internal'])}; "
            f"+ bottom (no top) {q(lambda x: x['la_exact_bottom'])}; + top (no bottom) {q(lambda x: x['la_exact_top'])}; "
            f"needs bottom (bottom-aware yes, internal no) {q(lambda x: x['la_exact_bottom'] and not x['la_internal'])}; "
            f"needs top/root honour only {q(lambda x: not x['la_exact_bottom'])}")
    log("\n================ annealed proposals (M = 8, L0 = 3): violations left at z_{M-1} ================")
    for i in (0, 1):
        aa = [x for x in ANN if x["i"] == i]
        n = sum(x["n"] for x in aa)
        d = sum(x["dead"] for x in aa)
        v = [t for x in aa for vv in x["viols"] for t in vv]
        log(f"--- root h={cl.HS[i]}: {d}/{n} dead; {len(v)} violated rows, mean {len(v) / max(d, 1):.1f} per dead proposal")
        c = Counter((cl.HS[t[0]], t[1]) for t in v)
        log("  by (level, rule): " + "  ".join(f"h{k[0]} {k[1]}: {pct(c_, len(v))}" for k, c_ in sorted(c.items(), key=lambda z: -z[1])))
        c = Counter(t[2] for t in v)
        log("  by row position:  " + "  ".join(f"{k}: {pct(c_, len(v))}" for k, c_ in c.most_common()))
    log("\n================ lookahead simulation (dead / passes, same roots) ================")
    for i in (0, 1):
        aa = [x for x in SIM if x["i"] == i]
        if aa:
            n = sum(x["n"] for x in aa)
            log(f"  root h={cl.HS[i]}: " + "  ".join(f"{k} {sum(x[k] for x in aa)}/{n}" for k in ("plain", "internal", "bottom", "pstrip", "pstrip-noroot", "full")))
    log("\n================ examples ================")
    for x in examples:
        inf = x["info"]
        log(f"[{x['world']} root h={cl.HS[x['i']]} {x['root']}] dead {x['kind']} at h={cl.HS[x['l']]} ({x['y']},{x['x']}) "
            f"pos {x['pos']}: parent (A,B)=({inf['AP']},{inf['BP']}) up={inf['up']} dn={inf['dn']} "
            f"sib={inf.get('sib')} tags={inf['tag']}; MUS {x['mus']}; culprit h={cl.HS[x['culprit'][0]]} "
            f"{x['culprit'][1:]} pos {x['culprit_pos']} value {x['culprit_val']}; k* h={cl.HS[x['kstar']]} "
            f"{x['mismatch']}; ancestor {x['ancestor']}; bnd {x['bnd']}; la_bottom {x['la_exact_bottom']} strip {x['la_strip']}")


if __name__ == "__main__":
    main()
