"""Reference energy of a sample: soft energy and hard violations per level.

FROZEN.  Plain numpy, written for clarity, not speed.  An instance exists only
when every cell it reads lies in the region (or is read through a pad); on a
periodic strip x wraps.  Hard instances are the instances of factors whose
table has an inf entry; an instance is violated when its entry is inf.

    S(x)   = sum of finite entries over all instances
    H_l(x) = number of violated hard instances homed at level l
    N_l(x) = number of hard instances homed at level l
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problem import Problem, Region


@dataclass
class Energy:
    soft: float
    viol: dict = field(default_factory=dict)      # level -> violated hard instances
    nhard: dict = field(default_factory=dict)     # level -> hard instances
    per_factor: dict = field(default_factory=dict)  # name -> (soft, violated, instances)

    def frac(self, l):
        return self.viol.get(l, 0) / max(self.nhard.get(l, 0), 1)


def _view(problem, x, cv):
    c, v = cv
    return problem.chan[c].views[v][np.asarray(x[c])]


def _gather(arr, Y, X, region, l, pad):
    """arr (h, w) at level l read at (Y, X); returns (values, ok).  Rows off the
    strip read pad (ok only if pad >= 0); x wraps or is cut by the window."""
    h, w = arr.shape
    ok = np.ones(Y.shape, bool)
    if region.periodic:
        Xr = np.mod(X, w)
    else:
        ok &= (X >= 0) & (X < w)
        Xr = np.clip(X, 0, w - 1)
    offrow = (Y < 0) | (Y >= h)
    Yr = np.clip(Y, 0, h - 1)
    val = arr[Yr, Xr]
    if pad >= 0:
        val = np.where(offrow, pad, val)
    else:
        ok &= ~offrow
    return val, ok


def instances(problem: Problem, f, x, region: Region):
    """Entries of factor f's table at every instance in the region (1-D)."""
    ca = problem.chan[f.a[0]]
    la = ca.level
    A = _view(problem, x, f.a)
    if f.kind == "unary":
        return f.table[A].ravel()
    B = _view(problem, x, f.b)
    lb = problem.chan[f.b[0]].level
    if f.kind == "count":
        h, w = B.shape
        s = np.zeros((h, w), np.int64)
        for k in f.subset:
            s += A[(k >> 1)::2, (k & 1)::2]
        return f.table[B, s].ravel()
    # pair: enumerate from a's cells, plus cells of b whose partner a is off the strip
    dy, dx = f.off
    h, w = A.shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    out = []
    if lb == la:
        Yb, Xb = Y + dy, X + dx
    else:
        Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
    vb, ok = _gather(B, Yb, Xb, region, lb, f.pad_b)
    if f.within is not None:
        L = f.within
        ok &= ((Y << la) >> L == (Yb << lb) >> L) & ((X << la) >> L == (Xb << lb) >> L)
    if f.by_pos:
        e = f.table[2 * (Y & 1) + (X & 1), A, vb]
    else:
        e = f.table[A, vb]
    out.append(e[ok])
    if f.pad_a >= 0 and lb == la and dy != 0:
        # b cells whose a = q - off is off the strip: rows of b near the edge
        hb, wb = B.shape
        Yq, Xq = np.meshgrid(np.arange(hb), np.arange(wb), indexing="ij")
        Ya, Xa = Yq - dy, Xq - dx
        offa = (Ya < 0) | (Ya >= h)
        if not region.periodic:
            offa &= (Xa >= 0) & (Xa < w)
        if f.within is not None:
            L = f.within
            offa &= ((Ya << la) >> L == (Yq << lb) >> L) & ((Xa << la) >> L == (Xq << lb) >> L)
        e2 = f.table[f.pad_a, B]
        out.append(e2[offa])
    return np.concatenate(out)


def energy(problem: Problem, x, region: Region) -> Energy:
    problem.check_sample(x, region)
    E = Energy(0.0)
    for l in range(problem.T + 1):
        E.viol[l], E.nhard[l] = 0, 0
    for i, f in enumerate(problem.factors):
        e = instances(problem, f, x, region)
        inf = np.isinf(e)
        s = float(e[~inf].sum())
        v = int(inf.sum())
        n = int(e.size) if f.hard else 0
        l = problem.chan[f.a[0]].level
        E.soft += s
        E.viol[l] += v
        E.nhard[l] += n
        E.per_factor[f.name or f"f{i}"] = (s, v, int(e.size))
    return E
