"""Graded (smooth) violation measures (ideas.md #21).  Not frozen; a diagnostic
companion to castlegen.bench.energy that reads a Problem's declarations only.

For every hard factor instance (a factor whose table has an inf entry) this
module gives an integer distance to feasibility d_f(x) >= 0 with

    d_f(x) = 0  <=>  the instance is satisfied (its table entry is finite),

and per level the "graded violation mass" D_l(x) = sum of d_f over the hard
instances homed at level l, next to energy.py's binary count H_l(x).

Construction (generic, from the tables; no per-problem code)
------------------------------------------------------------
Every instance reads a few cells (its scope) through views; the table is a
boolean feasibility pattern F over the view values (and, for count factors,
over the children's sum).  d_f is the smallest total movement that lands the
instance on a feasible entry,

    d_f = min_{feasible entry e'} sum_axes rho_axis(e_axis, e'_axis),

where the per-axis movement rho is

  * |v - v'| on an ORDERED axis: the axis is a sum (count factors), or the
    table's feasible sets along it are intervals whose endpoints move
    monotonically with the other axis (a "monotone band": <=, >=, =, a band
    between two thresholds).  This is exactly the idea's "distance in view
    value to the nearest feasible partner value" / "distance from the sum to
    the nearest finite entry";
  * 1[v != v'] on an UNORDERED axis (any other pattern, e.g. dimers'
    partner directions, columns' positions): the number of cells of the scope
    that must change (edit distance);
  * infinity unless v' = v on a FIXED cell: a designed channel, or a pad read
    for a row off the strip.  A fixed cell cannot move, so the other side
    carries the whole distance.

Being a minimum of a sum over a set that contains the instance itself when it
is feasible, d_f = 0 exactly on feasible instances, d_f is an integer, and for
a monotone band with one movable side it equals the one-sided view distance.
The band test is a property of the table, so relabelling an unordered domain
cannot turn it into a numeric one unless the relabelled table really is a
threshold pattern (with two values both measures coincide anyway).

What it gives on the suite
--------------------------
ground   support_l (B(p) <= A(below)):  d = B - A, the occupied columns of p
         without a full column under them; at level 0 one per floating tile.
         honourA_l (sum A_children >= A_parent): d = shortfall of full columns;
         honourB_l (sum B_children <= B_parent): d = excess of occupied columns.
toys     rho1 = sum t, rho2 = sum rho1, rho2 = 8: d = |sum - count|, |rho2 - 8|.
columns  n_l = sum (hard counts): d = |sum - n|.  The 'column' pair over the
         position k is not a band (k = 0 may sit above 0 or 12), so d = 1 per
         violated instance: graded = binary there.
dimers   every hard factor is over unordered directions, and every value has a
         feasible partner, so d = 1 per violated instance: graded = binary.
         (A world-level edit distance would need a global matching, which a
         per-factor measure does not see.)

API (stable; other variants import these)
----------------------------------------------
    instance_distances(problem, state, region) -> {factor name: (d, loc)}
        per-instance integer distances (energy.instances' order) and the cells each instance reads
    level_mass(problem, state, region) -> int array (T + 1,)     D_l per level
    graded_field(problem, state, region, level) -> float array (H >> level, W >> level)
        per cell, the summed d of the hard instances (any level) touching it (see its docstring)
    graded(problem, state, region) -> Graded   (viol, nhard, mass per level; per_factor name ->
        (viol, mass, n); inst name -> d array; hist())
    factor_distance(problem, f) -> FactorDistance   (the precomputed tables)
`state` holds every channel including the designed ones (as the bench's _with_designed gives).
numpy only, evaluated on the host; no jax twin yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from castlegen.bench.problem import Problem, Region

INF = np.inf


# ------------------------------------------------------------------ band test
def _is_interval(mask):
    idx = np.flatnonzero(mask)
    return idx.size == 0 or idx[-1] - idx[0] + 1 == idx.size


def _monotone(v):
    v = np.asarray(v)
    return v.size <= 1 or bool((np.diff(v) >= 0).all() or (np.diff(v) <= 0).all())


def axis_is_band(F, axis):
    """Is `axis` of the boolean feasibility pattern F (2-D) a monotone band?
    For every value of the other axis the feasible set along `axis` must be a
    contiguous interval, and the interval endpoints (over the non-empty slices,
    in order of the other axis) must each be monotone.  A pattern with no
    feasible entry is not a band."""
    F = np.asarray(F, bool)
    M = F if axis == 0 else F.T                 # M[v, o]: v along the tested axis, o the other
    if not M.any():
        return False
    lo, hi = [], []
    for o in range(M.shape[1]):
        col = M[:, o]
        if not _is_interval(col):
            return False
        idx = np.flatnonzero(col)
        if idx.size:
            lo.append(idx[0])
            hi.append(idx[-1])
    return _monotone(lo) and _monotone(hi)


def _rho(n, ordered):
    """(n, n) per-axis movement cost between two view values."""
    v = np.arange(n)
    if ordered:
        return np.abs(v[:, None] - v[None, :]).astype(float)
    return (v[:, None] != v[None, :]).astype(float)


def _l1_table(F, ordered_a, ordered_b, move_a=True, move_b=True):
    """d[a, b] = min over feasible (a', b') of rho_a(a, a') + rho_b(b, b'), where a
    side that cannot move has rho = inf off the diagonal.  F is (Va, Vb) bool."""
    F = np.asarray(F, bool)
    Va, Vb = F.shape
    ra = _rho(Va, ordered_a) if move_a else np.where(np.eye(Va, dtype=bool), 0.0, INF)
    rb = _rho(Vb, ordered_b) if move_b else np.where(np.eye(Vb, dtype=bool), 0.0, INF)
    # g[a, b'] = min over feasible a' in column b' of rho_a(a, a')
    g = np.full((Va, Vb), INF)
    for bp in range(Vb):
        idx = np.flatnonzero(F[:, bp])
        if idx.size:
            g[:, bp] = ra[:, idx].min(1)
    # d[a, b] = min over b' of g[a, b'] + rho_b(b, b')
    d = np.full((Va, Vb), INF)
    for b in range(Vb):
        d[:, b] = (g + rb[b][None, :]).min(1)
    return d


@dataclass
class FactorDistance:
    """Distance tables of one hard factor (see the module docstring)."""
    kind: str
    ordered: tuple                       # per axis: is it treated as ordered
    d: np.ndarray                        # both sides movable (count: parent movable)
    d_a: np.ndarray | None = None        # only side a movable (pair) / only the children (count)
    d_b: np.ndarray | None = None        # only side b movable (pair) / only the parent (count)

    def __post_init__(self):
        for t in (self.d, self.d_a, self.d_b):
            if t is not None:
                assert np.isfinite(t).all(), "a hard table with no feasible completion"
                assert (t >= 0).all() and (t == np.rint(t)).all()


def _pattern_tables(F, ordered_a, ordered_b):
    return (_l1_table(F, ordered_a, ordered_b, True, True),
            _l1_table(F, ordered_a, ordered_b, True, False),
            _l1_table(F, ordered_a, ordered_b, False, True))


def factor_distance(problem: Problem, f) -> FactorDistance:
    """Precompute d for every entry of a hard factor's table."""
    assert f.hard, f.name
    F = np.isfinite(f.table)
    if f.kind == "unary":
        ordered = _is_interval(F)
        v = np.arange(F.size)
        idx = np.flatnonzero(F)
        assert idx.size, "a hard unary with no feasible value"
        d = np.abs(v[:, None] - idx[None, :]).min(1).astype(float) if ordered else (~F).astype(float)
        return FactorDistance("unary", (ordered,), d)
    if f.kind == "count":
        # axis 0 the parent's view (band test), axis 1 the children's sum (always ordered)
        ob = axis_is_band(F, 0)
        d, d_b, d_a = _pattern_tables(F, ob, True)       # d_a: parent fixed (children move), d_b: children fixed
        return FactorDistance("count", (ob, True), d, d_a, d_b)
    # pair
    if f.by_pos:
        slices = [F[k] for k in range(4)]
        oa = all(axis_is_band(S, 0) for S in slices)
        ob = all(axis_is_band(S, 1) for S in slices)
        tabs = [_pattern_tables(S, oa, ob) for S in slices]
        d, d_a, d_b = (np.stack([t[i] for t in tabs]) for i in range(3))
    else:
        oa, ob = axis_is_band(F, 0), axis_is_band(F, 1)
        d, d_a, d_b = _pattern_tables(F, oa, ob)
    return FactorDistance("pair", (oa, ob), d, d_a, d_b)


# ---------------------------------------------------------------- instances
def _view(problem, x, cv):
    c, v = cv
    return problem.chan[c].views[v][np.asarray(x[c])]


def _gather(arr, Y, X, region, pad):
    """As energy._gather, also returning which reads were pad reads."""
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
    return val, ok, offrow & ok


def _factor_instances(problem: Problem, f, fd: FactorDistance, x, region: Region):
    """(entries, d, loc) over factor f's instances in the region, in energy.instances' order:
    entries are the table values (inf = violated), d the graded distances, loc the cells read:
    {"la", "Ya", "Xa", "lb", "Yb", "Xb"} (level coordinates, x unwrapped; -1 where the instance has no
    such cell: a pad read, a unary's b side, the a side of a pad_a instance).  Count instances report
    the parent as b and the parent's cell as a at level lb (its children are implied)."""
    ca = problem.chan[f.a[0]]
    la = ca.level
    A = _view(problem, x, f.a)
    a_fixed = ca.designed is not None
    if f.kind == "unary":
        h, w = A.shape
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        e = f.table[A].ravel()
        none = np.full(e.size, -1, np.int64)
        return e, fd.d[A].ravel(), {"la": la, "Ya": Y.ravel(), "Xa": X.ravel(), "lb": None, "Yb": none, "Xb": none}
    cb = problem.chan[f.b[0]]
    B = _view(problem, x, f.b)
    lb = cb.level
    b_fixed = cb.designed is not None
    if f.kind == "count":
        h, w = B.shape
        s = np.zeros((h, w), np.int64)
        for k in f.subset:
            s += A[(k >> 1)::2, (k & 1)::2]
        tab = fd.d_a if b_fixed else (fd.d_b if a_fixed else fd.d)
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        none = np.full(h * w, -1, np.int64)
        return (f.table[B, s].ravel(), tab[B, s].ravel(),
                {"la": lb, "Ya": none, "Xa": none, "lb": lb, "Yb": Y.ravel(), "Xb": X.ravel()})
    dy, dx = f.off
    h, w = A.shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    if lb == la:
        Yb, Xb = Y + dy, X + dx
    else:
        Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
    vb, ok, padded = _gather(B, Yb, Xb, region, f.pad_b)
    if f.within is not None:
        L = f.within
        ok &= ((Y << la) >> L == (Yb << lb) >> L) & ((X << la) >> L == (Xb << lb) >> L)
    pos = 2 * (Y & 1) + (X & 1)
    if f.by_pos:
        e = f.table[pos, A, vb]
        d_both, d_a, d_b = fd.d[pos, A, vb], fd.d_a[pos, A, vb], fd.d_b[pos, A, vb]
    else:
        e = f.table[A, vb]
        d_both, d_a, d_b = fd.d[A, vb], fd.d_a[A, vb], fd.d_b[A, vb]
    bfix = padded | b_fixed
    if a_fixed:
        d = np.where(bfix, np.where(np.isfinite(e), 0.0, 1.0), d_b)     # nothing movable: 1 if violated
    else:
        d = np.where(bfix, d_a, d_both)
    Yb_loc, Xb_loc = np.where(padded, -1, Yb), np.where(padded, -1, Xb)
    ents, ds = [e[ok]], [d[ok]]
    loc = {"la": la, "Ya": [Y[ok]], "Xa": [X[ok]], "lb": lb, "Yb": [Yb_loc[ok]], "Xb": [Xb_loc[ok]]}
    if f.pad_a >= 0 and lb == la and dy != 0:
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
        d2 = (np.where(np.isfinite(e2), 0.0, 1.0) if b_fixed else fd.d_b[f.pad_a, B])
        ents.append(e2[offa])
        ds.append(d2[offa])
        n2 = int(offa.sum())
        loc["Ya"].append(np.full(n2, -1, np.int64)); loc["Xa"].append(np.full(n2, -1, np.int64))
        loc["Yb"].append(Yq[offa]); loc["Xb"].append(Xq[offa])
    for k in ("Ya", "Xa", "Yb", "Xb"):
        loc[k] = np.concatenate(loc[k]).astype(np.int64)
    return np.concatenate(ents), np.concatenate(ds), loc


# ------------------------------------------------------------------ public API (stable)
def instance_distances(problem: Problem, state, region: Region):
    """{factor name: (d, loc)} for every HARD factor: d the per-instance integer distances (0 = satisfied),
    in castlegen.bench.energy.instances' order; loc the cells each instance reads (see _factor_instances),
    with "level" = the factor's home level.  `state` holds every channel (designed ones included)."""
    fds = distances(problem)
    out = {}
    for i, f in enumerate(problem.factors):
        if not f.hard:
            continue
        e, d, loc = _factor_instances(problem, f, fds[i], state, region)
        loc["level"] = problem.chan[f.a[0]].level
        loc["violated"] = np.isinf(e)
        out[f.name or f"f{i}"] = (np.rint(d).astype(np.int64), loc)
    return out


def level_mass(problem: Problem, state, region: Region):
    """Graded violation mass per level, array (T + 1,): D_l = sum of d over the hard instances homed at l."""
    D = np.zeros(problem.T + 1, np.int64)
    for d, loc in instance_distances(problem, state, region).values():
        D[loc["level"]] += int(d.sum())
    return D


def _footprint(l, Y, X, L, w_l):
    """Level-L cell rectangle [y0, y1) x [x0, x1) covered by the level-l cells (Y, X) (x taken mod the
    level-l width w_l first, so a footprint never crosses the periodic seam)."""
    X = np.mod(X, w_l) if w_l else X
    if l <= L:
        y0, x0 = Y >> (L - l), X >> (L - l)
        return y0, y0 + 1, x0, x0 + 1
    k = l - L
    return Y << k, (Y + 1) << k, X << k, (X + 1) << k


def _rect_add(acc, d, y0, y1, x0, x1):
    """acc (h + 1, w + 1) difference array += d on each rectangle (vectorised over instances)."""
    h, w = acc.shape[0] - 1, acc.shape[1] - 1
    y0, y1 = np.clip(y0, 0, h), np.clip(y1, 0, h)
    x0, x1 = np.clip(x0, 0, w), np.clip(x1, 0, w)
    np.add.at(acc, (y0, x0), d)
    np.add.at(acc, (y0, x1), -d)
    np.add.at(acc, (y1, x0), -d)
    np.add.at(acc, (y1, x1), d)


def graded_field(problem: Problem, state, region: Region, level: int, only_violated=True):
    """Graded violation field at `level`: float array (H >> level, W >> level) whose cell holds the summed
    distance d of the hard instances (of every level) that touch it.  An instance touches a level-L cell
    when the footprint of one of the cells it reads meets the cell's footprint: a finer cell maps to its
    level-L ancestor, a coarser cell spreads its full d (not split) to every level-L cell it covers, and a
    count instance covers its parent's footprint (the children lie inside it).  An instance is counted
    once per cell even when its two cells map to the same level-L cell.  Pad reads have no cell.  The
    field is 0 everywhere iff the state is valid."""
    T = problem.T
    assert 0 <= level <= T
    h, w = problem.H >> level, region.W >> level
    acc = np.zeros((h + 1, w + 1), np.float64)
    for name, (d, loc) in instance_distances(problem, state, region).items():
        m = d > 0 if only_violated else np.ones(d.shape, bool)
        if not m.any():
            continue
        d = d[m].astype(np.float64)
        rects = []
        for side in ("a", "b"):
            l = loc["l" + side]
            if l is None:
                continue
            Y, X = loc["Y" + side][m], loc["X" + side][m]
            has = Y >= 0
            w_l = (region.W >> l) if region.periodic else None
            r = _footprint(l, np.where(has, Y, 0), np.where(has, X, 0), level, w_l)
            rects.append((has, r))
        if len(rects) == 2:
            (ha, ra), (hb, rb) = rects
            _rect_add(acc, d * ha, *ra)
            _rect_add(acc, d * hb, *rb)
            # subtract the intersection so an instance counts once per cell
            iy0, iy1 = np.maximum(ra[0], rb[0]), np.minimum(ra[1], rb[1])
            ix0, ix1 = np.maximum(ra[2], rb[2]), np.minimum(ra[3], rb[3])
            both = ha & hb & (iy1 > iy0) & (ix1 > ix0)
            if both.any():
                _rect_add(acc, -d[both], iy0[both], iy1[both], ix0[both], ix1[both])
        else:
            has, r = rects[0]
            _rect_add(acc, d * has, *r)
    return np.cumsum(np.cumsum(acc, 0), 1)[:h, :w]


# ------------------------------------------------------------------ summary
@dataclass
class Graded:
    viol: dict = field(default_factory=dict)        # level -> violated hard instances (as energy.py)
    nhard: dict = field(default_factory=dict)       # level -> hard instances
    mass: dict = field(default_factory=dict)        # level -> graded violation mass sum d_f
    per_factor: dict = field(default_factory=dict)  # name -> (violated, mass, instances)
    inst: dict = field(default_factory=dict)        # name -> per-instance d (int array)
    level_of: dict = field(default_factory=dict)    # factor name -> home level

    def hist(self, l=None):
        """{d: number of violated instances with that distance} at level l (all levels if None)."""
        out = {}
        for name, d in self.inst.items():
            if l is not None and self.level_of[name] != l:
                continue
            v, c = np.unique(d[d > 0], return_counts=True)
            for a, b in zip(v.tolist(), c.tolist()):
                out[int(a)] = out.get(int(a), 0) + int(b)
        return dict(sorted(out.items()))


_FD_CACHE = {}


def distances(problem: Problem):
    """{factor index: FactorDistance} for the problem's hard factors (cached per problem object)."""
    key = id(problem)
    if key not in _FD_CACHE:
        _FD_CACHE[key] = {i: factor_distance(problem, f) for i, f in enumerate(problem.factors) if f.hard}
    return _FD_CACHE[key]


def graded(problem: Problem, x, region: Region) -> Graded:
    """Binary counts and graded violation mass of a complete sample (designed channels included)."""
    problem.check_sample(x, region)
    fds = distances(problem)
    G = Graded()
    for l in range(problem.T + 1):
        G.viol[l], G.nhard[l], G.mass[l] = 0, 0, 0
    for i, f in enumerate(problem.factors):
        if not f.hard:
            continue
        e, d, _ = _factor_instances(problem, f, fds[i], x, region)
        d = np.rint(d).astype(np.int64)
        l = problem.chan[f.a[0]].level
        name = f.name or f"f{i}"
        v, m = int(np.isinf(e).sum()), int(d.sum())
        G.viol[l] += v
        G.nhard[l] += int(e.size)
        G.mass[l] += m
        G.per_factor[name] = (v, m, int(e.size))
        G.inst[name] = d
        G.level_of[name] = l
    return G


def describe(problem: Problem):
    """One line per hard factor: kind, home level, which axes are ordered."""
    fds = distances(problem)
    out = []
    for i, fd in fds.items():
        f = problem.factors[i]
        l = problem.chan[f.a[0]].level
        if f.kind == "unary":
            ax = f"{f.a[0]}.{f.a[1]} {'ordered' if fd.ordered[0] else 'unordered'}"
        elif f.kind == "count":
            ax = (f"parent {f.b[0]}.{f.b[1]} {'ordered' if fd.ordered[0] else 'unordered'}, "
                  f"sum of {f.a[0]}.{f.a[1]} ordered")
        else:
            ax = (f"{f.a[0]}.{f.a[1]} {'ordered' if fd.ordered[0] else 'unordered'}, "
                  f"{f.b[0]}.{f.b[1]} {'ordered' if fd.ordered[1] else 'unordered'}")
        out.append(f"{f.name or f'f{i}'}: {f.kind}, level {l}, {ax}, max d {int(fd.d.max())}")
    return out
