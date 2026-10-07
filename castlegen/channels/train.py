"""Generic pieces for fitting learned tables by moment matching / self-play.

counts      sufficient statistics of named factors at the current state
site_probs  the kernel's one-site conditionals of a channel, softmax(-e / T)
rb_gap      Rao-Blackwellised expected change of the counts under one-site
            moves of a channel with given per-site probabilities
step        theta += eta * gap / scale (- prior pull), zero-mean per table

Sign convention: E = sum theta[n] * N[n], p ~ exp(-E); gap = (forward
counts) - (target counts), so an over-produced feature gains energy.

Pads: a pair term with an off-grid partner exists when the pad is >= 0 and
reads the entry (va, pad_b) (or (pad_a, vb) for a same-level pair seen from
b's side, a off the grid), exactly as the kernel does; counts and rb_gap
count these entries too, so E = sum theta * N is the energy whose one-site
conditionals the kernel samples."""
import numpy as np
from numba import njit

from .core import COUNT, PAIR, UNARY
from .kernel import _energies


# ----------------------------------------------------------------- counts
def _pair_counts(f, A, B, va, vb, shape):
    N = np.zeros(shape)
    ra, ca = A.grid.shape
    rb, cb = B.grid.shape
    yy, xx = np.mgrid[:ra, :ca]
    qy = (yy * A.h) // B.h + f.off[0]
    qx = (xx * A.h) // B.h + f.off[1]
    on = (qy >= 0) & (qy < rb) & (qx >= 0) & (qx < cb)
    a = va[A.grid]
    np.add.at(N, (a[on], vb[B.grid[qy[on], qx[on]]]), 1)
    if f.pad_b >= 0:
        np.add.at(N[:, f.pad_b], a[~on], 1)
    if f.pad_a >= 0 and A.h == B.h:                                 # b's cells whose a partner is off the grid
        yy, xx = np.mgrid[:rb, :cb]
        py, px = yy - f.off[0], xx - f.off[1]
        off = (py < 0) | (py >= ra) | (px < 0) | (px >= ca)
        np.add.at(N[f.pad_a], vb[B.grid[off]], 1)
    return N


def counts(model, names):
    """{name: N} with N shaped like the factor's table: pairs once each (a
    at p, b at q, plus pad entries), counts once per block, unaries per cell."""
    out = {}
    for f in model.factors:
        if f.name not in names:
            continue
        A = model.chan(f.a[0])
        va = A.views[f.a[1]]
        if f.kind == UNARY:
            out[f.name] = np.bincount(va[A.grid].ravel(), minlength=f.table.shape[0]).astype(np.float64)
            continue
        B = model.chan(f.b[0])
        vb = B.views[f.b[1]]
        if f.kind == PAIR:
            out[f.name] = _pair_counts(f, A, B, va, vb, f.table.shape)
        else:
            r = B.h // A.h
            rb, cb = B.grid.shape
            s = va[A.grid].reshape(rb, r, cb, r).sum(axis=(1, 3))
            N = np.zeros(f.table.shape)
            np.add.at(N, (vb[B.grid].ravel(), s.ravel()), 1)
            out[f.name] = N
    missing = set(names) - set(out)
    assert not missing, f"no factor named {missing}"
    return out


# ------------------------------------------------------------- site probs
@njit(cache=True)
def _all_energies(home, grids, hs, views, fac, tabs, D):
    g = grids[home]
    rows, cols = g.shape
    out = np.empty((rows, cols, D))
    e = np.empty(D)
    for y in range(rows):
        for x in range(cols):
            _energies(y, x, home, grids, hs, views, fac, tabs, e)
            out[y, x] = e
    return out


def site_probs(model, home, T=1.0, below=True):
    """(rows, cols, D): softmax(-e / T) of the kernel's candidate energies
    at every site of `home`; fixed cells (and sites with no finite
    candidate) get a one-hot on their current value."""
    P = model.compile(home, below)
    hc = model.chan(home)
    e = _all_energies(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, hc.D) / T
    m = e.min(axis=2, keepdims=True)
    dead = ~np.isfinite(m[..., 0])
    w = np.exp(-(e - np.where(np.isfinite(m), m, 0.0)))
    w /= np.maximum(w.sum(axis=2, keepdims=True), 1e-300)
    onehot = hc.fixed | dead
    w[onehot] = 0.0
    yy, xx = np.nonzero(onehot)
    w[yy, xx, hc.grid[yy, xx]] = 1.0
    return w


# ----------------------------------------------------------------- rb gap
@njit(cache=True)
def _rb(home, grids, hs, views, fac, sel, trans, base, ncol, probs, out):
    """For every site p and selected row f, out[entry(t)] += probs[p, t] and
    out[entry(cur)] -= probs[p, t]; entry in the factor's own orientation,
    flat index base[i] + row * ncol[i] + col."""
    g_home = grids[home]
    rows, cols = g_home.shape
    D = probs.shape[2]
    hc = hs[home]
    for y in range(rows):
        for x in range(cols):
            cur = g_home[y, x]
            pr = probs[y, x]
            for i in range(sel.shape[0]):
                f = sel[i]
                kind = fac[f, 0]
                av = views[fac[f, 2]]
                o, nc = base[i], ncol[i]
                if kind == 2:
                    for t in range(D):
                        out[o + av[t]] += pr[t]
                        out[o + av[cur]] -= pr[t]
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
                        if trans[i]:
                            out[o + vb * nc + av[t]] += pr[t]
                            out[o + vb * nc + av[cur]] -= pr[t]
                        else:
                            out[o + av[t] * nc + vb] += pr[t]
                            out[o + av[cur] * nc + vb] -= pr[t]
                elif kind == 1:                                 # count homed here: home is the fine side
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
                        out[o + vb * nc + s + av[t]] += pr[t]
                        out[o + vb * nc + s + av[cur]] -= pr[t]
                else:                                           # 3 / 4: home is the coarse side
                    r = hc // hb
                    vh = views[fac[f, 3]]
                    if kind == 3:
                        for yy in range(y * r, y * r + r):
                            for xx in range(x * r, x * r + r):
                                u = av[g[yy, xx]]
                                for t in range(D):
                                    out[o + u * nc + vh[t]] += pr[t]
                                    out[o + u * nc + vh[cur]] -= pr[t]
                    else:
                        s = 0
                        for yy in range(y * r, y * r + r):
                            for xx in range(x * r, x * r + r):
                                s += av[g[yy, xx]]
                        for t in range(D):
                            out[o + vh[t] * nc + s] += pr[t]
                            out[o + vh[cur] * nc + s] -= pr[t]


def rb_gap(model, home, names, probs):
    """{name: dN}: sum_p sum_t probs[p, t] (N(z with z_p = t) - N(z)), each
    site of `home` moved alone from the current state.  Read off the
    below=True packed rows of `home` (home-side, reflected and below rows),
    one entry change per row per site; factors that do not touch `home`
    give zeros.  Fixed cells contribute nothing if probs is one-hot there."""
    P = model.compile(home, below=True)
    idx = {f.name: i for i, f in enumerate(model.factors)}
    missing = set(names) - set(idx)
    assert not missing, f"no factor named {missing}"
    want = {idx[n]: n for n in names}
    shapes = {n: model.factors[idx[n]].table.shape for n in names}
    offs, o = {}, 0
    for n in names:
        offs[n] = o
        o += int(np.prod(shapes[n]))
    sel = [f for f in range(P.fac.shape[0]) if P.src[f] in want]
    nm = [want[P.src[f]] for f in sel]
    base = np.array([offs[n] for n in nm], np.int64)
    ncol = np.array([shapes[n][1] if len(shapes[n]) > 1 else 1 for n in nm], np.int64)
    probs = np.ascontiguousarray(probs, np.float64)
    assert probs.shape == model.chan(home).grid.shape + (model.chan(home).D,), probs.shape
    out = np.zeros(o)
    _rb(P.home, P.grids, P.hs, P.views, P.fac, np.array(sel, np.int64), P.transposed[sel].copy(),
        base, ncol, probs, out)
    return {n: out[offs[n]:offs[n] + int(np.prod(shapes[n]))].reshape(shapes[n]) for n in names}


# ------------------------------------------------------------------- step
def step(theta, gap, prior=None, eta=1.0, lam_prior=0.0, scale=None):
    """theta[k] + eta * gap[k] / scale[k] - lam_prior * (theta[k] - prior[k]),
    then zero-mean per table.  gap = forward counts - target counts (per
    site); keys missing from gap are only pulled and re-centred."""
    new = {}
    for k, th in theta.items():
        t = np.array(th, np.float64)
        if k in gap:
            s = 1.0 if scale is None else (scale[k] if isinstance(scale, dict) else scale)
            t = t + eta * np.asarray(gap[k]) / s
        if prior is not None and lam_prior:
            t = t - lam_prior * (np.asarray(th) - np.asarray(prior[k]))
        new[k] = t - t.mean()
    return new
