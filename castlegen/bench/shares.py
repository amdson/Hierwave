"""Per-block, per-level scores (bench version 2; notes/bench.md, "Block scores").

A block at level l is the 2 x 2 patch of level-l sites sharing one parent
(Y >> 1, X >> 1); at the coarsest level T the same rule makes a block 2 x 2
top cells (as if a level T + 1 existed).  For a state:

  logq share  [l][block] = sum of the per-draw log q of the draws at level l in the block
                           (Variant.score_sites gives the per-draw terms)
  weight share [l][block] = -(soft energy) - LAMBDA_SCORE * (violated hard instances) of the
                           factors homed at level l, each instance assigned to the block of
                           its finer-side cell p; a same-level pair whose cells lie in two
                           blocks gives half to each.  ("direct" weight, the spec.)
  subtree     [l][block] = weight share of the block plus those of all blocks below it
                           (levels < l inside its footprint): what the draw at level l
                           is implicitly deciding about.  Reported next to "direct".

Spearman rho per level between the two shares, pooled over own draws and probe
states (and each alone); starvation: fraction of probe blocks whose logq share
lies more than STARVE_NATS below the own-draw robust line at the same weight.
"""
from __future__ import annotations

import numpy as np

from . import energy as en

LAMBDA_SCORE = 10.0      # penalty per violated hard instance (cpass/cfast lam0)
STARVE_NATS = 5.0
TOP_BLOCK = 2            # coarsest-level blocks: 2 x 2 top cells


def _locs(problem, f, x, region):
    """Like energy.instances, plus the cells: returns (e, Yp, Xp, Yq, Xq); (Yp, Xp) the level-a
    cell of the instance, (Yq, Xq) the same-level partner cell (pairs at one level) or -1."""
    ca = problem.chan[f.a[0]]
    la = ca.level
    A = en._view(problem, x, f.a)
    h, w = A.shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    none = -np.ones(Y.size, np.int64)
    if f.kind == "unary":
        return f.table[A].ravel(), Y.ravel(), X.ravel(), none, none
    B = en._view(problem, x, f.b)
    lb = problem.chan[f.b[0]].level
    if f.kind == "count":
        hb, wb = B.shape
        s = np.zeros((hb, wb), np.int64)
        for k in f.subset:
            s += A[(k >> 1)::2, (k & 1)::2]
        Yb, Xb = np.meshgrid(np.arange(hb), np.arange(wb), indexing="ij")
        # the instance lives at level la; its block is the b cell itself (a 2 x 2 patch of a's)
        return f.table[B, s].ravel(), (2 * Yb).ravel(), (2 * Xb).ravel(), none[:Yb.size], none[:Yb.size]
    dy, dx = f.off
    if lb == la:
        Yb, Xb = Y + dy, X + dx
    else:
        Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
    vb, ok = en._gather(B, Yb, Xb, region, lb, f.pad_b)
    if f.within is not None:
        L = f.within
        ok &= ((Y << la) >> L == (Yb << lb) >> L) & ((X << la) >> L == (Xb << lb) >> L)
    e = f.table[2 * (Y & 1) + (X & 1), A, vb] if f.by_pos else f.table[A, vb]
    outs = [(e[ok], Y[ok], X[ok], (Yb[ok] if lb == la else -np.ones(ok.sum(), np.int64)),
             (np.mod(Xb[ok], w) if lb == la and region.periodic else (Xb[ok] if lb == la else -np.ones(ok.sum(), np.int64))))]
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
        outs.append((e2[offa], Yq[offa], Xq[offa], -np.ones(offa.sum(), np.int64), -np.ones(offa.sum(), np.int64)))
    return tuple(np.concatenate([o[i] for o in outs]) for i in range(5))


def _nblk(problem, region, l):
    return (-(-(problem.H >> l) // 2), -(-(region.W >> l) // 2))


def weight_shares_signed(problem, x, region):
    """Direct and subtree weight shares: -soft - LAMBDA_SCORE * violations, per block."""
    T = problem.T
    D = {l: np.zeros(_nblk(problem, region, l)) for l in range(T + 1)}
    for f in problem.factors:
        e, Yp, Xp, Yq, Xq = _locs(problem, f, x, region)
        if e.size == 0:
            continue
        l = problem.chan[f.a[0]].level
        w_l = region.W >> l
        val = np.where(np.isinf(e), -LAMBDA_SCORE, -np.where(np.isinf(e), 0.0, e))
        by, bx = Yp >> 1, (Xp % w_l) >> 1
        two = (Yq >= 0) & (Yq < (problem.H >> l))        # partner on the strip (pad rows: whole weight to p)
        qby, qbx = np.where(two, Yq, 0) >> 1, (np.where(two, Xq, 0) % w_l) >> 1
        split = two & ((qby != by) | (qbx != bx))
        np.add.at(D[l], (by, bx), np.where(split, 0.5, 1.0) * val)
        if split.any():
            np.add.at(D[l], (qby[split], qbx[split]), 0.5 * val[split])
    S = {0: D[0].copy()}
    for l in range(1, T + 1):
        ch = S[l - 1]
        hb, wb = D[l].shape
        pad = np.zeros((2 * hb, 2 * wb))
        pad[:ch.shape[0], :ch.shape[1]] = ch
        S[l] = D[l] + pad.reshape(hb, 2, wb, 2).sum(axis=(1, 3))
    return D, S


def logq_shares(problem, sites, region):
    """sites {channel: (h, w) per-draw log q} -> {l: (nby, nbx)} block sums."""
    out = {}
    for l in range(problem.T + 1):
        nb = _nblk(problem, region, l)
        a = np.zeros(nb)
        for c in problem.level_chans(l):
            if c.designed is not None:
                continue
            s = np.asarray(sites[c.name], float)
            pad = np.zeros((2 * nb[0], 2 * nb[1]))
            pad[:s.shape[0], :s.shape[1]] = s
            a += pad.reshape(nb[0], 2, nb[1], 2).sum(axis=(1, 3))
        out[l] = a
    return out


class Pool:
    """Collects, per level, flat arrays (logq share, direct weight, subtree weight, world id, block id)."""

    def __init__(self, T):
        self.T = T
        self.rows = {l: [] for l in range(T + 1)}

    def add(self, problem, region, sites, x, kind, wid):
        lq = logq_shares(problem, sites, region)
        D, S = weight_shares_signed(problem, x, region)
        for l in range(self.T + 1):
            n = lq[l].size
            self.rows[l].append((lq[l].ravel(), D[l].ravel(), S[l].ravel(),
                                 np.full(n, kind), np.full(n, wid), np.arange(n)))

    def arrays(self, l):
        r = self.rows[l]
        return [np.concatenate([a[i] for a in r]) for i in range(6)]


def _spearman(a, b):
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    from scipy.stats import spearmanr
    r = spearmanr(a, b)[0]
    return None if np.isnan(r) else float(r)


def _robust_line(x, y, rng, npair=20000):
    """Theil-Sen line on a random pair subsample; a flat median if x does not vary."""
    if np.ptp(x) == 0 or x.size < 3:
        return 0.0, float(np.median(y))
    i = rng.integers(0, x.size, npair)
    j = rng.integers(0, x.size, npair)
    ok = x[i] != x[j]
    s = np.median((y[i] - y[j])[ok] / (x[i] - x[j])[ok]) if ok.any() else 0.0
    return float(s), float(np.median(y - s * x))


def summarize(pool: Pool, shape_w, worst=5):
    """Per-level numbers for both weight definitions; kind 0 = own draws, 1 = probes."""
    out = {}
    rng = np.random.default_rng(0)
    for l in range(pool.T + 1):
        lq, dw, sw, kind, wid, bid = pool.arrays(l)
        res = {"n_own": int((kind == 0).sum()), "n_probe": int((kind == 1).sum())}
        for name, w in (("direct", dw), ("subtree", sw)):
            own, pr = kind == 0, kind == 1
            r = {"rho_pool": _spearman(w, lq), "rho_probe": _spearman(w[pr], lq[pr]),
                 "rho_own": _spearman(w[own], lq[own])}
            slope, icpt = _robust_line(w[own], lq[own], rng)
            deficit = (slope * w + icpt) - lq
            r.update({"slope": slope, "intercept": icpt,
                      "starved_probe": float((deficit[pr] > STARVE_NATS).mean()) if pr.any() else None,
                      "starved_own": float((deficit[own] > STARVE_NATS).mean()) if own.any() else None,
                      "median_deficit_probe": float(np.median(deficit[pr])) if pr.any() else None})
            idx = np.where(pr)[0]
            top = idx[np.argsort(-deficit[idx])[:worst]]
            nbx = shape_w[l]
            r["worst"] = [{"level": l, "probe": int(wid[i]), "block": [int(bid[i] // nbx), int(bid[i] % nbx)],
                           "deficit": float(deficit[i]), "logq": float(lq[i]), "weight": float(w[i])} for i in top]
            res[name] = r
        out[str(l)] = res
    return out
