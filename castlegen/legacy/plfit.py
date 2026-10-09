"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Pseudo-likelihood fit of promise energies (notes/history/promise_estimation.md section 5.3).

The energies of a promise level are log-linear in a feature map F (V, d) of
the promise values:

    unary   u(v)          = F[v] . theta
    pair    e_o(a, b)     = F[a]^T M_o F[b]          o = 0 left|right, 1 top/bottom
    parent  c_pos(v, p)   = F[v]^T P_pos F[p]        pos = child position 2*dy + dx

so a level has d + 2 d^2 + 4 d^2 parameters.  With one-hot features (d = V)
these are dense tables; with a few hand-picked features (e.g. the side bits
and the number of open sides) they are a quadratic metric on features.

The objective is the pseudo-likelihood the sampler actually uses: for every
corpus parent, the conditional probability of its observed 2x2 refinement
among the refinements consistent with the parent and the observed halo,

    loss = mean_parents [ E(observed) + log sum_k exp(-E(k)) ] + l2 * |params|^2,

E being unary + parent terms of the four children plus the pair terms of the
four internal seams and the eight seams to the fixed neighbouring children.
Inconsistent refinements are never in the sum, so the hard constraints do
not distort the fit.  The result is expanded into dense `promise.Tables`, so
`PromiseVar` uses it unchanged.

    python -m castlegen.legacy.plfit castle --features conn      # fits and caches tables
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from castlegen.legacy.promise import Tables

KIDS = ((0, 0), (0, 1), (1, 0), (1, 1))


# ------------------------------------------------------------------ features
def onehot(V):
    return np.eye(V, dtype=np.float32)


def conn_features(V=16):
    """[1, bit N, bit E, bit S, bit W, open sides / 4] for the 4-bit side set."""
    F = np.zeros((V, 6), np.float32)
    for v in range(V):
        bits = [(v >> d) & 1 for d in range(4)]
        F[v] = [1.0, *bits, sum(bits) / 4.0]
    return F


FEATURES = {"onehot": onehot, "conn": conn_features}


# --------------------------------------------------------------------- data
@dataclass
class Batch:
    cands: np.ndarray        # (B, Kmax, 4) candidate refinements, padded
    mask: np.ndarray         # (B, Kmax) valid candidates
    obs: np.ndarray          # (B,) index of the observed refinement
    parent: np.ndarray       # (B,)
    ext: np.ndarray          # (B, 4, 2) [kid, 0] vertical neighbour value, [kid, 1] horizontal
    skipped: int             # parents whose observed refinement was not a candidate


def build(grids_h, grids_2h, refine, halo_fn):
    """Batch of every parent in the corpus at one level.  grids_h: child grids,
    grids_2h: the parent grids (same order).  Levels whose child grid side is
    below 4 are skipped (a group's halo would be itself on the torus)."""
    cands, obs, parent, ext, skipped = [], [], [], [], 0
    for g, G in zip(grids_h, grids_2h):
        g0, g1 = g.shape
        if min(g0, g1) < 4:
            continue
        for py, px in np.ndindex(*G.shape):
            kids = tuple(int(g[2 * py + dy, 2 * px + dx]) for dy, dx in KIDS)
            cs = refine(int(G[py, px]), halo_fn(g, py, px))
            if kids not in cs:
                skipped += 1
                continue
            e = np.zeros((4, 2), np.int64)
            for i, (dy, dx) in enumerate(KIDS):
                y, x = 2 * py + dy, 2 * px + dx
                e[i, 0] = g[(y - 1) % g0, x] if dy == 0 else g[(y + 1) % g0, x]
                e[i, 1] = g[y, (x - 1) % g1] if dx == 0 else g[y, (x + 1) % g1]
            cands.append(np.array(cs, np.int64))
            obs.append(cs.index(kids))
            parent.append(int(G[py, px]))
            ext.append(e)
    if not cands:
        return None
    K = max(len(c) for c in cands)
    C = np.zeros((len(cands), K, 4), np.int64)
    M = np.zeros((len(cands), K), bool)
    for i, c in enumerate(cands):
        C[i, :len(c)] = c
        M[i, :len(c)] = True
    return Batch(C, M, np.array(obs), np.array(parent), np.stack(ext), skipped)


# ------------------------------------------------------------------- model
def expand(params, F):
    """-> dense U (V,), PAIR (2, V, V), PAR (4, V, V) from (theta, M, P)."""
    th, M, P = params
    U = F @ th
    PAIR = jnp.einsum("vd,ode,we->ovw", F, M, F)
    PAR = jnp.einsum("vd,pde,we->pvw", F, P, F)
    return U, PAIR, PAR


def energies(U, PAIR, PAR, cands, parent, ext):
    """(B, K) energy of every candidate refinement."""
    c = cands
    e = U[c].sum(-1)
    for i in range(4):
        e = e + PAR[i, c[..., i], parent[:, None]]
    e = e + PAIR[0, c[..., 0], c[..., 1]] + PAIR[0, c[..., 2], c[..., 3]] \
          + PAIR[1, c[..., 0], c[..., 2]] + PAIR[1, c[..., 1], c[..., 3]]
    for i, (dy, dx) in enumerate(KIDS):
        ci, nv, nh = c[..., i], ext[:, None, i, 0], ext[:, None, i, 1]
        e = e + (PAIR[1, nv, ci] if dy == 0 else PAIR[1, ci, nv])
        e = e + (PAIR[0, nh, ci] if dx == 0 else PAIR[0, ci, nh])
    return e


def nll(params, F, b, l2):
    U, PAIR, PAR = expand(params, F)
    e = energies(U, PAIR, PAR, b.cands, b.parent, b.ext)
    e = jnp.where(b.mask, e, jnp.inf)
    obs = jnp.take_along_axis(e, b.obs[:, None], 1)[:, 0]
    loss = (obs + jax.nn.logsumexp(-e, axis=1)).mean()
    reg = sum(jnp.sum(p ** 2) for p in params)
    return loss + l2 * reg, loss


def _pack(params):
    return np.concatenate([np.asarray(p, np.float64).ravel() for p in params])


def _unpack(x, d):
    th = x[:d]
    M = x[d:d + 2 * d * d].reshape(2, d, d)
    P = x[d + 2 * d * d:].reshape(4, d, d)
    return jnp.asarray(th, jnp.float32), jnp.asarray(M, jnp.float32), jnp.asarray(P, jnp.float32)


def fit_level(batch, F, l2=1e-3, maxiter=300):
    """-> (U, PAIR, PAR) as numpy, and a report dict."""
    d = F.shape[1]
    Fj = jnp.asarray(F)
    b = Batch(*(jnp.asarray(a) for a in (batch.cands, batch.mask, batch.obs, batch.parent, batch.ext)),
              batch.skipped)
    fg = jax.jit(jax.value_and_grad(lambda p: nll(p, Fj, b, l2)[0]))
    pl = jax.jit(lambda p: nll(p, Fj, b, l2)[1])

    def f(x):
        v, g = fg(_unpack(x, d))
        return float(v), _pack(g)

    x0 = np.zeros(d + 6 * d * d)
    flat = float(pl(_unpack(x0, d)))
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    params = _unpack(res.x, d)
    U, PAIR, PAR = (np.asarray(a) for a in expand(params, Fj))
    rep = dict(parents=int(batch.cands.shape[0]), skipped=batch.skipped,
               cands=float(batch.mask.sum(1).mean()), flat=flat, fitted=float(pl(params)),
               iters=int(res.nit), params=int(x0.size))
    return (U, PAIR, PAR), rep, params


def heldout(params, F, batch):
    """Mean pseudo-negative-log-likelihood of `batch` under fitted params."""
    b = Batch(*(jnp.asarray(a) for a in (batch.cands, batch.mask, batch.obs, batch.parent, batch.ext)), 0)
    return float(nll(params, jnp.asarray(F), b, 0.0)[1])


def fit(grids, refine, halo_fn, F, l2=0.1, maxiter=300, holdout=None, log=print):
    """grids: {h: [child grid, ...]} for every level (parents at 2h).
    -> (promise.Tables, {h: report}).  With holdout (same format) the report
    gains the held-out loss per level; if l2 is a sequence the value with the
    lowest held-out loss is used per level (the first value without holdout).
    A few hundred parents support l2 ~ 0.1; thousands allow 0.01."""
    V = F.shape[0]
    l2s = [l2] if np.ndim(l2) == 0 else list(l2)
    u, pair, parent, reports = {}, {}, {}, {}
    for h in sorted(grids):
        if 2 * h not in grids:
            continue
        batch = build(grids[h], grids[2 * h], refine, halo_fn)
        if batch is None:
            continue
        hb = build(holdout[h], holdout[2 * h], refine, halo_fn) if holdout is not None else None
        best = None
        for lam in (l2s if hb is not None else l2s[:1]):
            tabs, rep, params = fit_level(batch, F, lam, maxiter)
            rep["l2"] = lam
            if hb is not None:
                rep["heldout"] = heldout(params, F, hb)
                rep["heldout_flat"] = float(np.log(hb.mask.sum(1)).mean())
            if best is None or rep.get("heldout", 0) < best[1].get("heldout", 0):
                best = (tabs, rep)
        (U, PAIR, PAR), rep = best
        u[h], pair[h], parent[h] = U, PAIR, PAR
        reports[h] = rep
        log(f"h={h:4d}  parents {rep['parents']:5d} (skipped {rep['skipped']})  cands {rep['cands']:6.1f}  "
            f"flat {rep['flat']:.3f}  fitted {rep['fitted']:.3f}"
            + (f"  heldout {rep['heldout']:.3f} (flat {rep['heldout_flat']:.3f})" if "heldout" in rep else "")
            + f"  [l2 {rep['l2']:g}, {rep['params']} params, {rep['iters']} iters]")
    return Tables(V, u, pair, parent), reports


# ---------------------------------------------------------------------- cli
def main():
    from castlegen.legacy import corpus, tileset
    from castlegen.legacy import exemplar as ex
    from castlegen.legacy.quantities import connectivity as C
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("layout", help="exemplar layout name (castlegen/legacy/exemplars)")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--features", default="conn", choices=sorted(FEATURES))
    ap.add_argument("--l2", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--holdout", type=int, default=4, help="corpus samples held out")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    layout = ex.load_layout(args.layout)
    ts = tileset.load(layout.get("tileset", "demo"))
    tiles = corpus.samples(ts, layout)
    mined = [corpus.mine(ts, t, args.K) for t in tiles]
    grids = {h: [m[0][h] for m in mined[args.holdout:]] for h in mined[0][0]}
    held = {h: [m[0][h] for m in mined[:args.holdout]] for h in mined[0][0]} if args.holdout else None
    F = FEATURES[args.features](16)
    tab, _ = fit(grids, C.refine, corpus._halo, F, args.l2, holdout=held)
    out = args.out or os.path.join(corpus.CACHE_DIR, f"promise-tables-pl-{args.layout}-{args.features}.npz")
    tab.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
