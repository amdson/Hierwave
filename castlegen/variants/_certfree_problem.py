"""certfree (overnight 2026-10-16, agent certfree; ideas #22 target B): the certificate-free ground problem and the
bottom-up certificate derivation.

ground_cf(p): `p` (bench ground) without the certificate channels, without their honour factors and without the
certificate support factors of levels 1..T; kept: fill (designed), rho1..rho4, tiles, soft smoothness / count prior /
top noise, and support0 (tiles) as the only hard factor.  Same construction as reportables/2026-10-14_cmix/mix.py
`stripped`, except the problem keeps probes (filtered to the kept channels).  User decision: dropping the certificates
changes the target (the tile marginal loses ground's log #valid-certificates term); accepted for this line (ideas.md
#22, target B), so constraint 4 (objective fixed) is not the question here.

derive_certs: the certificates of a sampled tile / rho state, bottom-up, by the TIGHT MAP declared as an abstraction map
of the level below: for a level-l block, A = the largest value that satisfies every declared "honourA" count factor
(sum over the subset of children's A >= A_parent, i.e. min over child rows of the row sum of A) and B = the smallest
that satisfies every "honourB" factor (max over child rows of the row sum of B).  Both are read from the declared
factor subsets (Factor.subset of the honour factors), nothing else.  At level 0 the certificate is the tile.  On a
support-valid terrain this is ground's `lift` (A = full columns, B = occupied columns), and A <= B, the honour factors
and the support factors hold by construction (tests/variants/test_certfree.py).
"""
from __future__ import annotations

import numpy as np

from castlegen.bench.problem import Problem
from castlegen.bench.problems import ground as G


def make_ground_cf(p: Problem) -> Problem:
    chans = [c for c in p.chans if not c.name.startswith("cert")]
    keep = lambda n: not (n.startswith("honour") or (n.startswith("support") and n != "support0"))
    fs = [f for f in p.factors if keep(f.name)]
    names = {c.name for c in chans}
    probes = p.probes

    def pr(seed, region, n=4):
        return [{k: v for k, v in x.items() if k in names} for x in probes(seed, region, n)]

    return Problem(name=p.name + "_cf", T=p.T, H=p.H, chans=chans, factors=fs, W_train=p.W_train, W_eval=p.W_eval,
                   stats=p.stats, render=p.render, probes=pr, exact={}, tol=p.tol, version=p.version,
                   notes="certificate-free " + p.name)


def honour_rows(p: Problem, l: int):
    """(rows_A, rows_B): the declared child-subsets of the honour factors of level l (read from p.factors)."""
    ra = [f.subset for f in p.factors if f.name.startswith(f"honourA{l}")]
    rb = [f.subset for f in p.factors if f.name.startswith(f"honourB{l}")]
    return ra, rb


def _child_views(x, w):
    """x (..., h, w) -> the 4 children of each parent cell, k = 2 (Y & 1) + (X & 1): list of 4 (..., h/2, w/2)."""
    return [x[..., (k >> 1)::2, (k & 1)::2] for k in range(4)]


def derive_certs(p: Problem, t, xp=np, cert_index=None):
    """Certificates of every level 1..T from the tiles `t` (..., H, W), by the tight map above.
    Returns {cert{l}: (..., H/2^l, W/2^l) int index in certland order}.  `p` is the FULL ground problem (the honour
    factors are read from it).  xp = numpy or jax.numpy."""
    ci = cert_index or G.cert_index
    A = B = xp.asarray(t).astype(xp.int32)
    out = {}
    for l in range(1, p.T + 1):
        ra, rb = honour_rows(p, l)
        ca, cb = _child_views(A, None), _child_views(B, None)
        a = None
        for sub in ra:
            s = sum(ca[k] for k in sub)
            a = s if a is None else xp.minimum(a, s)
        b = None
        for sub in rb:
            s = sum(cb[k] for k in sub)
            b = s if b is None else xp.maximum(b, s)
        A, B = a, b
        idx = xp.asarray(ci(1 << l))
        out[f"cert{l}"] = idx[A, B]
    return out
