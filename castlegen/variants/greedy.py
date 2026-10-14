"""greedy: the untrained coloured pass, a floor for every other variant.

Levels coarse to fine; within a level, colours (Y mod 4, X mod 4) in
lexicographic order, then channels in declared order.  Each draw uses the
exact local conditional over known cells (castlegen/variants/_local.py),
keeps the least-violating values and samples them by Gumbel-max with
coordinate-keyed noise.  No lookahead, no learning, no repair.  An open
window is generated on a wider window and cropped; the margin bounds the
dependency radius (16 colour steps per level, each reading one factor reach,
plus the parent)."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from castlegen.bench import noise
from castlegen.bench.problem import Region
from castlegen.bench.variant import Variant, register

from ._local import cond, factor_reach

P = 4


def colour_classes(h, w, periodic, xoff=0):
    """Colours by absolute coordinates; xoff is the region's first column at this level."""
    px = P if (not periodic or w % P == 0) else w
    for cy in range(P):
        for cx in range(min(px, w)):
            Y, X = np.meshgrid(np.arange(cy, h, P), np.arange((cx - xoff) % px, w, px), indexing="ij")
            if Y.size:
                yield Y.ravel(), X.ravel()


def coloured_pass(problem, seed, region, forced=None):
    """Run the pass; returns (state, log q).  forced: a sample whose values are
    scored instead of drawn."""
    state = {**problem.designed_values(seed, region)}
    if forced is not None:
        state.update({k: np.asarray(forced[k]) for k in state if k in forced})
    for c in problem.sampled():
        state[c.name] = np.full((problem.H >> c.level, region.W >> c.level), -1, np.int64)
    logq = 0.0
    sites = {c.name: np.zeros((problem.H >> c.level, region.W >> c.level)) for c in problem.sampled()}
    for l in range(problem.T, -1, -1):
        chans = problem.level_chans(l)
        h, w = problem.H >> l, region.W >> l
        for Y, X in colour_classes(h, w, region.periodic, 0 if region.periodic else region.x0 >> l):
            for c in chans:
                if c.designed is not None:
                    continue
                soft, nv = cond(problem, state, region, c.name, Y, X)
                mask = nv == nv.min(1, keepdims=True)
                logits = np.where(mask, -soft, -np.inf)
                lz = logsumexp(logits, axis=1)
                if forced is None:
                    Xn = np.mod(X, w) if region.periodic else X + (region.x0 >> l)
                    g = noise.gumbel(np, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                    t = np.argmax(logits + g, axis=1)
                else:
                    t = np.asarray(forced[c.name])[Y, X]
                lt = logits[np.arange(Y.size), t] - lz
                sites[c.name][Y, X] = lt
                logq += float(np.sum(lt))
                state[c.name][Y, X] = t
    coloured_pass.sites = sites
    return state, logq


@register
class Greedy(Variant):
    name = "greedy"

    def margin(self):
        p = self.problem
        r = factor_reach(p)
        m = sum((P * P * r + 2) << l for l in range(p.T + 1))
        step = 1 << p.T
        return -(-m // step) * step

    def generate(self, seed, region):
        if region.periodic:
            x, _ = coloured_pass(self.problem, seed, region)
        else:
            m = self.margin()
            ext = Region(region.W + 2 * m, periodic=False, x0=region.x0 - m)
            xe, _ = coloured_pass(self.problem, seed, ext)
            x = {}
            for c in self.problem.chans:
                s = m >> c.level
                x[c.name] = xe[c.name][:, s:s + (region.W >> c.level)]
        x = {c.name: x[c.name] for c in self.problem.sampled()}
        return {"raw": x, "final": x}

    def score(self, x, region):
        assert region.periodic
        _, lq = coloured_pass(self.problem, 0, region, forced=x)
        return lq

    def score_sites(self, x, region):
        assert region.periodic
        coloured_pass(self.problem, 0, region, forced=x)
        return coloured_pass.sites

    def info(self):
        return {"margin": self.margin()}
