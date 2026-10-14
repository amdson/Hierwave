"""Training-only sampler of target-like coarse states (ideas #14, source for cstart).

For a start depth L, draws every sampled channel at levels >= L from the
designed energy restricted to those levels:

    p_L(x_{>=L})  propto  exp(-(1/tau) sum_{f : all participants at levels >= L} E_f)

with hard factors (inf entries) enforced.  Everything is read off the factor
declarations and the hierarchy (constraint 1): unary, pair and count tables,
views, pads, `within`, `by_pos`.  Nothing refers to a problem's channel names.

Moves (numpy, batched over worlds, periodic strips only):
  * single-site Gibbs per channel, over colour classes derived from the
    factors (two sites of one class never share an instance), restricted to the
    least-violating values at each site, so an invalid start is repaired and a
    valid state stays valid (then it is exact Gibbs on p_L);
  * a joint (chain) move over a run of count factors, each the only coupling
    between its child and parent channels (full subset, no same-channel pair on
    the child, no offset-0 pair between the two), chained when one factor's
    parent is the next one's child: upward sum messages over the 2x2 count
    trees, downward exact sampling.  This moves hard count equalities, which
    single-site moves cannot; channels in a chain skip the site sweeps.

Known limit (report 2026-10-14_ccoarse): channels whose restricted factors are
all hard (ground's certificates, columns' and the toys' counts) come out uniform
over the hard-feasible set and independent of channels they are only linked to
through finer levels (rho vs cert): not target-like.  informative_channels()
lists the channels that do carry soft restricted terms.

Training-only: global and iterative (the critic exemption of constraint 2); the
generation pass never calls it.  Randomness from castlegen.bench.noise keyed by
(world seed, tag, step, Y, X, value): deterministic per seed and independent of
the batch it is drawn in.
"""
from __future__ import annotations

import numpy as np

from castlegen.bench import noise
from castlegen.bench.problem import Region

TAG = noise.tag_of("coarse_prior")


# ------------------------------------------------------------------ helpers
def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def _gather(arr, Y, X, pad):
    """arr (B, h, w) at (Y, X) (h, w)-shaped grids; periodic x.  Rows off the strip
    read pad (ok only if pad >= 0).  Returns (values (B, ...), ok (...))."""
    h, w = arr.shape[1:]
    off = (Y < 0) | (Y >= h)
    v = arr[:, np.clip(Y, 0, h - 1), np.mod(X, w)]
    if pad >= 0:
        v = np.where(off[None], pad, v)
        ok = np.ones(Y.shape, bool)
    else:
        ok = ~off
    return v, ok


def _within(f, Ya, Xa, la, Yb, Xb, lb):
    if f.within is None:
        return True
    L = f.within
    return (((Ya << la) >> L) == ((Yb << lb) >> L)) & (((Xa << la) >> L) == ((Xb << lb) >> L))


class CoarsePrior:
    """Compiled restriction of a problem to levels >= L on a periodic strip."""

    def __init__(self, problem, region: Region, L: int):
        assert region.periodic, "the coarse prior runs on periodic training strips"
        self.p, self.region, self.L = problem, region, L
        self.lev = lambda c: problem.chan[c].level
        self.factors = [f for f in problem.factors
                        if self.lev(f.a[0]) >= L and (f.b is None or self.lev(f.b[0]) >= L)]
        self.chans = [c for c in problem.chans if c.level >= L and c.designed is None]
        self.hw = {l: (problem.H >> l, region.W >> l) for l in range(problem.T + 1)}
        self.touch = {c.name: [f for f in self.factors
                               if f.a[0] == c.name or (f.b is not None and f.b[0] == c.name)]
                      for c in self.chans}
        self.colours = {c.name: self._colouring(c) for c in self.chans}
        self.blocks = self._blocks()
        self.chains = self._chains()

    # -------------------------------------------------------------- structure
    def _colouring(self, c):
        """Per-axis modulus so two sites of one class never share an instance."""
        my = mx = 1
        for f in self.touch[c.name]:
            if f.kind == "pair" and f.a[0] == c.name and f.b[0] == c.name:
                my, mx = max(my, abs(f.off[0]) + 1), max(mx, abs(f.off[1]) + 1)
            if f.kind == "count" and f.a[0] == c.name:
                my, mx = max(my, 2), max(mx, 2)
        h, w = self.hw[c.level]
        while w % mx:
            mx += 1
        return my, mx

    def _blocks(self):
        out = []
        cnt = {}
        for f in self.factors:
            if f.kind == "count":
                cnt.setdefault((f.a[0], f.b[0]), []).append(f)
        for (a, b), fs in cnt.items():
            if self.p.chan[a].designed is not None or self.p.chan[b].designed is not None:
                continue
            if len(fs) != 1 or tuple(sorted(fs[0].subset)) != (0, 1, 2, 3):
                continue
            bad = any(f.kind == "pair" and ((f.a[0] == a and f.b[0] == a)
                                            or ({f.a[0], f.b[0]} == {a, b} and f.off == (0, 0)))
                      for f in self.factors)
            if not bad:
                out.append(fs[0])
        return out

    def _chains(self):
        """Maximal runs of block factors where one factor's parent is the next one's child."""
        by_child = {f.a[0]: f for f in self.blocks}
        parents = {f.b[0] for f in self.blocks}
        out = []
        for f in self.blocks:
            if f.a[0] in parents:
                continue                                     # not the bottom of a run
            run = [f]
            while run[-1].b[0] in by_child:
                run.append(by_child[run[-1].b[0]])
            out.append(run)
        return out

    # -------------------------------------------------------------- conditional
    def view(self, x, cv):
        c, v = cv
        return self.p.chan[c].views[v][x[c]]

    def cond(self, x, cname, skip=None):
        """Conditional energy of every site of cname for every value given the
        rest of x: soft (B, D, h, w) and number of violated instances."""
        c = self.p.chan[cname]
        l = c.level
        h, w = self.hw[l]
        B = x[cname].shape[0]
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        soft = np.zeros((B, c.D, h, w))
        nv = np.zeros((B, c.D, h, w), np.int32)

        def add(e, ok=True):
            e = np.where(ok, e, 0.0)
            inf = np.isinf(e)
            np.add(soft, np.where(inf, 0.0, e), out=soft)
            np.add(nv, inf, out=nv, casting="unsafe")

        V = lambda cv: self.p.chan[cv[0]].views[cv[1]]                  # (D,) view table
        for f in self.touch[cname]:
            if skip is not None and id(f) in skip:
                continue
            if f.kind == "unary":
                add(f.table[V(f.a)][None, :, None, None])
                continue
            la, lb = self.lev(f.a[0]), self.lev(f.b[0])
            if f.kind == "count":
                if f.b[0] == cname:                                       # c is the parent
                    A = self.view(x, f.a)
                    s = sum(A[:, (k >> 1)::2, (k & 1)::2] for k in f.subset)
                    add(f.table[V(f.b)[None, :, None, None], s[:, None]])
                if f.a[0] == cname:                                       # c is a child
                    A = self.view(x, f.a)
                    Bv = self.view(x, f.b)
                    kpos = 2 * (Y & 1) + (X & 1)
                    par = Bv[:, Y >> 1, X >> 1]
                    tot = np.zeros((B, h, w), np.int64)
                    for k in f.subset:
                        sib = A[:, (Y >> 1) * 2 + (k >> 1), (X >> 1) * 2 + (k & 1)]
                        tot = tot + np.where(kpos == k, 0, sib)
                    ins = np.isin(kpos, f.subset)
                    e = f.table[par[:, None], np.minimum(tot[:, None] + V(f.a)[None, :, None, None],
                                                         f.table.shape[1] - 1)]
                    add(e, ins[None, None])
                continue
            # pair
            dy, dx = f.off
            if f.a[0] == cname:                                           # c on the a side
                if lb == la:
                    Yb, Xb = Y + dy, X + dx
                else:
                    Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
                vb, ok = _gather(self.view(x, f.b), Yb, Xb, f.pad_b)
                ok = ok & _within(f, Y, X, la, Yb, Xb, lb)
                va = V(f.a)[None, :, None, None]
                if f.by_pos:
                    e = f.table[(2 * (Y & 1) + (X & 1))[None, None], va, vb[:, None]]
                else:
                    e = f.table[va, vb[:, None]]
                add(e, ok[None, None])
            if f.b[0] == cname:                                           # c on the b side
                vbv = V(f.b)[None, :, None, None]
                if lb == la:
                    Ya, Xa = Y - dy, X - dx
                    va, ok = _gather(self.view(x, f.a), Ya, Xa, -1)
                    ok = ok & _within(f, Ya, Xa, la, Y, X, lb)
                    add(f.table[va[:, None], vbv], ok[None, None])
                    if f.pad_a >= 0 and dy != 0:
                        offa = ((Ya < 0) | (Ya >= self.hw[la][0])) & _within(f, Ya, Xa, la, Y, X, lb)
                        add(f.table[f.pad_a, vbv], offa[None, None])
                else:
                    for k in range(4):
                        Ya, Xa = 2 * (Y - dy) + (k >> 1), 2 * (X - dx) + (k & 1)
                        va, ok = _gather(self.view(x, f.a), Ya, Xa, -1)
                        ok = ok & _within(f, Ya, Xa, la, Y, X, lb)
                        e = f.table[k, va[:, None], vbv] if f.by_pos else f.table[va[:, None], vbv]
                        add(e, ok[None, None])
        return soft, nv

    # -------------------------------------------------------------- moves
    def _gumbel(self, seeds, step, shape):
        """(B, *shape) Gumbel noise keyed by (seed, step, index)."""
        idx = np.arange(int(np.prod(shape))).reshape(shape)
        u = noise.uniform(np, seeds.reshape((-1,) + (1,) * len(shape)), TAG, step, idx, 0)
        return -np.log(-np.log(u.astype(np.float64)))

    def site_sweep(self, x, cname, seeds, step, tau):
        c = self.p.chan[cname]
        h, w = self.hw[c.level]
        my, mx = self.colours[cname]
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        for cy in range(my):
            for cx in range(mx):
                sel = ((Y % my) == cy) & ((X % mx) == cx)
                if not sel.any():
                    continue
                soft, nv = self.cond(x, cname)
                ys, xs = np.nonzero(sel)
                soft, nv = soft[:, :, ys, xs], nv[:, :, ys, xs]                  # (B, D, n)
                vmin = nv.min(1, keepdims=True)
                logit = np.where(nv == vmin, -soft / tau, -np.inf)
                g = self._gumbel(seeds, step, (c.D, ys.size))
                x[cname][:, ys, xs] = np.argmax(logit + g, axis=1)
                step += 1
        return step

    def _child_logw(self, m, Va, Sv):
        """m (B, Da, 2h, 2w) value log-weights -> per child k (B, Sv, h, w) view log-weights."""
        oneh = (Va[:, None] == np.arange(Sv)[None]).astype(float)
        out = []
        for k in range(4):
            e = m[:, :, (k >> 1)::2, (k & 1)::2]
            mx = np.max(e, 1, keepdims=True)
            mx = np.where(np.isfinite(mx), mx, 0.0)
            with np.errstate(divide="ignore"):
                out.append(np.log(np.einsum("bdhw,ds->bshw", np.exp(e - mx), oneh)) + mx)
        return out

    @staticmethod
    def _partials(wk):
        P = [wk[0]]
        Sv = wk[0].shape[1]
        for k in range(1, 4):
            prev = P[-1]
            B, n0, h, w = prev.shape
            cur = np.full((B, n0 + Sv - 1, h, w), -np.inf)
            for s in range(Sv):
                cur[:, s:s + n0] = np.logaddexp(cur[:, s:s + n0], prev + wk[k][:, s:s + 1])
            P.append(cur)
        return P

    def chain_move(self, x, chain, seeds, step, tau):
        """Exact joint Gibbs draw of every channel in a chain of count factors
        (chain[i] links child c_i to parent c_{i+1}), the rest of x fixed: upward
        sum messages over the count trees, then downward sampling."""
        K = len(chain)
        cs = [chain[0].a[0]] + [f.b[0] for f in chain]
        m, Ps, Zs = [], [], []
        for i, cn in enumerate(cs):
            skip = set()
            if i > 0:
                skip.add(id(chain[i - 1]))
            if i < K:
                skip.add(id(chain[i]))
            so, nv = self.cond(x, cn, skip=skip)
            e = np.where(nv == nv.min(1, keepdims=True), -so / tau, -np.inf)
            if i > 0:
                f = chain[i - 1]
                Va = self.p.chan[f.a[0]].views[f.a[1]]
                Vb = self.p.chan[f.b[0]].views[f.b[1]]
                P = self._partials(self._child_logw(m[-1], Va, int(Va.max()) + 1))
                S = f.table.shape[1]
                Z = P[3][:, :S]
                zm = np.max(Z, 1, keepdims=True)
                zm = np.where(np.isfinite(zm), zm, 0.0)
                M = np.exp(-f.table / tau)                                       # (Vbv, S), hard -> 0
                with np.errstate(divide="ignore"):
                    msg = np.log(np.einsum("vs,bshw->bvhw", M, np.exp(Z - zm), optimize=True)) + zm
                e = e + msg[:, Vb]
                Ps.append(P)
                Zs.append(Z)
            m.append(e)
        top = cs[-1]
        D, (h, w) = self.p.chan[top].D, self.hw[self.p.chan[top].level]
        g = self._gumbel(seeds, step, (D, h, w))
        step += 1
        lg = m[-1] + g
        v = np.argmax(lg, axis=1)
        if not np.isfinite(np.take_along_axis(m[-1], v[:, None], 1)).all():
            return step                                     # no valid joint value somewhere: skip
        new = {top: v}
        for i in range(K - 1, -1, -1):
            f = chain[i]
            ca = self.p.chan[f.a[0]]
            Va = ca.views[f.a[1]]
            Vb = self.p.chan[f.b[0]].views[f.b[1]]
            vb = new[f.b[0]]
            P, Z = Ps[i], Zs[i]
            S = Z.shape[1]
            h, w = vb.shape[1:]
            ls = -f.table[Vb[vb]] / tau                     # (B, h, w, S)
            ls = np.moveaxis(ls, -1, 1) + Z
            g = self._gumbel(seeds, step, (S, h, w))
            step += 1
            r = np.argmax(ls + g, axis=1)
            child = np.zeros((vb.shape[0], 2 * h, 2 * w), np.int64)
            for k in range(3, -1, -1):
                e = m[i][:, :, (k >> 1)::2, (k & 1)::2]
                rem = r[:, None] - Va[None, :, None, None]
                if k > 0:
                    Pk = P[k - 1]
                    okr = (rem >= 0) & (rem < Pk.shape[1])
                    lw = np.where(okr, np.take_along_axis(Pk, np.clip(rem, 0, Pk.shape[1] - 1), 1),
                                  -np.inf)
                else:
                    lw = np.where(rem == 0, 0.0, -np.inf)
                g = self._gumbel(seeds, step, (ca.D, h, w))
                step += 1
                vk = np.argmax(e + lw + g, axis=1)
                child[:, (k >> 1)::2, (k & 1)::2] = vk
                r = r - Va[vk]
            new[f.a[0]] = child
        x.update(new)
        return step


# ------------------------------------------------------------------ public API
def init_state(problem, region, L, B, init="min"):
    """Constant start: every sampled channel at levels >= L at value 0 ('min') or
    D - 1 ('max').  Need not be valid: the least-violating moves repair it."""
    out = {}
    for c in problem.chans:
        if c.level >= L and c.designed is None:
            h, w = problem.H >> c.level, region.W >> c.level
            out[c.name] = np.full((B, h, w), 0 if init == "min" else c.D - 1, np.int64)
    return out


def sample_coarse_batch(problem, region, L, seeds, sweeps=8, tau=1.0, init="mix", start=None,
                        prior=None):
    """{channel: (B, h, w) int64} for every channel (sampled and designed) at levels >= L.

    seeds: (B,) world seeds (designed channels and noise are keyed by them).
    init: 'min' | 'max' | 'mix' (min for even seed, max for odd); start: optional
    {channel: (B, h, w)} start state (e.g. the model's own draw) overriding init."""
    seeds = np.asarray(seeds, np.int64).reshape(-1)
    B = seeds.size
    cp = prior if prior is not None else CoarsePrior(problem, region, L)
    if start is not None:
        x = {c.name: np.asarray(start[c.name], np.int64).copy() for c in cp.chans}
    elif init == "mix":
        lo, hi = init_state(problem, region, L, B, "min"), init_state(problem, region, L, B, "max")
        odd = (seeds % 2 == 1)[:, None, None]
        x = {k: np.where(odd, hi[k], lo[k]) for k in lo}
    else:
        x = init_state(problem, region, L, B, init)
    for c in problem.chans:
        if c.designed is not None and c.level >= L:
            x[c.name] = np.stack([problem.designed_values(int(s), region)[c.name] for s in seeds])
    step = 0
    order = sorted(cp.chans, key=lambda c: -c.level)
    inchain = {f.a[0] for ch in cp.chains for f in ch} | {f.b[0] for ch in cp.chains for f in ch}
    order = [c for c in order if c.name not in inchain]     # the chain move draws these jointly
    for _ in range(sweeps):
        for c in order:
            step = cp.site_sweep(x, c.name, seeds, step, tau)
        for ch in cp.chains:
            step = cp.chain_move(x, ch, seeds, step, tau)
    return x


def sample_coarse(problem, region, L, seed, sweeps=8, tau=1.0, init="mix"):
    """One world: {channel: (h, w) int64} for every channel at levels >= L."""
    x = sample_coarse_batch(problem, region, L, [seed], sweeps, tau, init)
    return {k: v[0] for k, v in x.items()}


def coarse_clamps(problem, seeds, L, region=None, sweeps=8, tau=1.0, init="mix"):
    """Interface for cstart: a list (one per world seed) of clamp dicts
    {sampled channel at levels >= L: (h, w) int32}, the format
    _creplay_clamp.clamped_pass / _cstart_core.train_pass_ctx take.
    Designed channels are not included (the pass computes them)."""
    region = region or Region(problem.W_train)
    x = sample_coarse_batch(problem, region, L, seeds, sweeps, tau, init)
    names = [c.name for c in problem.chans if c.level >= L and c.designed is None]
    return [{n: x[n][i].astype(np.int32) for n in names} for i in range(len(seeds))]


def hard_violations(problem, x, region, L):
    """Violated hard instances of the restricted factors in a batched state, per world."""
    from castlegen.bench.energy import instances
    fs = [f for f in problem.factors if f.hard and problem.chan[f.a[0]].level >= L
          and (f.b is None or problem.chan[f.b[0]].level >= L)]
    B = next(iter(x.values())).shape[0]
    out = np.zeros(B, np.int64)
    for i in range(B):
        xi = {k: v[i] for k, v in x.items()}
        out[i] = sum(int(np.isinf(instances(problem, f, xi, region)).sum()) for f in fs)
    return out


_PRIORS = {}


def _prior(problem, region, L):
    key = (problem.name, region.W, L)
    if key not in _PRIORS:
        _PRIORS[key] = CoarsePrior(problem, region, L)
    return _PRIORS[key]


def coarse_clamp_batch(problem, seeds, L, region=None, sweeps=8, tau=1.0, init="mix"):
    """Batched clamp {sampled channel at levels >= L: (B, h, w) int32} (the vmapped
    clamped_pass format)."""
    region = region or Region(problem.W_train)
    x = sample_coarse_batch(problem, region, L, seeds, sweeps, tau, init, prior=_prior(problem, region, L))
    return {c.name: x[c.name].astype(np.int32) for c in problem.chans
            if c.level >= L and c.designed is None}


def coarse_contexts(problem, seeds, Ls, region=None, sweeps=8, tau=1.0, init="mix", chans=None):
    """cstart's context format: {chan: (B, h, w) int64} for the sampled channels
    `chans` (default: every sampled channel at levels >= 1), world b filled from
    the restricted-energy prior at levels >= Ls[b] and 0 below (unused there);
    worlds with Ls[b] > T are left at 0.  Deterministic per (seed, L)."""
    region = region or Region(problem.W_train)
    seeds = np.asarray(seeds, np.int64).reshape(-1)
    Ls = np.broadcast_to(np.asarray(Ls, np.int64), seeds.shape)
    if chans is None:
        chans = [c for c in problem.chans if c.designed is None and c.level >= 1]
    out = {c.name: np.zeros((seeds.size, problem.H >> c.level, region.W >> c.level), np.int64)
           for c in chans}
    for L in np.unique(Ls):
        if L > problem.T:
            continue
        who = np.nonzero(Ls == L)[0]
        x = sample_coarse_batch(problem, region, int(L), seeds[who], sweeps, tau, init,
                                prior=_prior(problem, region, int(L)))
        for c in chans:
            if c.level >= L:
                out[c.name][who] = x[c.name]
    return out


def informative_channels(problem, L):
    """Sampled channels at levels >= L that carry a finite (soft) restricted factor
    entry, i.e. whose restricted-energy marginal is not just uniform over the
    hard-feasible set.  Read off the tables only."""
    cp = _prior(problem, Region(problem.W_train), L)
    out = []
    for c in cp.chans:
        soft = any(np.isfinite(f.table).any() and np.ptp(f.table[np.isfinite(f.table)]) > 0
                   for f in cp.touch[c.name])
        if soft:
            out.append(c.name)
    return out
