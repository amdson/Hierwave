"""Annealed subtree proposals for certland (notes/certland_chain_spec.md,
section certland_anneal; the plan doc's "Hard constraints: annealed
proposals").

At a coarse root an ordered proposal pass with hard rows can paint itself
into a corner, so plain proposals die.  Here a proposal is

  z_0 ~ q            the soft pass (pass_subtree at L0: a hard row costs L0
                     per violation instead of excluding; never dies)
  z_m ~ T_m(z_{m-1}) m = 1..M, one MH sweep over the subtree leaving pi_m
                     invariant

on the path

  pi_m(z) ~ q(z)^(1 - b_m) exp(-b_m E_soft(z) - L_m viol(z))

with b_m = m / M and L_0 = 0 (so pi_0 = q exactly), L_1..L_{M-1} geometric
L0 -> Lmax, L_M = inf: pi_M is p* (exp(-soft) on valid states).  log q(z)
of an arbitrary state is the forced pass at L0, recomputed in full for
every MH proposal (an incremental delta-log-q is a later optimisation).
E_soft and viol come from subtree_energy.

MH proposals (moves="both", the default; "sub" or "site" for one kind).
A sub-subtree move at every non-tile cell (l, y, x) of the subtree is an
independence MH proposal from the chain's ordered pass on the subtree of
(l, y, x) at L = L_m (pass_subtree(L0=L_m), inf at m = M); the pass reads
nothing of the region it draws, so q_sub is a fixed function of the rest
and the reverse probability is the forced pass of the old values.  These
repair corners that need several cells at once.  A non-tile cell
(l, ch, y, x) proposes t' from its predictor conditional
c_z(t) ~ softmax(-cell_conditional(world, preds, l, ch, y, x, L=L_m))
(designed soft + bias + L_m per violation against the
current cells; L_M = inf excludes); the reverse probability c_{z'}(t) is
evaluated at the proposed state, so the ratio is right even if the
conditional depended on the cell's own value.  Tiles are proposed per
2 x 2 block from the exact 16-state designed conditional (cl.window_terms
at level 4, L_m per violation), which does not depend on the block's own
values.  Each update is an MH kernel reversible w.r.t. pi_m; the sweep T_m
is their composition in a fixed order (levels top-down, raster; per cell
the sub-subtree move, then rho, then cert; then the tile blocks), so its
reversal R_m is the same updates in the reverse order.  Cost: every
accepted-or-not proposal recomputes subtree_energy and (b < 1) the full
forced pass for log q (incremental delta-log-q is a later optimisation).

Path options (anneal_path): Lmax, L1 (start of the geometric L schedule)
and bsteps (beta reaches 1 after bsteps steps; default M, the spec's
linear path); order = the pass order of every pass used.

Weights (extended-space AIS inside i-SIR).  Forward proposal density
Q(z_0:M) = q(z_0) prod_m T_m(z_{m-1}, z_m); extended target
P(z_0:M) = p*(z_M) prod_m R_m(z_m, z_{m-1}).  Then
  log P/Q = sum_{m=1..M} [log pi~_m(z_{m-1}) - log pi~_{m-1}(z_{m-1})] + const,
-inf when z_{M-1} violates (dead; T_M is then not applied).  Particle 0
(the current valid state x) gets a path drawn from P(z_0:M-1 | z_M = x):
z_{m-1} = R_m(z_m) for m = M..1 (reversed sweeps, starting at pi_M), and
its weight is the same sum evaluated on that path.  Conditional SIR on the
extended space with this refreshed particle 0 leaves P, hence p* on z_M,
invariant: selecting j ~ w and installing z_M^(j) is exact."""
import numpy as np
from numba import njit

from . import certland as cl
from . import certland_chain as ch_

NT = cl.NL - 1                                  # the tile level


# ------------------------------------------------------------------ path
def anneal_path(M, L0, Lmax=None, bsteps=None, L1=None):
    """betas (M+1,): 0 -> 1 linearly over the first bsteps steps (default M:
    the spec's linear path), then 1.  Ls (M+1,): Ls[0] = 0 (so pi_0 = q
    exactly), Ls[1..M-1] geometric L1 (default L0) -> Lmax (default 8 L0),
    Ls[M] = inf."""
    assert M >= 1
    if Lmax is None:
        Lmax = 8.0 * L0
    if L1 is None:
        L1 = L0
    if bsteps is None:
        bsteps = M
    betas = np.minimum(np.arange(M + 1) / bsteps, 1.0)
    Ls = np.zeros(M + 1)
    if M >= 2:
        k = np.arange(M - 1)
        Ls[1:M] = L1 * (Lmax / L1) ** (k / max(M - 2, 1))
    Ls[M] = np.inf
    return betas, Ls


def _logpi(b, L, logq, E, nv):
    """log pi~ for (beta, L) of a state with components (logq, E, nv)."""
    if nv > 0 and L == np.inf:
        return -np.inf
    v = -b * E - (L * nv if nv > 0 else 0.0)
    if b < 1.0:
        if logq == -np.inf:
            return -np.inf
        v += (1.0 - b) * logq
    return v


# ------------------------------------------------------------------ tiles
@njit(cache=True)
def _tile_block_energies(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                         Er, Ec, Epr, Epc, th, m, learned, by, bx, L, out):
    """out[c] = soft + L * viol of tile block (by, bx) set to state c (bit k
    of c = tile k in raster order of the 2 x 2), against the current
    cells; +inf for a violation when L = inf.  The block is restored."""
    y0, x0 = 2 * by, 2 * bx
    save = rho[y0:y0 + 2, x0:x0 + 2].copy()
    for c in range(16):
        rho[y0, x0] = c & 1
        rho[y0, x0 + 1] = (c >> 1) & 1
        rho[y0 + 1, x0] = (c >> 2) & 1
        rho[y0 + 1, x0 + 1] = (c >> 3) & 1
        s, nv = cl.window_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                                Er, Ec, Epr, Epc, th, m, learned, by, bx)
        if nv == 0:
            out[c] = s
        elif L == np.inf:
            out[c] = np.inf
        else:
            out[c] = s + L * nv
    rho[y0:y0 + 2, x0:x0 + 2] = save


def _tile_args(world):
    a = list(world.args(NT))
    a[-3], a[-2], a[-1] = np.zeros(1), 0, 0
    return tuple(a)


def _block_get(t, by, bx):
    b = t[2 * by:2 * by + 2, 2 * bx:2 * bx + 2]
    return int(b[0, 0] | (b[0, 1] << 1) | (b[1, 0] << 2) | (b[1, 1] << 3))


def _block_set(t, by, bx, c):
    t[2 * by, 2 * bx] = c & 1
    t[2 * by, 2 * bx + 1] = (c >> 1) & 1
    t[2 * by + 1, 2 * bx] = (c >> 2) & 1
    t[2 * by + 1, 2 * bx + 1] = (c >> 3) & 1


# ------------------------------------------------------------------ sweep
def _sites(i, ry, rx, moves="both"):
    """The fixed site order of T_m (levels top-down, raster):
      ('s', l, y, x)       sub-subtree move at every non-tile cell (moves "sub"/"both")
      ('c', l, ch, y, x)   single-cell move, rho then cert (moves "site"/"both")
      ('t', by, bx)        2 x 2 tile block (moves "site"/"both")"""
    assert moves in ("sub", "site", "both")
    out = []
    for (l, y0, y1, x0, x1) in ch_.subtree_slices(i, ry, rx):
        if l < NT:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if moves != "site":
                        out.append(('s', l, y, x))
                    if moves != "sub":
                        out.append(('c', l, cl.RHO, y, x))
                        out.append(('c', l, cl.CERT, y, x))
        elif moves != "sub":
            for by in range(y0 // 2, y1 // 2):
                for bx in range(x0 // 2, x1 // 2):
                    out.append(('t', by, bx))
    return out


def _logsoftmax(e):
    """log softmax(-e) with +inf entries -> -inf."""
    mn = e.min()
    if not np.isfinite(mn):
        return None
    z = -(e - mn)
    return z - np.log(np.exp(z).sum())


class _Chain:
    """The subtree state's components (logq, E, nv) and its MH sweeps."""

    def __init__(self, world, preds, i, ry, rx, L0, rng, info, moves="both", order="raster"):
        self.w, self.p, self.i, self.ry, self.rx = world, preds, i, ry, rx
        self.L0, self.rng, self.info = L0, rng, info
        self.sites = _sites(i, ry, rx, moves)
        self.order = order
        self.targs = _tile_args(world)
        self.e16 = np.empty(16)
        self.logq = None
        self.E, self.nv = ch_.subtree_energy(world, i, ry, rx)

    def q(self):
        if self.logq is None:
            lq, dead = ch_.pass_subtree(self.w, self.p, self.i, self.ry, self.rx, self.rng,
                                        forced=True, L0=self.L0, order=self.order)
            self.logq = -np.inf if dead else float(lq)
            self.info["forced"] = self.info.get("forced", 0) + 1
        return self.logq

    def logpi(self, b, L):
        if b < 1.0 and not (self.nv > 0 and L == np.inf):
            self.q()
        return _logpi(b, L, self.logq, self.E, self.nv)

    def sweep(self, b, L, reverse=False):
        cur = self.logpi(b, L)
        sites = self.sites[::-1] if reverse else self.sites
        for s in sites:
            cur = self._site(s, b, L, cur)

    def _site(self, s, b, L, cur):
        w = self.w
        if s[0] == 's':
            # independence MH on the sub-subtree of (l, y, x): proposal = the ordered pass at L
            # (it reads nothing of the region it draws, so q_sub is a fixed function of the rest)
            _, l, y, x = s
            sold = ch_.get_subtree(w, l, y, x)
            lrev, d = ch_.pass_subtree(w, self.p, l, y, x, self.rng, forced=True, L0=L, order=self.order)
            if d:
                lrev = -np.inf
            lfwd, d = ch_.pass_subtree(w, self.p, l, y, x, self.rng, forced=False, L0=L, order=self.order)
            restore = lambda: ch_.set_subtree(w, l, y, x, sold)
            if d or lrev == -np.inf:
                restore()
                return cur
        elif s[0] == 'c':
            _, l, chn, y, x = s
            g = w.rho[l] if chn == cl.RHO else w.cert[l]
            old = int(g[y, x])
            lf = _logsoftmax(ch_.cell_conditional(w, self.p, l, chn, y, x, L=L))
            if lf is None:
                return cur
            new = int(self.rng.choice(len(lf), p=np.exp(lf)))
            if new == old:
                return cur
            g[y, x] = new
            lr = _logsoftmax(ch_.cell_conditional(w, self.p, l, chn, y, x, L=L))
            restore = lambda: g.__setitem__((y, x), old)
            lfwd, lrev = lf[new], (lr[old] if lr is not None else -np.inf)
        else:
            _, by, bx = s
            t = w.rho[NT]
            old = _block_get(t, by, bx)
            _tile_block_energies(*self.targs, by, bx, L, self.e16)
            lf = _logsoftmax(self.e16)
            if lf is None:
                return cur
            new = int(self.rng.choice(16, p=np.exp(lf)))
            if new == old:
                return cur
            _block_set(t, by, bx, new)
            restore = lambda: _block_set(t, by, bx, old)
            lfwd, lrev = lf[new], lf[old]                # the 16-state conditional ignores the block's values
        self.info["prop"] = self.info.get("prop", 0) + 1
        saved = (self.logq, self.E, self.nv)
        self.logq = None
        self.E, self.nv = ch_.subtree_energy(w, self.i, self.ry, self.rx)
        nxt = self.logpi(b, L)
        a = nxt - cur + lrev - lfwd
        if nxt > -np.inf and (a >= 0 or self.rng.random() < np.exp(a)):
            self.info["acc"] = self.info.get("acc", 0) + 1
            return nxt
        restore()
        self.logq, self.E, self.nv = saved
        return cur


# ------------------------------------------------------------------ AIS runs
def annealed_forward(world, preds, i, ry, rx, M, L0, rng, info=None, moves="both", order="raster", **path):
    """Soft pass, then M MH sweeps T_1..T_M; returns (logw, dead) with logw
    the AIS weight sum_m [log pi~_m(z_{m-1}) - log pi~_{m-1}(z_{m-1})] (-inf
    and dead = 1 when z_{M-1} still violates; T_M is then skipped).  Leaves
    the final state z_M in the world.  path: anneal_path's Lmax, bsteps."""
    info = {} if info is None else info
    betas, Ls = anneal_path(M, L0, **path)
    logq, dead = ch_.pass_subtree(world, preds, i, ry, rx, rng, forced=False, L0=L0, order=order)
    if dead or not np.isfinite(logq):
        return -np.inf, 1
    c = _Chain(world, preds, i, ry, rx, L0, rng, info, moves, order)
    c.logq = float(logq)
    info.setdefault("nv0", []).append(c.nv)
    lw = 0.0
    for m in range(1, M + 1):
        if m == M:
            info.setdefault("nv_end", []).append(c.nv)
        lw += c.logpi(betas[m], Ls[m]) - c.logpi(betas[m - 1], Ls[m - 1])
        if lw == -np.inf:
            return -np.inf, 1
        c.sweep(betas[m], Ls[m])
    return float(lw), 0


def annealed_backward(world, preds, i, ry, rx, M, L0, rng, info=None, moves="both", order="raster", **path):
    """Particle 0: from the current (valid) state z_M, z_{m-1} = R_m(z_m) for
    m = M..1 (R_m = T_m's updates in reverse order), weight sum_m
    [log pi~_m(z_{m-1}) - log pi~_{m-1}(z_{m-1})] on that path.  Leaves z_0
    in the world (the caller restores)."""
    info = {} if info is None else info
    betas, Ls = anneal_path(M, L0, **path)
    c = _Chain(world, preds, i, ry, rx, L0, rng, info, moves, order)
    assert c.nv == 0, "particle 0 must be valid"
    lw = 0.0
    for m in range(M, 0, -1):
        c.sweep(betas[m], Ls[m], reverse=True)
        lw += c.logpi(betas[m], Ls[m]) - c.logpi(betas[m - 1], Ls[m - 1])
    assert np.isfinite(lw), lw
    return float(lw)


def _ncells(s0, s1):
    n = 0
    for (r0, c0), (r1, c1) in zip(s0, s1):
        n += int((r0 != r1).sum())
        if c0 is not r0:
            n += int((c0 != c1).sum())
    return n


def annealed_move(world, preds, i, ry, rx, K, rng, collect=None, M=8, L0=3.0, info=None,
                  moves="both", order="raster", **path):
    """subtree_move's contract with annealed proposals: particle 0 the
    current state with its backward-run weight, K forward annealed
    proposals; select j ~ w, collect, install.  Returns dict(logw (K+1,),
    chosen, dead, ess, changed)."""
    info = {} if info is None else info
    s0 = ch_.get_subtree(world, i, ry, rx)
    logw = np.empty(K + 1)
    states = [s0]
    logw[0] = annealed_backward(world, preds, i, ry, rx, M, L0, rng, info, moves, order, **path)
    dead = 0
    for j in range(1, K + 1):
        ch_.set_subtree(world, i, ry, rx, s0)
        lw, d = annealed_forward(world, preds, i, ry, rx, M, L0, rng, info, moves, order, **path)
        logw[j] = lw
        dead += d
        states.append(ch_.get_subtree(world, i, ry, rx))
    mx = logw.max()
    w = np.exp(logw - mx)
    wbar = w / w.sum()
    chosen = int(rng.choice(K + 1, p=wbar))
    ess = float(1.0 / (wbar ** 2).sum())
    if collect is not None:
        ch_.set_subtree(world, i, ry, rx, s0)
        collect(world, i, ry, rx, states, wbar)
    ch_.set_subtree(world, i, ry, rx, states[chosen])
    return dict(logw=logw, chosen=chosen, dead=dead, ess=ess, changed=_ncells(s0, states[chosen]),
                n=ch_.flat_size(i))
