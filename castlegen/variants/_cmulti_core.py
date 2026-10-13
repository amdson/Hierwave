"""Core of cmulti (ideas.md #3): P coloured passes per level.

Schedule.  Levels T..0.  At each level, pass 1 is cfast's coloured pass (16
colours, channels in declared order, one-step lookahead).  Passes p = 2..P
sweep the same 16 colours again and redraw every site given the current
values of all its neighbours (same level now fully known, coarser levels
final, finer levels unknown: lookahead as in pass 1).  Each (level, channel,
pass) has its own actor.  After pass p >= 2 of a level, a backward-kernel
"r-step" scores the values the pass overwrote:
    r_p(x^{p-1} | x^p) = prod_sites r_phi(x^{p-1}_s | x^p in the actor window of s),
a normalised distribution over the earlier pass given the later one.

Objective (auxiliary-variable bound).  With the trajectory z = (x^1..x^P),
    J = E_q[ S(x^P) + lam c(x^P) + tau (log q(z) - log r(x^{<P} | x^P)) ]
      = E_q[ S + lam c + tau log q(x^P) ] + tau E_q KL(q(x^{<P} | x^P) || r)
     >= the stated objective, with equality iff r is q's posterior.
The per-draw costs telescope to J's integrand: a pass-1 draw costs its closed
energy (cfast); a pass-p >= 2 draw from old value o to a costs
soft(a) - soft(o) + lam (nv(a) - nv(o)) + tau log q(a) (same-colour sites
share no factor, so the changes add); an r-step costs -tau log r(o | x^p).

Dependency cones (MCone) generalise _cfast_cone.Cone to this step list.  The
r-step and later-pass draws read the site's own earlier value, which lies
inside the hull of any draw that changed it, so the cones stay conservative."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from ._cfast_cone import _cdiv, _fdiv, fine_reach
from ._cpass_core import ACTOR_WIN, NCOL, P as PL, key_of, mlp


def gkey(l, cname, p):
    """Parameter / record key of the (level, channel, pass) draw group."""
    return key_of(l, cname) if p == 1 else f"{key_of(l, cname)}@p{p}"


def rkey(l, cname, p):
    return f"{key_of(l, cname)}@r{p}"


def noise_tag(cname, p):
    return noise.tag_of(cname if p == 1 else f"{cname}#pass{p}")


class Sched:
    """Global step order for P passes per level.  groups[key] = (l, ci, cname,
    p, kind) with kind 'draw' or 'r'; every group has 16 steps (colours)."""

    def __init__(self, lay, P):
        self.lay, self.P = lay, P
        self.groups, self.first = {}, {}
        steps, s = [], 0
        for l in lay.levels:
            chans = lay.chans[l]
            if not chans:
                continue
            for p in range(1, P + 1):
                for ci, c in enumerate(chans):
                    self.groups[gkey(l, c.name, p)] = (l, ci, c.name, p, "draw")
                    self.first[gkey(l, c.name, p)] = s + ci
                for k in range(NCOL):
                    for ci, c in enumerate(chans):
                        steps.append((l, k, gkey(l, c.name, p)))
                s += NCOL * len(chans)
                if p >= 2:
                    for ci, c in enumerate(chans):
                        self.groups[rkey(l, c.name, p)] = (l, ci, c.name, p, "r")
                        self.first[rkey(l, c.name, p)] = s + ci
                    for k in range(NCOL):
                        for ci, c in enumerate(chans):
                            steps.append((l, k, rkey(l, c.name, p)))
                    s += NCOL * len(chans)
        self.steps, self.nsteps = steps, s

    def step(self, key, k):
        l, ci, cname, p, kind = self.groups[key]
        return self.first[key] + k * len(self.lay.chans[l])

    def draw_keys(self):
        return [k for k, g in self.groups.items() if g[4] == "draw"]

    def r_keys(self):
        return [k for k, g in self.groups.items() if g[4] == "r"]


# ------------------------------------------------------------------ pass
def multi_pass(sch, params, rparams, seed, designed, tau, lam, record=True, forced=None, override=None):
    """One world.  forced: {group key: (h, w) values} replaces the draws of a
    group (trajectory scoring).  override: (step, site, value, on) replaces
    one draw (cone tests).  Returns (state, recs) with recs[key] =
    {feat, soft, nv, mask, a, valid, logq, cost, dv} (draw groups, each
    (16, n, ...)) or {feat, o, valid, cost} (r groups); with record=False,
    recs[key] = cost (16, n) only."""
    lay, P = sch.lay, sch.P
    p_ = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p_.chans}
    recs = {}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue
        h, w = lay.hw[l]
        for p in range(1, P + 1):
            prev = {c.name: state[c.name] for c in chans}

            def body(state, k, l=l, chans=chans, p=p):
                Y, X, valid = lay.sites(l, k)
                Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
                Yc, Xc = jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)
                idx = jnp.arange(Y.shape[0])
                outs = []
                for ci, c in enumerate(chans):
                    key = gkey(l, c.name, p)
                    soft, nv, look = lay.cond(state, c.name, Y, X)
                    v = nv + look
                    mask = (v == jnp.min(v, axis=1, keepdims=True)) | ~valid[:, None]
                    feat = lay.features(state, l, Y, X, ACTOR_WIN)
                    b = mlp(params[key], feat)
                    logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, axis=1)
                    if forced is not None:
                        a = forced[key][Yc, Xc]
                    else:
                        g = noise.gumbel(jnp, seed, noise_tag(c.name, p), l, Y, Xn, c.D)
                        a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                    if override is not None:
                        os_, oj, oval, on = override
                        st = sch.first[key] + k * len(chans)
                        a = jnp.where(on & (st == os_) & (idx == oj), oval, a)
                    la = logq[idx, a]
                    if p == 1:
                        dv = nv[idx, a]
                        e = soft[idx, a] + lam[l] * nv[idx, a]
                    else:
                        o = state[c.name][Yc, Xc]
                        dv = nv[idx, a] - nv[idx, o]
                        e = soft[idx, a] - soft[idx, o] + lam[l] * dv
                    cost = jnp.where(valid, e + tau * la, 0.0)
                    if record:
                        outs.append({"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a,
                                     "valid": valid, "logq": logq, "cost": cost,
                                     "dv": jnp.where(valid, dv, 0)})
                    else:
                        outs.append(cost)
                    Ys = jnp.where(valid, Y, h)
                    state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
                return state, outs

            state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
            for ci, c in enumerate(chans):
                recs[gkey(l, c.name, p)] = ys[ci]
            if p >= 2:                                   # backward kernel scores x^{p-1} given x^p
                def rbody(_, k, l=l, chans=chans, p=p, prev=prev, state=state):
                    Y, X, valid = lay.sites(l, k)
                    Yc, Xc = jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)
                    idx = jnp.arange(Y.shape[0])
                    feat = lay.features(state, l, Y, X, ACTOR_WIN)
                    outs = []
                    for c in chans:
                        o = prev[c.name][Yc, Xc]
                        lr = jax.nn.log_softmax(mlp(rparams[rkey(l, c.name, p)], feat), axis=1)[idx, o]
                        cost = jnp.where(valid, -tau * lr, 0.0)
                        outs.append({"feat": feat, "o": o, "valid": valid, "cost": cost} if record else cost)
                    return None, outs

                _, ys = jax.lax.scan(rbody, None, jnp.arange(NCOL))
                for ci, c in enumerate(chans):
                    recs[rkey(l, c.name, p)] = ys[ci]
    return state, recs


def margin(problem, P):
    """Fine cells of padding bounding the dependency radius of P passes per
    level plus one repair round (cpass's bound with P sweeps per level)."""
    m = sum((P * NCOL * 3 + NCOL * 2 + 8) << l for l in range(problem.T + 1))
    step = PL << problem.T
    return -(-m // step) * step


def check_colouring(problem):
    """Same-colour sites (PL apart) must share no factor: every same-level
    pair offset is shorter than PL (count siblings differ in colour)."""
    for f in problem.factors:
        if f.kind == "pair" and problem.chan[f.a[0]].level == problem.chan[f.b[0]].level:
            assert abs(f.off[0]) < PL and abs(f.off[1]) < PL, f.name


# ------------------------------------------------------------------ cones
class MCone:
    """Dependency-cone boxes and gather tables over a Sched's step list (see
    _cfast_cone.Cone: same hull walk, generic step list)."""

    def __init__(self, sch, win):
        lay = sch.lay
        assert lay.periodic and lay.x0 == 0
        self.sch, self.lay = sch, lay
        p = lay.p
        self.reach = [fine_reach(p, l, sch.groups[key][2], win) for (l, k, key) in sch.steps]
        self.ii_off, off = {}, 1
        for s, (l, k, key) in enumerate(sch.steps):
            ny, nx = lay.nyx[l]
            self.ii_off[s] = off
            off += (ny + 1) * (2 * nx + 1)
        self.ii_size = off
        self.idx, self.boxes, self.s_first = {}, {}, {}
        for key in sch.groups:
            self.boxes[key], self.idx[key], self.s_first[key] = self._tables(key)

    def _tables(self, key):
        sch, lay = self.sch, self.lay
        l0 = sch.groups[key][0]
        n = lay.nsite(l0)
        s_first = min(sch.step(key, k) for k in range(NCOL))
        S = sch.nsteps - s_first
        boxes = np.zeros((NCOL, n, S, 4), np.int64)
        boxes[..., 0], boxes[..., 1] = 1, 0
        idx = np.zeros((NCOL, n, S, 4), np.int32)
        k0 = np.repeat(np.arange(NCOL), n)
        ny0, nx0 = lay.nyx[l0]
        iy, ix = np.divmod(np.tile(np.arange(n), NCOL), nx0)
        Y0 = k0 // PL + PL * iy
        X0 = k0 % PL + PL * ix
        s0 = np.array([sch.step(key, k) for k in range(NCOL)])[k0]
        hy0, hy1 = Y0 << l0, ((Y0 + 1) << l0) - 1
        hx0, hx1 = X0 << l0, ((X0 + 1) << l0) - 1
        bx_ = boxes.reshape(NCOL * n, S, 4)
        ix_ = idx.reshape(NCOL * n, S, 4)
        for s in range(s_first, sch.nsteps):
            l, k, _ = sch.steps[s]
            ay, by, ax, bx = self.reach[s]
            h, w = lay.hw[l]
            ny, nx = lay.nyx[l]
            cy, cx = k // PL, k % PL
            u = 1 << l
            Ylo = np.maximum(_fdiv(hy0 - by, u), 0)
            Yhi = np.minimum(_fdiv(hy1 + ay, u), h - 1)
            iy0 = np.maximum(_cdiv(Ylo - cy, PL), 0)
            iy1 = _fdiv(Yhi - cy, PL)
            ix0 = _cdiv(_fdiv(hx0 - bx, u) - cx, PL)
            ix1 = _fdiv(_fdiv(hx1 + ax, u) - cx, PL)
            full = ix1 - ix0 + 1 >= nx
            ix0 = np.where(full, 0, ix0)
            ix1 = np.where(full, nx - 1, ix1)
            on = (s > s0) & (iy0 <= iy1) & (ix0 <= ix1)
            j = s - s_first
            bx_[on, j] = np.stack([iy0, iy1, ix0, ix1], 1)[on]
            hy0 = np.where(on, np.minimum(hy0, (cy + PL * iy0) << l), hy0)
            hy1 = np.where(on, np.maximum(hy1, ((cy + PL * iy1 + 1) << l) - 1), hy1)
            hx0 = np.where(on, np.minimum(hx0, (cx + PL * ix0) << l), hx0)
            hx1 = np.where(on, np.maximum(hx1, ((cx + PL * ix1 + 1) << l) - 1), hx1)
            a = np.mod(ix0, nx)
            b = a + (ix1 - ix0)
            Wd = 2 * nx + 1
            o = self.ii_off[s]
            r0, r1, c0, c1 = iy0, iy1 + 1, a, b + 1
            g = np.stack([r1 * Wd + c1, r0 * Wd + c1, r1 * Wd + c0, r0 * Wd + c0], 1) + o
            ix_[on, j] = g[on]
        return boxes, idx, s_first

    def integral(self, costs):
        """costs: {group key: (B, 16, n)} -> (B, ii_size) integral images."""
        sch, lay = self.sch, self.lay
        Bn = next(iter(costs.values())).shape[0]
        parts = [None] * sch.nsteps
        for key, c in costs.items():
            l = sch.groups[key][0]
            ny, nx = lay.nyx[l]
            img = c.reshape(Bn, NCOL, ny, nx)
            img = jnp.concatenate([img, img], -1)
            ii = jnp.cumsum(jnp.cumsum(img, -2), -1)
            ii = jnp.pad(ii, ((0, 0), (0, 0), (1, 0), (1, 0)))
            for k in range(NCOL):
                parts[sch.step(key, k)] = ii[:, k].reshape(Bn, -1)
        return jnp.concatenate([jnp.zeros((Bn, 1), jnp.float32)] + parts, -1)

    def returns(self, costs, keys=None):
        ii = self.integral(costs)
        G = {}
        for key in (keys or self.sch.draw_keys()):
            v = ii[:, jnp.asarray(self.idx[key])]
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        return G

    def member(self, key, k, j):
        """{key2: (16, n2) bool} steps' sites inside the cone of draw (key, k, j)."""
        sch, lay = self.sch, self.lay
        bx = self.boxes[key][k, j]
        out = {}
        for key2, (l, ci, cn, p, kind) in sch.groups.items():
            ny, nx = lay.nyx[l]
            m = np.zeros((NCOL, ny, nx), bool)
            for kk in range(NCOL):
                jj = sch.step(key2, kk) - self.s_first[key]
                if jj < 0:
                    continue
                iy0, iy1, ix0, ix1 = bx[jj]
                if iy0 > iy1:
                    continue
                cols = np.mod(np.arange(ix0, ix1 + 1), nx)
                m[kk, iy0:iy1 + 1][:, cols] = True
            out[key2] = m.reshape(NCOL, -1)
        return out
