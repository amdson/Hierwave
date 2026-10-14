"""Core of crevise (ideas.md #19, constitution Amendment 2026-10-14 (b)):
the coloured pass followed by a fixed number of revising steps per level,
trained by the path KL.

Generation (one world; vmapped by callers).  Levels T..0.  At level l, step
m = 0 is cfast's coloured pass (16 colours, channels in declared order,
one-step lookahead, Gumbel-max with coordinate-keyed noise).  Steps
m = 1..R_l are revising steps:
  * the declared noising erases each (channel, site) of level l independently
    with probability rho (a coin hashed from (seed, "revise:<chan>#m", l, Y,
    X): coordinate-keyed, fixed, not learned);
  * every erased cell is unknown (-1) until redrawn; the 16 colours are then
    swept again and the erased cells of each colour are redrawn given
    everything known at that moment (unerased same-level cells, redrawn
    erased cells of earlier colours, coarser levels), with the same mask /
    lookahead rule and their own Gumbel noise (tag "<chan>#rev<m>").
Then cpass's one repair round (in the variant).  Fixed work: (1 + R_l) sweeps
per level; finite support with the dependency radius of (1 + R_l) sweeps
(`margin`).

Objective: the path KL.  Q is the generator's trajectory distribution (pass-1
values, masks, redrawn values, level by level); P = p*(x_0) prod_k r(x_k |
x_{k-1}) where the fixed noising r, read from the final state backwards,
erases level l's revising steps in reverse order (each cell independently
with probability rho, refilled uniformly over its D values) and then erases
the level.  The mask probabilities are the same in Q and P and cancel, so
    KL(Q || P) + log Z = E_Q[ S(x_0) + sum_l lam_l H_l(x_0) + tau log Q(traj) + tau sum_masked log D ]
(tau = 1 at the end of training; tau multiplies both log Q and -log r).  It
upper-bounds the reverse KL of x_0 by E_Q KL(Q(earlier | x_0) || r(earlier | x_0))
and equals it when R = 0 (then r is deterministic and the objective is
cfast's).  Per-draw costs that sum to the integrand:
  * pass-1 draw of value a:  soft(a) + lam_l nv(a) + tau log q(a)   (closed terms, as cfast)
  * redraw of an erased cell from old value o to a:
        [soft_h(a) - soft_h(o)] + lam_l [nv_h(a) - nv_h(o)] + tau (log q(a) + log D)
    where soft_h / nv_h are the closed terms in the "hybrid" state (erased
    cells of earlier colours of this step at their new values, erased cells
    of this and later colours at their old values, everything else as is).
    Colour by colour these differences telescope to E(x^{m+1}) - E(x^m), so the
    sum over the trajectory is S(x_0) + lam H(x_0) + tau log Q + tau |M| log D
    (checked by tests/variants/test_crevise.py).  log D is constant over the
    candidate values, so it adds no gradient; it is kept so the summed costs
    are the bound's integrand.
The dependency cones (cmulti's MCone over this step list) make cfast's cone
returns, critic control variate and lambda control apply unchanged.

Guided revising steps (coordinator / user decision, 2026-10-14 evening; ideas
#21; an allowed freedom: more information to learned parts).  With guide=True
the bias network of a revising step also reads a violation field over its
window: before the step erases anything, every (channel, cell) of the level
and of the parent level gets the number of violated hard instances touching
it in the current state (nv of the current value, from the same local
conditional the draw uses: the binary stand-in for graded.py's graded
distance, which has no per-cell JAX form yet), and the draw's features are
cfast's plus this field over the actor's same-level and parent windows
(zero where the window leaves the region).  The field is fixed for the step,
so it reads only values fixed before the draw; its reach is the window's plus
the factor scopes of the window cells (`RCone`, `margin`).  Revising steps
then have their own actors (the input differs from the pass's).  The noising
r is unchanged: a fixed rate rho, independent of the field.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from castlegen.bench import noise

from ._cfast_cone import fine_reach
from ._cfast_core import _cols, _encode, _geom, fast_features
from ._cmulti_core import MCone
from ._cpass_core import NCOL, P as PL, Window, key_of, mlp


def gkey(l, cname, m):
    """Record / critic key of the (level, channel, step) draw group; m = 0 is the pass."""
    return key_of(l, cname) if m == 0 else f"{key_of(l, cname)}@rev{m}"


def noise_tag(cname, m):
    return noise.tag_of(cname if m == 0 else f"{cname}#rev{m}")


def mask_tag(cname, m):
    return noise.tag_of(f"revise:{cname}#{m}")


def steps_per_level(problem, R):
    """R: int (every level) or a sequence indexed by level -> {level: R_l}."""
    T = problem.T
    if isinstance(R, (list, tuple)):
        assert len(R) == T + 1, (R, T)
        return {l: int(R[l]) for l in range(T + 1)}
    return {l: int(R) for l in range(T + 1)}


class Sched:
    """Step order for the pass plus R_l revising steps per level.  groups[key]
    = (l, ci, cname, m, "draw"); every group has 16 steps (colours).  The
    interface _cmulti_core.MCone reads (lay, groups, steps, step, nsteps,
    draw_keys)."""

    def __init__(self, lay, R):
        self.lay = lay
        self.R = steps_per_level(lay.p, R)
        self.groups, self.first = {}, {}
        steps, s = [], 0
        for l in lay.levels:
            chans = lay.chans[l]
            if not chans:
                continue
            for m in range(self.R[l] + 1):
                for ci, c in enumerate(chans):
                    self.groups[gkey(l, c.name, m)] = (l, ci, c.name, m, "draw")
                    self.first[gkey(l, c.name, m)] = s + ci
                for k in range(NCOL):
                    for ci, c in enumerate(chans):
                        steps.append((l, k, gkey(l, c.name, m)))
                s += NCOL * len(chans)
        self.steps, self.nsteps = steps, s

    def step(self, key, k):
        l = self.groups[key][0]
        return self.first[key] + k * len(self.lay.chans[l])

    def draw_keys(self):
        return list(self.groups)

    def actor_key(self, key, share, guide=False):
        l, ci, cname, m, _ = self.groups[key]
        return key_of(l, cname) if ((share and not guide) or m == 0) else key

    def field_dim(self, l):
        """Width of the guide features at level l: the actor's same-level window over the level's
        sampled channels plus its parent window over the parent's sampled channels."""
        lay = self.lay
        win = lay.actor_win
        n = len(win.same) * len(lay.chans[l])
        if l < lay.T:
            n += len(win.parent) * len(lay.chans[l + 1])
        return n


# ------------------------------------------------------------------ pass
def violation_field(lay, state, l):
    """(h * w, n_chans) float32: for every sampled channel of level l, the number of violated hard
    instances touching each cell in `state` (nv of the current value); 0 for unknown cells."""
    h, w = lay.hw[l]
    Y, X = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    Y, X = Y.ravel(), X.ravel()
    cols = []
    for c in lay.chans[l]:
        _, nv, _ = lay.cond(state, c.name, Y, X)
        cur = state[c.name].reshape(-1)
        f = nv[jnp.arange(Y.shape[0]), jnp.maximum(cur, 0)]
        cols.append(jnp.where(cur >= 0, f, 0).astype(jnp.float32))
    return jnp.stack(cols, -1)


def guide_features(lay, state, l, g):
    """(16, n, field_dim) guide features of every site of level l (all colours) from the violation
    fields of level l and of the parent level over the actor's windows (0 outside the region)."""
    Fs = violation_field(lay, state, l)                               # (h w, nc)
    f = (Fs[g["iw"]] * g["ok_s"]).reshape(g["iw"].shape[:2] + (-1,))          # ok_s is (16, n, m, 1)
    out = [f]
    if l < lay.T and lay.chans[l + 1]:
        Fp = violation_field(lay, state, l + 1)
        out.append((Fp[g["ip"]] * g["ok_p"]).reshape(g["ip"].shape[:2] + (-1,)))
    return jnp.concatenate(out, -1)


def revise_pass(sch, params, seed, designed, tau, lam, rho, share=True, record=True, forced=None,
                override=None, mask_seed=None, guide=False):
    """One world.  forced: {group key: (h, w) values} replaces the draws of a
    group (trajectory scoring; erased cells read their forced value).
    override: (step, site, value, on) replaces one draw (cone tests).
    mask_seed: seed of the erase coins (default: the world seed).  Returns
    (state, recs) with recs[key] = {feat, soft, nv, mask, a, valid, logq,
    cost, dv} each (16, n, ...): valid marks the group's draws (at revising
    steps: the erased sites), cost the per-draw path-KL cost, dv the draw's
    change in closed violations; with record=False, recs[key] = cost."""
    lay = sch.lay
    assert not lay.resid, "crevise: count-residual features are not supported"
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs = {}
    Ep = None
    ms = seed if mask_seed is None else mask_seed
    for l in lay.levels:
        chans = lay.chans[l]
        Es = _encode(lay, l, state)
        if not chans:
            Ep = Es
            continue
        g = {k: jnp.asarray(v) for k, v in _geom(lay, l).items()}
        h, w = lay.hw[l]
        cols = _cols(lay, l)
        for m in range(sch.R[l] + 1):
            gum = [] if forced is not None else [
                noise.gumbel(jnp, seed, noise_tag(c.name, m), l, g["Y"], g["Xn"], c.D) for c in chans]
            if m == 0:
                msk = [g["valid"] for c in chans]
                hyb = None
            else:
                msk = [(noise.uniform(jnp, ms, mask_tag(c.name, m), l, g["Y"], g["Xn"]) < rho) & g["valid"]
                       for c in chans]
                gf = guide_features(lay, state, l, g) if guide else None     # violations of the state being revised
                hyb = dict(state)                                          # old values, for the energy differences
                for ci, c in enumerate(chans):                            # erase: unknown until redrawn
                    Ys = jnp.where(msk[ci], g["Y"], h)
                    state = {**state, c.name: state[c.name].at[Ys, g["X"]].set(-1, mode="drop")}
                    o, kk = cols[c.name]
                    sidx = jnp.where(msk[ci], g["sidx"], h * w)
                    Es = Es.at[sidx[..., None], jnp.arange(o, o + kk + 1)[None, None, :]].set(0.0, mode="drop")
            xs = {**g, "k": jnp.arange(NCOL), "gum": gum, "msk": msk}
            if m > 0 and guide:
                xs["gf"] = gf
            Ep_ = Ep

            def body(carry, x, l=l, chans=chans, m=m, Ep_=Ep_, h=h, w=w):
                state, hyb, Es = carry
                Y, X, valid = x["Y"], x["X"], x["valid"]
                k = x["k"]
                Yc, Xc = jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)
                idx = jnp.arange(Y.shape[0])
                outs = []
                for ci, c in enumerate(chans):
                    key = gkey(l, c.name, m)
                    soft, nv, look = lay.cond(state, c.name, Y, X)
                    v = nv + look
                    mask = (v == jnp.min(v, axis=1, keepdims=True)) | ~valid[:, None]
                    feat = fast_features(lay, l, Es, Ep_, x)
                    if m > 0 and guide:
                        feat = jnp.concatenate([feat, x["gf"]], -1)
                    b = mlp(params[sch.actor_key(key, share, guide)], feat)
                    logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, axis=1)
                    if forced is not None:
                        a = forced[key][Yc, Xc].astype(jnp.int32)
                    else:
                        a = jnp.argmax(logits + x["gum"][ci], axis=1).astype(jnp.int32)
                    if override is not None:
                        os_, oj, oval, on = override
                        st = sch.first[key] + k * len(chans)
                        a = jnp.where(on & (st == os_) & (idx == oj), oval, a)
                    draw = x["msk"][ci]
                    la = logq[idx, a]
                    if m == 0:
                        dv = nv[idx, a]
                        e = soft[idx, a] + lam[l] * dv
                        extra = 0.0
                    else:
                        soft_h, nv_h, _ = lay.cond(hyb, c.name, Y, X)
                        o = hyb[c.name][Yc, Xc]
                        dv = nv_h[idx, a] - nv_h[idx, o]
                        e = soft_h[idx, a] - soft_h[idx, o] + lam[l] * dv
                        extra = tau * float(np.log(c.D))
                    cost = jnp.where(draw, e + tau * la + extra, 0.0)
                    if record:
                        outs.append({"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a,
                                     "valid": draw, "logq": logq, "cost": cost, "dv": jnp.where(draw, dv, 0)})
                    else:
                        outs.append(cost)
                    if m == 0:
                        aw = a
                    else:
                        aw = jnp.where(draw, a, hyb[c.name][Yc, Xc])
                    Ys = jnp.where(valid, Y, h)
                    state = {**state, c.name: state[c.name].at[Ys, X].set(aw, mode="drop")}
                    if hyb is not None:
                        hyb = {**hyb, c.name: hyb[c.name].at[Ys, X].set(aw, mode="drop")}
                    rows = jnp.concatenate([lay.emb[c.name][aw], jnp.ones((aw.shape[0], 1), jnp.float32)], -1)
                    o_, kk = cols[c.name]
                    Es = Es.at[x["sidx"][:, None], jnp.arange(o_, o_ + kk + 1)[None, :]].set(rows, mode="drop")
                return (state, hyb, Es), outs

            (state, hyb, Es), ys = jax.lax.scan(body, (state, hyb, Es), xs)
            for ci, c in enumerate(chans):
                recs[gkey(l, c.name, m)] = ys[ci]
        Ep = Es
    return state, recs


def margin(problem, R, guide=False):
    """Fine cells of padding bounding the dependency radius of (1 + R_l)
    sweeps per level (each 16 colour steps reading <= 3 cells away; a guided
    revising sweep reads the violation field over the window, <= 3 + 3 cells)
    plus one repair round (cpass's bound with more sweeps; the erase coins are
    coordinate-keyed and read nothing)."""
    Rl = steps_per_level(problem, R)
    rr = 6 if guide else 3
    m = sum((NCOL * 3 + Rl[l] * NCOL * rr + NCOL * 2 + 8) << l for l in range(problem.T + 1))
    step = PL << problem.T
    return -(-m // step) * step


def check_colouring(problem):
    """Same-colour sites (PL apart) must share no factor (the pass and the
    erase bookkeeping both rely on it): every same-level pair offset is
    shorter than PL."""
    for f in problem.factors:
        if f.kind == "pair" and problem.chan[f.a[0]].level == problem.chan[f.b[0]].level:
            assert abs(f.off[0]) < PL and abs(f.off[1]) < PL, f.name


# ------------------------------------------------------------------ cones
class RCone(MCone):
    """MCone with the reach of guided revising steps widened by the factor
    scopes of the window cells (same level and parent level): the guide
    field at a window cell depends on every cell of the instances touching it."""

    def __init__(self, sch, win, guide=False):
        if not guide:
            super().__init__(sch, win)
            return
        lay = sch.lay
        p = lay.p
        empty = Window(same=(), parent=())
        extra = {}
        for l in range(lay.T + 1):
            cs = list(lay.chans[l]) + (list(lay.chans[l + 1]) if l < lay.T else [])
            rs = [fine_reach(p, c.level, c.name, empty) for c in cs]
            extra[l] = tuple(max(r[i] for r in rs) if rs else 0 for i in range(4))
        self._extra = extra
        super().__init__(sch, win)
        # recompute: MCone.__init__ set self.reach from fine_reach(win) only; widen the guided groups and rebuild
        reach = []
        for (l, k, key) in sch.steps:
            r = fine_reach(p, l, sch.groups[key][2], win)
            if sch.groups[key][3] > 0:
                r = tuple(a + b for a, b in zip(r, extra[l]))
            reach.append(r)
        self.reach = reach
        for key in sch.groups:
            self.boxes[key], self.idx[key], self.s_first[key] = self._tables(key)
