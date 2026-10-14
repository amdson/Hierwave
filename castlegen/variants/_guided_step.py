"""Guided revising step (user direction for ideas #19-#21, 2026-10-14 evening).

Given a model q of the cpass family (a Layout and its actor parameters, frozen)
and a COMPLETE state, one guided step

    1. computes the guidance field: per level, per cell, the violation mass of
       the hard instances touching the cell (stand-in: the binary count of
       violated hard instances from exact per-instance maps, HardMaps, which
       mirror castlegen/bench/energy.py; the graded distance of ideas #21,
       castlegen/graded.py, can replace it through the same `field` argument);
    2. selects the cells to revise: every cell with a nonzero field plus its
       neighbourhood (dilation by `dilate` cells at the cell's level), and,
       with `descend`, the descendants of every selected coarse cell, so the
       children of a redrawn parent are redrawn after it;
    3. erases the selected sampled cells and redraws them with the model's own
       conditional: the coloured pass (levels T..0, 16 colours, channels in
       declared order; least-violating mask with lookahead, logits -soft/tau -
       b_theta(features), Gumbel-max) restricted to the selected cells, every
       other cell teacher-forced (kept, read as known context).  Noise is keyed
       by (seed, step index, channel, level, absolute coordinates).

The step is dense (every site is computed, the mask decides what is written),
so its cost is fixed whatever the field looks like: no cap is needed for a
fixed-cost step.  R steps are R fixed passes; each reads a fixed local window,
so the dependency radius grows by R x margin_step() (constraint 2).

Feature slot (variant B).  If the actor of a (level, channel) has an input
dimension equal to the pass features plus FIELD_FEATS, the step appends
field_feats(): log(1 + field) at the site and at its parent.  Frozen models
trained without it never see it; a sampler trained with it (path KL, #19)
gets a revising conditional that reads the guidance.

API for other samplers (#19):
    hm = HardMaps(lay); field = violation_field(hm, state)      # {level: (h, w) int32}
    new_state, log_q_step, cells_changed = guided_step(lay, params, state, field, seed, step_index)
    state, info = guided_revise(lay, params, state, seed, R)      # R steps, info per step
log_q_step is the sum of log q of the redrawn values (the step's proposal
density under q, teacher-forced context); cells_changed counts redrawn cells
whose value changed.  `seed` is the world seed (int32 scalar), as everywhere
in castlegen.bench.noise; fold a PRNG key down to one int before calling.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from . import _cpass_core as core

FIELD_FEATS = 2
REACH = 3            # cells read by one colour step at its level (as _cpass_core.margin)


# ------------------------------------------------------------- violation maps
class HardMaps:
    """Violated-instance indicator maps of every hard factor, mirroring energy.instances.

    specs: (kind, factor, inf table, index level, dilation, member levels).  "pair" / "unary"
    maps are indexed by the a cell, "count" and "edge" (pad_a rows) maps by the b cell.
    `dilation`: how far (cells of the index level) the index cell can lie from another member
    cell: |off| for same-level pairs, 2 |off| + 1 for parent-level pairs, 0 otherwise."""

    def __init__(self, lay):
        self.lay, p = lay, lay.p
        self.specs = []
        self.view = {}
        for c in p.chans:
            for v, arr in c.views.items():
                self.view[(c.name, v)] = jnp.asarray(arr, jnp.int32)
        for f in p.factors:
            if not f.hard:
                continue
            la = p.chan[f.a[0]].level
            inf = jnp.asarray(np.isinf(f.table), jnp.int32)
            if f.kind == "unary":
                self.specs.append(("unary", f, inf, la, 0, (la,)))
            elif f.kind == "count":
                self.specs.append(("count", f, inf, la + 1, 0, (la, la + 1)))
            else:
                lb = p.chan[f.b[0]].level
                o = max(abs(f.off[0]), abs(f.off[1]))
                self.specs.append(("pair", f, inf, la, o if lb == la else 2 * o + 1, (la, lb)))
                if f.pad_a >= 0 and lb == la and f.off[0] != 0:
                    self.specs.append(("edge", f, inf, lb, 0, (lb,)))
        self.max_dil = max([s[4] for s in self.specs], default=0)

    def _gather(self, arr, Y, X, l):
        h, w = self.lay.hw[l]
        if self.lay.periodic:
            ok = jnp.ones(Y.shape, bool)
            Xr = jnp.mod(X, w)
        else:
            ok = (X >= 0) & (X < w)
            Xr = jnp.clip(X, 0, w - 1)
        offrow = (Y < 0) | (Y >= h)
        return arr[jnp.clip(Y, 0, h - 1), Xr], ok, offrow

    def maps(self, state):
        """[(h, w) int32] per spec: 1 where the instance indexed by that cell is violated."""
        p, out = self.lay.p, []
        for kind, f, inf, li, _, _ in self.specs:
            A = self.view[f.a][jnp.maximum(state[f.a[0]], 0)]
            la = p.chan[f.a[0]].level
            if kind == "unary":
                out.append(inf[A])
                continue
            B = self.view[f.b][jnp.maximum(state[f.b[0]], 0)]
            lb = p.chan[f.b[0]].level
            if kind == "count":
                s = sum(A[(k >> 1)::2, (k & 1)::2] for k in f.subset)
                out.append(inf[B, s])
                continue
            dy, dx = f.off
            if kind == "edge":
                hb, wb = self.lay.hw[lb]
                h, w = self.lay.hw[la]
                Yq, Xq = jnp.meshgrid(jnp.arange(hb), jnp.arange(wb), indexing="ij")
                Ya, Xa = Yq - dy, Xq - dx
                offa = (Ya < 0) | (Ya >= h)
                if not self.lay.periodic:
                    offa &= (Xa >= 0) & (Xa < w)
                if f.within is not None:
                    L = f.within
                    offa &= ((Ya << la) >> L == (Yq << lb) >> L) & ((Xa << la) >> L == (Xq << lb) >> L)
                out.append(jnp.where(offa, inf[f.pad_a, B], 0))
                continue
            h, w = self.lay.hw[la]
            Y, X = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
            if lb == la:
                Yb, Xb = Y + dy, X + dx
            else:
                Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
            vb, ok, offrow = self._gather(B, Yb, Xb, lb)
            if f.pad_b >= 0:
                vb = jnp.where(offrow, f.pad_b, vb)
            else:
                ok &= ~offrow
            if f.within is not None:
                L = f.within
                ok &= ((Y << la) >> L == (Yb << lb) >> L) & ((X << la) >> L == (Xb << lb) >> L)
            e = inf[2 * (Y & 1) + (X & 1), A, vb] if f.by_pos else inf[A, vb]
            out.append(jnp.where(ok, e, 0))
        return out

    def total(self, state):
        return sum(jnp.sum(m) for m in self.maps(state))

    def per_level(self, state):
        """Violated hard instances per home level (energy.py's E.viol), (T + 1,) int32."""
        out = jnp.zeros(self.lay.T + 1, jnp.int32)
        for (kind, f, _, _, _, _), m in zip(self.specs, self.maps(state)):
            out = out.at[self.lay.p.chan[f.a[0]].level].add(jnp.sum(m))
        return out

    def block_sums(self, maps, L, Y, X, depth=1):
        """(n,) violated instances touching the blocks (level-L cells (Y, X) plus their descendants
        down to L - depth): every relevant map summed over the block's footprint at the map's index
        level, dilated by the map's dilation (used by the jitter-repair baseline)."""
        changed = set(range(L - depth, L + 1))
        tot = jnp.zeros(Y.shape, jnp.int32)
        for (kind, f, inf, li, dil, mem), m in zip(self.specs, maps):
            if not changed.intersection(mem):
                continue
            if li <= L:
                size = 1 << (L - li)
                y0, x0 = Y << (L - li), X << (L - li)
            else:
                size = 1
                y0, x0 = Y >> (li - L), X >> (li - L)
            d = jnp.arange(-dil, size + dil)
            Yw = y0[:, None, None] + d[None, :, None]
            Xw = x0[:, None, None] + d[None, None, :]
            v, ok, offrow = self._gather(m, Yw, Xw, li)
            tot = tot + jnp.sum(jnp.where(ok & ~offrow, v, 0), axis=(1, 2))
        return tot


def violation_field(hm: HardMaps, state):
    """{level: (h, w) int32}: for every cell, the number of violated hard instances touching it
    (the binary stand-in for the graded field of ideas #21; same shape and zero set)."""
    lay, p = hm.lay, hm.lay.p
    field = {l: jnp.zeros(lay.hw[l], jnp.int32) for l in range(lay.T + 1)}
    for (kind, f, inf, li, _, _), m in zip(hm.specs, hm.maps(state)):
        field[li] = field[li] + m
        la = p.chan[f.a[0]].level
        if kind == "unary" or kind == "edge":
            continue
        if kind == "count":                                   # members: the b cell and its children
            for k in f.subset:
                field[la] = field[la].at[(k >> 1)::2, (k & 1)::2].add(m)
            continue
        lb = p.chan[f.b[0]].level                             # pair: the b cell
        h, w = lay.hw[la]
        Y, X = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        dy, dx = f.off
        if lb == la:
            Yb, Xb = Y + dy, X + dx
        else:
            Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
        hb, wb = lay.hw[lb]
        if lay.periodic:
            Xb = jnp.mod(Xb, wb)
        Yb = jnp.where((Yb >= 0) & (Yb < hb), Yb, hb)         # rows off the strip (pads) write nowhere
        field[lb] = field[lb].at[Yb, Xb].add(m, mode="drop")
    return field


# -------------------------------------------------------------------- selection
def _dilate(lay, m, l, r):
    """Binary dilation by r cells (Chebyshev); x wraps on a periodic strip, is cut on a window."""
    if r <= 0:
        return m
    h, w = lay.hw[l]
    mp = jnp.pad(m, ((r, r), (r, r)), mode="wrap" if lay.periodic else "constant")
    if lay.periodic:
        mp = mp.at[:r].set(False).at[r + h:].set(False)      # rows do not wrap
    out = jnp.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = out | mp[r + dy:r + dy + h, r + dx:r + dx + w]
    return out


def select_cells(lay, field, dilate=1, descend=True):
    """{level: (h, w) bool}: cells with a nonzero field, dilated by `dilate` cells at their level,
    plus (descend) every descendant of a selected coarse cell."""
    sel = {}
    for l in range(lay.T, -1, -1):
        m = _dilate(lay, field[l] > 0, l, dilate)
        if descend and l < lay.T:
            m = m | jnp.repeat(jnp.repeat(sel[l + 1], 2, 0), 2, 1)
        sel[l] = m
    return sel


def field_feats(lay, field, l, Y, X):
    """(n, FIELD_FEATS) float32: log(1 + field) at the site and at its parent (0 at level T)."""
    h, w = lay.hw[l]
    Yc = jnp.clip(Y, 0, h - 1)
    Xc = jnp.mod(X, w) if lay.periodic else jnp.clip(X, 0, w - 1)
    own = jnp.log1p(field[l][Yc, Xc].astype(jnp.float32))
    if l < lay.T:
        hp, wp = lay.hw[l + 1]
        par = jnp.log1p(field[l + 1][jnp.clip(Y >> 1, 0, hp - 1),
                                     jnp.mod(X >> 1, wp) if lay.periodic else jnp.clip(X >> 1, 0, wp - 1)
                                     ].astype(jnp.float32))
    else:
        par = jnp.zeros_like(own)
    return jnp.stack([own, par], 1)


# --------------------------------------------------------------------- the step
def guided_step(lay, params, state, field, seed, step_index, tau=1.0, dilate=1, descend=True, sel=None):
    """One guided revising step (module docstring).  Returns (new_state, log_q_step, cells_changed)."""
    sel = select_cells(lay, field, dilate, descend) if sel is None else sel
    st = dict(state)
    for l in range(lay.T + 1):                       # erase the selected sampled cells
        for c in lay.chans[l]:
            st[c.name] = jnp.where(sel[l], -1, state[c.name])
    lq_tot = jnp.zeros(())
    nchg = jnp.zeros((), jnp.int32)
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue

        def body(carry, k, l=l, chans=chans):
            st, lq_tot, nchg = carry
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Yc, Xc = jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)
            Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
            inside = valid & sel[l][Yc, Xc]
            idx = jnp.arange(Y.shape[0])
            for c in chans:
                soft, nv, look = lay.cond(st, c.name, Y, X)
                v = nv + look
                mask = (v == jnp.min(v, axis=1, keepdims=True)) | ~valid[:, None]
                feat = lay.features(st, l, Y, X, lay.actor_win)
                pr = params[core.key_of(l, c.name)]
                if pr["W1"].shape[0] == feat.shape[1] + FIELD_FEATS:      # variant B: the field as an input
                    feat = jnp.concatenate([feat, field_feats(lay, field, l, Y, X)], 1)
                b = core.mlp(pr, feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                g = noise.gumbel(jnp, seed, noise.tag_of(f"guided{step_index}:{c.name}"), l, Y, Xn, c.D)
                a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                old = state[c.name][Yc, Xc]
                lq_tot = lq_tot + jnp.sum(jnp.where(inside, logq[idx, a], 0.0))
                nchg = nchg + jnp.sum((inside & (a != old)).astype(jnp.int32))
                Ys = jnp.where(inside, Y, h)
                st = {**st, c.name: st[c.name].at[Ys, X].set(a, mode="drop")}
            return (st, lq_tot, nchg), None

        (st, lq_tot, nchg), _ = jax.lax.scan(body, (st, lq_tot, nchg), jnp.arange(core.NCOL))
    return st, lq_tot, nchg


def guided_revise(lay, params, state, seed, R, tau=1.0, dilate=1, descend=True, hm=None, record=False):
    """R guided steps from a complete state.  info: per step, (log_q_step, cells_changed,
    violations per level before the step); record=True also returns the state after every step."""
    hm = HardMaps(lay) if hm is None else hm
    lqs, nch, vls, states = [], [], [], []
    for s in range(1, R + 1):
        field = violation_field(hm, state)
        vls.append(hm.per_level(state))
        state, lq, n = guided_step(lay, params, state, field, seed, s, tau, dilate, descend)
        lqs.append(lq)
        nch.append(n)
        if record:
            states.append(state)
    info = {"log_q": jnp.stack(lqs) if lqs else jnp.zeros(0), "changed": jnp.stack(nch) if nch else jnp.zeros(0, jnp.int32),
            "viol_before": jnp.stack(vls) if vls else jnp.zeros((0, lay.T + 1), jnp.int32)}
    if record:
        info["states"] = states
    return state, info


def margin_step(problem, dilate=1):
    """Fine cells of padding one guided step adds to the dependency radius: per level, 16 colour
    steps each reading REACH cells (as the pass), plus the field (instances within 1 cell) and the
    selection dilation."""
    return sum((core.NCOL * REACH + 2 * dilate + 2) << l for l in range(problem.T + 1))


def guided_margin(problem, R, dilate=1):
    """Padding for R steps, rounded up to a multiple of P << T (added to _cpass_core.margin)."""
    step = core.P << problem.T
    m = R * margin_step(problem, dilate)
    return -(-m // step) * step
