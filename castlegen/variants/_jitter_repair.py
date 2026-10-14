"""Jitter repair (ideas.md #20 (a)): a stronger repair with a fixed budget.

BASELINE ONLY (2026-10-14 evening): the user redirected the work to the guided revising step
(_guided_step.py); this module is kept as the comparison baseline of that report and is not
wired into any registered variant.

Runs after the coloured pass and after cpass's one-round repair
(_cpass_core.repair).  R rounds; each round visits the levels L = T .. 1
(coarse to fine) and, at each level, the 16 colour groups (Y mod 4, X mod 4,
absolute coordinates) of level-L cells.  For every cell of the group whose
BLOCK (the cell plus its four children at L - 1) touches a violated hard
instance, the block's sampled channels are erased and redrawn in pass order
(cell, then the children by position; channels in declared order) from the
designed conditional with lookahead: among the values minimising closed
violations + lookahead, with probability proportional to exp(-soft), by
Gumbel-max with noise keyed by (seed, round, L, channel, level, absolute
coordinates).  The redraw is ACCEPTED only if the number of violated hard
instances touching the block does not increase; otherwise the block keeps
its old values.  Blocks of one group are 4 cells apart at level L, so the
instances touching different blocks are disjoint and the per-block decisions
add up: the total number of violated hard instances never increases
(invariant "repair never increases violations").

Violations are counted by exact per-instance indicator maps built from the
factor declarations alone (HardMaps mirrors castlegen/bench/energy.py:
periodic wrap or window cut, pads, within, by_pos, pad_a edge instances).
Everything is derived from the declarations and the hierarchy (constraint 1);
every step reads a fixed local window (constraint 2); the budget R is fixed
and declared (constraint 7).  Dependency radius: jitter_margin() adds, per
round and level L, 16 group steps each reaching at most 3 cells at level L,
to _cpass_core.margin (pass + one repair round).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from . import _cpass_core as core

CHILD_POS = ((0, 0), (0, 1), (1, 0), (1, 1))
REACH = 3          # cells at level L read by one group step (same bound as the pass, _cpass_core.margin)


from ._guided_step import HardMaps  # exact per-instance violation maps (shared with the guided step)


# ------------------------------------------------------------------ the repair
def group_sites(lay, L, k, stride=core.P):
    """Group k of level L: cells with (Y mod stride, X mod stride) = (k // stride, k % stride) in
    absolute coordinates; (Y, X, valid) in region coordinates (like Layout.sites)."""
    h, w = lay.hw[L]
    ny, nx = -(-h // stride), -(-w // stride)
    cy, cx = k // stride, k % stride
    ox = (cx - (lay.x0 >> L)) % stride
    Y = cy + stride * jnp.arange(ny)
    X = ox + stride * jnp.arange(nx)
    Y, X = jnp.meshgrid(Y, X, indexing="ij")
    Y, X = Y.ravel(), X.ravel()
    return Y, X, (Y < h) & (X < w)


def _draw(lay, state, seed, tag, l, Y, X, valid):
    """Redraw every sampled channel of level l at (Y, X) from the designed conditional with
    lookahead (least-violating values, proportional to exp(-soft)), coordinate-keyed noise."""
    h, w = lay.hw[l]
    Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
    Ys = jnp.where(valid, Y, h)
    for c in lay.chans[l]:
        soft, nv, look = lay.cond(state, c.name, Y, X)
        v = nv + look
        mask = v == jnp.min(v, axis=1, keepdims=True)
        g = noise.gumbel(jnp, seed, noise.tag_of(tag + c.name), l, Y, Xn, c.D)
        a = jnp.argmax(jnp.where(mask, -soft, -jnp.inf) + g, axis=1).astype(jnp.int32)
        state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
    return state


def _erase(lay, state, l, Y, X, valid):
    h, _ = lay.hw[l]
    Ys = jnp.where(valid, Y, h)
    for c in lay.chans[l]:
        state = {**state, c.name: state[c.name].at[Ys, X].set(-1, mode="drop")}
    return state


def jitter_step(lay, hm: HardMaps, state, seed, r, L, k, stride=core.P):
    """One group step: redraw the blocks of group k at level L whose block touches a violation;
    keep each redraw only if its touched violations do not increase."""
    Y, X, valid = group_sites(lay, L, k, stride)
    maps0 = hm.maps(state)
    before = hm.block_sums(maps0, L, Y, X)
    tag = f"jitter{r}:L{L}:"
    new = _erase(lay, state, L, Y, X, valid)
    for i, j in CHILD_POS:
        new = _erase(lay, new, L - 1, 2 * Y + i, 2 * X + j, valid)
    new = _draw(lay, new, seed, tag, L, Y, X, valid)
    for i, j in CHILD_POS:
        new = _draw(lay, new, seed, tag, L - 1, 2 * Y + i, 2 * X + j, valid)
    after = hm.block_sums(hm.maps(new), L, Y, X)
    acc = valid & (before > 0) & (after <= before)
    h, w = lay.hw[L]
    accL = jnp.zeros((h, w), bool).at[jnp.where(valid, Y, h), X].set(acc, mode="drop")
    accC = jnp.repeat(jnp.repeat(accL, 2, 0), 2, 1)
    out = dict(state)
    for c in lay.chans[L]:
        out[c.name] = jnp.where(accL, new[c.name], state[c.name])
    for c in lay.chans[L - 1]:
        out[c.name] = jnp.where(accC, new[c.name], state[c.name])
    return out, jnp.sum(acc.astype(jnp.int32))


def jitter_repair(lay, state, seed, R, base_repair=True, stride=core.P):
    """cpass's one-round repair (base_repair) followed by R jitter rounds.  Returns the state."""
    hm = HardMaps(lay)
    assert stride - 1 > 2 * hm.max_dil, "jitter_repair: group stride too small for the factors' reach"
    if base_repair:
        state = core.repair(lay, state, seed)
    for r in range(1, R + 1):
        for L in range(lay.T, 0, -1):
            if not lay.chans[L] and not lay.chans[L - 1]:
                continue

            def body(st, k, L=L, r=r):
                st, n = jitter_step(lay, hm, st, seed, r, L, k, stride)
                return st, n

            state, _ = jax.lax.scan(body, state, jnp.arange(stride * stride))
    return state


def jitter_margin(problem, R, stride=core.P):
    """Extra fine cells of padding for R rounds: per round and level L, stride^2 group steps each
    reading at most REACH cells at level L (block, its neighbours' instances, the acceptance
    window), rounded up to a multiple of P << T (added to _cpass_core.margin)."""
    m = R * sum(stride * stride * REACH << L for L in range(1, problem.T + 1))
    step = core.P << problem.T
    return -(-m // step) * step
