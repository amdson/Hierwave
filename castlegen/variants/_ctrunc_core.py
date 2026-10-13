"""Core of the ctrunc variant: one level of the coloured pass with records,
per-site cost grids, local box sums, and the learned per-cell value.

The generator is cpass's (castlegen.variants._cpass_core.coloured_pass at
tau = 1, then repair); only training differs.  Training runs the pass level by
level (a chain, coarse to fine) so that every level can be given its own batch
and its own truncated return.

Value.  v_m (m = 0 .. T-1) is a per-cell MLP on the cells of level m + 1: it
reads the cell's window (same encoder as the sampler, VALUE_WIN, plus tau and
lambda) and estimates the expected cost of everything drawn below level m + 1
inside the cell (all draws of levels m .. 0 that are its descendants).
V_m(x) = sum over level-(m + 1) cells of v_m."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from . import _cpass_core as core
from ._cpass_core import ACTOR_WIN, CRITIC_WIN, NCOL, Window, key_of, mlp

VALUE_WIN = Window(same=tuple((dy, dx) for dy in range(-2, 4) for dx in range(-2, 4)),
                   parent=tuple((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)))


def vkey(m):
    return f"v{m}"


def level_pass(lay: core.Layout, params, seed, state, tau, lam, l, critic_extra):
    """Draw level l of one world given every coarser level in `state` (finer
    levels unknown, -1).  Returns (state, recs {key: arrays (16, n, ...)})."""
    chans = lay.chans[l]

    def body(state, k):
        Y, X, valid = lay.sites(l, k)
        h, w = lay.hw[l]
        Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
        outs = []
        for ci, c in enumerate(chans):
            soft, nv, look = lay.cond(state, c.name, Y, X)
            v = nv + look
            vmin = jnp.min(v, axis=1, keepdims=True)
            mask = (v == vmin) | ~valid[:, None]
            feat = lay.features(state, l, Y, X, ACTOR_WIN)
            b = mlp(params[key_of(l, c.name)], feat)
            logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
            logq = jax.nn.log_softmax(logits, axis=1)
            g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
            a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
            st = lay.step(l, 0, ci) + k * len(chans)
            ex = jnp.concatenate([critic_extra, jnp.array([st / lay.nsteps])])
            cfeat = lay.features(state, l, Y, X, CRITIC_WIN, ex)
            outs.append({"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a, "valid": valid,
                         "logq": logq, "cfeat": cfeat})
            Ys = jnp.where(valid, Y, h)
            state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
        return state, outs

    state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
    return state, {key_of(l, c.name): ys[ci] for ci, c in enumerate(chans)}


def site_index(lay, l):
    """Static (16, n) arrays Y, X, valid of every colour's sites at level l."""
    P = core.P
    h, w = lay.hw[l]
    ny, nx = lay.nyx[l]
    Ys, Xs = [], []
    for k in range(NCOL):                         # numpy copy of Layout.sites (static under jit)
        cy, cx = k // P, k % P
        ox = (cx - (lay.x0 >> l)) % P
        Y, X = np.meshgrid(cy + P * np.arange(ny), ox + P * np.arange(nx), indexing="ij")
        Ys.append(Y.ravel())
        Xs.append(X.ravel())
    Y, X = np.stack(Ys), np.stack(Xs)
    return Y, X, (Y < h) & (X < w)


def draw_cost(r, tau, lam_l):
    """Cost ell = s + lambda c + tau log q of each recorded draw (..., 16, n)."""
    a = r["a"][..., None]
    s = jnp.take_along_axis(r["soft"], a, -1)[..., 0]
    c = jnp.take_along_axis(r["nv"], a, -1)[..., 0]
    lq = jnp.take_along_axis(r["logq"], a, -1)[..., 0]
    return jnp.where(r["valid"], s + lam_l * c + tau * lq, 0.0)


def cost_grid(lay, l, recs, tau, lam_l):
    """(S_l, h, w) grid of draw costs, step s = k * nchan + ci (one world)."""
    chans = lay.chans[l]
    h, w = lay.hw[l]
    Y, X, V = site_index(lay, l)
    S = NCOL * len(chans)
    grid = jnp.zeros((S, h, w))
    for ci, c in enumerate(chans):
        cost = draw_cost(recs[key_of(l, c.name)], tau, lam_l)               # (16, n)
        s = (np.arange(NCOL) * len(chans) + ci)[:, None] * np.ones_like(Y)
        Ys = np.where(V, Y, h)
        grid = grid.at[s, Ys, X].set(cost, mode="drop")
    return grid


def gather_steps(lay, l, gmap):
    """{key: (16, n)} values of a (S_l, h, w) map at each draw's own step and site."""
    chans = lay.chans[l]
    h, w = lay.hw[l]
    Y, X, V = site_index(lay, l)
    out = {}
    for ci, c in enumerate(chans):
        s = (np.arange(NCOL) * len(chans) + ci)[:, None] * np.ones_like(Y)
        out[key_of(l, c.name)] = gmap[s, np.clip(Y, 0, h - 1), np.clip(X, 0, w - 1)]
    return out


def box(a, R, periodic=True):
    """Sum over the (2R + 1)^2 box around every cell of the last two axes; x
    wraps (periodic training strip), rows off the strip contribute 0."""
    s = sum(jnp.roll(a, dx, -1) for dx in range(-R, R + 1))
    h = a.shape[-2]
    p = jnp.pad(s, [(0, 0)] * (a.ndim - 2) + [(R, R), (0, 0)])
    return sum(p[..., R + dy:R + dy + h, :] for dy in range(-R, R + 1))


def down(a):
    """(..., h, w) -> (..., h/2, w/2): sum of each parent's 2 x 2 children."""
    return a[..., 0::2, 0::2] + a[..., 0::2, 1::2] + a[..., 1::2, 0::2] + a[..., 1::2, 1::2]


def value_map(lay, vp, state, m, extra, scale):
    """v_m on every level-(m + 1) cell of one world: (h, w)."""
    l = m + 1
    h, w = lay.hw[l]
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    f = lay.features(state, l, jnp.asarray(Y.ravel()), jnp.asarray(X.ravel()), VALUE_WIN, extra)
    return (mlp(vp, f)[:, 0] * scale).reshape(h, w)
