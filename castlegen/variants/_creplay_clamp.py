"""Diagnostics only (not used in training or generation): the cfast training
pass with every draw of the levels >= L clamped to given values, so the finer
levels' behaviour can be measured in a chosen coarse context (e.g. the coarse
channels of a valid probe state that follows the designed fill)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from castlegen.bench import noise

from ._cpass_core import ACTOR_WIN, NCOL, key_of, mlp


def clamped_pass(lay, params, seed, designed, tau, lam, clamp):
    """clamp: {chan name: (h, w) int32} for channels whose draws are replaced.
    Returns (state, {key: per-site cost (16, n)}, {key: per-site nv of the drawn value})."""
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    costs, nvs = {}, {}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue

        def body(state, k, l=l, chans=chans):
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
                if c.name in clamp:
                    a = clamp[c.name][jnp.minimum(Y, h - 1), jnp.mod(X, w)]
                idx = jnp.arange(Y.shape[0])
                cost = jnp.where(valid, soft[idx, a] + lam[l] * nv[idx, a] + tau * logq[idx, a], 0.0)
                outs.append((cost, jnp.where(valid, nv[idx, a], 0)))
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, outs

        state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            costs[key_of(l, c.name)], nvs[key_of(l, c.name)] = ys[ci]
    return state, costs, nvs
