"""Diagnostics only.  A copy of _creplay_clamp.clamped_pass (the cfast
training pass with every draw at levels >= L replaced by given values) that
also records, per draw: soft, nv, log q of the drawn value, |A| (size of the
least-violating set) and whether that set still violates (no valid option).
Per-level sums are returned per world.  Checked against clamped_pass."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from castlegen.bench import noise
from castlegen.variants._cpass_core import ACTOR_WIN, NCOL, key_of, mlp

FIELDS = ("n", "cost", "soft", "nv", "logq", "forced", "noopt", "noopt_now")


def clamped_pass_stats(lay, params, seed, designed, tau, lam, clamp):
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    T = lay.T
    acc = jnp.zeros((T + 1, len(FIELDS)))
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue

        def body(carry, k, l=l, chans=chans):
            state, acc = carry
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
            for c in chans:
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
                vf = valid.astype(jnp.float32)
                sa, na, la = soft[idx, a], nv[idx, a], logq[idx, a]
                cost = sa + lam[l] * na + tau * la
                row = jnp.stack([vf.sum(), jnp.sum(jnp.where(valid, cost, 0.0)),
                                 jnp.sum(jnp.where(valid, sa, 0.0)), jnp.sum(vf * na),
                                 jnp.sum(jnp.where(valid, la, 0.0)),
                                 jnp.sum(vf * (jnp.sum(mask, 1) == 1)),
                                 jnp.sum(vf * (vmin[:, 0] > 0)),
                                 jnp.sum(vf * (jnp.min(nv, 1) > 0))])
                acc = acc.at[l].add(row)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return (state, acc), None

        (state, acc), _ = jax.lax.scan(body, (state, acc), jnp.arange(NCOL))
    return state, acc
