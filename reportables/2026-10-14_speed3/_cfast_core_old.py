"""The coloured pass for cfast training: the same draws as
_cpass_core.coloured_pass (same features, conditional, logits and noise), with
per-site costs recorded (for cone returns), no critic features recorded (the
critic reads the actor features plus a few globals), and an override that can
replace any one draw (step s, site j) for coupled replays and cone tests."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from castlegen.bench import noise

from ._cpass_core import NCOL, key_of, mlp


def train_pass(lay, params, seed, designed, tau, lam, override=None, record=True):
    """One world.  override: (step, site, value, on).  Returns (state, recs)
    with recs[key] = {feat, soft, nv, mask, a, valid, logq, cost} each
    (16, n, ...), or (state, costs) with costs[key] (16, n) when not record."""
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs = {}
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
                feat = lay.features(state, l, Y, X, lay.actor_win)
                b = mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                st = lay.step(l, 0, ci) + k * len(chans)
                if override is not None:
                    os_, oj, oval, on = override
                    hit = on & (st == os_) & (jnp.arange(Y.shape[0]) == oj)
                    a = jnp.where(hit, oval, a)
                idx = jnp.arange(Y.shape[0])
                la = logq[idx, a]
                cost = jnp.where(valid, soft[idx, a] + lam[l] * nv[idx, a] + tau * la, 0.0)
                if record:
                    outs.append({"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a,
                                 "valid": valid, "logq": logq, "cost": cost})
                else:
                    outs.append(cost)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, outs

        state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            recs[key_of(l, c.name)] = ys[ci]
    return state, recs
