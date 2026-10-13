"""ctemp core: cpass's coloured pass (_cpass_core.coloured_pass) with one
change: the actor's context features can carry a vector of conditioning
inputs (`actor_extra`, e.g. log tau), appended to every site's features, so
one network represents q(. | tau).  Everything else is _cpass_core."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from castlegen.bench import noise

from ._cpass_core import ACTOR_WIN, CRITIC_WIN, NCOL, Layout, key_of, mlp  # noqa: F401


def coloured_pass(lay: Layout, params, seed, designed, tau, lam, forced=None, override=None,
                  record=False, critic_extra=None, actor_extra=None):
    """_cpass_core.coloured_pass with per-level temperatures and inputs:
    tau (T + 1,) is level l's temperature; actor_extra (T + 1, n) or None,
    row l appended to level l's actor features."""
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    totals = jnp.zeros(lay.nsteps)
    logq_total = jnp.zeros(())
    recs = {}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue
        tl = tau[l]
        axl = None if actor_extra is None else actor_extra[l]

        def body(carry, k, l=l, chans=chans, tl=tl, axl=axl):
            state, totals, lqt = carry
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
            outs = []
            for ci, c in enumerate(chans):
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = lay.features(state, l, Y, X, ACTOR_WIN, axl)
                b = mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tl - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                if forced is not None:
                    a = forced[c.name][jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)]
                else:
                    g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                    a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                if override is not None and l == lay.T:
                    ok_, oci, oj, oval, on = override
                    hit = on & (k == ok_) & (ci == oci) & (jnp.arange(Y.shape[0]) == oj)
                    a = jnp.where(hit, oval, a)
                idx = jnp.arange(Y.shape[0])
                la = logq[idx, a]
                cost = soft[idx, a] + lam[l] * nv[idx, a] + tl * la
                cost = jnp.where(valid, cost, 0.0)
                lqt = lqt + jnp.sum(jnp.where(valid, la, 0.0))
                st = lay.step(l, 0, ci) + k * len(chans)
                totals = totals.at[st].add(jnp.sum(cost))
                if record:
                    rec = {"feat": feat, "soft": soft, "nv": nv, "mask": mask, "a": a,
                           "valid": valid, "logq": logq}
                    if critic_extra is not None:
                        ex = jnp.concatenate([critic_extra, jnp.array([st / lay.nsteps])])
                        rec["cfeat"] = lay.features(state, l, Y, X, CRITIC_WIN, ex)
                    outs.append(rec)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return (state, totals, lqt), (outs if record else None)

        (state, totals, logq_total), ys = jax.lax.scan(body, (state, totals, logq_total), jnp.arange(NCOL))
        if record:
            for ci, c in enumerate(chans):
                recs[key_of(l, c.name)] = ys[ci]
    return state, totals, logq_total, (recs if record else None)
