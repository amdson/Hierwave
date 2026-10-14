"""Sweep 1 (c): log q of a GIVEN full state under the coloured pass (teacher
forcing) and a weighted maximum-likelihood update.

The pass draws, per (level, colour step, channel), the sites of one colour
independently from  q_j = softmax(mask ? -soft/tau - b(features) : -inf)  with
the context (features, soft, mask) read from the state built so far.  Forcing
every draw to the value x_j of a given state therefore gives the exact log q(x)
the sampler assigns to x at that tau, as the sum of the per-draw log q_j.

Infeasible draws.  If x_j lies outside the least-violating set A_j (mask false)
then q_j(x_j) = 0.  Such states are reported (n_infeasible > 0, total = -inf).
Option maskfree=True instead scores the draw with the mask-free logits
log_softmax(-soft/tau - b) over all D values: NOT the sampler's q, only a
finite fallback (documented; ml_update drops infeasible states by default).

Everything is jit-able and differentiable w.r.t. params (infeasible draws
contribute 0 to the differentiable part; the -inf is applied outside).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.variants import _cpass_core as core
from castlegen.variants._cpass_core import NCOL, key_of, mlp
from castlegen.variants.cpass import adam_init, adam_step, clip_by_level


def get_layout(var_or_lay, region=None):
    if hasattr(var_or_lay, "layout") and region is not None:
        return var_or_lay.layout(region)
    return var_or_lay


def _forced_pass_ref(lay, params, state_in, tau, maskfree, score_levels=None):
    """Reference implementation: the MLP is evaluated inside the colour scan."""
    """One state.  Returns (per-draw log q dict key -> (16, n) with 0 at invalid
    sites, feasible dict key -> (16, n) bool True at invalid sites)."""
    p = lay.p
    state = {c.name: (state_in[c.name].astype(jnp.int32) if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    lq, ok = {}, {}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue
        if score_levels is not None and l not in score_levels:
            # level not scored: its draws are the given values (the pass would end with exactly these),
            # so finer levels see the same context without running the level's scan
            state = {**state, **{c.name: state_in[c.name].astype(jnp.int32) for c in chans}}
            continue

        def body(state, k, l=l, chans=chans):
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            outs = []
            for c in chans:
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = lay.features(state, l, Y, X, lay.actor_win)
                b = mlp(params[key_of(l, c.name)], feat)
                a = state_in[c.name][jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)].astype(jnp.int32)
                idx = jnp.arange(Y.shape[0])
                if maskfree:
                    logq = jax.nn.log_softmax(-soft / tau - b, axis=1)
                    feas = jnp.ones(Y.shape, bool)
                else:
                    logq = jax.nn.log_softmax(jnp.where(mask, -soft / tau - b, -jnp.inf), axis=1)
                    feas = mask[idx, a]
                feas_all = mask[idx, a]
                la = jnp.where(valid & feas, logq[idx, a], 0.0)
                outs.append((la, feas_all | ~valid))
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, outs

        state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            lq[key_of(l, c.name)], ok[key_of(l, c.name)] = ys[ci]
    return lq, ok


def _collect_inputs(lay, state_in, score_levels=None):
    """Parameter-free half of the teacher-forced pass.  Scans the colour steps of every scored level
    to read, for each draw, the context the pass would see: features, soft energies, validity,
    least-violating mask and the forced value.  Returns {key: dict of (16, n, ...) arrays}."""
    p = lay.p
    state = {c.name: (state_in[c.name].astype(jnp.int32) if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    out = {}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue
        if score_levels is not None and l not in score_levels:
            state = {**state, **{c.name: state_in[c.name].astype(jnp.int32) for c in chans}}
            continue

        def body(state, k, l=l, chans=chans):
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            outs = []
            for c in chans:
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = lay.features(state, l, Y, X, lay.actor_win)
                a = state_in[c.name][jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)].astype(jnp.int32)
                outs.append(dict(feat=feat, soft=soft, mask=mask, a=a, valid=valid))
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, outs

        state, ys = jax.lax.scan(body, state, jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            out[key_of(l, c.name)] = jax.lax.stop_gradient(ys[ci])
    return out


def _forced_pass(lay, params, state_in, tau, maskfree, score_levels=None):
    """Same outputs as _forced_pass_ref.  The contexts are collected once without parameters, then the
    MLPs run over all 16 colour steps of a channel at once, so autodiff has no scan to unroll."""
    ins = _collect_inputs(lay, state_in, score_levels)
    lq, ok = {}, {}
    for key, d in ins.items():
        b = jax.vmap(lambda f: mlp(params[key], f))(d["feat"])                    # (16, n, D)
        soft, mask, a, valid = d["soft"], d["mask"], d["a"], d["valid"]
        n = a.shape[1]
        idx = jnp.arange(n)[None, :]
        pick = lambda x: jnp.take_along_axis(x, a[..., None], -1)[..., 0]
        if maskfree:
            logq = jax.nn.log_softmax(-soft / tau - b, axis=-1)
            feas = jnp.ones(a.shape, bool)
        else:
            logq = jax.nn.log_softmax(jnp.where(mask, -soft / tau - b, -jnp.inf), axis=-1)
            feas = pick(mask)
        feas_all = pick(mask)
        lq[key] = jnp.where(valid & feas, pick(logq), 0.0)
        ok[key] = feas_all | ~valid
    return lq, ok


def logq_of_state(var_or_lay, params, state, region=None, tau=1.0, maskfree=False, score_levels=None):
    """Total log q of one full state {channel: (h, w) int}, by teacher forcing.

    Returns dict:
      total        scalar log q (-inf if any draw infeasible and not maskfree)
      total_finite scalar sum over feasible draws (differentiable; infeasible draws count 0
                   unless maskfree, where they are scored by the mask-free logits)
      per_level    (T + 1,) finite log q per level
      per_draw     {key: (16, n)} per-draw log q (0 at invalid / skipped sites)
      n_infeasible int, number of draws whose forced value is outside A
      feasible     bool
    score_levels: optional set of levels whose draws are scored (the others are taken as given and
      contribute nothing: gradients w.r.t. the scored levels' parameters are unchanged, since a
      level's q depends only on coarser levels' values, not their parameters).
    Not batched; vmap over states (see logq_batch)."""
    lay = get_layout(var_or_lay, region)
    st = {k: jnp.asarray(v) for k, v in state.items()}
    lq, ok = _forced_pass(lay, params, st, tau, maskfree, score_levels)
    per_level = jnp.zeros(lay.T + 1)
    for key, v in lq.items():
        per_level = per_level.at[int(key[1:key.index("_")])].add(v.sum())
    n_inf = sum(jnp.sum(~o) for o in ok.values())
    fin = per_level.sum()
    total = jnp.where((n_inf > 0) & (not maskfree), -jnp.inf, fin)
    return {"total": total, "total_finite": fin, "per_level": per_level, "per_draw": lq,
            "n_infeasible": n_inf, "feasible": n_inf == 0}


def logq_batch(var_or_lay, params, states, region=None, tau=1.0, maskfree=False):
    """vmap of logq_of_state over a batch {name: (B, h, w)}."""
    lay = get_layout(var_or_lay, region)
    return jax.vmap(lambda s: logq_of_state(lay, params, s, tau=tau, maskfree=maskfree))(
        {k: jnp.asarray(v) for k, v in states.items()})


def make_ml_update(var_or_lay, region=None, tau=1.0, clip=10.0, maskfree=False):
    """Compile one weighted-ML Adam step.

    step(params, opt, states, w, lr, level_on) -> (params, opt, info) with
      loss = -sum_i w_i log q(x_i) / sum_i w_i  over feasible states (infeasible ones
             get weight 0 and are counted in info['n_infeasible'] unless maskfree),
      level_on: (T + 1,) 0/1 floats; levels with 0 get no gradient and no Adam move.
    Weights need not be normalised."""
    lay = get_layout(var_or_lay, region)
    levels = list(range(lay.T + 1))

    def loss_fn(params, states, w):
        r = jax.vmap(lambda s: logq_of_state(lay, params, s, tau=tau, maskfree=maskfree))(states)
        wf = w * r["feasible"].astype(w.dtype) if not maskfree else w
        den = jnp.maximum(wf.sum(), 1e-30)
        loss = -jnp.sum(wf * r["total_finite"]) / den
        return loss, (r["n_infeasible"].sum(), r["feasible"].sum())

    @jax.jit
    def step(params, opt, states, w, lr, level_on):
        (loss, (ninf, nfeas)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, states, w)
        g = clip_by_level(g, levels, clip)
        lv = lambda k: level_on[int(k[1:k.index("_")])]
        g = {k: jax.tree_util.tree_map(lambda x: x * lv(k), v) for k, v in g.items()}
        new, nopt = adam_step(params, g, opt, lr)
        # levels switched off keep params and Adam moments exactly
        keep = lambda k: lv(k) > 0
        new = {k: jax.tree_util.tree_map(lambda a, b: jnp.where(keep(k), a, b), new[k], params[k]) for k in params}
        nm = {k: jax.tree_util.tree_map(lambda a, b: jnp.where(keep(k), a, b), nopt["m"][k], opt["m"][k]) for k in params}
        nv = {k: jax.tree_util.tree_map(lambda a, b: jnp.where(keep(k), a, b), nopt["v"][k], opt["v"][k]) for k in params}
        return new, {"m": nm, "v": nv, "t": nopt["t"]}, {"loss": loss, "n_infeasible": ninf, "n_feasible": nfeas}
    step.levels = levels
    return step


def ml_update(params, opt, states, weights, step_fn, lr, level_on=None):
    """Thin wrapper: one weighted-ML Adam step with a step_fn from make_ml_update.
    opt = adam_init(params) with opt['t'] an int32 array."""
    if level_on is None:
        level_on = jnp.ones(len(step_fn.levels), jnp.float32)
    return step_fn(params, opt, {k: jnp.asarray(v) for k, v in states.items()},
                   jnp.asarray(weights, jnp.float32), jnp.float32(lr), jnp.asarray(level_on, jnp.float32))


def ml_opt_init(params):
    o = adam_init(params)
    o["t"] = jnp.int32(0)
    return o
