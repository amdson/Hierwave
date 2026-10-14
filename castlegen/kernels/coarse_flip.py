"""Sweep 2 (coarse half): coarse-cell flip kernel and exact single-cell coarse
Gibbs (notes/decision_2026-10-14_kernel_route.md; constitution amendment
2026-10-14: "coarse-cell flips with the sampler redrawing the subtree").

Flip move (MH, optionally multiple-try).  Pick a level-L cell (Y, X), L >= 1.
The proposal draws the cell's own sampled channels from a FLIP distribution
with full support over the least-violating set A (the pass's mask given every
value outside the block; the block's values are unknown):
  mode "uniform":  uniform over A;
  mode "q":        the pass's conditional at that site tempered at tau_c
                   (whole logit / tau_c);
then redraws the cell's whole subtree (levels < L) from the coloured pass q
at temperature tau, given the outside and the new cell value.  The proposal
density is  flip density x subtree pass density;  the reverse density is the
same function teacher-forced on the old block values.  The acceptance is the
exact (independent-proposal, multiple-try) MH ratio, identical to
block_mh.mh_move with the proposal replaced.  Target ratio from the closed
energies the pass accumulates on the block's steps (block_mh's argument):
every block-touching instance is counted once, so it is the exact energy
difference; lam = inf (hard) or finite lam <= 20 prices violations.

mode "q" with tau_c = tau is exactly block_mh.mh_move (tested).

Coarse Gibbs.  The exact single-cell Gibbs move at any level is
exact_block.ExactBlockGibbs(problem, region, level=L, depth=0) (CPU) or its
JAX port amortised.JaxBlockGibbs(problem, region, L, 0, lam) (GPU): the cell's
sampled channels given everything else, enumerated jointly over their
product, energies from every instance touching the cell (for ground's coarse
cells: count prior to children and parent, smoothness, honour certificates to
children and parent, support, top fill noise).  Rejection-free.
`coarse_gibbs` below only builds that kernel.

Generic: reads only the factors (through the pass's CondJax) and the layout.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.variants._cpass_core import NCOL, key_of, mlp
from .block_mh import block_masks, _logp, _lse


def flip_pass(lay, params, key, state, inb, Ltop, tau, tau_c, mode, Lmax, forced=None):
    """block_mh.block_pass with the proposal at the block's top level Ltop
    (may be traced) replaced by the flip distribution.  Returns
    (new_state, logq_total, E_soft, n_viol, logq_flip)."""
    st = {k: (jnp.where(inb[k], -1, v) if k in inb else v) for k, v in state.items()}
    lq = jnp.zeros(())
    lqf = jnp.zeros(())
    es = jnp.zeros(())
    nvs = jnp.zeros((), jnp.int32)
    for l in lay.levels:
        if l > Lmax:
            continue
        chans = lay.chans[l]
        if not chans:
            continue

        def body(carry, k, l=l, chans=chans):
            st, lq, lqf, es, nvs, key = carry
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Yc, Xc = jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)
            top = jnp.asarray(l) == Ltop
            for ci, c in enumerate(chans):
                inside = valid & inb[c.name][Yc, Xc]
                soft, nv, look = lay.cond(st, c.name, Y, X)
                v = nv + look
                mask = (v == jnp.min(v, axis=1, keepdims=True)) | ~valid[:, None]
                feat = lay.features(st, l, Y, X, lay.actor_win)
                b = mlp(params[key_of(l, c.name)], feat)
                raw = -soft - b
                if mode == "uniform":
                    flip_lg = jnp.zeros_like(raw)
                else:
                    flip_lg = raw / tau_c
                logits = jnp.where(top, flip_lg, raw / tau)
                logits = jnp.where(mask, logits, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                if forced is None:
                    key, k1 = jax.random.split(key)
                    a = jnp.argmax(logits + jax.random.gumbel(k1, logits.shape), axis=1).astype(jnp.int32)
                else:
                    a = forced[c.name][Yc, Xc]
                cur = st[c.name][Yc, Xc]
                a = jnp.where(inside, a, cur)
                idx = jnp.arange(Y.shape[0])
                term = jnp.sum(jnp.where(inside, logq[idx, jnp.maximum(a, 0)], 0.0))
                lq = lq + term
                lqf = lqf + jnp.where(top, term, 0.0)
                es = es + jnp.sum(jnp.where(inside, soft[idx, jnp.maximum(a, 0)], 0.0))
                nvs = nvs + jnp.sum(jnp.where(inside, nv[idx, jnp.maximum(a, 0)], 0))
                Ys = jnp.where(inside, Y, h)
                st = {**st, c.name: st[c.name].at[Ys, X].set(a, mode="drop")}
            return (st, lq, lqf, es, nvs, key), None

        (st, lq, lqf, es, nvs, key), _ = jax.lax.scan(body, (st, lq, lqf, es, nvs, key), jnp.arange(NCOL))
    return st, lq, es, nvs, lqf


def flip_move(lay, params, key, state, L, Y, X, tau=1.0, tau_c=1.0, mode="uniform", lam=20.0, K=1, Lmax=None):
    """One coarse-flip MH / multiple-try move on one world.  Returns (state', record)."""
    Lmax = lay.T if Lmax is None else Lmax
    inb = block_masks(lay, L, Y, X)
    keys = jax.random.split(key, K + 2)
    cst, clq, ces, cnv, _ = jax.vmap(lambda kk: flip_pass(lay, params, kk, state, inb, L, tau, tau_c, mode, Lmax))(keys[:K])
    _, xlq, xes, xnv, _ = flip_pass(lay, params, keys[K], state, inb, L, tau, tau_c, mode, Lmax, forced=state)
    lp_c, lp_x = _logp(ces, cnv, lam), _logp(xes, xnv, lam)
    lw = lp_c - clq
    lwx = lp_x - xlq
    j = jax.random.categorical(keys[K + 1], jnp.where(jnp.isnan(lw), -jnp.inf, lw))
    others = jnp.where(jnp.arange(K) == j, lwx, lw)
    log_r = _lse(lw) - _lse(others)
    log_r = jnp.where(jnp.isfinite(_lse(lw)), log_r, -jnp.inf)
    u = jax.random.uniform(jax.random.fold_in(keys[K + 1], 1))
    acc = jnp.log(u) < log_r
    new = {k: jnp.where(acc, cst[k][j], v) for k, v in state.items()}
    rec = {"L": L, "log_r": jnp.minimum(log_r, 0.0), "acc": acc, "dE": -(lp_c[j] - lp_x),
           "dlogq": clq[j] - xlq, "nv_new": cnv[j], "nv_old": xnv}
    return new, rec


def make_flip_chain(lay, Ls, tau=1.0, tau_c=1.0, mode="uniform", lam=20.0, K=1):
    """Jitted vmapped chain of flip moves (L uniform from Ls, then a uniform
    level-L cell): chain(params, keys (B,), states, nmoves) -> (states, records)."""
    Ls = jnp.asarray(Ls, jnp.int32)
    Lmax = int(np.max(np.asarray(Ls)))
    hs = jnp.asarray([lay.hw[l][0] for l in range(lay.T + 1)])
    ws = jnp.asarray([lay.hw[l][1] for l in range(lay.T + 1)])

    def one(params, key, state, nmoves):
        def step(carry, i):
            state, key = carry
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            L = Ls[jax.random.randint(k1, (), 0, Ls.shape[0])]
            Y = jax.random.randint(k2, (), 0, hs[L])
            X = jax.random.randint(k3, (), 0, ws[L])
            state, rec = flip_move(lay, params, k4, state, L, Y, X, tau, tau_c, mode, lam, K, Lmax)
            return (state, key), rec
        (state, _), recs = jax.lax.scan(step, (state, key), jnp.arange(nmoves))
        return state, recs

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None)), static_argnums=(3,))


def coarse_gibbs(problem, region, L, lam=None, backend="jax"):
    """Exact single-cell Gibbs on every level-L cell (colour-parallel sweep)."""
    if backend == "jax":
        from .amortised import JaxBlockGibbs
        return JaxBlockGibbs(problem, region, L, 0, lam=lam)
    from .exact_block import ExactBlockGibbs
    return ExactBlockGibbs(problem, region, level=L, depth=0, lam=lam, max_joint=10 ** 6)
