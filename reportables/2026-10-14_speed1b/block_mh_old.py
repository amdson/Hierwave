"""Sweep 1 (b): generic Metropolis-Hastings block kernel with the learned
coloured pass q as the proposal (notes/decision_2026-10-14_kernel_route.md;
constitution amendment 2026-10-14).

Block.  A level-L cell (Y, X) and its whole subtree: every sampled channel at
levels l <= L on the cells (y, x) with y >> (L - l) == Y, x >> (L - l) == X.
Everything else (coarser levels, other cells, designed channels) is fixed.

Proposal q(x_B | x_out).  The coloured pass restricted to the block: the state
starts with every outside value KNOWN and the block's values unknown (-1); the
pass's steps (levels <= L, 16 colours, channels in order) draw only the block's
sites, from the pass's own conditional  softmax(mask ? (-soft - b(feat)) / tau
: -inf)  (whole-logit tempering by tau; the pass's training tau is 1).  Fresh
jax.random Gumbel noise per move.  Teacher forcing the same function on the
old block values gives q(x_B | x_out) exactly (the reverse proposal density).

Target.  log p*(x) = -(S(x) + lam * V(x)) with V the hard violations; lam may
be inf (hard: p* = 0 off support; then the chain must start valid).  Only the
block-touching instances change, and the closed energies the pass accumulates
at the block's steps count every block-touching instance exactly once (the
step where its last unknown member is drawn), so
E_B(x) = sum_block-steps soft[a] + lam nv[a]  gives  log p*(x') - log p*(x) =
-(E_B(x') - E_B(x)) with the same outside.

Move.  Multiple-try with K independent candidates y_1..y_K ~ q(.|x_out),
weights w = p*/q (block parts), pick j ~ w, accept with
min(1, sum_i w(y_i) / (sum_{i != j} w(y_i) + w(x_B))).  K = 1 is plain MH:
p*(y) q(x|.) / (p*(x) q(y|.)).  (q does not depend on the old block values
given the outside, so this is the independent-proposal MTM, Liu et al. 2000.)

Delayed acceptance is not implemented: the first-stage screen would need the
pass to stop after level L, but the full block pass is one vmapped scan, so
the screen saves no compute here.

Generic: reads only the problem's factors (through the pass's CondJax) and
the layout; no problem-specific code.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.variants._cpass_core import NCOL, key_of, mlp


def block_masks(lay, L, Y, X):
    """{channel: bool (h_l, w_l)} for the sampled channels in the subtree of
    level-L cell (Y, X) (Y, X may be traced)."""
    out = {}
    for c in lay.p.sampled():
        l = c.level
        h, w = lay.hw[l]
        if l > lay.T:
            continue
        yy = jnp.arange(h)[:, None]
        xx = jnp.arange(w)[None, :]
        sh = jnp.maximum(L - l, 0)
        m = ((yy >> sh) == Y) & ((xx >> sh) == X) & (l <= L)
        out[c.name] = m
    return out


def block_pass(lay, params, key, state, inb, tau, lam, Lmax, forced=None, drop_mask=False):
    """One world.  state: {name: int32 (h, w)} complete (all channels).  inb:
    block masks.  forced: None (draw) or {name: values} to score.  Levels
    > Lmax are skipped (static).  Returns (new_state, logq, E_soft, n_viol)
    summed over the block's sites; E_soft excludes lam * n_viol."""
    st = {k: (jnp.where(inb[k], -1, v) if k in inb else v) for k, v in state.items()}
    lq = jnp.zeros(())
    es = jnp.zeros(())
    nvs = jnp.zeros((), jnp.int32)
    for l in lay.levels:
        if l > Lmax:
            continue
        chans = lay.chans[l]
        if not chans:
            continue

        def body(carry, k, l=l, chans=chans):
            st, lq, es, nvs, key = carry
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Yc, Xc = jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)
            for ci, c in enumerate(chans):
                inside = valid & inb[c.name][Yc, Xc]
                soft, nv, look = lay.cond(st, c.name, Y, X)
                v = nv + look
                mask = (v == jnp.min(v, axis=1, keepdims=True)) | ~valid[:, None]
                feat = lay.features(st, l, Y, X, lay.actor_win)
                b = mlp(params[key_of(l, c.name)], feat)
                raw = (-soft - b) / tau
                logits = raw if drop_mask else jnp.where(mask, raw, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                if forced is None:
                    key, k1 = jax.random.split(key)
                    g = jax.random.gumbel(k1, logits.shape)
                    a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                else:
                    a = forced[c.name][Yc, Xc]
                cur = st[c.name][Yc, Xc]
                a = jnp.where(inside, a, cur)
                idx = jnp.arange(Y.shape[0])
                lq = lq + jnp.sum(jnp.where(inside, logq[idx, jnp.maximum(a, 0)], 0.0))
                es = es + jnp.sum(jnp.where(inside, soft[idx, jnp.maximum(a, 0)], 0.0))
                nvs = nvs + jnp.sum(jnp.where(inside, nv[idx, jnp.maximum(a, 0)], 0))
                Ys = jnp.where(inside, Y, h)
                st = {**st, c.name: st[c.name].at[Ys, X].set(a, mode="drop")}
            return (st, lq, es, nvs, key), None

        (st, lq, es, nvs, key), _ = jax.lax.scan(body, (st, lq, es, nvs, key), jnp.arange(NCOL))
    return st, lq, es, nvs


def _logp(es, nvs, lam):
    """Block part of log p*: -(soft + lam nv); lam = inf -> -inf when nv > 0."""
    if np.isinf(lam):
        return jnp.where(nvs > 0, -jnp.inf, -es)
    return -(es + lam * nvs)


def _lse(x):
    m = jnp.max(x)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    return m + jnp.log(jnp.sum(jnp.exp(x - m)))


def mh_move(lay, params, key, state, L, Y, X, tau=1.0, lam=20.0, K=1, Lmax=None, drop_q=False):
    """One MTM / MH block move on one world.  Returns (state', log record)."""
    Lmax = lay.T if Lmax is None else Lmax
    inb = block_masks(lay, L, Y, X)
    keys = jax.random.split(key, K + 2)
    # candidates
    cand = jax.vmap(lambda kk: block_pass(lay, params, kk, state, inb, tau, lam, Lmax))(keys[:K])
    cst, clq, ces, cnv = cand
    # reverse: q(x_B | x_out) and the old block energy
    _, xlq, xes, xnv = block_pass(lay, params, keys[K], state, inb, tau, lam, Lmax, forced=state)
    lp_c, lp_x = _logp(ces, cnv, lam), _logp(xes, xnv, lam)
    if drop_q is not False:                      # deliberately wrong ratio (test power only):
        sc = 0.0 if drop_q is True else float(drop_q)   # True drops q; a float scales log q
        clq, xlq = sc * clq, sc * xlq
    lw = lp_c - clq                              # (K,)
    lwx = lp_x - xlq
    j = jax.random.categorical(keys[K + 1], jnp.where(jnp.isnan(lw), -jnp.inf, lw))
    others = jnp.where(jnp.arange(K) == j, lwx, lw)
    log_r = _lse(lw) - _lse(others)
    log_r = jnp.where(jnp.isfinite(_lse(lw)), log_r, -jnp.inf)
    u = jax.random.uniform(jax.random.fold_in(keys[K + 1], 1))
    acc = jnp.log(u) < log_r
    new = {k: jnp.where(acc, cst[k][j], v) for k, v in state.items()}
    size = sum(jnp.sum(m) for m in inb.values())
    rec = {"L": L, "size": size, "log_r": jnp.minimum(log_r, 0.0), "acc": acc,
           "dE": -(lp_c[j] - lp_x), "dlogq": clq[j] - xlq, "nv_new": cnv[j], "nv_old": xnv}
    return new, rec


def make_chain(lay, Ls, tau=1.0, lam=20.0, K=1, drop_q=False, n_save=1):
    """Jitted vmapped chain: chain(params, keys (B,), states, nmoves) -> (states,
    records (B, nmoves), snaps of channel 't'-like fine arrays every nmoves // n_save).
    Each move picks L uniformly from Ls, then a uniform level-L cell."""
    Ls = jnp.asarray(Ls, jnp.int32)
    Lmax = int(np.max(np.asarray(Ls)))

    def one(params, key, state, nmoves):
        def step(carry, i):
            state, key = carry
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            L = Ls[jax.random.randint(k1, (), 0, Ls.shape[0])]
            hL = jnp.asarray([lay.hw[l][0] for l in range(lay.T + 1)])[L]
            wL = jnp.asarray([lay.hw[l][1] for l in range(lay.T + 1)])[L]
            Y = jax.random.randint(k2, (), 0, hL)
            X = jax.random.randint(k3, (), 0, wL)
            state, rec = mh_move(lay, params, k4, state, L, Y, X, tau, lam, K, Lmax, drop_q)
            return (state, key), rec
        (state, _), recs = jax.lax.scan(step, (state, key), jnp.arange(nmoves))
        return state, recs

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None)), static_argnums=(3,))


def make_accept(lay, L, tau=1.0, lam=20.0, K=1):
    """Jitted: one move per (world, cell) without chaining (acceptance
    diagnostic).  f(params, keys (N,), states (N, ...), Y (N,), X (N,))."""
    def one(params, key, state, Y, X):
        return mh_move(lay, params, key, state, jnp.int32(L), Y, X, tau, lam, K, L)
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0)))
