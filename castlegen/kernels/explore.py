"""wave2-explore: kernels that try to move the sampler's OWN draws out of a
collapsed coarse mode (constitution amendment 2026-10-14: p*-invariant kernels
built from factors + layout + q; multiple-try; annealed block proposals with
importance weights; delayed acceptance with the coarse-level factors).

1. make_flip_chain_tracked: coarse_flip.flip_move chained exactly as
   coarse_flip.make_flip_chain, but each record also carries the old and new
   value of the flipped cell's FIRST sampled channel (generic; on ground this
   is rho_L), and the sum of the finest level's first channel under the cell
   before and after (on ground: the dirt tiles of the block), so moves that
   take a pure fine block (all dirt) to a partial one can be counted.

2. (delayed acceptance: not built in this file yet; see report.)

3. annealed_candidates: annealed importance proposals for a level-L subtree.
   For each world and each of K tries: x_0 ~ q(subtree | outside) (block pass
   with the least-violating mask DROPPED, so the start proposal has full
   support and covers f_0; density known), then speed-2's anneal.anneal_block_valid (tau tau_hi -> 1,
   lambda schedule, hard final sweeps).  The importance weight of the end state
   for the target p*(subtree | outside) is
       log w = [ -beta_0 (S_0 + lam_0 N_0) - log q(x_0 | out) ] + log w_AIS,
   i.e. q is the AIS start proposal and f_0 its first target, so the weights
   are exact AIS weights (unbiased for Z(subtree | out)); invalid end states
   have w = 0.  full_support=False keeps q's least-violating mask for x_0:
   q then does not cover f_0 and the weights are heuristic (biased; the toy
   test shows -5 to -6 nats on log Z), but starts are more often valid.  Self-normalised per world, they are the multiple-try
   candidates' importance weights used by the weighted ML update.

4. make_tempered_transition: Neal's tempered transitions on a level-L subtree
   (heat with masked colour-Gibbs sweeps tau 1 -> tau_hi, cool back with the
   reversed sweeps, MH accept with the product of tempered ratios): a
   p*-invariant annealed block move started from the CURRENT (own-draw) block.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.kernels import anneal as an
from castlegen.kernels import colour_gibbs as cg
from castlegen.kernels.block_mh import block_masks, block_pass
from castlegen.kernels.coarse_flip import flip_move


def _first_val(lay, state, Ls, L, Y, X):
    """value of the first sampled channel at level L (traced) at (Y, X)."""
    out = jnp.int32(0)
    for l in Ls:
        c = lay.chans[l][0].name
        h, w = lay.hw[l]
        out = jnp.where(L == l, state[c][jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)], out)
    return out


def _fine_sum(lay, state, L, Y, X):
    """(sum, max possible) of the finest level's first channel under level-L cell (Y, X)."""
    c = lay.chans[0][0]
    m = block_masks(lay, L, Y, X)[c.name]
    return jnp.sum(jnp.where(m, state[c.name], 0)), jnp.sum(m) * (c.D - 1)


def _pick(lay, Ls_j, key):
    hs = jnp.asarray([lay.hw[l][0] for l in range(lay.T + 1)])
    ws = jnp.asarray([lay.hw[l][1] for l in range(lay.T + 1)])
    k1, k2, k3 = jax.random.split(key, 3)
    L = Ls_j[jax.random.randint(k1, (), 0, Ls_j.shape[0])]
    return L, jax.random.randint(k2, (), 0, hs[L]), jax.random.randint(k3, (), 0, ws[L])


def make_flip_chain_tracked(lay, Ls, tau=1.0, tau_c=1.0, mode="uniform", lam=20.0, K=1):
    Ls_s = [int(l) for l in Ls]
    Ls_j = jnp.asarray(Ls_s, jnp.int32)
    Lmax = max(Ls_s)

    def one(params, key, state, nmoves):
        def step(carry, i):
            state, key = carry
            key, kp, km = jax.random.split(key, 3)
            L, Y, X = _pick(lay, Ls_j, kp)
            v_old = _first_val(lay, state, Ls_s, L, Y, X)
            f_old, f_max = _fine_sum(lay, state, L, Y, X)
            state, rec = flip_move(lay, params, km, state, L, Y, X, tau, tau_c, mode, lam, K, Lmax)
            rec = {**rec, "v_old": v_old, "v_new": _first_val(lay, state, Ls_s, L, Y, X),
                   "f_old": f_old, "f_new": _fine_sum(lay, state, L, Y, X)[0], "f_max": f_max}
            return (state, key), rec
        (state, _), recs = jax.lax.scan(step, (state, key), jnp.arange(nmoves))
        return state, recs

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None)), static_argnums=(3,))


# ------------------------------------------------------------------ annealed candidates
def make_annealed_candidates(lay, L, K, tau_hi=8.0, n_steps=64, lam=20.0, lam_kind="const", n_hard=2, tau_q=1.0, full_support=True):
    """Returns f(params, key, states (B,...), cells (B, 2)) ->
    (cand states (B*K, ...), logw (B, K), valid (B, K), info).  See module doc (3)."""
    lam0 = float(an.lam_schedule(lam_kind, n_steps, lam)[0])
    beta0 = 1.0 / float(cg.tau_schedule(tau_hi, n_steps)[0])
    lam_q = float(lam)

    @jax.jit
    def qdraw(params, keys, states, cells):
        def one(k, st, c):
            inb = block_masks(lay, jnp.int32(L), c[0], c[1])
            s, lq, es, nv = block_pass(lay, params, k, st, inb, tau_q, lam_q, L, drop_mask=full_support)
            return s, lq
        return jax.vmap(one)(keys, states, cells)

    def f(params, key, states, cells):
        B = cells.shape[0]
        rep = lambda x: jnp.repeat(x, K, axis=0)
        st = {k: rep(v) for k, v in states.items()}
        cr = rep(jnp.asarray(cells, jnp.int32))
        k1, k2 = jax.random.split(key)
        st0, lq = qdraw(params, jax.random.split(k1, B * K), st, cr)
        mask = cg.subtree_mask(lay, L, cr)
        st1, logw_ais, valid, info = an.anneal_block_valid(lay, st0, mask, k2, tau_hi, n_steps, lam_kind, lam,
                                                           final="hard", n_hard=n_hard)
        logw = -beta0 * (info["S0"] + lam0 * info["N0"]) - lq + logw_ais
        logw = jnp.where(valid, logw, -jnp.inf)
        return st1, logw.reshape(B, K), valid.reshape(B, K), {"lq": lq.reshape(B, K), "S_end": info["S_end"].reshape(B, K)}
    return f


def snis(logw):
    """(B, K) log weights -> normalised weights per row (rows with no finite weight -> 0) and per-row ESS."""
    lw = np.asarray(logw, np.float64)
    m = np.where(np.isfinite(lw).any(1), np.max(np.where(np.isfinite(lw), lw, -np.inf), 1), 0.0)
    w = np.where(np.isfinite(lw), np.exp(lw - m[:, None]), 0.0)
    s = w.sum(1, keepdims=True)
    wn = np.where(s > 0, w / np.maximum(s, 1e-300), 0.0)
    ess = np.where(s[:, 0] > 0, 1.0 / np.maximum((wn ** 2).sum(1), 1e-300), 0.0)
    return wn, ess


# ------------------------------------------------------------------ tempered transitions
def _ordered_sweep(lay, state, levels, beta, lam, key, mask, reverse):
    """One masked sweep at inverse temperature beta over `levels`; forward order =
    levels as given, colours 0..15, channels in declared order; reverse = the exact
    reverse order (levels reversed, colours 15..0, channels reversed), so that the
    reverse sweep is the reversal of the forward sweep w.r.t. the tempered target."""
    B = state[next(iter(state))].shape[0]
    ds = jnp.zeros(B)
    dn = jnp.zeros(B, jnp.int32)
    lv = list(levels)[::-1] if reverse else list(levels)
    cols = jnp.arange(16)[::-1] if reverse else jnp.arange(16)
    for l in lv:
        chans = lay.chans[l][::-1] if reverse else lay.chans[l]
        if not chans:
            continue

        def body(carry, k, chans=chans, l=l):
            st, key, ds, dn = carry
            for c in chans:
                key, sub = jax.random.split(key)
                st, a, b = cg._sweep_class(lay, st, l, c.name, k, beta, lam, False, sub, mask)
                ds, dn = ds + a, dn + b
            return (st, key, ds, dn), None
        key, sub = jax.random.split(key)
        (state, _, ds, dn), _ = jax.lax.scan(body, (state, sub, ds, dn), cols)
    return state, ds, dn


def make_tempered_transition(lay, L, tau_hi=4.0, n_steps=16, lam=20.0, lam_hi=None):
    """Neal (1996) tempered transitions on a level-L subtree per world, target
    p_0 = p*(subtree | outside) with violations priced at lam (finite), p_j its
    tempering at beta_j = 1/tau_j, tau_j geometric 1 -> tau_hi.  Up: x^_j =
    forward sweep at beta_j of x^_{j-1} (j = 1..n); down from x_n: x~_{j-1} =
    reverse sweep at beta_j of x~_j (j = n..1).  Accept x~_0 with
      log a = sum_{j=0}^{n-1} -(beta_{j+1} - beta_j) E(x^_j)
            + sum_{j=1}^{n}  -(beta_{j-1} - beta_j) E(x~_j),
    E = soft + lam nv of the instances touching the subtree.  p*-invariant
    (detailed balance for p_0).  lam_hi: optional violation price at the hot
    end (geometric lam -> lam_hi along the ladder): p_j = exp(-beta_j (S + lam_j N));
    the ratios then use S and N separately (log p_{j+1}/p_j (x) =
    -(beta_{j+1}(S + lam_{j+1} N) - beta_j (S + lam_j N))); p_0 is unchanged.  Returns f(key, states (B,...), cells (B,2)) ->
    (states, log_a (B,), acc (B,), info)."""
    taus = np.geomspace(1.0, tau_hi, n_steps + 1)
    betas = jnp.asarray(1.0 / taus, jnp.float32)
    lams = jnp.asarray(np.geomspace(lam, lam if lam_hi is None else lam_hi, n_steps + 1), jnp.float32)
    levels = list(range(L, -1, -1))
    lamv = jnp.float32(lam)
    cg.get_plan(lay)                      # build eagerly (never inside a trace)

    @jax.jit
    def f(key, states, cells):
        mask = cg.subtree_mask(lay, L, cells)
        S0, N0 = cg._state_energy(lay, states, mask)
        lp = lambda j, S, N: -betas[j] * (S + lams[j] * N)

        def up(carry, j):
            st, key, S, N, la = carry
            la = la + lp(j, S, N) - lp(j - 1, S, N)
            key, sub = jax.random.split(key)
            st, ds, dn = _ordered_sweep(lay, st, levels, betas[j], lams[j], sub, mask, False)
            return (st, key, S + ds, N + dn, la), None
        (st, key, S, N, la), _ = jax.lax.scan(up, (states, key, S0, N0, jnp.zeros_like(S0)), jnp.arange(1, n_steps + 1))

        def down(carry, j):
            st, key, S, N, la = carry
            la = la + lp(j - 1, S, N) - lp(j, S, N)
            key, sub = jax.random.split(key)
            st, ds, dn = _ordered_sweep(lay, st, levels, betas[j], lams[j], sub, mask, True)
            return (st, key, S + ds, N + dn, la), None
        (st, key, S, N, la), _ = jax.lax.scan(down, (st, key, S, N, la), jnp.arange(n_steps, 0, -1))
        E = S + lamv * N
        E0 = S0 + lamv * N0
        u = jax.random.uniform(jax.random.fold_in(key, 7), la.shape)
        acc = jnp.log(u) < la
        new = {k: jnp.where(acc.reshape((-1,) + (1,) * (v.ndim - 1)), st[k], v) for k, v in states.items()}
        return new, jnp.minimum(la, 0.0), acc, {"E0": E0, "E_prop": E}
    return f
