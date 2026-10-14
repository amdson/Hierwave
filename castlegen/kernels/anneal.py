"""Annealed blocks with a lambda schedule and a hard final phase (speed-2, part 2).

Target at step j:  f_j(x) = exp(-beta_j (S(x) + lam_j N(x))),  S soft energy, N hard
violations (both of the factor instances touching the mask), beta_j = 1/tau_j.
x_j = one masked colour_gibbs sweep invariant for f_j.  AIS weight
    log w = sum_{j=1..n} [log f_j(x_{j-1}) - log f_{j-1}(x_{j-1})]
          = - sum_j [ beta_j (S_{j-1} + lam_j N_{j-1}) - beta_{j-1} (S_{j-1} + lam_{j-1} N_{j-1}) ].
S, N of x_{j-1} are tracked exactly from the sweeps' energy differences.
final="hard": after step n, n_hard sweeps with violating values excluded.  The last target is
f_hard = exp(-beta_n S) 1[N = 0]; f_hard / f_n (x) = 1[N=0] exp(beta_n lam_n N) = 1[N=0], so
log w is set to -inf for blocks with N > 0 and unchanged otherwise (hard sweeps leave f_hard
invariant and add no weight).  `valid` = N == 0 at the end (touching the mask).
The weights are exact AIS weights when x_0 ~ f_0; from any other start (probe, q draw) they
are heuristic importance weights of a tempered perturbation, not exact.

joint: optional list of ExactBlockGibbs kernels (kernel-a, numpy, whole region, no tau argument)
applied `joint_sweeps` times at the very end (target lam_n, tau = 1): p*-invariant, weight unchanged.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.kernels import colour_gibbs as cg


def lam_schedule(kind, n, lam_end=20.0):
    """lam_0..lam_n.  const: lam_end throughout; ramp: linear 0 -> lam_end; geo: geometric 0.5 -> lam_end."""
    if kind == "const":
        return np.full(n + 1, lam_end)
    if kind == "ramp":
        return np.linspace(0.0, lam_end, n + 1)
    if kind == "geo":
        return np.geomspace(0.5, lam_end, n + 1)
    raise ValueError(kind)


def anneal_block_valid(lay, state, mask, key, tau_hi=8.0, n_steps=32, lam_kind="ramp", lam_end=20.0,
                       final="hard", n_hard=2, levels=None, tau_kind="geometric", joint=None, joint_sweeps=1):
    """Returns (state, logw (B,), valid (B,) bool, info)."""
    taus = cg.tau_schedule(tau_hi, n_steps, tau_kind)
    betas = jnp.asarray(1.0 / taus, jnp.float32)
    lams = jnp.asarray(lam_schedule(lam_kind, n_steps, lam_end), jnp.float32)
    if levels is None:
        levels = sorted({c.level for c in lay.p.chans if c.name in mask}, reverse=True)
    hard_final = final == "hard"
    has_joint = bool(joint)

    def fn(state, key, mask, betas, lams):
        S0, N0 = cg._state_energy(lay, state, mask)

        def step(carry, j):
            st, key, S, N, logw = carry
            logw = logw - (betas[j] * (S + lams[j] * N) - betas[j - 1] * (S + lams[j - 1] * N))
            for l in levels:
                if lay.chans[l]:
                    key, sub = jax.random.split(key)
                    st, ds, dn = cg._level_body(lay, l, betas[j], lams[j], False, mask)(st, sub)
                    S, N = S + ds, N + dn
            return (st, key, S, N, logw), None

        (st, key, S, N, logw), _ = jax.lax.scan(step, (state, key, S0, N0, jnp.zeros_like(S0)),
                                                jnp.arange(1, n_steps + 1))
        if hard_final:
            if not has_joint:
                logw = jnp.where(N > 0, -jnp.inf, logw)
            for _ in range(n_hard):
                for l in levels:
                    if lay.chans[l]:
                        key, sub = jax.random.split(key)
                        st, ds, dn = cg._level_body(lay, l, betas[-1], jnp.float32(0), True, mask)(st, sub)
                        S, N = S + ds, N + dn
        return st, logw, N, dict(S0=S0, N0=N0, S_end=S, N_end=N)

    f = cg._jit(lay, "anneal_valid", fn, (tuple(levels), n_steps, hard_final, n_hard, has_joint))
    st, logw, N, info = f(state, key, mask, betas, lams)
    if joint:
        st = apply_joint(lay, st, joint, joint_sweeps)
        S1, N = cg.state_energy(lay, st, mask)
        info.update(S_end=S1, N_end=N)
        if hard_final:       # the joint move leaves f_n invariant; the hard-phase ratio is taken after it
            logw = jnp.where(N > 0, -jnp.inf, logw)
    return st, logw, N == 0, info


def apply_joint(lay, state, kernels, sweeps=1, seed=0):
    """kernel-a exact block Gibbs on every world (CPU, whole region)."""
    rng = np.random.default_rng(seed)
    names = [c.name for c in lay.p.chans]
    B = state[names[0]].shape[0]
    host = {k: np.asarray(v) for k, v in state.items()}
    K0 = kernels[0]
    out = {k: np.empty_like(v) for k, v in host.items()}
    for b in range(B):
        v = K0.flat.flatten({k: host[k][b] for k in names})
        for _ in range(sweeps):
            for K in kernels:
                K.sweep(v, rng)
        for k, a in K0.flat.unflatten(v).items():
            out[k][b] = a
    return {k: jnp.asarray(a, jnp.int32) for k, a in out.items()}


def ess(logw):
    w = np.asarray(logw, np.float64)
    w = np.exp(w - w[np.isfinite(w)].max()) if np.isfinite(w).any() else np.zeros_like(w)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))
