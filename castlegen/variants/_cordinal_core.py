"""Core of the cordinal variant (ideas.md #6): cpass with ordinal encodings and
smooth output heads for large channels.  Everything not defined here is
cpass's (_cpass_core), unchanged.

(a) Inputs.  A view with more than 8 values is encoded by a piecewise-linear
    thermometer over 16 bins of u = view / (n - 1), plus a one-hot of the exact
    view value when n <= 32 (cpass: u and u^2).  Views with at most 8 values
    keep their one-hot.
(b) Output.  A channel with D > 8 gets b(v) = sum_views sum_k c_{view,k}
    phi_k(u_view(v)) + rscale * r_v with phi_k(u) = cos(pi k u), k = 1..K.
    The actor MLP outputs D residuals r followed by K coefficients per smooth
    view (all zero-initialised).  rscale < 1 slows the residual under Adam so
    the smooth part dominates early.  Channels with D <= 8 keep per-value b.
(c) Mask-relative coordinates.  u_view(v) = clip((view(v) - lo) / max(hi - lo,
    1), 0, 1) with lo / hi the min / max of view over the draw's least-
    violating set A (the mask).  With rel off, lo = 0 and hi = n - 1.

Smooth views of a channel: every view other than "id" with more than 8
values; when there is none, the id view itself (the id order is then taken as
the order; on the suite this is the case for counts, which are ordered, and
for columns' k, whose id order is whatever the problem declared).  Decided
from domain sizes and views only."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from ._cpass_core import (ACTOR_WIN, CRITIC_WIN, NCOL, P, Layout, Window, key_of, margin, mlp,  # noqa: F401
                          mlp_init, repair)

NBIN = 16
ONEHOT_MAX = 32
SMALL = 8


# ----------------------------------------------------------------- encoders
def thermometer(view, n, nbin=NBIN):
    """(D, nbin) piecewise-linear thermometer of u = view / (n - 1): bin j is
    clip(nbin u - j, 0, 1), so every value gets a distinct code."""
    u = view / max(n - 1, 1)
    return np.clip(nbin * u[:, None] - np.arange(nbin)[None, :], 0.0, 1.0)


def channel_embedding(c, enc="thermo"):
    """(D, k) fixed features of a channel's values.  enc "poly" is cpass's."""
    cols = []
    for name, view in sorted(c.views.items()):
        n = int(view.max()) + 1
        if n <= 1:
            continue
        if n <= SMALL:
            cols.append(np.eye(n)[view])
        elif enc == "poly":
            u = view / (n - 1)
            cols.append(np.stack([u, u * u], 1))
        else:
            cols.append(thermometer(view, n))
            if n <= ONEHOT_MAX:
                cols.append(np.eye(n)[view])
    if not cols:
        cols = [np.zeros((c.D, 1))]
    return np.concatenate(cols, 1).astype(np.float32)


class OLayout(Layout):
    def __init__(self, problem, W, periodic, x0=0, enc="thermo"):
        super().__init__(problem, W, periodic, x0)
        self.emb = {c.name: jnp.asarray(channel_embedding(c, enc)) for c in problem.chans}


# -------------------------------------------------------------------- heads
@dataclass
class HeadSpec:
    D: int
    views: tuple          # ((view (D,) float32, n), ...)
    K: int
    rscale: float
    rel: bool

    @property
    def dout(self):
        return self.D + self.K * len(self.views)


def smooth_views(c):
    big = [(k, v) for k, v in sorted(c.views.items()) if k != "id" and int(v.max()) + 1 > SMALL]
    if not big:
        big = [("id", c.views["id"])]
    return tuple((np.asarray(v, np.float32), int(v.max()) + 1) for _, v in big)


def head_spec(c, K=8, rscale=0.1, rel=True, smooth=True):
    """None (per-value logits, cpass) for D <= 8 or smooth off."""
    if not smooth or c.D <= SMALL:
        return None
    return HeadSpec(c.D, smooth_views(c), K, rscale, rel)


def head(spec, p, feat, mask):
    """b (..., D) from the actor MLP p on feat (..., din); mask (..., D) is the
    draw's least-violating set A."""
    out = mlp(p, feat)
    if spec is None:
        return out
    D, K = spec.D, spec.K
    b = spec.rscale * out[..., :D]
    kk = jnp.arange(1, K + 1, dtype=jnp.float32)
    for i, (view, n) in enumerate(spec.views):
        coef = out[..., D + K * i:D + K * (i + 1)]                        # (..., K)
        vv = jnp.asarray(view)
        if spec.rel:
            lo = jnp.min(jnp.where(mask, vv, jnp.inf), -1, keepdims=True)
            hi = jnp.max(jnp.where(mask, vv, -jnp.inf), -1, keepdims=True)
        else:
            lo, hi = jnp.zeros(mask.shape[:-1] + (1,)), jnp.full(mask.shape[:-1] + (1,), float(n - 1))
        u = jnp.clip((vv - lo) / jnp.maximum(hi - lo, 1.0), 0.0, 1.0)       # (..., D)
        phi = jnp.cos(jnp.pi * u[..., None] * kk)                         # (..., D, K)
        b = b + jnp.einsum("...dk,...k->...d", phi, coef)
    return b


# ---------------------------------------------------------------- the pass
def coloured_pass(lay, params, specs, seed, designed, tau, lam, forced=None, override=None,
                  record=False, critic_extra=None):
    """cpass's coloured_pass with b from head(specs[key], ...)."""
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

        def body(carry, k, l=l, chans=chans):
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
                feat = lay.features(state, l, Y, X, ACTOR_WIN)
                key = key_of(l, c.name)
                b = head(specs[key], params[key], feat, mask)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
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
                cost = soft[idx, a] + lam[l] * nv[idx, a] + tau * la
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
