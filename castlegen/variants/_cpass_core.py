"""Core of the cpass variant (the "Coloured-pass sampler: final implementation
spec"): schedule, context features, MLPs, the coloured pass and repair, in
JAX.  The pass runs for one world; callers vmap it over worlds.

Steps.  Levels T..0; within a level 16 colours (absolute Y mod 4, X mod 4);
within a colour, the level's sampled channels in declared order.  A step is
one (level, colour, channel); all its sites are drawn in parallel.

Draw.  v = closed violations + lookahead; A = argmin v; q proportional to
1[A] exp(-s / tau - b_theta(h)), s the closed soft energy.  Gumbel-max with
coordinate-keyed noise.  Cost of a draw: ell(t) = s(t) + lambda_l c(t)
+ tau log q(t)."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from ._cond_jax import CondJax

P = 4
NCOL = P * P


# ----------------------------------------------------------------- encoders
def channel_embedding(c):
    """(D, k) fixed features of a channel's values: one-hot for views with at
    most 8 values, (v / (n - 1), its square) for larger views."""
    cols = []
    for name, view in sorted(c.views.items()):
        n = int(view.max()) + 1
        if n <= 1:
            continue
        if n <= 8:
            cols.append(np.eye(n)[view])
        else:
            u = view / (n - 1)
            cols.append(np.stack([u, u * u], 1))
    if not cols:
        cols = [np.zeros((c.D, 1))]
    return np.concatenate(cols, 1).astype(np.float32)


@dataclass
class Window:
    """Cells read around a site: same-level offsets from the parent block's
    corner (2 (Y >> 1), 2 (X >> 1)), parent-level offsets from the parent."""
    same: tuple      # ((dy, dx), ...)
    parent: tuple


ACTOR_WIN = Window(same=tuple((dy, dx) for dy in range(-1, 3) for dx in range(-1, 3)),
                   parent=tuple((dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)))
CRITIC_WIN = Window(same=tuple((dy, dx) for dy in range(-3, 5) for dx in range(-3, 5)),
                    parent=tuple((dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)))


class Layout:
    """Static schedule and feature layout for one region shape."""

    def __init__(self, problem, W, periodic, x0=0):
        self.p, self.W, self.periodic, self.x0 = problem, W, periodic, x0
        T = problem.T
        self.T = T
        self.levels = list(range(T, -1, -1))
        self.hw = {l: (problem.H >> l, W >> l) for l in range(T + 1)}
        self.nyx = {l: (-(-h // P), -(-w // P)) for l, (h, w) in self.hw.items()}
        for l, (h, w) in self.hw.items():
            assert not periodic or w % P == 0 or w < 1, f"periodic width at level {l} must be a multiple of {P}"
        self.chans = {l: [c for c in problem.level_chans(l) if c.designed is None] for l in range(T + 1)}
        self.all_chans = {l: problem.level_chans(l) for l in range(T + 1)}
        self.emb = {c.name: jnp.asarray(channel_embedding(c)) for c in problem.chans}
        # global step index of (level, colour, channel)
        self.base, s = {}, 0
        for l in self.levels:
            self.base[l] = s
            s += NCOL * len(self.chans[l])
        self.nsteps = s
        self.cond = CondJax(problem, W, periodic)

    def step(self, l, k, ci):
        return self.base[l] + k * len(self.chans[l]) + ci

    def sites(self, l, k):
        """Colour k's sites at level l: (Y, X, valid), each (n,)."""
        h, w = self.hw[l]
        ny, nx = self.nyx[l]
        cy, cx = k // P, k % P
        ox = (cx - (self.x0 >> l)) % P            # colours are absolute
        Y = cy + P * jnp.arange(ny)
        X = ox + P * jnp.arange(nx)
        Y, X = jnp.meshgrid(Y, X, indexing="ij")
        Y, X = Y.ravel(), X.ravel()
        return Y, X, (Y < h) & (X < w)

    def nsite(self, l):
        ny, nx = self.nyx[l]
        return ny * nx

    # ----------------------------------------------------------- features
    def _read(self, state, cname, l, Y, X):
        h, w = self.hw[l]
        a = state[cname]
        if self.periodic:
            ok = jnp.ones(Y.shape, bool)
            Xr = jnp.mod(X, w)
        else:
            ok = (X >= 0) & (X < w)
            Xr = jnp.clip(X, 0, w - 1)
        ok = ok & (Y >= 0) & (Y < h)
        v = a[jnp.clip(Y, 0, h - 1), Xr]
        return v, ok

    def _enc_cells(self, state, l, Ys, Xs):
        """Ys, Xs (n, m): features (n, m * sum_c (k_c + 2)) of every channel of level l."""
        out = []
        for c in self.all_chans[l]:
            v, ok = self._read(state, c.name, l, Ys, Xs)
            known = ok & (v >= 0)
            e = self.emb[c.name][jnp.maximum(v, 0)] * known[..., None]
            out += [e, known[..., None].astype(jnp.float32), ok[..., None].astype(jnp.float32)]
        f = jnp.concatenate(out, -1)
        return f.reshape(f.shape[0], -1)

    def features(self, state, l, Y, X, win: Window, extra=None):
        h, _ = self.hw[l]
        by, bx = 2 * (Y >> 1), 2 * (X >> 1)
        dy = jnp.array([d[0] for d in win.same])
        dx = jnp.array([d[1] for d in win.same])
        f = [self._enc_cells(state, l, by[:, None] + dy, bx[:, None] + dx)]
        if l < self.T:
            pdy = jnp.array([d[0] for d in win.parent])
            pdx = jnp.array([d[1] for d in win.parent])
            f.append(self._enc_cells(state, l + 1, (Y >> 1)[:, None] + pdy, (X >> 1)[:, None] + pdx))
        pos = 2 * (Y & 1) + (X & 1)
        f.append(jax.nn.one_hot(pos, 4))
        f.append((Y / max(h - 1, 1))[:, None].astype(jnp.float32))
        if extra is not None:
            f.append(jnp.broadcast_to(extra[None, :], (Y.shape[0], extra.shape[0])))
        return jnp.concatenate(f, -1).astype(jnp.float32)

    def feat_dim(self, l, win, n_extra=0):
        st = {c.name: jnp.zeros(self.hw[c.level], jnp.int32) for c in self.p.chans}
        Y, X, _ = self.sites(l, 0)
        return int(self.features(st, l, Y, X, win, jnp.zeros(n_extra) if n_extra else None).shape[1])


# --------------------------------------------------------------------- MLPs
def mlp_init(key, din, dout, width=64):
    k1, k2 = jax.random.split(key)
    return {"W1": jax.random.normal(k1, (din, width)) / np.sqrt(din), "b1": jnp.zeros(width),
            "W2": jax.random.normal(k2, (width, width)) / np.sqrt(width), "b2": jnp.zeros(width),
            "W3": jnp.zeros((width, dout)), "b3": jnp.zeros(dout)}


def mlp(p, x):
    h = jnp.tanh(x @ p["W1"] + p["b1"])
    h = jnp.tanh(h @ p["W2"] + p["b2"])
    return h @ p["W3"] + p["b3"]


def key_of(l, cname):
    return f"l{l}_{cname}"


# ---------------------------------------------------------------- the pass
def coloured_pass(lay: Layout, params, seed, designed, tau, lam, forced=None, override=None,
                  record=False, critic_extra=None):
    """One world.  designed: {name: (h, w) int32}.  lam: (T + 1,) penalties.
    forced: {name: array} values to score instead of draw.  override: (k, ci, j,
    value, on) replaces one draw at the coarsest level.  Returns (state,
    totals (nsteps,), logq_total, records or None)."""
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
                b = mlp(params[key_of(l, c.name)], feat)
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
                    if critic_extra is not None:           # read before the draw is written
                        ex = jnp.concatenate([critic_extra, jnp.array([st / lay.nsteps])])
                        rec["cfeat"] = lay.features(state, l, Y, X, CRITIC_WIN, ex)
                    outs.append(rec)
                Ys = jnp.where(valid, Y, h)               # invalid sites write nowhere
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return (state, totals, lqt), (outs if record else None)

        (state, totals, logq_total), ys = jax.lax.scan(body, (state, totals, logq_total), jnp.arange(NCOL))
        if record:
            for ci, c in enumerate(chans):
                recs[key_of(l, c.name)] = ys[ci]
    return state, totals, logq_total, (recs if record else None)


def repair(lay: Layout, state, seed):
    """One round, levels coarse to fine, colours, channels: redraw every site
    touching a violated instance among its least-violating values with
    probability proportional to exp(-soft), all incident factors included."""
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue

        def body(state, k, l=l, chans=chans):
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
            for c in chans:
                soft, nv, _ = lay.cond(state, c.name, Y, X)
                cur = state[c.name][jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)]
                idx = jnp.arange(Y.shape[0])
                marked = valid & (nv[idx, cur] > 0)
                mask = nv == jnp.min(nv, axis=1, keepdims=True)
                g = noise.gumbel(jnp, seed, noise.tag_of("repair:" + c.name), l, Y, Xn, c.D)
                a = jnp.argmax(jnp.where(mask, -soft, -jnp.inf) + g, axis=1).astype(jnp.int32)
                a = jnp.where(marked, a, cur)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, None

        state, _ = jax.lax.scan(body, state, jnp.arange(NCOL))
    return state


def margin(problem):
    """Fine cells of padding that bound the dependency radius of pass + repair:
    per level 16 colour steps each reading at most 3 cells away (context
    window, parent, factors, lookahead) plus a repair round reading 2."""
    m = sum((NCOL * 3 + NCOL * 2 + 8) << l for l in range(problem.T + 1))
    step = P << problem.T
    return -(-m // step) * step
