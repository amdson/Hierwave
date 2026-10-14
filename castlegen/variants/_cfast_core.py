"""The coloured pass for cfast training: the same draws as
_cpass_core.coloured_pass (same features, conditional, logits and noise), with
per-site costs recorded (for cone returns), no critic features recorded (the
critic reads the actor features plus a few globals), and an override that can
replace any one draw (step s, site j) for coupled replays and cone tests.

speed-3 (2026-10-14): same values as before, fewer device ops per colour step.
  * Everything that depends only on (level, colour) -- site coordinates, window
    gather indices and in-range flags, position one-hots, the row coordinate
    feature -- is built once in numpy (`_geom`) and enters the scan as xs.
  * The Gumbel noise depends on (seed, tag, level, site, value) only, so the
    noise of all 16 colours of a channel is computed in one batch before the
    scan (same hash, same bits; replay stays faithful: noise per domain value).
  * Context features read from an encoded image per (level, channel) -- the
    embedding times known, and known, per cell -- kept in the scan carry and
    updated by the scatter that writes the draw, instead of re-encoding every
    window cell from the raw state at every step.  The encoded image of the
    parent level is the final carry of the previous level.
  Layouts with count-residual features (lay.resid) keep the old path."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from castlegen.bench import noise

from ._cpass_core import NCOL, P, key_of, mlp


def _sites_np(lay, l, k):
    h, w = lay.hw[l]
    ny, nx = lay.nyx[l]
    cy, cx = k // P, k % P
    ox = (cx - (lay.x0 >> l)) % P
    Y = cy + P * np.arange(ny)
    X = ox + P * np.arange(nx)
    Y, X = np.meshgrid(Y, X, indexing="ij")
    Y, X = Y.ravel(), X.ravel()
    return Y, X, (Y < h) & (X < w)


def _read_idx(lay, l, Y, X):
    """numpy twin of Layout._read's addressing: flat index and ok flag."""
    h, w = lay.hw[l]
    if lay.periodic:
        ok = np.ones(Y.shape, bool)
        Xr = np.mod(X, w)
    else:
        ok = (X >= 0) & (X < w)
        Xr = np.clip(X, 0, w - 1)
    ok = ok & (Y >= 0) & (Y < h)
    return (np.clip(Y, 0, h - 1) * w + Xr).astype(np.int32), ok.astype(np.float32)[..., None]


def _geom(lay, l):
    cache = lay.__dict__.setdefault("_geom_cache", {})
    if l in cache:
        return cache[l]
    h, w = lay.hw[l]
    win = lay.actor_win
    out = {k: [] for k in ("Y", "X", "Xn", "valid", "sidx", "sw", "iw", "ok_s", "ip", "ok_p", "pos")}
    dy = np.array([d[0] for d in win.same]); dx = np.array([d[1] for d in win.same])
    pdy = np.array([d[0] for d in win.parent]); pdx = np.array([d[1] for d in win.parent])
    for k in range(NCOL):
        Y, X, valid = _sites_np(lay, l, k)
        out["Y"].append(Y); out["X"].append(X); out["valid"].append(valid)
        out["Xn"].append(np.mod(X, w) if lay.periodic else X + (lay.x0 >> l))
        out["sidx"].append(np.where(valid, Y * w + X, h * w).astype(np.int32))
        by, bx = 2 * (Y >> 1), 2 * (X >> 1)
        i, o = _read_idx(lay, l, by[:, None] + dy, bx[:, None] + dx)
        out["iw"].append(i); out["ok_s"].append(o)
        if l < lay.T:
            i, o = _read_idx(lay, l + 1, (Y >> 1)[:, None] + pdy, (X >> 1)[:, None] + pdx)
            out["ip"].append(i); out["ok_p"].append(o)
        pos = 2 * (Y & 1) + (X & 1)
        out["pos"].append(np.eye(4, dtype=np.float32)[pos])
    g = {}
    for key, v in out.items():
        if v:
            a = np.stack(v)
            g[key] = a.astype(np.int32) if a.dtype == np.int64 else a
    cache[l] = g
    return g


def _cols(lay, l):
    """Column ranges (start, k) of each channel of level l in the encoded image."""
    o, out = 0, {}
    for c in lay.all_chans[l]:
        k = lay.emb[c.name].shape[1]
        out[c.name] = (o, k)
        o += k + 1
    return out


def _encode(lay, l, state):
    """Encoded image of level l: (h*w, sum_c (k_c + 1)), per channel [emb * known, known]."""
    E = []
    for c in lay.all_chans[l]:
        v = state[c.name].reshape(-1)
        known = v >= 0
        e = lay.emb[c.name][jnp.maximum(v, 0)] * known[..., None]
        E += [e, known[..., None].astype(jnp.float32)]
    return jnp.concatenate(E, -1)


def _gather_enc(lay, l, E, idx, ok):
    """Window features (n, m * sum_c (k_c + 2)) from the encoded image; same
    layout as Layout._enc_cells."""
    G = E[idx]                                              # (n, m, sum_c (k_c + 1))
    out = []
    for c in lay.all_chans[l]:
        o, k = _cols(lay, l)[c.name]
        out += [G[..., o:o + k] * ok, G[..., o + k:o + k + 1] * ok, ok]
    f = jnp.concatenate(out, -1)
    return f.reshape(f.shape[0], -1)


def fast_features(lay, l, Es, Ep, g):
    f = [_gather_enc(lay, l, Es, g["iw"], g["ok_s"])]
    if l < lay.T:
        f.append(_gather_enc(lay, l + 1, Ep, g["ip"], g["ok_p"]))
    f.append(g["pos"])
    # row coordinate: divided at run time exactly as Layout.features does (XLA may turn a constant
    # division into a multiplication by the reciprocal, so a numpy constant would differ by an ulp)
    f.append((g["Y"] / max(lay.hw[l][0] - 1, 1))[:, None].astype(jnp.float32))
    return jnp.concatenate(f, -1).astype(jnp.float32)


def train_pass(lay, params, seed, designed, tau, lam, override=None, record=True):
    """One world.  override: (step, site, value, on).  Returns (state, recs)
    with recs[key] = {feat, soft, nv, mask, a, valid, logq, cost} each
    (16, n, ...), or (state, costs) with costs[key] (16, n) when not record."""
    if lay.resid:
        return _train_pass_slow(lay, params, seed, designed, tau, lam, override, record)
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs = {}
    Ep = None
    for l in lay.levels:
        chans = lay.chans[l]
        Es = _encode(lay, l, state)
        if not chans:
            Ep = Es
            continue
        g = {k: jnp.asarray(v) for k, v in _geom(lay, l).items()}
        h, w = lay.hw[l]
        # noise of all 16 colours of each channel, before the scan
        gum = [noise.gumbel(jnp, seed, noise.tag_of(c.name), l, g["Y"], g["Xn"], c.D) for c in chans]
        xs = {**g, "k": jnp.arange(NCOL), "gum": gum}
        Ep_ = Ep

        def body(carry, x, l=l, chans=chans, Ep_=Ep_, h=h, w=w):
            state, Es = carry
            Y, X, valid = x["Y"], x["X"], x["valid"]
            k = x["k"]
            outs = []
            for ci, c in enumerate(chans):
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = fast_features(lay, l, Es, Ep_, x)
                b = mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                a = jnp.argmax(logits + x["gum"][ci], axis=1).astype(jnp.int32)
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
                rows = jnp.concatenate([lay.emb[c.name][a], jnp.ones((a.shape[0], 1), jnp.float32)], -1)
                o, kk = _cols(lay, l)[c.name]
                Es = Es.at[x["sidx"][:, None], jnp.arange(o, o + kk + 1)[None, :]].set(rows, mode="drop")
            return (state, Es), outs

        (state, Es), ys = jax.lax.scan(body, (state, Es), xs)
        Ep = Es
        for ci, c in enumerate(chans):
            recs[key_of(l, c.name)] = ys[ci]
    return state, recs


def _train_pass_slow(lay, params, seed, designed, tau, lam, override=None, record=True):
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


def fast_coloured_pass(lay, params, seed, designed, tau, lam, forced=None, override=None):
    """_cpass_core.coloured_pass for record=False, no critic features: the same
    draws, totals and log q with the hoisted geometry / noise and the encoded
    images of train_pass.  Returns (state, totals, logq_total, None)."""
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    totals = jnp.zeros(lay.nsteps)
    logq_total = jnp.zeros(())
    Ep = None
    for l in lay.levels:
        chans = lay.chans[l]
        Es = _encode(lay, l, state)
        if not chans:
            Ep = Es
            continue
        g = {k: jnp.asarray(v) for k, v in _geom(lay, l).items()}
        h, w = lay.hw[l]
        gum = [] if forced is not None else [
            noise.gumbel(jnp, seed, noise.tag_of(c.name), l, g["Y"], g["Xn"], c.D) for c in chans]
        xs = {**g, "k": jnp.arange(NCOL), "gum": gum}

        def body(carry, x, l=l, chans=chans, Ep_=Ep, h=h, w=w):
            state, Es, totals, lqt = carry
            Y, X, valid = x["Y"], x["X"], x["valid"]
            k = x["k"]
            for ci, c in enumerate(chans):
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = fast_features(lay, l, Es, Ep_, x)
                b = mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                if forced is not None:
                    a = forced[c.name][jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)]
                else:
                    a = jnp.argmax(logits + x["gum"][ci], axis=1).astype(jnp.int32)
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
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
                rows = jnp.concatenate([lay.emb[c.name][a], jnp.ones((a.shape[0], 1), jnp.float32)], -1)
                o, kk = _cols(lay, l)[c.name]
                Es = Es.at[x["sidx"][:, None], jnp.arange(o, o + kk + 1)[None, :]].set(rows, mode="drop")
            return (state, Es, totals, lqt), None

        (state, Es, totals, logq_total), _ = jax.lax.scan(body, (state, Es, totals, logq_total), xs)
        Ep = Es
    return state, totals, logq_total, None
