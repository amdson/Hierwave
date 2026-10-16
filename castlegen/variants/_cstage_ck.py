"""Coordinate-keyed noise for the cstage stage (wave-3 agent stage2, reportables/2026-10-16_stage2).

Copies of colour_gibbs' per-class step and of JaxBlockGibbs' fast sweep in which every Gumbel draw comes from
castlegen.bench.noise keyed by (world seed, tag, level, Y, absolute X, value): tag = (phase, round, sweep,
channel or block class), (level, Y, X) = the updated cell, or the block's first variable for a block draw, and
value = the domain value or the block completion index.  The kernels' own code is not edited (constraint:
never edit castlegen/kernels); only the noise source differs.  x0 = absolute fine x of the layout's column 0;
it must be a multiple of 64 (= colour period 4 x 2^T) so that colours and block tilings are absolute."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import NCOL


def tag(*parts):
    return noise.tag_of("cstage/" + "/".join(str(p) for p in parts))


# ------------------------------------------------------------------ single-cell sweeps
def _step_world(lay, st, level, cname, k, beta, lam, seed, tg, x0, mask_w, group):
    pl = cg.get_plan(lay)
    h, w = lay.hw[level]
    Y, X, valid = lay.sites(level, k)
    soft, nv, _ = lay.cond(st, cname, Y, X)
    soft = soft.astype(jnp.float32)
    off = pl.flat.off[cname]
    Yc, Xm = jnp.clip(Y, 0, h - 1), jnp.mod(X, w)
    gid = off + Yc * w + Xm
    if pl.ngroups > 1:
        valid = valid & (pl.jgroup[gid] == group)
    if mask_w is not None:
        valid = valid & mask_w[Yc, Xm]
    logits = -beta * (soft + lam * nv)
    g = noise.gumbel(jnp, seed, tg, level, Yc, Xm + jnp.right_shift(x0, level), logits.shape[1])
    new = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
    ok = jnp.isfinite(logits).any(1)
    cur = st[cname][Yc, Xm]
    new = jnp.where(valid & ok, new, cur)
    Ys = jnp.where(valid, Yc, h)
    return {**st, cname: st[cname].at[Ys, Xm].set(new, mode="drop")}


def _level_fn(lay, level, masked):
    chans = lay.chans[level]
    pl = cg.get_plan(lay)

    def run(state, seeds, tags, lam, mask, x0):
        def body(state, k):
            for ci, c in enumerate(chans):
                for gi in range(pl.ngroups):
                    if masked:
                        f = jax.vmap(lambda s, sd, m: _step_world(lay, s, level, c.name, k, 1.0, lam, sd, tags[ci], x0, m, gi))
                        state = f(state, seeds, mask[c.name])
                    else:
                        f = jax.vmap(lambda s, sd: _step_world(lay, s, level, c.name, k, 1.0, lam, sd, tags[ci], x0, None, gi))
                        state = f(state, seeds)
            return state, None
        state, _ = jax.lax.scan(body, state, jnp.arange(NCOL))
        return state
    return jax.jit(run)


_LF = {}


def sweep_level(lay, state, level, seeds, tags, lam, x0=0, mask=None):
    k = (id(lay), level, mask is None)
    if k not in _LF:
        cg.get_plan(lay)
        _LF[k] = (lay, _level_fn(lay, level, mask is not None))
    return _LF[k][1](state, seeds, jnp.asarray(tags, jnp.int32), jnp.float32(lam), mask, jnp.int32(x0))


# ------------------------------------------------------------------ block sweeps
def block_anchors(J, flat, p):
    """Per colour class: (level, Y, X) arrays of each block's first variable."""
    names = sorted(p.chans, key=lambda c: flat.off[c.name])
    starts = np.array([flat.off[c.name] for c in names])
    out = []
    for BV, nbk, *_ in J.classes:
        g = np.asarray(BV)[:, 0]
        ci = np.searchsorted(starts, g, side="right") - 1
        lev = np.array([names[i].level for i in ci]); w = np.array([flat.shape[names[i].name][1] for i in ci])
        r = g - starts[ci]
        out.append((jnp.asarray(lev, jnp.int32), jnp.asarray(r // w, jnp.int32), jnp.asarray(r % w, jnp.int32)))
    return out


def _keyed_gumbel(seeds, tg, lev, Yb, Xb, x0, C):
    Xa = Xb + jnp.right_shift(x0, lev)
    u = noise.uniform(jnp, seeds[:, None, None], tg, lev[None, :, None], Yb[None, :, None],
                      Xa[None, :, None], jnp.arange(C)[None, None, :])
    return -jnp.log(-jnp.log(u))


def make_block_sweep(J, flat, p):
    """jit(v, seeds (Wb,), tags (nclasses,), x0) -> v : JaxBlockGibbs._sweep_fast (or, when the fast
    factorisation does not apply, _sweep with one chunk) with keyed Gumbel noise."""
    anchors = block_anchors(J, flat, p)
    comps = jnp.asarray(J.comps, jnp.int32)
    C = comps.shape[0]
    hp = jax.lax.Precision.HIGHEST

    def run_fast(v, seeds, tags, x0):
        Wb = v.shape[0]
        for i, ((BV, nbk, single, isoft, inv), (lev, Yb, Xb)) in enumerate(zip(J.fclasses, anchors)):
            fs, fn = J._class_field(v, single, nbk)
            mm = lambda f: jnp.matmul(f.reshape(Wb * nbk, J.K.nv * J.Dmax), J.onehot, precision=hp).reshape(Wb, nbk, C)
            lg = -(isoft[None] + mm(fs) + J.lam * jnp.round(inv[None] + mm(fn)))
            sc = jnp.where(jnp.isfinite(lg), lg + _keyed_gumbel(seeds, tags[i], lev, Yb, Xb, x0, C), -jnp.inf)
            arg = jnp.argmax(sc, -1)
            anyok = jnp.isfinite(lg).any(-1)
            old = v[:, BV]
            v = v.at[:, BV].set(jnp.where(anyok[..., None], comps[arg], old))
        return v

    def run_slow(v, seeds, tags, x0):
        for i, ((BV, nbk, gs), (lev, Yb, Xb)) in enumerate(zip(J.classes, anchors)):
            soft, nvl = J._class_energy(v, gs, nbk, comps)
            lg = -(soft + J.lam * nvl)
            sc = jnp.where(jnp.isfinite(lg), lg + _keyed_gumbel(seeds, tags[i], lev, Yb, Xb, x0, C), -jnp.inf)
            arg = jnp.argmax(sc, -1)
            anyok = jnp.isfinite(lg).any(-1)
            old = v[:, BV]
            v = v.at[:, BV].set(jnp.where(anyok[..., None], comps[arg], old))
        return v
    return jax.jit(run_fast if J.fast else run_slow)
