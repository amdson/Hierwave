"""GPU-batched, generic, single-cell exact Gibbs on the coloured layout (speed-2).

Defined only from the problem's factor declarations and the cpass Layout
(castlegen.variants._cpass_core.Layout, whose CondJax gives, for every site of
a channel, the exact soft energy and violation count of every domain value
given all other cells).  Not problem specific.

State: dict channel -> int32 (B, h_l, w_l), B worlds (designed channels are
included and never touched; every sampled cell must hold a valid value).

Step.  gibbs_sweep resamples every cell of one (level, channel, colour) class in
parallel in all B worlds from  p(v) ~ exp(-(soft(v) + lam nv(v)) / tau)  over the
channel's full domain (lam=None: violating values are excluded, untempered).
Colour = (Y mod 4, X mod 4), absolute, as in the pass (lay.sites).

Colouring validity.  Cells of one class are exactly conditionally independent
iff no factor instance contains two of them.  `conflicts(lay)` enumerates the
instances (exact_block.build_instances: periodic wrap, pads, within, by_pos,
counts) and reports every instance with two cells of one (channel, colour)
class.  Where there are none (every factor of toy_modes, toy_support, ground)
a class is one parallel group.  Where there are, the class is split by a greedy
colouring of its conflict graph and the groups are updated one after another
(sequential sub-sweeps keep exactness).  Count factors couple a parent and its
four children; siblings differ in (Y&1, X&1), so they are different colours.

Energy bookkeeping.  Every sweep returns per-world (dsoft, dnv): the change of
the sum of all factor instances touching the resampled cells (exact because a
group shares no instance).  `state_energy` gives the absolute value for the
instances touching a mask (default: all), so AIS weights use absolute energies.

anneal_block: annealed importance sampling inside a mask.  beta_j = 1/tau_j runs
from 1/tau_hi to 1 (geometric in tau by default); x_0 is the given state (it
should come from pi_{beta_0} for exact AIS); log w = sum_j (beta_j - beta_{j-1})
(-E(x_{j-1})), x_j = one masked sweep at beta_j; E = soft + lam nv (lam
finite) or soft (hard: the indicator is not tempered and cancels in the ratio).
log w is the log of an unbiased estimate of Z_1 / Z_{beta_0} of the conditional
of the mask given everything outside.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.kernels.exact_block import Flat, build_instances
from castlegen.variants._cpass_core import NCOL, P


# ------------------------------------------------------------------ plan
class Plan:
    """Static, per-Layout data: instances (energy), conflict groups."""

    def __init__(self, lay):
        self.lay = lay
        p = lay.p
        assert lay.periodic, "colour_gibbs: periodic regions only"
        self.flat = Flat(p, Region(lay.W))
        self.insts = build_instances(self.flat)
        n = self.flat.n
        # class label of every flat cell: (chan index, colour)
        cls = np.full(n, -1, np.int64)
        self.colour_of = np.zeros(n, np.int64)
        for ci, c in enumerate(p.chans):
            h, w = lay.hw[c.level]
            Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            col = (Y % P) * P + (X + (lay.x0 >> c.level)) % P
            o = self.flat.off[c.name]
            self.colour_of[o:o + h * w] = col.ravel()
            cls[o:o + h * w] = ci * NCOL + col.ravel()
        self.cls = cls
        self.conflict_list = []
        edges = {}
        for d in self.insts:
            G = np.stack([g for g, _, _ in d["members"]], 1)
            m = G.shape[1]
            for i in range(m):
                for j in range(i + 1, m):
                    gi, gj = G[:, i], G[:, j]
                    ok = (gi >= 0) & (gj >= 0)
                    same = ok & (cls[np.maximum(gi, 0)] == cls[np.maximum(gj, 0)])
                    if same.any():
                        self.conflict_list.append((p.factors[d["f"]].name, int(same.sum())))
                        for a, b in zip(gi[same], gj[same]):
                            if a != b:
                                edges.setdefault(int(a), set()).add(int(b))
                                edges.setdefault(int(b), set()).add(int(a))
                            else:
                                edges.setdefault(int(a), set())
        self.self_coupled = [k for k, v in edges.items() if k in v]
        # group id per cell (greedy colouring of conflict graph; 0 if none)
        self.group = np.zeros(n, np.int64)
        for a in sorted(edges):
            used = {self.group[b] for b in edges[a] if b < a}
            g = 0
            while g in used:
                g += 1
            self.group[a] = g
        self.ngroups = int(self.group.max()) + 1
        # jax copies of instance index arrays for energies
        self.jinst = []
        for d in self.insts:
            mem = [(jnp.asarray(g, jnp.int32), jnp.asarray(c, jnp.int32), jnp.asarray(v, jnp.int32))
                   for g, c, v in d["members"]]
            self.jinst.append(dict(kind=d["kind"], table=d["table"], mem=mem,
                                   pos=None if d["pos"] is None else jnp.asarray(d["pos"], jnp.int32),
                                   by_pos=d.get("by_pos", False)))
        self.jgroup = jnp.asarray(self.group, jnp.int32)
        self._tab = {}

    def flatten(self, state):
        return jnp.concatenate([state[c.name].reshape(state[c.name].shape[0], -1)
                                for c in self.lay.p.chans], 1)




def get_plan(lay):
    if not hasattr(lay, "_cg_plan"):
        lay._cg_plan = Plan(lay)
    return lay._cg_plan


def conflicts(lay):
    """[(factor name, #instances with two cells of one (channel, colour) class)]; [] = valid."""
    return get_plan(lay).conflict_list


# ------------------------------------------------------------------ energies
def _state_energy(lay, state, mask=None):
    """(soft, nv) per world (B,) of every factor instance touching a mask cell
    (mask: dict channel -> bool (B,h,w); None = all instances)."""
    pl = get_plan(lay)
    v = pl.flatten(state)                                    # (B, n)
    mflat = None
    if mask is not None:
        mflat = jnp.concatenate([mask[c.name].reshape(v.shape[0], -1) if c.name in mask
                                 else jnp.zeros((v.shape[0], int(np.prod(lay.hw[c.level]))), bool)
                                 for c in lay.p.chans], 1)
    B = v.shape[0]
    soft = jnp.zeros(B)
    nv = jnp.zeros(B, jnp.int32)
    for d in pl.jinst:
        vals, touch = [], jnp.zeros((B, d["mem"][0][0].shape[0]), bool)
        for g, c, view in d["mem"]:
            gg = jnp.maximum(g, 0)
            val = jnp.where(g[None] >= 0, view[v[:, gg]], c[None])
            vals.append(val)
            if mflat is not None:
                touch = touch | ((g[None] >= 0) & mflat[:, gg])
        tab = jnp.asarray(np.where(np.isinf(d["table"]), 0.0, d["table"]), jnp.float32)
        inf = jnp.asarray(np.isinf(d["table"]))
        if d["kind"] == "unary":
            idx = (vals[0],)
        elif d["kind"] == "count":
            idx = (vals[0], sum(vals[1:]))
        elif d["by_pos"]:
            idx = (jnp.broadcast_to(d["pos"][None], vals[0].shape), vals[0], vals[1])
        else:
            idx = (vals[0], vals[1])
        e, iv = tab[idx], inf[idx]
        if mflat is None:
            soft = soft + e.sum(1)
            nv = nv + iv.sum(1).astype(jnp.int32)
        else:
            soft = soft + jnp.where(touch, e, 0.0).sum(1)
            nv = nv + (touch & iv).sum(1).astype(jnp.int32)
    return soft, nv


def state_energy(lay, state, mask=None):
    f = _jit(lay, "energy", lambda s, m: _state_energy(lay, s, m), (mask is None,))
    return f(state, mask)


# ------------------------------------------------------------------ one step
def _step_world(lay, st, level, cname, k, beta, lam, hard, key, mask_w, group=0):
    """One world.  Returns (new state dict, dsoft, dnv)."""
    pl = get_plan(lay)
    c = lay.p.chan[cname]
    h, w = lay.hw[level]
    Y, X, valid = lay.sites(level, k)
    soft, nv, _ = lay.cond(st, cname, Y, X)
    soft, nv = soft.astype(jnp.float32), nv
    off = pl.flat.off[cname]
    gid = off + jnp.clip(Y, 0, h - 1) * w + jnp.mod(X, w)
    if pl.ngroups > 1:
        valid = valid & (pl.jgroup[gid] == group)
    if mask_w is not None:
        valid = valid & mask_w[jnp.clip(Y, 0, h - 1), jnp.mod(X, w)]
    if hard:
        logits = jnp.where(nv > 0, -jnp.inf, -beta * soft)
    else:
        logits = -beta * (soft + lam * nv)
    g = jax.random.gumbel(key, logits.shape)
    new = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
    ok = jnp.isfinite(logits).any(1)
    cur = st[cname][jnp.clip(Y, 0, h - 1), jnp.mod(X, w)]
    new = jnp.where(valid & ok, new, cur)
    idx = jnp.arange(Y.shape[0])
    upd = valid & ok
    dsoft = jnp.where(upd, soft[idx, new] - soft[idx, cur], 0.0).sum()
    dnv = jnp.where(upd, nv[idx, new] - nv[idx, cur], 0).sum()
    Ys = jnp.where(valid, jnp.clip(Y, 0, h - 1), h)
    arr = st[cname].at[Ys, jnp.mod(X, w)].set(new, mode="drop")
    return {**st, cname: arr}, dsoft, dnv


def _lam_arg(lam):
    hard = lam is None or (isinstance(lam, float) and np.isinf(lam))
    return hard, jnp.float32(0.0 if hard else lam)


def _sweep_class(lay, state, level, cname, k, beta, lam, hard, key, mask):
    """All groups of one class, vmapped over worlds (k may be traced)."""
    pl = get_plan(lay)
    B = state[next(iter(state))].shape[0]
    dsoft = jnp.zeros(B)
    dnv = jnp.zeros(B, jnp.int32)
    for gi in range(pl.ngroups):
        keys = jax.random.split(jax.random.fold_in(key, gi), B)
        if mask is None:
            f = jax.vmap(lambda s, kk: _step_world(lay, s, level, cname, k, beta, lam, hard, kk, None, gi))
            state, ds, dn = f(state, keys)
        else:
            f = jax.vmap(lambda s, kk, m: _step_world(lay, s, level, cname, k, beta, lam, hard, kk, m, gi))
            state, ds, dn = f(state, keys, mask[cname])
        dsoft, dnv = dsoft + ds, dnv + dn
    return state, dsoft, dnv


# jit cache: layout identity + static choices
_JIT = {}


def _jit(lay, name, fn, static):
    get_plan(lay)                              # build eagerly (never inside a trace)
    key = (id(lay), name, static)
    if key not in _JIT:
        _JIT[key] = (lay, jax.jit(fn))          # keep lay alive so id stays unique
    return _JIT[key][1]


def gibbs_sweep(lay, state, level, channel, colour, tau, lam, key, mask=None, return_delta=False):
    """Resample every cell of the (level, channel, colour) class in all worlds."""
    hard, lamv = _lam_arg(lam)

    def fn(state, k, tau, lamv, key, mask):
        return _sweep_class(lay, state, level, channel, k, 1.0 / tau, lamv, hard, key, mask)

    f = _jit(lay, "sweep", fn, (level, channel, hard, mask is None))
    st, ds, dn = f(state, jnp.int32(colour), jnp.float32(tau), lamv, key, mask)
    return (st, ds, dn) if return_delta else st


def _level_body(lay, level, beta, lam, hard, mask):
    """Function (state, key) -> (state, dsoft, dnv): 16 colours x channels (lax.scan)."""
    chans = lay.chans[level]

    def body(carry, k):
        state, key, ds, dn = carry
        for c in chans:
            key, sub = jax.random.split(key)
            state, a, b = _sweep_class(lay, state, level, c.name, k, beta, lam, hard, sub, mask)
            ds, dn = ds + a, dn + b
        return (state, key, ds, dn), None

    def run(state, key):
        B = state[next(iter(state))].shape[0]
        carry = (state, key, jnp.zeros(B), jnp.zeros(B, jnp.int32))
        (state, _, ds, dn), _ = jax.lax.scan(body, carry, jnp.arange(NCOL))
        return state, ds, dn
    return run


def sweep_level(lay, state, level, tau, lam, key, mask=None, return_delta=False):
    """One systematic sweep of a level: 16 colours, channels in declared order."""
    hard, lamv = _lam_arg(lam)

    def fn(state, tau, lamv, key, mask):
        return _level_body(lay, level, 1.0 / tau, lamv, hard, mask)(state, key)

    f = _jit(lay, "level", fn, (level, hard, mask is None))
    st, ds, dn = f(state, jnp.float32(tau), lamv, key, mask)
    return (st, ds, dn) if return_delta else st


def sweep_all(lay, state, tau, lam, key, levels=None, mask=None):
    """Levels coarse to fine (or `levels` in the given order)."""
    levels = lay.levels if levels is None else levels
    for l in levels:
        if lay.chans[l]:
            key, sub = jax.random.split(key)
            state = sweep_level(lay, state, l, tau, lam, sub, mask)
    return state


# ------------------------------------------------------------------ masks
def subtree_mask(lay, L, cells):
    """Per world, the subtree of the level-L cell cells[b] = (Y, X): every
    sampled-channel cell at levels <= L under it.  Returns dict channel -> bool (B, h, w)."""
    cells = jnp.asarray(cells, jnp.int32)
    out = {}
    for c in lay.p.chans:
        if c.designed is not None or c.level > L:
            continue
        h, w = lay.hw[c.level]
        s = L - c.level
        Y = jnp.arange(h)[None, :, None] >> s
        X = jnp.arange(w)[None, None, :] >> s
        out[c.name] = (Y == cells[:, 0, None, None]) & (X == cells[:, 1, None, None])
    return out


def full_mask(lay, B, levels):
    return {c.name: jnp.ones((B,) + lay.hw[c.level], bool)
            for c in lay.p.chans if c.designed is None and c.level in levels}


# ------------------------------------------------------------------ AIS
def tau_schedule(tau_hi, n, kind="geometric"):
    """taus[1..n], with taus[0] = tau_hi the start; ends at 1.  (n + 1,) values."""
    if kind == "geometric":
        return np.geomspace(tau_hi, 1.0, n + 1)
    b = np.linspace(1.0 / tau_hi, 1.0, n + 1)
    return 1.0 / b


def anneal_block(lay, state, mask, key, tau_hi=8.0, n_steps=32, lam=20.0, levels=None, kind="geometric"):
    """Annealed importance sampling inside `mask` (see module doc).
    Returns (state, logw (B,), info) with info = dict(E0, E_end, nv_end)."""
    hard, lamv = _lam_arg(lam)
    taus = jnp.asarray(tau_schedule(tau_hi, n_steps, kind), jnp.float32)
    betas = 1.0 / taus
    if levels is None:
        levels = sorted({c.level for c in lay.p.chans if c.name in mask}, reverse=True)

    def fn(state, key, mask, betas, lamv):
        s0, n0 = _state_energy(lay, state, mask)
        E0 = s0 if hard else s0 + lamv * n0

        def step(carry, j):
            st, key, E, logw = carry
            logw = logw - (betas[j] - betas[j - 1]) * E
            for l in levels:
                if lay.chans[l]:
                    key, sub = jax.random.split(key)
                    st, ds, dn = _level_body(lay, l, betas[j], lamv, hard, mask)(st, sub)
                    E = E + ds + (0.0 if hard else lamv * dn)
            return (st, key, E, logw), None

        (st, _, E, logw), _ = jax.lax.scan(step, (state, key, E0, jnp.zeros_like(E0)),
                                           jnp.arange(1, n_steps + 1))
        s1, n1 = _state_energy(lay, st, mask)
        return st, logw, dict(E0=E0, E_end=E, soft_end=s1, nv_end=n1)

    f = _jit(lay, "anneal", fn, (hard, tuple(levels), n_steps))
    return f(state, key, mask, betas, lamv)


def conditional(lay, state, level, channel, colour):
    """Exact (soft, nv) of every domain value at every site of the class, per world:
    (B, n, D) each, plus the sites (Y, X, valid).  For checks and diagnostics."""
    Y, X, valid = lay.sites(level, colour)
    f = jax.jit(jax.vmap(lambda s: lay.cond(s, channel, Y, X)[:2]))
    soft, nv = f(state)
    return soft, nv, (Y, X, valid)
