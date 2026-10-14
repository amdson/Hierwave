"""exactopt (ideas.md #10): optimise the cpass actor with EXACT expectations of
the objective on small enumerable hierarchies, and compare with sampled-
gradient training (cpass's and cfast's estimators) from the same
initialisation and the same number of optimiser steps.  A diagnostic, not a
bench variant.

Enumeration.  Every reachable draw outcome x of the pass (masks fixed by the
layout, independent of the parameters) is listed by _gradcheck.reachable_states.
One forced pass per outcome records, for each valid draw, its context (actor
features, closed soft energies, mask) and its action.  Contexts are deduplicated
per (level, channel) network, so log q(x) for any parameters is

    log q(x) = sum_draws log_softmax(mask ? -soft / tau - b(context) : -inf)[a]

with b from one of three actors:
  mlp         the cpass actor (same parameters, features and init);
  table_feat  one free logit vector per distinct feature vector (same
              information as the MLP, unlimited capacity);
  table_hist  one free logit vector per (draw, full state before its step):
              any distribution over the reachable outcomes that the masks and
              the draw order allow (the order's own limit only).

J(theta, tau) = sum_x q(x) (E(x) + tau log q(x)), E = S + lam H (H per drawing
level, as the pass closes it); grad J = sum_x grad q(x) (E + tau log q).  At
tau = 1, KL(q || p_lam) = J + log Z_lam, Z_lam = sum over reachable outcomes of
exp(-E): the reachable set contains every valid outcome (checked), so with lam
large this is the KL to the target.

Problems: (a) _gradcheck.tiny_problem (T = 1, H = 2, periodic W = 8);
(b) toy_support and (c) toy_modes on ONE 4 x 4 block: the periodic width at
every level must be a multiple of 4, so a one-block strip is illegal; the block
is the same frozen factors (all confined by `within` / block-local counts) on a
non-periodic H = 4, W = 4 region.  Blocks are independent under the target, so
the per-block target is the problem's exact target (toys.py)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import toys

from . import _cpass_core as core
from . import _gradcheck as gc
from .cfast import CFast
from .cpass import CPass, adam_init, adam_step, clip_by_level


# ------------------------------------------------------------------ problems
def block_problem(which):
    """toy_support / toy_modes on one 4 x 4 block (non-periodic H = 4, W = 4)."""
    p = toys.build_toy_support() if which == "support" else toys.build_toy_modes()
    return dataclasses.replace(p, name=p.name + "_block", H=4, W_train=4, W_eval=4, stats=None, probes=None,
                               render=None, exact={}, tol={})


def get_problem(name):
    """(problem, region)."""
    if name == "tiny":
        return gc.tiny_problem(), Region(8)
    return block_problem(name), Region(4, periodic=False)


def _nonperiodic(cls):
    """A subclass whose every layout is the non-periodic region of its width
    (CPass / CFast build a periodic layout of W_train in __init__)."""
    class NP(cls):
        name = cls.name + "_np"

        def layout(self, region):
            if getattr(self.problem, "_np", False):
                region = Region(region.W, periodic=False)
            return cls.layout(self, region)
    NP.__name__ = cls.__name__ + "NP"
    return NP


class _NoCone:
    """Stand-in for cfast's Cone on a non-periodic layout (cone = False never
    reads it; the cone needs a periodic strip)."""

    def __init__(self, lay, win):
        self.idx = {core.key_of(l, c.name): np.zeros((1,), np.int32) for l in lay.levels for c in lay.chans[l]}


class _CFastBlock(CFast):
    name = "cfast_block"

    def _build_train(self, lay):
        if lay.periodic:
            return CFast._build_train(self, lay)
        from . import cfast as cf
        orig = cf.Cone
        cf.Cone = _NoCone
        try:
            return CFast._build_train(self, lay)
        finally:
            cf.Cone = orig


CPassNP, CFastNP = _nonperiodic(CPass), _nonperiodic(_CFastBlock)


def make_variant(kind, problem, region, seed, **cfg):
    problem._np = not region.periodic
    cls = {"cpass": CPassNP, "cfast": CFastNP}[kind]
    if kind == "cfast" and not region.periodic:
        cfg = {"cone": False, **cfg}        # the cone needs a periodic layout; on one block it is everything later
    return cls(problem, seed, f"/tmp/exactopt_{kind}", **cfg)


# ------------------------------------------------------- extra actor nets
def net_spec(kind):
    """'net-<act>-<width>x<depth>' -> (act, width, depth); act in tanh | ln (LayerNorm + SiLU)."""
    _, act, wd = kind.split("-")
    w, d = wd.split("x")
    return act, int(w), int(d)


def net_init(key, din, dout, act, width, depth):
    """Same init convention as core.mlp_init (scaled normal hidden layers, zero
    output layer, so b = 0 at step 0 for every actor)."""
    ks = jax.random.split(key, depth)
    p, d = {}, din
    for i in range(depth):
        p[f"W{i}"] = jax.random.normal(ks[i], (d, width)) / np.sqrt(d)
        p[f"b{i}"] = jnp.zeros(width)
        if act == "ln":
            p[f"g{i}"] = jnp.ones(width)
            p[f"c{i}"] = jnp.zeros(width)
        d = width
    p["Wo"] = jnp.zeros((d, dout))
    p["bo"] = jnp.zeros(dout)
    return p


def net(p, x, act, depth):
    h = x
    for i in range(depth):
        h = h @ p[f"W{i}"] + p[f"b{i}"]
        if act == "ln":
            mu = h.mean(-1, keepdims=True)
            var = ((h - mu) ** 2).mean(-1, keepdims=True)
            h = jax.nn.silu((h - mu) * jax.lax.rsqrt(var + 1e-5) * p[f"g{i}"] + p[f"c{i}"])
        else:
            h = jnp.tanh(h)
    return h @ p["Wo"] + p["bo"]


def set_actor_window(spec):
    """'lo:hi' or 'lo:hi,plo:phi': widen the actor window (same-level offsets
    lo..hi from the parent block corner, parent offsets plo..phi).  Must be
    called before the variant and the Enum are built (it rebinds
    core.ACTOR_WIN, which coloured_pass reads at trace time)."""
    if not spec:
        return
    parts = spec.split(",")
    lo, hi = map(int, parts[0].split(":"))
    plo, phi = map(int, parts[1].split(":")) if len(parts) > 1 else (-1, 1)
    core.ACTOR_WIN = core.Window(same=tuple((dy, dx) for dy in range(lo, hi + 1) for dx in range(lo, hi + 1)),
                                 parent=tuple((dy, dx) for dy in range(plo, phi + 1) for dx in range(plo, phi + 1)))


# ---------------------------------------------------------------- enumeration
def _dedupe(rows, table):
    """rows (m, k) float32 -> global ids, extending table {bytes: id}."""
    u, inv = np.unique(np.ascontiguousarray(rows).view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))),
                       return_inverse=True)
    ids = np.empty(len(u), np.int64)
    for i, b in enumerate(u):
        b = b.tobytes()
        if b not in table:
            table[b] = len(table)
        ids[i] = table[b]
    return ids[inv.ravel()]


class Enum:
    """All reachable outcomes of one layout with per-draw contexts."""

    _SAVE = ("X", "N", "keys", "level", "D", "vidx", "ctx", "act", "Ufeat", "Usoft", "Umask", "fid", "nfeat",
             "S", "H", "hid", "nhist")

    def __init__(self, problem, region, params, lam, chunk=4096, ref_check=200, verbose=True, cache=None):
        t0 = time.time()
        self.p, self.region = problem, region
        self.lay = lay = core.Layout(problem, region.W, region.periodic, region.x0)
        self.lam = np.asarray(lam, np.float64)
        self._jits = {}
        if cache is not None and os.path.exists(cache):
            import pickle
            with open(cache, "rb") as fh:
                self.__dict__.update(pickle.load(fh))
            self.set_lam(lam)
            return
        self.X = gc.reachable_states(lay)
        N = self.N = len(next(iter(self.X.values())))
        if verbose:
            print(f"[enum] {problem.name}: {N} reachable outcomes ({time.time() - t0:.1f}s)", flush=True)
        T = problem.T
        lam0 = jnp.zeros(T + 1, jnp.float32)

        def one(forced):
            _, _, lq, rec = core.coloured_pass(lay, params, 0, {}, 1.0, lam0, forced=forced, record=True)
            return rec

        f = jax.jit(jax.vmap(one))
        self.keys = [core.key_of(l, c.name) for l in lay.levels for c in lay.chans[l]]
        self.level = {core.key_of(l, c.name): l for l in lay.levels for c in lay.chans[l]}
        self.D = {core.key_of(l, c.name): c.D for l in lay.levels for c in lay.chans[l]}
        ctab = {k: {} for k in self.keys}
        ftab = {k: {} for k in self.keys}
        ctx = {k: [] for k in self.keys}
        act = {k: [] for k in self.keys}
        U = {k: {} for k in self.keys}        # ctx id -> (feat, soft, mask), first occurrence
        S = np.zeros(N)
        H = np.zeros((N, T + 1))
        self.vidx = {}
        for s in range(0, N, chunk):
            rec = jax.device_get(f({k: jnp.asarray(v[s:s + chunk]) for k, v in self.X.items()}))
            for k in self.keys:
                r = rec[k]
                if k not in self.vidx:
                    self.vidx[k] = np.nonzero(r["valid"][0])         # (colour, site) of valid draws
                kk, jj = self.vidx[k]
                feat = r["feat"][:, kk, jj]                          # (n, m, F)
                soft = r["soft"][:, kk, jj]
                nv = r["nv"][:, kk, jj]
                mask = r["mask"][:, kk, jj]
                a = r["a"][:, kk, jj]
                n, m = a.shape
                rows = np.concatenate([feat, soft, mask.astype(np.float32)], -1).reshape(n * m, -1)
                ids = _dedupe(rows.astype(np.float32), ctab[k])
                sf, mk = soft.reshape(n * m, -1), mask.reshape(n * m, -1)
                for i, j in zip(*np.unique(ids, return_index=True)):
                    if i not in U[k]:
                        U[k][i] = (rows[j, :feat.shape[-1]], sf[j], mk[j])
                ctx[k].append(ids.reshape(n, m))
                act[k].append(a)
                ia = a[..., None]
                S[s:s + n] += np.take_along_axis(soft, ia, -1)[..., 0].sum(1)
                H[s:s + n, self.level[k]] += np.take_along_axis(nv, ia, -1)[..., 0].sum(1)
        self.ctx = {k: np.concatenate(v) for k, v in ctx.items()}
        self.act = {k: np.concatenate(v) for k, v in act.items()}
        C = {k: len(ctab[k]) for k in self.keys}
        self.Ufeat = {k: np.stack([U[k][i][0] for i in range(C[k])]) for k in self.keys}
        self.Usoft = {k: np.stack([U[k][i][1] for i in range(C[k])]).astype(np.float32) for k in self.keys}
        self.Umask = {k: np.stack([U[k][i][2] for i in range(C[k])]) for k in self.keys}
        self.fid = {k: _dedupe(self.Ufeat[k].astype(np.float32), ftab[k]) for k in self.keys}
        self.nfeat = {k: len(ftab[k]) for k in self.keys}
        self.S, self.H = S, H
        self._check_reference(ref_check)
        self._history_ids()
        self.set_lam(lam)
        if verbose:
            print(f"[enum] contexts {C}, feature contexts {self.nfeat}, history contexts {self.nhist} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        if cache is not None:
            import pickle
            with open(cache, "wb") as fh:
                pickle.dump({k: getattr(self, k) for k in self._SAVE}, fh)

    def _check_reference(self, n):
        rng = np.random.default_rng(0)
        for i in rng.choice(self.N, min(n, self.N), replace=False):
            e = en.energy(self.p, {k: a[i] for k, a in self.X.items()}, self.region)
            assert abs(e.soft - self.S[i]) < 1e-3 and abs(sum(e.viol.values()) - self.H[i].sum()) < 1e-6, \
                (i, e.soft, self.S[i], e.viol, self.H[i])

    def _history_ids(self):
        """Per draw: id of the full state before its step (outcome prefix)."""
        lay = self.lay
        cols, steps = [], []
        for l in lay.levels:
            for ci, c in enumerate(lay.chans[l]):
                k = core.key_of(l, c.name)
                for i, kk in enumerate(self.vidx[k][0]):
                    cols.append((k, i))
                    steps.append(lay.step(l, int(kk), ci))
        steps = np.array(steps)
        A = np.stack([self.act[k][:, i] for k, i in cols], 1).astype(np.int16)    # (N, ndraws)
        self.hid = {k: np.zeros_like(self.ctx[k]) for k in self.keys}
        self.nhist = {k: 0 for k in self.keys}
        cache = {}
        for d, (k, i) in enumerate(cols):
            s = steps[d]
            if s not in cache:
                pre = A[:, steps < s]
                if pre.shape[1] == 0:
                    cache[s] = (np.zeros(self.N, np.int64), 1)
                else:
                    u, inv = np.unique(pre, axis=0, return_inverse=True)
                    cache[s] = (inv.ravel(), len(u))
            inv, nu = cache[s]
            self.hid[k][:, i] = self.nhist[k] + inv
            self.nhist[k] += nu

    def _seq_ids(self):
        """Per draw: id of every EARLIER draw in a total order (steps, then
        site index within the step).  Unlike table_hist this lets a draw see
        the same-step draws before it: the fully expressive chain rule, i.e. no
        within-colour independence.  A diagnostic, not a valid sampler."""
        if hasattr(self, "sid"):
            return
        lay = self.lay
        cols, steps = [], []
        for l in lay.levels:
            for ci, c in enumerate(lay.chans[l]):
                k = core.key_of(l, c.name)
                for i, kk in enumerate(self.vidx[k][0]):
                    cols.append((k, i))
                    steps.append(lay.step(l, int(kk), ci))
        order = np.argsort(np.array(steps), kind="stable")
        A = np.stack([self.act[k][:, i] for k, i in cols], 1).astype(np.int16)[:, order]
        self.sid = {k: np.zeros_like(self.ctx[k]) for k in self.keys}
        self.nseq = {k: 0 for k in self.keys}
        for pos, d in enumerate(order):
            k, i = cols[d]
            if pos == 0:
                inv, nu = np.zeros(self.N, np.int64), 1
            else:
                u, inv = np.unique(A[:, :pos], axis=0, return_inverse=True)
                inv, nu = inv.ravel(), len(u)
            self.sid[k][:, i] = self.nseq[k] + inv
            self.nseq[k] += nu

    def set_lam(self, lam):
        self.lam = np.asarray(lam, np.float64)
        self.E = self.S + self.H @ self.lam
        m = (-self.E).max()
        self.logZ = float(m + np.log(np.exp(-self.E - m).sum()))
        self.p_target = np.exp(-self.E - self.logZ)
        self._jits = {}

    # ------------------------------------------------------------ actors
    def init_params(self, kind, mlp_params=None, seed=0):
        if kind == "mlp":
            return mlp_params
        if kind.startswith("net"):
            act, w, d = net_spec(kind)
            keys = jax.random.split(jax.random.PRNGKey(seed), len(self.keys))
            return {k: net_init(kk, self.Ufeat[k].shape[1], self.D[k], act, w, d) for k, kk in zip(self.keys, keys)}
        if kind == "table_seq":
            self._seq_ids()
            return {k: jnp.zeros((self.nseq[k], self.D[k]), jnp.float32) for k in self.keys}
        n = self.nfeat if kind == "table_feat" else self.nhist
        return {k: jnp.zeros((n[k], self.D[k]), jnp.float32) for k in self.keys}

    def _logq(self, kind):
        d = {k: dict(ctx=jnp.asarray(self.ctx[k]), act=jnp.asarray(self.act[k]),
                     feat=jnp.asarray(self.Ufeat[k]), soft=jnp.asarray(self.Usoft[k]),
                     mask=jnp.asarray(self.Umask[k]), fid=jnp.asarray(self.fid[k]),
                     hid=jnp.asarray(self.hid[k]),
                     sid=jnp.asarray(self.sid[k]) if kind == "table_seq" else None) for k in self.keys}

        def logq(params, tau):
            tot = 0.0
            for k in self.keys:
                e = d[k]
                if kind == "mlp":
                    b = core.mlp(params[k], e["feat"])[e["ctx"]]
                elif kind.startswith("net"):
                    act, _, depth = net_spec(kind)
                    b = net(params[k], e["feat"], act, depth)[e["ctx"]]
                elif kind == "table_feat":
                    b = params[k][e["fid"]][e["ctx"]]
                elif kind == "table_seq":
                    b = params[k][e["sid"]]
                else:
                    b = params[k][e["hid"]]
                lg = jnp.where(e["mask"][e["ctx"]], -e["soft"][e["ctx"]] / tau - b, -jnp.inf)
                ls = jax.nn.log_softmax(lg, -1)
                tot = tot + jnp.sum(jnp.take_along_axis(ls, e["act"][..., None], -1)[..., 0], -1)
            return tot
        return logq

    def fns(self, kind):
        if kind not in self._jits:
            logq = self._logq(kind)
            E = jnp.asarray(self.E, jnp.float32)

            def surrogate(params, tau):
                lq = logq(params, tau)
                w = jax.lax.stop_gradient(jnp.exp(lq) * (E + tau * lq))
                return jnp.sum(w * lq), lq

            def jg(params, tau):
                (_, lq), g = jax.value_and_grad(surrogate, has_aux=True)(params, tau)
                return lq, g
            self._jits[kind] = (jax.jit(logq), jax.jit(jg))
        return self._jits[kind]

    def J(self, lq, tau):
        lq = np.asarray(lq, np.float64)
        q = np.exp(lq)
        return float(np.sum(q * (self.E + tau * lq))), q

    def evaluate(self, kind, params, stat_fn=None):
        """Exact metrics at tau = 1."""
        lq = np.asarray(self.fns(kind)[0](params, jnp.float32(1.0)), np.float64)
        J, q = self.J(lq, 1.0)
        out = {"J": J, "kl": J + self.logZ, "sum_q": float(q.sum()), "p_viol": float(q @ (self.H.sum(1) > 0)),
               "entropy": float(-(q * lq).sum()), "kl_coarse": self.kl_coarse(q)}
        if stat_fn is not None:
            if getattr(self, "_stats", (None,))[0] is not stat_fn:
                self._stats = (stat_fn, stat_fn(self.X))
            for k, v in self._stats[1].items():
                out[k] = float(q @ v)
        return out

    def kl_coarse(self, q):
        """KL of the joint marginal of every coarse level (>= 1), q vs the target."""
        if not hasattr(self, "_cinv"):
            top = np.concatenate([self.X[c.name].reshape(self.N, -1) for c in self.p.sampled() if c.level >= 1], 1)
            self._cinv = np.unique(top, axis=0, return_inverse=True)[1].ravel()
        inv = self._cinv
        qm = np.bincount(inv, q)
        pm = np.bincount(inv, self.p_target)
        ok = qm > 0
        return float(np.sum(qm[ok] * (np.log(qm[ok]) - np.log(np.maximum(pm[ok], 1e-300)))))


def stat_fn_for(name):
    if name == "support":
        return lambda X: {k: toys.support_block_stats(X["t"])[k] for k in ("flat_block_frac", "empty_block_frac",
                                                                           "dirt_frac")}
    if name == "modes":
        return lambda X: {k: toys.modes_block_stats(X["t"])[k] for k in ("mode_other", "mode_L", "broken_bonds")}
    return None


# ------------------------------------------------------------------ training
def tau_at(i, steps, tau0=2.0, anneal=0.2):
    return 1.0 + (tau0 - 1.0) * max(0.0, 1 - i / max(anneal * steps, 1e-9))


def train_exact(enum, kind, params, steps, lr, lam, every, stat_fn, clip=10.0, tau0=2.0, log=print):
    _, jg = enum.fns(kind)
    opt = adam_init(params)
    opt["t"] = jnp.int32(0)
    levels = list(range(enum.p.T + 1))

    @jax.jit
    def update(params, opt, tau):
        _, g = jg(params, tau)
        if kind == "mlp" or kind.startswith("net"):
            g = clip_by_level(g, levels, clip)
        return adam_step(params, g, opt, lr)
    curve = []
    t0 = time.time()
    for i in range(steps + 1):
        if i % every == 0 or i == steps:
            m = {"step": i, "t": time.time() - t0, **enum.evaluate(kind, params, stat_fn)}
            curve.append(m)
            log(f"[{kind}] {i} kl {m['kl']:.4f} coarse {m['kl_coarse']:.4f} " +
                " ".join(f"{k} {m[k]:.3f}" for k in m if k in ("flat_block_frac", "mode_other", "p_viol")))
        if i == steps:
            break
        params, opt = update(params, opt, jnp.float32(tau_at(i, steps, tau0)))
    return params, curve


def train_sampled(var, kind, enum, steps, lam, every, stat_fn, B, tau0=2.0, log=print):
    """cpass's / cfast's own update (critic, replays / returns, clipping, Adam)
    on the same layout, with tau annealed per step and lam fixed."""
    cfg = var.cfg
    region = enum.region
    lay = var.layout(region)
    lamj = jnp.asarray(lam, jnp.float32)
    curve = []
    t0 = time.time()
    seed0 = 1000 * var.seed + 7
    if kind == "cpass":
        var._build_train(lay)
    else:
        fn = var.fn = var.train_fns(lay)
        ones = {k: jnp.float32(1.0) for k in var.params}
        out = fn["fused"](var.params, var.cparams, var.opt, var.copt, ones,
                          jnp.asarray(np.arange(900_000, 900_000 + B), jnp.int32), {}, jnp.float32(1.0),
                          lamj, 0.0, 0.0, cfg["clip"], jax.random.PRNGKey(0))
        var.gs = {k: jnp.float32(max(float(v), 1.0)) for k, v in out[-1].items()}
    for i in range(steps + 1):
        if i % every == 0 or i == steps:
            m = {"step": i, "t": time.time() - t0, **enum.evaluate("mlp", var.params, stat_fn)}
            curve.append(m)
            log(f"[{kind} B={B}] {i} kl {m['kl']:.4f} coarse {m['kl_coarse']:.4f} " +
                " ".join(f"{k} {m[k]:.3f}" for k in m if k in ("flat_block_frac", "mode_other", "p_viol")))
        if i == steps:
            break
        tau = jnp.float32(tau_at(i, steps, tau0))
        seeds = jnp.asarray(np.arange(seed0 + i * B, seed0 + (i + 1) * B), jnp.int32)
        if kind == "cpass":
            extra = jnp.concatenate([jnp.array([tau]), lamj]).astype(jnp.float32)
            st, tot, recs = lay.jit_rec(var.params, seeds, {}, tau, lamj, extra)
            G, after = var._returns(lay, tot)
            if var.gscale is None:
                var.gscale = float(max(np.std(np.asarray(after)), 1.0))
            br = var._branches(lay, recs, after, seeds, {}, tau, lamj)
            var.params, var.cparams, var.opt, var.copt, _, _ = lay.jit_step(
                var.params, var.cparams, var.opt, var.copt, recs, G, tau, lamj, br, jnp.float32(var.gscale),
                cfg["lr"], cfg["lr_critic"], cfg["clip"])
        else:
            out = var.fn["fused"](var.params, var.cparams, var.opt, var.copt, var.gs, seeds, {}, tau, lamj,
                                  cfg["lr"], cfg["lr_critic"], cfg["clip"], jax.random.PRNGKey(i + 7919 * var.seed))
            _, var.params, var.cparams, var.opt, var.copt = out[:5]
    return var.params, curve


# ----------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["tiny", "support", "modes"])
    ap.add_argument("--runs", default="exact_mlp,exact_table_feat,exact_table_hist,cpass,cfast")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=None, help="exact runs (default: cfast's 3e-3)")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--out", default="reportables/2026-10-13_exactopt/runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--win", default="", help="actor window 'lo:hi[,plo:phi]' (default -1:2,-1:1)")
    a = ap.parse_args()
    set_actor_window(a.win)
    problem, region = get_problem(a.problem)
    lam = np.full(problem.T + 1, a.lam)
    base = make_variant("cpass", problem, region, a.seed)
    init = jax.tree_util.tree_map(lambda x: x, base.params)
    enum = Enum(problem, region, init, lam, cache=f"/tmp/exactopt_enum_{a.problem}{('_w' + a.win.replace(':', '_').replace(',', '_')) if a.win else ''}.pkl")
    stat_fn = stat_fn_for(a.problem)
    os.makedirs(a.out, exist_ok=True)
    meta = {"problem": a.problem, "N": enum.N, "logZ": enum.logZ, "lam": a.lam, "steps": a.steps,
            "nctx": {k: len(v) for k, v in enum.Usoft.items()}, "nfeat": enum.nfeat, "nhist": enum.nhist,
            "win": a.win, "feat_dim": {k: int(v.shape[1]) for k, v in enum.Ufeat.items()},
            "target": ({k: float(enum.p_target @ v) for k, v in stat_fn(enum.X).items()} if stat_fn else {}),
            "target_p_viol": float(enum.p_target @ (enum.H.sum(1) > 0))}
    if a.problem != "tiny":
        valid = enum.H.sum(1) == 0
        meta["n_valid_reachable"] = int(valid.sum())
        meta["F_valid_per_cell"] = float(-np.log(np.exp(-enum.S[valid]).sum()) / 16)
    print(json.dumps(meta), flush=True)
    for run in a.runs.split(","):
        t0 = time.time()
        if run.startswith("exact_"):
            kind = run[len("exact_"):]
            lr = a.lr if a.lr is not None else 3e-3
            if kind.startswith("table") and a.lr is None:
                lr = 3e-2
            if "@" in kind:                                 # per-run lr: exact_mlp@0.01
                kind, lr = kind.split("@")[0], float(kind.split("@")[1])
            params = enum.init_params(kind, init)
            _, curve = train_exact(enum, kind, params, a.steps, lr, lam, a.every, stat_fn, tau0=a.tau0)
            info = {"lr": lr, "win": a.win}
        else:
            var = make_variant(run, problem, region, a.seed, B=a.B)
            assert all(np.allclose(np.asarray(x), np.asarray(y)) for x, y in
                       zip(jax.tree_util.tree_leaves(var.params), jax.tree_util.tree_leaves(init)))
            _, curve = train_sampled(var, run, enum, a.steps, lam, a.every, stat_fn, a.B, tau0=a.tau0)
            info = {"lr": var.cfg["lr"], "B": a.B}
        name = f"{a.problem}_{run}{('_' + a.tag) if a.tag else ''}"
        with open(os.path.join(a.out, name + ".json"), "w") as fh:
            json.dump({"meta": meta, "run": run, "info": info, "curve": curve, "secs": time.time() - t0}, fh)
        print(f"== {name}: final kl {curve[-1]['kl']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
