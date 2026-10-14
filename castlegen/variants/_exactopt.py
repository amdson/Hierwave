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


def dimers_problem(H=4):
    """Dimers (triangular lattice, hard perfect matching, T = 0, no soft terms) on H rows with
    top / bottom edges (designed channel), same factors as the bench problem."""
    from castlegen.bench.problem import Chan
    from castlegen.bench.problems import dimers as dm
    p = dm.build_dimers()

    def edge(seed, Y, X, period, H=H):
        Y = np.asarray(Y, np.int64)
        return np.where(Y == 0, 1, np.where(Y == H - 1, 2, 0)).astype(np.int64)
    chans = [Chan("edge", 0, 3, designed=edge) if c.name == "edge" else c for c in p.chans]
    return dataclasses.replace(p, name=f"dimers_{H}", H=H, chans=chans, stats=None, probes=None, render=None,
                               exact={}, tol={})


def reachable_designed(lay, designed):
    """gc.reachable_states with designed channels held at their values."""
    p = lay.p
    st = {c.name: (np.asarray(designed[c.name], np.int32)[None] if c.designed is not None else
                   np.full((1,) + lay.hw[c.level], -1, np.int32)) for c in p.chans}
    for l in lay.levels:
        for k in range(core.NCOL):
            Y, X, valid = (np.asarray(a) for a in lay.sites(l, k))
            for c in lay.chans[l]:
                def masks(state, c=c, Y=Y, X=X):
                    _, nv, look = lay.cond(state, c.name, jnp.asarray(Y), jnp.asarray(X))
                    v = nv + look
                    return v == jnp.min(v, axis=1, keepdims=True)
                m = np.asarray(jax.vmap(masks)({n: jnp.asarray(a) for n, a in st.items()}))
                for s in np.flatnonzero(valid):
                    rows, vals = np.nonzero(m[:, s])
                    st = {n: a[rows] for n, a in st.items()}
                    m = m[rows]
                    st[c.name][:, Y[s], X[s]] = vals
    return {c.name: st[c.name] for c in p.sampled()}


def get_problem(name):
    """(problem, region)."""
    if name.startswith("dimers"):                      # dimers4 (H = 4), dimers4x8: periodic W
        H, _, W = name[6:].partition("x")
        return dimers_problem(int(H)), Region(int(W) if W else 4)
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
    return net_hidden(p, x, act, depth) @ p["Wo"] + p["bo"]


def net_hidden(p, x, act, depth):
    h = x
    for i in range(depth):
        h = h @ p[f"W{i}"] + p[f"b{i}"]
        if act == "ln":
            mu = h.mean(-1, keepdims=True)
            var = ((h - mu) ** 2).mean(-1, keepdims=True)
            h = jax.nn.silu((h - mu) * jax.lax.rsqrt(var + 1e-5) * p[f"g{i}"] + p[f"c{i}"])
        else:
            h = jnp.tanh(h)
    return h


# ---- cdim3 extra actors (all full-domain per-channel outputs; constraint 5) ----
# cnet-<act>-<W>x<D>  candidate-scoring head (cconv-style): b(t) = h Wo + bo + sum_j w_j
#                     SiLU(h A + e(t) E + c)_j, e(t) = (channel embedding of t, s(t)), h the net's
#                     last hidden layer; Wo and w start at zero (b = 0 at step 0).
# hnet-<act>-<W>x<D>  learned embeddings of the discrete window pattern as extra INPUTS: the whole
#                     actor input vector and the 3 x 3 same-level cells around the site (with the
#                     in-block position) are each hashed (crc32) to a table of HASH_ROWS rows of
#                     HASH_DIM; [feat, emb_full, emb_local] feeds the net.  Generic and local (a
#                     function of the actor window only); an input table, not an output table.
HASH_ROWS, HASH_DIM, CAND_HC = 4096, 16, 32


def window_hash_ids(lay, l, U, rows=HASH_ROWS):
    """U (C, F) actor inputs at level l -> (full ids, local 3 x 3 ids), each (C,) in [0, rows)."""
    import zlib
    win = lay.actor_win
    fcell = sum(int(lay.emb[c.name].shape[1]) + 2 for c in lay.all_chans[l])
    lo = min(d[0] for d in win.same)
    n = int(round(np.sqrt(len(win.same))))
    assert len(win.same) == n * n and not lay.resid
    pos = U[:, -5:-1].argmax(1)                 # in-block position one-hot (features() layout)
    cells = U[:, :n * n * fcell].reshape(len(U), n, n, fcell)
    full = np.array([zlib.crc32(r.tobytes()) % rows for r in np.ascontiguousarray(U, np.float32)])
    loc = np.empty(len(U), np.int64)
    for i in range(len(U)):
        oy, ox = (pos[i] >> 1) - lo, (pos[i] & 1) - lo
        blk = np.ascontiguousarray(cells[i, oy - 1:oy + 2, ox - 1:ox + 2], np.float32)
        loc[i] = zlib.crc32(blk.tobytes() + bytes([int(pos[i])])) % rows
    return full, loc


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
        self.des = {k: jnp.asarray(v, jnp.int32) for k, v in problem.designed_values(0, region).items()}
        self.X = reachable_designed(lay, self.des) if self.des else gc.reachable_states(lay)
        N = self.N = len(next(iter(self.X.values())))
        if verbose:
            print(f"[enum] {problem.name}: {N} reachable outcomes ({time.time() - t0:.1f}s)", flush=True)
        T = problem.T
        lam0 = jnp.zeros(T + 1, jnp.float32)

        def one(forced):
            _, _, lq, rec = core.coloured_pass(lay, params, 0, self.des, 1.0, lam0, forced=forced, record=True)
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
            e = en.energy(self.p, {**{k: np.asarray(v) for k, v in self.des.items()},
                                   **{k: a[i] for k, a in self.X.items()}}, self.region)
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
        if kind.startswith(("cnet", "hnet")):
            act, w, d = net_spec(kind)
            keys = jax.random.split(jax.random.PRNGKey(seed), len(self.keys))
            out = {}
            for k, kk in zip(self.keys, keys):
                k1, k2, k3, k4 = jax.random.split(kk, 4)
                din = self.Ufeat[k].shape[1] + (2 * HASH_DIM if kind.startswith("hnet") else 0)
                q = net_init(k1, din, self.D[k], act, w, d)
                if kind.startswith("hnet"):
                    q["Hf"] = jax.random.normal(k2, (HASH_ROWS, HASH_DIM))
                    q["Hl"] = jax.random.normal(k3, (HASH_ROWS, HASH_DIM))
                else:
                    ke = self._cand_emb(k).shape[1] + 1
                    q["A"] = jax.random.normal(k2, (w, CAND_HC)) / np.sqrt(w)
                    q["E"] = jax.random.normal(k3, (ke, CAND_HC)) / np.sqrt(ke)
                    q["c"] = jnp.zeros(CAND_HC)
                    q["w"] = jnp.zeros(CAND_HC)
                out[k] = q
            return out
        if kind == "table_seq":
            self._seq_ids()
            return {k: jnp.zeros((self.nseq[k], self.D[k]), jnp.float32) for k in self.keys}
        n = self.nfeat if kind == "table_feat" else self.nhist
        return {k: jnp.zeros((n[k], self.D[k]), jnp.float32) for k in self.keys}

    def _cand_emb(self, k):
        l = self.level[k]
        c = next(c for c in self.lay.chans[l] if core.key_of(l, c.name) == k)
        return np.asarray(self.lay.emb[c.name])

    def hash_ids(self, k):
        if not hasattr(self, "_hids"):
            self._hids = {}
        if k not in self._hids:
            self._hids[k] = window_hash_ids(self.lay, self.level[k], self.Ufeat[k])
        return self._hids[k]

    def actor_b(self, kind, q, k, feat, soft, tau):
        """b over the unique contexts of network k (cnet / hnet kinds)."""
        act, _, depth = net_spec(kind)
        if kind.startswith("hnet"):
            hf, hl = (jnp.asarray(a) for a in self.hash_ids(k))
            x = jnp.concatenate([feat, q["Hf"][hf], q["Hl"][hl]], -1)
            return net(q, x, act, depth)
        h = net_hidden(q, feat, act, depth)
        emb = jnp.asarray(self._cand_emb(k))                                    # (D, ke - 1)
        e = jnp.concatenate([jnp.broadcast_to(emb[None], soft.shape + emb.shape[1:]), soft[..., None] / tau], -1)
        z = (h @ q["A"])[:, None, :] + e @ q["E"] + q["c"]
        return h @ q["Wo"] + q["bo"] + jax.nn.silu(z) @ q["w"]

    def b_unique(self, kind, params, k, tau=1.0):
        """b over the unique (feature, soft, mask) contexts of network k: (C, D)."""
        feat, soft = jnp.asarray(self.Ufeat[k]), jnp.asarray(self.Usoft[k])
        if kind == "mlp":
            return core.mlp(params[k], feat)
        if kind.startswith(("cnet", "hnet")):
            return self.actor_b(kind, params[k], k, feat, soft, tau)
        if kind.startswith("net"):
            act, _, depth = net_spec(kind)
            return net(params[k], feat, act, depth)
        if kind == "table_feat":
            return params[k][jnp.asarray(self.fid[k])]
        raise ValueError(kind)

    def ctx_weights(self, k, how="target"):
        """Per unique context of network k: expected number of visits under the
        target ('target'), or 1 for every reachable context ('all')."""
        C = len(self.Usoft[k])
        if how == "all":
            return np.ones(C)
        ctx = self.ctx[k]
        return np.bincount(ctx.ravel(), np.repeat(self.p_target, ctx.shape[1]), minlength=C)

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
                elif kind.startswith(("cnet", "hnet")):
                    b = self.actor_b(kind, params[k], k, e["feat"], e["soft"], tau)[e["ctx"]]
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
        if self.p.T == 0:
            return 0.0                      # no coarse level
        if not hasattr(self, "_cinv"):
            top = np.concatenate([self.X[c.name].reshape(self.N, -1) for c in self.p.sampled() if c.level >= 1], 1)
            self._cinv = np.unique(top, axis=0, return_inverse=True)[1].ravel()
        inv = self._cinv
        qm = np.bincount(inv, q)
        pm = np.bincount(inv, self.p_target)
        ok = qm > 0
        return float(np.sum(qm[ok] * (np.log(qm[ok]) - np.log(np.maximum(pm[ok], 1e-300)))))


def dimer_stats(X):
    """Per outcome: orientation fractions h / v / diag (partner direction // 2)."""
    o = np.asarray(X["p"]).reshape(len(X["p"]), -1) // 2
    return {"horizontal": (o == 0).mean(1), "vertical": (o == 1).mean(1), "diagonal": (o == 2).mean(1)}


def stat_fn_for(name):
    if name == "support":
        return lambda X: {k: toys.support_block_stats(X["t"])[k] for k in ("flat_block_frac", "empty_block_frac",
                                                                           "dirt_frac")}
    if name.startswith("dimers"):
        return dimer_stats
    if name == "modes":
        return lambda X: {k: toys.modes_block_stats(X["t"])[k] for k in ("mode_other", "mode_L", "broken_bonds")}
    return None


# ------------------------------------------------------------- distillation
def distill(enum, src_kind, src_params, kind, params, steps=3000, lr=3e-3, how="target", every=500, log=print):
    """cdim3: supervised fit of actor `kind` to the per-context policies of
    `src_kind` (tau = 1, masks applied): minimise sum_c w_c KL(pi_src(.|c) || pi(.|c)) / sum_c w_c,
    w from Enum.ctx_weights(how).  Full batch Adam.  Returns (params, curve)."""
    tgt, wts, msk = {}, {}, {}
    for k in enum.keys:
        m = jnp.asarray(enum.Umask[k])
        lg = jnp.where(m, -jnp.asarray(enum.Usoft[k]) - enum.b_unique(src_kind, src_params, k), -jnp.inf)
        tgt[k] = jax.nn.softmax(lg, -1)
        w = enum.ctx_weights(k, how)
        wts[k] = jnp.asarray(w / w.sum(), jnp.float32)
        msk[k] = m

    def loss(params):
        tot = 0.0
        for k in enum.keys:
            lg = jnp.where(msk[k], -jnp.asarray(enum.Usoft[k]) - enum.b_unique(kind, params, k), -jnp.inf)
            ls = jax.nn.log_softmax(lg, -1)
            kl = jnp.sum(jnp.where(tgt[k] > 0, tgt[k] * (jnp.log(jnp.maximum(tgt[k], 1e-30)) - ls), 0.0), -1)
            tot = tot + jnp.sum(wts[k] * kl)
        return tot
    opt = adam_init(params)
    opt["t"] = jnp.int32(0)

    @jax.jit
    def step(params, opt):
        v, g = jax.value_and_grad(loss)(params)
        p2, o2 = adam_step(params, g, opt, lr)
        return p2, o2, v
    curve = []
    for i in range(steps + 1):
        if i % every == 0 or i == steps:
            v = float(loss(params))
            m = {"step": i, "distill_loss": v, **enum.evaluate(kind, params, stat_fn_for("dimers"))}
            curve.append(m)
            log(f"[distill {kind}<-{src_kind} {how}] {i} loss {v:.4f} kl {m['kl']:.4f} "
                f"h/v/d {m.get('horizontal', 0):.3f}/{m.get('vertical', 0):.3f}/{m.get('diagonal', 0):.3f}")
        if i == steps:
            break
        params, opt, _ = step(params, opt)
    return params, curve


# ------------------------------------------------------------------ training
def tau_at(i, steps, tau0=2.0, anneal=0.2):
    return 1.0 + (tau0 - 1.0) * max(0.0, 1 - i / max(anneal * steps, 1e-9))


def train_exact(enum, kind, params, steps, lr, lam, every, stat_fn, clip=10.0, tau0=2.0, log=print,
                anneal=0.2, warmup=0, hook=None):
    """anneal: fraction of the steps over which tau goes tau0 -> 1; warmup: linear
    lr warm-up steps (cdim3; defaults reproduce cdim2).  hook(i, params, tau) is
    called at every logged step (diagnostics)."""
    _, jg = enum.fns(kind)
    opt = adam_init(params)
    opt["t"] = jnp.int32(0)
    levels = list(range(enum.p.T + 1))

    @jax.jit
    def update(params, opt, tau, lr_i):
        _, g = jg(params, tau)
        if not kind.startswith("table"):
            g = clip_by_level(g, levels, clip)
        return adam_step(params, g, opt, lr_i)
    curve = []
    t0 = time.time()
    for i in range(steps + 1):
        if i % every == 0 or i == steps:
            m = {"step": i, "t": time.time() - t0, **enum.evaluate(kind, params, stat_fn)}
            m["tau"] = tau_at(i, steps, tau0, anneal)
            curve.append(m)
            log(f"[{kind}] {i} kl {m['kl']:.4f} H/cell {m['entropy'] / enum.region.W / enum.p.H:.3f} " +
                " ".join(f"{k} {m[k]:.3f}" for k in m if k in ("flat_block_frac", "mode_other", "p_viol", "horizontal", "vertical", "diagonal")))
            if hook is not None:
                hook(i, params, m["tau"])
        if i == steps:
            break
        lr_i = lr * min(1.0, (i + 1) / warmup) if warmup else lr
        params, opt = update(params, opt, jnp.float32(tau_at(i, steps, tau0, anneal)), jnp.float32(lr_i))
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
                " ".join(f"{k} {m[k]:.3f}" for k in m if k in ("flat_block_frac", "mode_other", "p_viol", "horizontal", "vertical", "diagonal")))
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
    ap.add_argument("problem", help="tiny | support | modes | dimers<H>[x<W>] (periodic width W, default 4)")
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
    ap.add_argument("--anneal", type=float, default=0.2, help="fraction of steps for tau0 -> 1 (cdim3)")
    ap.add_argument("--warmup", type=int, default=0, help="linear lr warm-up steps, exact runs (cdim3)")
    ap.add_argument("--init", default="", help="pickle of params to start exact runs from (cdim3)")
    ap.add_argument("--save", default="", help="pickle the final params of exact runs here (cdim3)")
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
        meta["F_valid_per_cell"] = float(-np.log(np.exp(-enum.S[valid]).sum()) / (problem.H * region.W))
        meta["log_nvalid_per_cell"] = float(np.log(valid.sum()) / (problem.H * region.W))
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
            if a.init:
                import pickle
                with open(a.init, "rb") as fh:
                    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(fh))
            params, curve = train_exact(enum, kind, params, a.steps, lr, lam, a.every, stat_fn, tau0=a.tau0,
                                        anneal=a.anneal, warmup=a.warmup)
            if a.save:
                import pickle
                with open(a.save, "wb") as fh:
                    pickle.dump(jax.device_get(params), fh)
            info = {"lr": lr, "win": a.win, "anneal": a.anneal, "warmup": a.warmup, "init": a.init, "tau0": a.tau0}
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
