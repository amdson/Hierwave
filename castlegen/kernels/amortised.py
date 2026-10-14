"""Sweep 2: the amortised loop  (draws -> p*-invariant kernel cycles -> weighted ML).

Generic: built only from the problem's factors, the layout and the kernels.

JaxBlockGibbs.  A GPU (JAX) port of exact_block.ExactBlockGibbs, batched over
worlds.  It reuses ExactBlockGibbs's structure (blocks, colouring, instance
groups) and enumerates a completion set `comps` shared by every block:
  - default: the block's full domain product (the kernel's shared path);
  - `internal_pruned(K)`: the full product pruned by the hard instances whose
    members all lie inside one block (a superset of every block's feasible set,
    whatever the outside values are), so the conditional over it is still exact.
For each colour class the conditional energy of every completion of every block
is  soft + lam * violations  from the instances touching the block, and one
completion is drawn by Gumbel-max (chunked over completions, exact).  lam=None
excludes violating completions (hard target; a block with no valid completion
keeps its value); a finite lam (<= 20) prices them.

Energy.  `make_energy(flat, insts)` gives per-world total soft energy and
violation count of a flat state (same instance rules as bench/energy.py), and
the soft energy of instances that touch a level <= `fine_max`.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from .exact_block import ExactBlockGibbs, _lookup


def _lookup_j(kind, by_pos, table, vals, pos):
    if kind == "unary":
        return table[vals[0]]
    if kind == "count":
        return table[vals[0], sum(vals[1:])]
    if by_pos:
        return table[pos, vals[0], vals[1]]
    return table[vals[0], vals[1]]


def internal_pruned(K: ExactBlockGibbs, max_out=2_000_000):
    """Completions consistent with every hard instance lying wholly inside one
    block (slot-level structure is the same for every block of a periodic
    layout; checked on block 0's rows).  Built slot by slot."""
    # internal hard rows of block 0: all members are slots of block 0
    rows = []
    for k, (blocks, groups) in enumerate(K.cls):
        for gr in groups:
            if not gr["d"]["hard"]:
                continue
            for i in np.where(gr["block"] == 0)[0]:
                sl = gr["slots"][i]
                if (sl >= 0).all() and (gr["G"][i] >= 0).all():
                    pos = None if gr["pos"] is None else gr["pos"][i]
                    rows.append((gr["d"], sl, pos))
        if any(gr["block"].size and (gr["block"] == 0).any() for gr in groups):
            break
    last = [int(sl.max()) for _, sl, _ in rows]
    comps = np.zeros((1, 0), np.int64)
    for j in range(K.nv):
        n = comps.shape[0]
        comps = np.concatenate([np.repeat(comps, K.D[j], 0), np.tile(np.arange(K.D[j]), n)[:, None]], 1)
        ok = np.ones(comps.shape[0], bool)
        for (d, sl, pos), ls in zip(rows, last):
            if ls != j:
                continue
            vals = [view[np.minimum(comps[:, s], view.size - 1)] for (_, _, view), s in zip(d["members"], sl)]
            ok &= ~np.isinf(_lookup(d, vals, None if pos is None else pos))
        comps = comps[ok]
        assert comps.shape[0] <= max_out, f"internal-pruned set too large at slot {j}: {comps.shape[0]}"
    return comps


class JaxBlockGibbs:
    def __init__(self, problem, region, level, depth, lam=None, comps="full", chunk=None, fast=True):
        K = ExactBlockGibbs(problem, region, level=level, depth=depth, lam=lam, max_joint=10 ** 12 if comps == "full" else 0)
        self.K, self.lam, self.level, self.depth = K, lam, level, depth
        if isinstance(comps, str):
            comps = K.comps if comps == "full" else internal_pruned(K)
        self.comps = np.asarray(comps, np.int32)
        C = self.comps.shape[0]
        self.chunk = chunk or C
        nchunk = -(-C // self.chunk)
        pad = nchunk * self.chunk - C
        self.valid_c = np.concatenate([np.ones(C, bool), np.zeros(pad, bool)]).reshape(nchunk, self.chunk)
        self.comps_c = np.concatenate([self.comps, np.zeros((pad, K.nv), np.int32)]).reshape(nchunk, self.chunk, K.nv)
        self.classes = []
        for blocks, groups in K.cls:
            rowmap = np.full(K.nb, -1, np.int64)
            rowmap[blocks] = np.arange(blocks.size)
            gs = []
            for gr in groups:
                d = gr["d"]
                gs.append(dict(kind=d["kind"], by_pos=bool(d.get("by_pos")), table=jnp.asarray(d["table"], jnp.float32),
                               views=[jnp.asarray(view, jnp.int32) for _, _, view in d["members"]],
                               G=jnp.asarray(gr["G"], jnp.int32), Cst=jnp.asarray(gr["C"], jnp.int32),
                               slots=jnp.asarray(gr["slots"], jnp.int32), r=jnp.asarray(rowmap[gr["block"]], jnp.int32),
                               pos=None if gr["pos"] is None else jnp.asarray(gr["pos"], jnp.int32)))
            self.classes.append((jnp.asarray(K.BV[blocks], jnp.int32), blocks.size, gs))
        self.nchunk = nchunk
        self.fast = bool(fast) and self._build_fast()
        self.sweep_ref = jax.jit(self._sweep)
        self.sweep = jax.jit(self._sweep_fast) if self.fast else self.sweep_ref

    # ---------------------------------------------------------------- fast path
    def _build_fast(self):
        """Factor each colour class's energy.  Every instance row has either all its
        members inside the block (energy depends on the completion only: computed once,
        here) or exactly one member inside (energy is a per-world field over that
        variable's domain: computed per sweep over D values instead of over every
        completion, then summed over the block's variables).  Anything else -> slow path."""
        K, nv = self.K, self.K.nv
        Dmax = int(max(K.D))
        self.Dmax = Dmax
        comps = jnp.asarray(self.comps, jnp.int32)
        C = comps.shape[0]
        # one-hot of (variable, value) per completion: (nv*Dmax, C)
        oh = np.zeros((nv * Dmax, C), np.float32)
        oh[(np.arange(nv)[:, None] * Dmax + self.comps.T), np.arange(C)[None, :]] = 1.0
        self.onehot = jnp.asarray(oh)
        fcls = []
        for (BV, nbk, gs), (blocks, groups) in zip(self.classes, K.cls):
            internal, single = [], []
            for g, gr in zip(gs, groups):
                nin = (gr["slots"] >= 0).sum(1)
                nm = gr["slots"].shape[1]
                if not np.all((nin == nm) | (nin == 1)):
                    return False
                if nm == 1:
                    return False
                ii, si = np.where(nin == nm)[0], np.where(nin == 1)[0]
                if ii.size:
                    internal.append({**g, **{k: (None if g[k] is None else g[k][jnp.asarray(ii)]) for k in ("G", "Cst", "slots", "r", "pos")}})
                if si.size:
                    sl = gr["slots"][si]
                    star = np.argmax(sl >= 0, axis=1)
                    j = sl[np.arange(si.size), star]
                    rowmap = np.full(K.nb, -1, np.int64)
                    rowmap[blocks] = np.arange(blocks.size)
                    single.append(dict(kind=g["kind"], by_pos=g["by_pos"], table=g["table"], views=g["views"],
                                       G=g["G"][jnp.asarray(si)], Cst=g["Cst"][jnp.asarray(si)], star=jnp.asarray(star, jnp.int32),
                                       seg=jnp.asarray(rowmap[gr["block"][si]] * nv + j, jnp.int32),
                                       pos=None if g["pos"] is None else g["pos"][jnp.asarray(si)]))
            # world-independent part: all-internal rows over every completion
            if internal:
                isoft, inv = self._class_energy(jnp.zeros((1, K.flat.n), jnp.int32), internal, nbk, comps)
                isoft, inv = isoft[0], inv[0]
            else:
                isoft = inv = jnp.zeros((nbk, C))
            fcls.append((BV, nbk, single, isoft, inv))
        self.fclasses = fcls
        return True

    def _class_field(self, v, single, nbk):
        """Per-world fields (Wb, nbk*nv, Dmax) of soft and nviol from the one-member-inside rows."""
        Wb, nv, Dmax = v.shape[0], self.K.nv, self.Dmax
        d = jnp.arange(Dmax)
        soft = jnp.zeros((nbk * nv, Wb, Dmax))
        nviol = jnp.zeros((nbk * nv, Wb, Dmax))
        for g in single:
            vals = []
            for m, view in enumerate(g["views"]):
                Gm = g["G"][:, m]
                fixed = jnp.where(Gm >= 0, view[v[:, jnp.maximum(Gm, 0)]], g["Cst"][:, m])          # (Wb, rows)
                inside = view[jnp.minimum(d, view.shape[0] - 1)]                                       # (Dmax,)
                vals.append(jnp.where((g["star"] == m)[None, :, None], inside[None, None], fixed[:, :, None]))
            pos = None if g["pos"] is None else g["pos"][None, :, None]
            e = _lookup_j(g["kind"], g["by_pos"], g["table"], vals, pos)                            # (Wb, rows, Dmax)
            inf = jnp.isinf(e)
            et = jnp.moveaxis(e, 1, 0)
            it = jnp.moveaxis(inf, 1, 0)
            soft = soft + jax.ops.segment_sum(jnp.where(it, 0.0, et), g["seg"], num_segments=nbk * nv)
            nviol = nviol + jax.ops.segment_sum(it.astype(jnp.float32), g["seg"], num_segments=nbk * nv)
        return jnp.moveaxis(soft, 1, 0), jnp.moveaxis(nviol, 1, 0)

    def _sweep_fast(self, v, key):
        Wb, nv, Dmax = v.shape[0], self.K.nv, self.Dmax
        changed = jnp.zeros(Wb, jnp.int32)
        comps = jnp.asarray(self.comps, jnp.int32)
        C = comps.shape[0]
        hp = jax.lax.Precision.HIGHEST
        for BV, nbk, single, isoft, inv in self.fclasses:
            key, sub = jax.random.split(key)
            fs, fn = self._class_field(v, single, nbk)
            mm = lambda f: jnp.matmul(f.reshape(Wb * nbk, nv * Dmax), self.onehot, precision=hp).reshape(Wb, nbk, C)
            soft = isoft[None] + mm(fs)
            nvl = inv[None] + mm(fn)
            nvl = jnp.round(nvl)
            if self.lam is None:
                lg = jnp.where(nvl > 0, -jnp.inf, -soft)
            else:
                lg = -(soft + self.lam * nvl)
            # the reference path's noise (per chunk, padded shape), so equal keys give equal draws
            gmb = jnp.concatenate([jax.random.gumbel(jax.random.fold_in(sub, ci), (Wb, nbk, self.chunk)) for ci in range(self.nchunk)], -1)[..., :C]
            sc = jnp.where(jnp.isfinite(lg), lg + gmb, -jnp.inf)
            arg = jnp.argmax(sc, -1)
            anyok = jnp.isfinite(lg).any(-1)
            new = comps[arg]
            old = v[:, BV]
            newv = jnp.where(anyok[..., None], new, old)
            changed = changed + (newv != old).any(-1).sum(-1).astype(jnp.int32)
            v = v.at[:, BV].set(newv)
        return v, changed

    def _class_energy(self, v, gs, nbk, comps):
        """v (Wb, n) ; comps (Cc, nv) -> soft, nv (Wb, nbk, Cc)."""
        Wb, Cc = v.shape[0], comps.shape[0]
        soft = jnp.zeros((Wb, nbk, Cc))
        nviol = jnp.zeros((Wb, nbk, Cc))
        for g in gs:
            vals = []
            for m, view in enumerate(g["views"]):
                Gm, sm = g["G"][:, m], g["slots"][:, m]
                fixed = jnp.where(Gm >= 0, view[v[:, jnp.maximum(Gm, 0)]], g["Cst"][:, m])          # (Wb, n)
                cv = view[jnp.minimum(comps[:, jnp.maximum(sm, 0)].T, view.shape[0] - 1)]          # (n, Cc)
                vals.append(jnp.where((sm >= 0)[None, :, None], cv[None], fixed[:, :, None]))
            pos = None if g["pos"] is None else g["pos"][None, :, None]
            e = _lookup_j(g["kind"], g["by_pos"], g["table"], vals, pos)                            # (Wb, n, Cc)
            inf = jnp.isinf(e)
            seg = lambda x: jax.vmap(lambda y: jax.ops.segment_sum(y, g["r"], num_segments=nbk))(x)
            soft = soft + seg(jnp.where(inf, 0.0, e))
            nviol = nviol + seg(inf.astype(jnp.float32))
        return soft, nviol

    def _sweep(self, v, key):
        """One systematic sweep over the colour classes. v (Wb, n) int32.
        Returns (v, changed blocks per world)."""
        changed = jnp.zeros(v.shape[0], jnp.int32)
        comps_c, valid_c = jnp.asarray(self.comps_c), jnp.asarray(self.valid_c)
        for BV, nbk, gs in self.classes:
            key, sub = jax.random.split(key)

            def chunk_body(carry, xs):
                best, arg, anyok = carry
                comps, okc, ci = xs
                soft, nv = self._class_energy(v, gs, nbk, comps)
                if self.lam is None:
                    lg = jnp.where(nv > 0, -jnp.inf, -soft)
                else:
                    lg = -(soft + self.lam * nv)
                lg = jnp.where(okc[None, None], lg, -jnp.inf)
                gmb = jax.random.gumbel(jax.random.fold_in(sub, ci), lg.shape)
                sc = jnp.where(jnp.isfinite(lg), lg + gmb, -jnp.inf)
                a = jnp.argmax(sc, -1)
                m = jnp.max(sc, -1)
                upd = m > best
                return (jnp.where(upd, m, best), jnp.where(upd, a + ci * self.chunk, arg),
                        anyok | jnp.isfinite(lg).any(-1)), None

            init = (jnp.full((v.shape[0], nbk), -jnp.inf), jnp.zeros((v.shape[0], nbk), jnp.int32),
                    jnp.zeros((v.shape[0], nbk), bool))
            (best, arg, anyok), _ = jax.lax.scan(chunk_body, init, (comps_c, valid_c, jnp.arange(comps_c.shape[0])))
            flatc = comps_c.reshape(-1, comps_c.shape[-1])
            new = flatc[arg]                                         # (Wb, nbk, nv)
            old = v[:, BV]
            newv = jnp.where(anyok[..., None], new, old)
            changed = changed + (newv != old).any(-1).sum(-1).astype(jnp.int32)
            v = v.at[:, BV].set(newv)
        return v, changed


def make_energy(flat, insts, fine_max=2):
    """jit fn v (Wb, n) -> (soft, nviol, soft_fine) per world; soft_fine sums
    instances with a member at level <= fine_max."""
    lev_of = np.zeros(flat.n, np.int64)
    for c in flat.p.chans:
        o = flat.off[c.name]
        lev_of[o:o + int(np.prod(flat.shape[c.name]))] = c.level
    gs = []
    for d in insts:
        G = np.stack([g for g, _, _ in d["members"]], 1)
        lv = np.where(G >= 0, lev_of[np.maximum(G, 0)], 99).min(1)
        gs.append(dict(kind=d["kind"], by_pos=bool(d.get("by_pos")), table=jnp.asarray(d["table"], jnp.float32),
                       views=[jnp.asarray(view, jnp.int32) for _, _, view in d["members"]],
                       G=jnp.asarray(G, jnp.int32), Cst=jnp.asarray(np.stack([c for _, c, _ in d["members"]], 1), jnp.int32),
                       fine=jnp.asarray(lv <= fine_max), pos=None if d["pos"] is None or not d.get("by_pos") else jnp.asarray(d["pos"], jnp.int32)))

    @jax.jit
    def energy(v):
        soft = nv = sf = 0.0
        for g in gs:
            vals = [jnp.where(g["G"][:, m] >= 0, view[v[:, jnp.maximum(g["G"][:, m], 0)]], g["Cst"][:, m])
                    for m, view in enumerate(g["views"])]
            e = _lookup_j(g["kind"], g["by_pos"], g["table"], vals, None if g["pos"] is None else g["pos"][None])
            inf = jnp.isinf(e)
            e0 = jnp.where(inf, 0.0, e)
            soft = soft + e0.sum(-1)
            nv = nv + inf.sum(-1)
            sf = sf + jnp.where(g["fine"][None], e0, 0.0).sum(-1)
        return soft, nv, sf
    return energy


def refine(kernels, v, key, n_cycles, energy=None):
    """Apply n_cycles cycles of [K.sweep for K in kernels] to a batch v (Wb, n).
    Returns (v, per-cycle list of dicts with energies and changed counts)."""
    curve = []
    if energy is not None:
        s, nv, sf = energy(v)
        curve.append({"cycle": 0, "soft": np.asarray(s), "nv": np.asarray(nv), "soft_fine": np.asarray(sf)})
    for c in range(n_cycles):
        ch = []
        for K in kernels:
            key, sub = jax.random.split(key)
            v, n = K.sweep(v, sub)
            ch.append(np.asarray(n))
        if energy is not None:
            s, nv, sf = energy(v)
            curve.append({"cycle": c + 1, "soft": np.asarray(s), "nv": np.asarray(nv), "soft_fine": np.asarray(sf),
                          "changed": ch})
    return v, curve


def make_ml_update_drawwise(lay, tau=1.0, clip=10.0, score_levels=None):
    """Weighted ML with infeasible draws dropped individually (instead of whole
    states): loss = -sum_i w_i sum_{feasible draws j} log q_j(x_ij) / sum_i w_i.
    Every kept term is the pass's own conditional at that draw given the state's
    context; only draws whose value lies outside the pass's least-violating set
    (q_j = 0) are left out.  Same Adam / level mask as logq.make_ml_update."""
    from castlegen.kernels.logq import logq_of_state
    from castlegen.variants.cpass import adam_step, clip_by_level
    levels = list(range(lay.T + 1))

    def loss_fn(params, states, w):
        r = jax.vmap(lambda s: logq_of_state(lay, params, s, tau=tau, score_levels=score_levels))(states)
        loss = -jnp.sum(w * r["total_finite"]) / jnp.maximum(w.sum(), 1e-30)
        return loss, (r["n_infeasible"].sum(), r["feasible"].sum())

    @jax.jit
    def step(params, opt, states, w, lr, level_on):
        (loss, (ninf, nfeas)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, states, w)
        g = clip_by_level(g, levels, clip)
        lv = lambda k: level_on[int(k[1:k.index("_")])]
        g = {k: jax.tree_util.tree_map(lambda x: x * lv(k), v) for k, v in g.items()}
        new, nopt = adam_step(params, g, opt, lr)
        keep = lambda k: lv(k) > 0
        sel = lambda a, b, k: jax.tree_util.tree_map(lambda x, y: jnp.where(keep(k), x, y), a, b)
        new = {k: sel(new[k], params[k], k) for k in params}
        nm = {k: sel(nopt["m"][k], opt["m"][k], k) for k in params}
        nv = {k: sel(nopt["v"][k], opt["v"][k], k) for k in params}
        return new, {"m": nm, "v": nv, "t": nopt["t"]}, {"loss": loss, "n_infeasible": ninf, "n_feasible": nfeas}
    return step


def make_refine_scan(kernels, n_cycles):
    """jit fn (v, key) -> (v, changed_last) running n_cycles cycles of [K.sweep for K in kernels]
    under one lax.scan (no per-cycle energy / host traffic).  Same sweeps as `refine` (keys differ:
    one split per cycle).  changed_last: list over kernels of (Wb,) blocks changed in the last cycle."""
    fns = [(K._sweep_fast if K.fast else K._sweep) for K in kernels]

    @jax.jit
    def run(v, key):
        def body(v, k):
            ks = jax.random.split(k, len(fns))
            ch = []
            for f, kk in zip(fns, ks):
                v, n = f(v, kk)
                ch.append(n)
            return v, jnp.stack(ch)
        v, ch = jax.lax.scan(body, v, jax.random.split(key, n_cycles))
        return v, ch[-1]
    return run


def make_multi_update(step):
    """Fuse several ML updates (same states pool, different minibatch indices) in one jit.
    multi(params, opt, states, idx (U, B), w, lr, on) -> (params, opt, infos with leading axis U).
    Each inner update is exactly `step` on states[idx[u]]."""
    @jax.jit
    def multi(params, opt, states, idx, w, lr, on):
        def body(carry, ix):
            p, o = carry
            p, o, info = step(p, o, {k: x[ix] for k, x in states.items()}, w, lr, on)
            return (p, o), info
        (params, opt), infos = jax.lax.scan(body, (params, opt), idx)
        return params, opt, infos
    return multi
