"""Sweep 1 (a): a generic exact block-Gibbs kernel (p*-invariant).

Defined only from the problem's factor declarations and the layout.

Block.  A block spec (L, depth) names, for every level-L cell, the sampled
channels of that cell and of all its descendants down to level L - depth
(depth = L: down to the fine tiles).  E.g. (1, 1) on ground is rho1, cert1 and
the four child tiles (5 x 6 x 16 = 480 joint values); (2, 0) is the level-2
cell's own channels (rho2, cert2) alone.

Conditional.  Every factor instance is listed once as member variable ids in a
flat state vector (the same instance rules as castlegen/bench/energy.py:
periodic x, pads, `within`, by_pos, pad_a rows).  For one block, the instances
touching its variables give its exact conditional energy over every joint
completion:  E(c) = soft(c) + lam * violations(c), or with lam = None the
violating completions are excluded (the hard target).  A completion is drawn
from exp(-E).  Blocks that share no instance are updated in parallel: blocks
are greedily coloured on the "shares an instance" graph.

Enumeration.  If the product of the block's domains is <= max_joint, the full
product is enumerated once and shared by every block (violations handled by
the energy).  Otherwise the completions are built variable by variable
(children before parents, post-order) and pruned by the hard instances whose
members are all known; the pruned set is cached by the block's hard-instance
structure and outside values.

Only periodic regions are supported.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse

from castlegen.bench.problem import Problem, Region


# ------------------------------------------------------------------ flat layout
class Flat:
    def __init__(self, problem: Problem, region: Region):
        assert region.periodic, "exact_block: periodic regions only"
        self.p, self.region = problem, region
        self.off, self.shape = {}, {}
        n = 0
        for c in problem.chans:
            h, w = problem.H >> c.level, region.W >> c.level
            self.off[c.name], self.shape[c.name] = n, (h, w)
            n += h * w
        self.n = n

    def gid(self, name, Y, X):
        h, w = self.shape[name]
        return self.off[name] + np.asarray(Y) * w + np.mod(X, w)

    def flatten(self, x):
        v = np.zeros(self.n, np.int64)
        for c in self.p.chans:
            o = self.off[c.name]
            v[o:o + int(np.prod(self.shape[c.name]))] = np.asarray(x[c.name]).ravel()
        return v

    def unflatten(self, v):
        out = {}
        for c in self.p.chans:
            o, (h, w) = self.off[c.name], self.shape[c.name]
            out[c.name] = v[o:o + h * w].reshape(h, w).copy()
        return out


# ------------------------------------------------------------------ instances
def build_instances(flat: Flat):
    """Per factor: dict(table, kind, by_pos pos (n,), members: list of
    (gid (n,) int, const (n,) int, view (D,))); gid -1 reads const (pad)."""
    p, W = flat.p, flat.region.W
    out = []
    for fi, f in enumerate(p.factors):
        ca = p.chan[f.a[0]]
        la, ha, wa = ca.level, p.H >> ca.level, W >> ca.level
        va = ca.views[f.a[1]]
        if f.kind == "unary":
            Y, X = np.meshgrid(np.arange(ha), np.arange(wa), indexing="ij")
            g = flat.gid(ca.name, Y.ravel(), X.ravel())
            out.append(dict(f=fi, kind="unary", table=f.table, pos=None,
                            members=[(g, np.zeros_like(g), va)]))
            continue
        cb = p.chan[f.b[0]]
        lb, hb, wb = cb.level, p.H >> cb.level, W >> cb.level
        vb = cb.views[f.b[1]]
        if f.kind == "count":
            Y, X = np.meshgrid(np.arange(hb), np.arange(wb), indexing="ij")
            Y, X = Y.ravel(), X.ravel()
            gb = flat.gid(cb.name, Y, X)
            mem = [(gb, np.zeros_like(gb), vb)]
            for k in f.subset:
                g = flat.gid(ca.name, 2 * Y + (k >> 1), 2 * X + (k & 1))
                mem.append((g, np.zeros_like(g), va))
            out.append(dict(f=fi, kind="count", table=f.table, pos=None, members=mem))
            continue
        dy, dx = f.off
        Y, X = np.meshgrid(np.arange(ha), np.arange(wa), indexing="ij")
        Y, X = Y.ravel(), X.ravel()
        if lb == la:
            Yb, Xb = Y + dy, X + dx
        else:
            Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
        off = (Yb < 0) | (Yb >= hb)
        ok = np.ones(Y.shape, bool) if f.pad_b >= 0 else ~off
        if f.within is not None:
            L = f.within
            ok &= ((Y << la) >> L == (Yb << lb) >> L) & ((X << la) >> L == (Xb << lb) >> L)
        ga = flat.gid(ca.name, Y, X)
        gb = np.where(off, -1, flat.gid(cb.name, np.clip(Yb, 0, hb - 1), Xb))
        cbv = np.where(off, max(f.pad_b, 0), 0)
        pos = 2 * (Y & 1) + (X & 1)
        out.append(dict(f=fi, kind="pair", table=f.table, by_pos=f.by_pos, pos=pos[ok],
                        members=[(ga[ok], np.zeros(ok.sum(), np.int64), va), (gb[ok], cbv[ok], vb)]))
        if f.pad_a >= 0 and lb == la and dy != 0:
            Yq, Xq = np.meshgrid(np.arange(hb), np.arange(wb), indexing="ij")
            Yq, Xq = Yq.ravel(), Xq.ravel()
            Ya, Xa = Yq - dy, Xq - dx
            offa = (Ya < 0) | (Ya >= ha)
            if f.within is not None:
                L = f.within
                offa &= ((Ya << la) >> L == (Yq << lb) >> L) & ((Xa << la) >> L == (Xq << lb) >> L)
            g = flat.gid(cb.name, Yq[offa], Xq[offa])
            n = int(offa.sum())
            out.append(dict(f=fi, kind="pair", table=f.table, by_pos=False, pos=np.zeros(n, np.int64),
                            members=[(np.full(n, -1), np.full(n, f.pad_a), np.arange(va.max() + 1)),
                                     (g, np.zeros(n, np.int64), vb)]))
    for d in out:
        d["hard"] = bool(np.isinf(d["table"]).any())
    return out


def _lookup(d, vals, pos):
    """Table entries for member values vals (list of arrays, broadcastable)."""
    t = d["table"]
    if d["kind"] == "unary":
        return t[vals[0]]
    if d["kind"] == "count":
        return t[vals[0], sum(vals[1:])]
    if d.get("by_pos"):
        return t[pos, vals[0], vals[1]]
    return t[vals[0], vals[1]]


def total_energy(flat, insts, v):
    """(soft, violations) of a flat state, for checks."""
    soft, nv = 0.0, 0
    for d in insts:
        vals = [view[np.where(g >= 0, v[np.maximum(g, 0)], c)] if True else None
                for g, c, view in d["members"]]
        # pads hold view values already: undo the view for pad entries
        vals = [np.where(g >= 0, view[v[np.maximum(g, 0)]], c) for g, c, view in d["members"]]
        pos = d["pos"] if d.get("by_pos") else None
        e = _lookup(d, vals, pos)
        inf = np.isinf(e)
        soft += float(e[~inf].sum())
        nv += int(inf.sum())
    return soft, nv


# ------------------------------------------------------------------ the kernel
class ExactBlockGibbs:
    def __init__(self, problem: Problem, region: Region, level=1, depth=None, lam=None,
                 max_joint=200_000, seed=0):
        self.p, self.region = problem, region
        self.L = level
        self.depth = level if depth is None else depth
        self.lam = lam
        self.flat = Flat(problem, region)
        self.insts = build_instances(self.flat)
        self.rng = np.random.default_rng(seed)
        fl, W, L = self.flat, region.W, level
        hL, wL = problem.H >> L, W >> L
        YL, XL = np.meshgrid(np.arange(hL), np.arange(wL), indexing="ij")
        YL, XL = YL.ravel(), XL.ravel()
        # block variables in post-order (descendants before ancestors), one column per slot
        cols, doms = [], []
        for l in range(L - self.depth, L + 1):
            s = 1 << (L - l)
            chans = [c for c in problem.level_chans(l) if c.designed is None]
            for dy in range(s):
                for dx in range(s):
                    for c in chans:
                        cols.append(fl.gid(c.name, YL * s + dy, XL * s + dx))
                        doms.append(c.D)
        self.BV = np.stack(cols, 1)                      # (nb, nv)
        self.D = np.array(doms)
        nb, nv = self.BV.shape
        self.nb, self.nv = nb, nv
        self.block_of = np.full(fl.n, -1, np.int64)
        self.slot_of = np.full(fl.n, -1, np.int64)
        self.block_of[self.BV.ravel()] = np.repeat(np.arange(nb), nv)
        self.slot_of[self.BV.ravel()] = np.tile(np.arange(nv), nb)
        # touching blocks per instance, conflict graph, colouring
        nbr = [set() for _ in range(nb)]
        self.touch = []
        for d in self.insts:
            G = np.stack([g for g, _, _ in d["members"]], 1)
            Bt = np.where(G >= 0, self.block_of[np.maximum(G, 0)], -1)
            self.touch.append(Bt)
            for row in Bt[(Bt >= 0).sum(1) >= 2]:
                u = set(row[row >= 0].tolist())
                for a in u:
                    nbr[a] |= u - {a}
        colour = np.full(nb, -1, np.int64)
        for b in range(nb):
            used = {colour[q] for q in nbr[b] if colour[q] >= 0}
            k = 0
            while k in used:
                k += 1
            colour[b] = k
        self.colour = colour
        self.ncol = int(colour.max()) + 1
        # per class, per instance group: rows touching the class
        self.cls = []
        for k in range(self.ncol):
            inclass = colour == k
            groups = []
            for d, Bt in zip(self.insts, self.touch):
                hit = np.where(Bt >= 0, inclass[np.maximum(Bt, 0)], False)
                rows = np.where(hit.any(1))[0]
                if rows.size == 0:
                    continue
                Bh = np.where(hit[rows], Bt[rows], -1)
                brow = Bh.max(1)                       # the one class block it touches
                assert ((Bh == brow[:, None]) | (Bh < 0)).all()
                G = np.stack([g[rows] for g, _, _ in d["members"]], 1)
                slots = np.where(hit[rows], self.slot_of[np.maximum(G, 0)], -1)
                groups.append(dict(d=d, rows=rows, block=brow, slots=slots, G=G,
                                   C=np.stack([c[rows] for _, c, _ in d["members"]], 1),
                                   pos=d["pos"][rows] if d.get("by_pos") else None))
            blocks = np.where(inclass)[0]
            rowmap = np.full(nb, -1, np.int64)
            rowmap[blocks] = np.arange(blocks.size)
            for gr in groups:                          # sparse (class blocks x rows) summing matrix
                r = rowmap[gr["block"]]
                gr["M"] = sparse.csr_matrix((np.ones(r.size), (r, np.arange(r.size))),
                                            shape=(blocks.size, r.size))
            self.cls.append((blocks, groups))
        self.joint = int(np.prod(self.D.astype(float)))
        self.shared = self.joint <= max_joint
        if self.shared:
            grids = np.meshgrid(*[np.arange(D) for D in self.D], indexing="ij")
            self.comps = np.stack([g.ravel() for g in grids], 1).astype(np.int64)
        self._cache = {}

    # ---------------------------------------------------------------- energies
    def _fixed_vals(self, v, gr):
        """Member values (n, k) read from the state v (view applied; pads as declared)."""
        G, C = gr["G"], gr["C"]
        out = np.empty(G.shape, np.int64)
        for m, (_, _, view) in enumerate(gr["d"]["members"]):
            out[:, m] = np.where(G[:, m] >= 0, view[v[np.maximum(G[:, m], 0)]], C[:, m])
        return out

    def energies(self, v, k, blocks_sel=None, comps=None):
        """Energies (soft, nviol) of every completion for the class-k blocks:
        each (nb_k, C).  comps: (C, nv) shared completions (default self.comps)."""
        blocks, groups = self.cls[k]
        comps = self.comps if comps is None else comps
        rowmap = np.full(self.nb, -1, np.int64)
        rowmap[blocks] = np.arange(blocks.size)
        if blocks_sel is not None:
            keep = np.zeros(self.nb, bool)
            keep[blocks_sel] = True
        nsel = blocks.size
        soft = np.zeros((nsel, comps.shape[0]))
        nv = np.zeros((nsel, comps.shape[0]), np.int64)
        for gr in groups:
            sel = slice(None) if blocks_sel is None else keep[gr["block"]]
            fixed = self._fixed_vals(v, gr)[sel]
            slots = gr["slots"][sel]
            if slots.shape[0] == 0:
                continue
            vals = []
            for m, (_, _, view) in enumerate(gr["d"]["members"]):
                cv = view[np.minimum(comps[:, np.maximum(slots[:, m], 0)].T, view.size - 1)]   # (n, C); clipped where fixed
                vals.append(np.where(slots[:, m:m + 1] >= 0, cv, fixed[:, m:m + 1]))
            pos = None if gr["pos"] is None else gr["pos"][sel][:, None]
            e = _lookup(gr["d"], vals, pos)
            inf = np.isinf(e)
            if blocks_sel is None:
                soft += gr["M"] @ np.where(inf, 0.0, e)
                nv += (gr["M"] @ inf.astype(np.float64)).astype(np.int64)
            else:
                r = rowmap[gr["block"][sel]]
                np.add.at(soft, r, np.where(inf, 0.0, e))
                np.add.at(nv, r, inf.astype(np.int64))
        return soft, nv

    def _logits(self, soft, nv):
        if self.lam is None:
            return np.where(nv > 0, -np.inf, -soft)
        return -(soft + self.lam * nv)

    # ---------------------------------------------------------------- pruned enumeration
    def _pruned(self, v, b, k):
        """Completions of block b (class k) consistent with every hard instance
        whose members are all known, built slot by slot (cached)."""
        _, groups = self.cls[k]
        hard = []
        for gr in groups:
            if not gr["d"]["hard"]:
                continue
            sel = gr["block"] == b
            if sel.any():
                hard.append((gr, sel))
        key = []
        for gr, sel in hard:
            fx = np.where(gr["slots"][sel] >= 0, -1, self._fixed_vals(v, gr)[sel])   # outside values only
            key += [gr["d"]["f"], gr["slots"][sel].tobytes(), fx.tobytes(),
                    b"" if gr["pos"] is None else gr["pos"][sel].tobytes()]
        h = hashlib.sha1(repr(key).encode()).hexdigest()
        if h in self._cache:
            return self._cache[h]
        last = []
        for gr, sel in hard:
            sl = gr["slots"][sel]
            last.append(np.where((sl >= 0).any(1), sl.max(1), -1))
        comps = np.zeros((1, 0), np.int64)
        for j in range(self.nv):
            n = comps.shape[0]
            comps = np.concatenate([np.repeat(comps, self.D[j], 0),
                                    np.tile(np.arange(self.D[j]), n)[:, None]], 1)
            ok = np.ones(comps.shape[0], bool)
            for (gr, sel), ls in zip(hard, last):
                rows = ls == j
                if not rows.any():
                    continue
                slots = gr["slots"][sel][rows]
                fixed = self._fixed_vals(v, gr)[sel][rows]
                vals = []
                for m, (_, _, view) in enumerate(gr["d"]["members"]):
                    cv = view[np.minimum(comps[:, np.clip(slots[:, m], 0, j)].T, view.size - 1)]
                    vals.append(np.where(slots[:, m:m + 1] >= 0, cv, fixed[:, m:m + 1]))
                pos = None if gr["pos"] is None else gr["pos"][sel][rows][:, None]
                ok &= ~np.isinf(_lookup(gr["d"], vals, pos)).any(0)
            comps = comps[ok]
        self._cache[h] = comps
        return comps

    # ---------------------------------------------------------------- sweeps
    def sweep(self, v, rng=None):
        """One systematic sweep over the colour classes; v (flat state) updated in place.
        Returns the number of blocks whose value changed."""
        rng = self.rng if rng is None else rng
        changed = 0
        for k in range(self.ncol):
            blocks, _ = self.cls[k]
            if self.shared:
                soft, nv = self.energies(v, k)
                lg = self._logits(soft, nv)
                pick = self._draw(lg, rng)
                okb = np.isfinite(lg).any(1)
                new = self.comps[pick]
                old = v[self.BV[blocks]]
                newv = np.where(okb[:, None], new, old)
                changed += int((newv != old).any(1).sum())
                v[self.BV[blocks]] = newv
            else:
                for i, b in enumerate(blocks):
                    comps = self._pruned(v, b, k)
                    if comps.shape[0] == 0:
                        continue
                    soft, nv = self.energies(v, k, blocks_sel=[b], comps=comps)
                    lg = self._logits(soft[i:i + 1], nv[i:i + 1])
                    if not np.isfinite(lg).any():
                        continue
                    new = comps[self._draw(lg, rng)[0]]
                    changed += int((new != v[self.BV[b]]).any())
                    v[self.BV[b]] = new
        return changed

    @staticmethod
    def _draw(lg, rng):
        g = rng.gumbel(size=lg.shape)
        return np.argmax(np.where(np.isfinite(lg), lg + g, -np.inf), 1)


def cycle(kernels, v, n_sweeps=1, rng=None):
    """Apply each kernel's sweep in order, n_sweeps times (a p*-invariant cycle)."""
    for _ in range(n_sweeps):
        for K in kernels:
            K.sweep(v, rng)
    return v
