"""dimers-kernel: generic exact Gibbs on a k x k FINE footprint (p*-invariant).

exact_block.ExactBlockGibbs defines a block by a level and a depth (a level-L
cell plus its descendants).  On a problem with T = 0 there is no level 1, so
the largest exact block is one cell, and on a hard target such as perfect
matchings a single-cell move is frozen.  FineBlockGibbs takes the block to be a
k x k square of level-0 cells (all their sampled channels), tiled at stride k
from an offset (oy, ox); x wraps (periodic regions), tiles that would leave
the rows [0, H) are dropped.  A sweep draws a random offset from `offsets`
(a mixture of p*-invariant kernels is p*-invariant) and updates every tile in
colour classes exactly as exact_block does (greedy colouring on the shared-
instance graph, so tiles of one class share no factor instance and are
conditionally independent given the rest).

Completions of a tile given its boundary: the product of the tile's domains,
pruned by the hard instances whose members all lie inside the tile
(amortised.internal_pruned: a superset of every tile's feasible set, whatever
the outside values), then every instance touching the tile prices the
completion exactly (soft + lam * violations, or lam = None: violating
completions excluded).  The conditional over that set is therefore the exact
Gibbs conditional of the tile.  Generic: factors and layout only.

Classes
  FineBlockGibbs(ExactBlockGibbs)   numpy, per state (structure; reference)
  JaxFineBlockGibbs                 GPU, batched over worlds (amortised.JaxBlockGibbs
                                    on the same structure); .sweep(v, key)
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from castlegen.bench.problem import Problem, Region

from .amortised import JaxBlockGibbs, internal_pruned
from .exact_block import ExactBlockGibbs, Flat, build_instances


def default_offsets(k):
    s = max(k // 2, 1)
    return [(0, 0), (s, 0), (0, s), (s, s)] if k > 1 else [(0, 0)]


class FineBlockGibbs(ExactBlockGibbs):
    """One offset.  Attributes as ExactBlockGibbs (BV, D, cls, colour, ...)."""

    def __init__(self, problem: Problem, region: Region, k=2, offset=(0, 0), lam=None, seed=0,
                 internal=True, max_out=2_000_000):
        self.p, self.region = problem, region
        self.L, self.depth, self.k, self.offset = 0, 0, k, tuple(offset)
        self.lam = lam
        self.flat = Flat(problem, region)
        self.insts = build_instances(self.flat)
        self.rng = np.random.default_rng(seed)
        fl, W, H = self.flat, region.W, problem.H
        oy, ox = offset
        ny, nx = (H - oy) // k, W // k
        assert ny >= 1 and nx >= 1
        TY, TX = np.meshgrid(np.arange(ny) * k + oy, (np.arange(nx) * k + ox) % W, indexing="ij")
        TY, TX = TY.ravel(), TX.ravel()
        chans = [c for c in problem.level_chans(0) if c.designed is None]
        cols, doms = [], []
        for dy in range(k):
            for dx in range(k):
                for c in chans:
                    cols.append(fl.gid(c.name, TY + dy, TX + dx))
                    doms.append(c.D)
        self.BV = np.stack(cols, 1)
        self.D = np.array(doms)
        self._structure()
        # completion set shared by all tiles (internal-pruned superset)
        self.joint = int(np.prod(self.D.astype(float)))
        if internal == "boundary":              # per-tile enumeration pruned by the boundary (numpy only; any k)
            self.shared = False
        elif internal:
            self.comps = internal_pruned(self, max_out=max_out)
            self.shared = True
        else:
            grids = np.meshgrid(*[np.arange(D) for D in self.D], indexing="ij")
            self.comps = np.stack([g.ravel() for g in grids], 1).astype(np.int64)
            self.shared = True
        self._cache = {}

    def _structure(self):
        """blocks -> conflict graph -> colour classes -> per-class instance groups
        (verbatim from ExactBlockGibbs.__init__ after BV)."""
        fl = self.flat
        nb, nv = self.BV.shape
        self.nb, self.nv = nb, nv
        self.block_of = np.full(fl.n, -1, np.int64)
        self.slot_of = np.full(fl.n, -1, np.int64)
        self.block_of[self.BV.ravel()] = np.repeat(np.arange(nb), nv)
        self.slot_of[self.BV.ravel()] = np.tile(np.arange(nv), nb)
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
            c = 0
            while c in used:
                c += 1
            colour[b] = c
        self.colour = colour
        self.ncol = int(colour.max()) + 1
        self.cls = []
        for c in range(self.ncol):
            inclass = colour == c
            groups = []
            for d, Bt in zip(self.insts, self.touch):
                hit = np.where(Bt >= 0, inclass[np.maximum(Bt, 0)], False)
                rows = np.where(hit.any(1))[0]
                if rows.size == 0:
                    continue
                Bh = np.where(hit[rows], Bt[rows], -1)
                brow = Bh.max(1)
                assert ((Bh == brow[:, None]) | (Bh < 0)).all()
                G = np.stack([g[rows] for g, _, _ in d["members"]], 1)
                slots = np.where(hit[rows], self.slot_of[np.maximum(G, 0)], -1)
                groups.append(dict(d=d, rows=rows, block=brow, slots=slots, G=G,
                                   C=np.stack([cc[rows] for _, cc, _ in d["members"]], 1),
                                   pos=d["pos"][rows] if d.get("by_pos") else None))
            blocks = np.where(inclass)[0]
            rowmap = np.full(nb, -1, np.int64)
            rowmap[blocks] = np.arange(blocks.size)
            for gr in groups:
                r = rowmap[gr["block"]]
                gr["M"] = sparse.csr_matrix((np.ones(r.size), (r, np.arange(r.size))), shape=(blocks.size, r.size))
            self.cls.append((blocks, groups))


class _Shell:
    """Minimal ExactBlockGibbs stand-in carrying what JaxBlockGibbs reads."""


class JaxFineBlockGibbs(JaxBlockGibbs):
    """JaxBlockGibbs on a FineBlockGibbs structure (same sweep code)."""

    def __init__(self, K: FineBlockGibbs, fast=True):
        import jax
        import jax.numpy as jnp
        self.K, self.lam, self.level, self.depth = K, K.lam, 0, 0
        self.comps = np.asarray(K.comps, np.int32)
        C = self.comps.shape[0]
        self.chunk = C
        self.valid_c = np.ones((1, C), bool)
        self.comps_c = self.comps[None]
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
        self.fast = bool(fast) and self._build_fast()
        self.sweep_ref = jax.jit(self._sweep)
        self.sweep = jax.jit(self._sweep_fast) if self.fast else self.sweep_ref


class FineBlockKernel:
    """A mixture over offsets: sweep(v, key) picks an offset uniformly at random
    (host-side rng) and runs that offset's sweep.  p*-invariant."""

    def __init__(self, problem, region, k=2, lam=None, offsets=None, seed=0, jax_backend=True):
        self.k = k
        self.offsets = default_offsets(k) if offsets is None else list(offsets)
        internal = "boundary" if (not jax_backend and k >= 4) else True
        self.Ks = [FineBlockGibbs(problem, region, k, o, lam=lam, seed=seed, internal=internal) for o in self.offsets]
        self.Js = [JaxFineBlockGibbs(K) for K in self.Ks] if jax_backend else None
        self.rng = np.random.default_rng(seed + 17)
        self.flat = self.Ks[0].flat
        self.ncomps = self.Ks[0].comps.shape[0] if hasattr(self.Ks[0], "comps") else None

    def sweep(self, v, key):
        i = int(self.rng.integers(len(self.Js)))
        return self.Js[i].sweep(v, key)

    def sweep_np(self, v, rng=None):
        i = int(self.rng.integers(len(self.Ks)))
        return self.Ks[i].sweep(v, rng)
