"""Exact light cone (dependency reach, fine columns) of the cstage schedule, by boolean propagation over the
factor-instance graph through the stage's exact update order (colour_gibbs level sweeps T..0: 16 colours x
channels; JaxBlockGibbs L1 classes; FineBlockKernel 2x2 classes at offset r mod 4; certificate-only refill
sweeps T..1).  Seed: every cell covering a 64-column band at the strip centre; reach = growth on each side.
Per-step reaches add (upper bound for the composition)."""
import sys, json
import numpy as np, scipy.sparse as sp
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.kernels import colour_gibbs as cg
from castlegen.kernels.amortised import JaxBlockGibbs
from castlegen.kernels.fine_block import FineBlockKernel
from castlegen.variants._cpass_core import Layout, NCOL
from castlegen.variants.cstage import graded_problem

W = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SINGLE = NR <= 4
p = get("ground"); T = p.T
lay = Layout(graded_problem(p, 20.0), W, True)
pl = cg.get_plan(lay); flat = pl.flat; n = flat.n
print("ngroups", pl.ngroups, "n", n, flush=True)
# fine-column extent of every flat cell, its level and channel
lo = np.zeros(n, np.int64); hi = np.zeros(n, np.int64); chan = np.empty(n, object)
for c in p.chans:
    o = flat.off[c.name]; h, w = flat.shape[c.name]
    X = np.tile(np.arange(w), h)
    lo[o:o + h * w] = X << c.level; hi[o:o + h * w] = ((X + 1) << c.level) - 1; chan[o:o + h * w] = c.name
# adjacency: cells sharing a factor instance
rows, cols = [], []
for d in pl.insts:
    G = np.stack([g for g, _, _ in d["members"]], 1)
    for i in range(G.shape[1]):
        for j in range(G.shape[1]):
            ok = (G[:, i] >= 0) & (G[:, j] >= 0)
            rows.append(G[ok, i]); cols.append(G[ok, j])
A = sp.csr_matrix((np.ones(sum(r.size for r in rows), np.int8), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
A = ((A + sp.identity(n, dtype=np.int8, format="csr")) > 0).astype(np.int8).tocsr()
colour = pl.colour_of; group = pl.group


def nb(S):
    return (A @ S.astype(np.int8)) > 0


def cell_sweep(S, levels, only=None):
    for l in levels:
        for k in range(NCOL):
            for c in lay.chans[l]:
                if only and not only(c):
                    continue
                o = flat.off[c.name]; h, w = flat.shape[c.name]
                for gi in range(pl.ngroups):
                    idx = o + np.arange(h * w)
                    idx = idx[(colour[idx] == k) & (group[idx] == gi)]
                    S[idx] |= nb(S)[idx]
    return S


def block_sweep(S, J):
    for BV, nbk, *_ in J.classes:
        BV = np.asarray(BV)
        N = nb(S)
        hit = N[BV].any(1)
        S[BV[hit].ravel()] = True
    return S


def extent(S):
    return int(hi[S].max()), int(lo[S].min())


def seed():
    c0 = W // 2 - 32
    return (hi >= c0 + 1) & (lo <= c0 + 62)   # misaligned by one column at both ends (worst block phase)


reg = Region(W)
J1 = JaxBlockGibbs(p, reg, 1, 1, lam=20.0)
F2 = FineBlockKernel(p, reg, k=2, lam=20.0)
s0 = seed(); h0, l0 = extent(s0)
out = {}
for name, fn in ([] if not SINGLE else [("cell_all", lambda S: cell_sweep(S, range(T, -1, -1))),
                 ("J1", lambda S: block_sweep(S, J1))] + \
                [(f"F2_{i}", (lambda i: lambda S: block_sweep(S, F2.Js[i]))(i)) for i in range(len(F2.Js))] + \
                [("refill", lambda S: cell_sweep(S, range(T, 0, -1), only=lambda c: c.name.startswith("cert")))]):
    S = fn(s0.copy()); h1, l1 = extent(S)
    out[name] = (h1 - h0, l0 - l1)
    print(name, "right", h1 - h0, "left", l0 - l1, flush=True)
# four full rounds (one offset cycle) composed exactly, then 3 refill sweeps composed
S = s0.copy()
for r in range(NR):
    S = cell_sweep(S, range(T, -1, -1)); S = block_sweep(S, J1); S = block_sweep(S, F2.Js[r % 4])
    h1, l1 = extent(S); out[f"rounds_{r+1}"] = (h1 - h0, l0 - l1); print("rounds", r + 1, out[f"rounds_{r+1}"], flush=True)
for j in range(3):
    S = cell_sweep(S, range(T, 0, -1), only=lambda c: c.name.startswith("cert"))
    h1, l1 = extent(S); out[f"rounds4_refill{j+1}"] = (h1 - h0, l0 - l1); print("refill", j + 1, out[f"rounds4_refill{j+1}"], flush=True)
json.dump(out, open(f"reportables/2026-10-16_stage2/runs/reach_R{NR}.json", "w"), indent=1)
