"""Learned chi from a non-circular source: plain texture synthesis (no
promises), projected to a valid heightmap, corrected with M sweeps of the
exact target chain; record (coordinate at level h, realised promise value).
Then: held-out calibration, and 8 promise-pipeline maps with distance vs
learned chi.
    .venv/bin/python notes/experiments/chi_texture.py OUTDIR [N M]"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import chi_quick as CQ                                   # noqa: E402  (generation + diversity helpers)
from castlegen import average_var as AVV
from castlegen import exchain as XC
from castlegen import heights as HT
from castlegen import hier, texsyn
from castlegen import refheights as RH
from castlegen.quantities import average as AV

out = sys.argv[1]
N, M = (int(a) for a in (sys.argv[2:4] if len(sys.argv) > 3 else (16, 100)))
ts, lg, E, G, K, m = CQ.ts, CQ.lg, CQ.E, CQ.G, CQ.K, CQ.m
W = {4: 5.12, 8: 20.48, 16: 81.92}
an = texsyn.Analysis(ts, E, n_pca=32, w_sock=0.5)
r = [1, 1] + [0] * (an.L - 1)

recs = []
for s in range(N):
    coord = texsyn.CoordVar(an, r, kappa=4.0, corrections=2, first_corrected=3)
    ctx = hier.Ctx(2000 + s)
    S1 = hier.run([coord], [], 128, ctx).vars[coord.name]
    X = E[S1[..., 0], S1[..., 1]]
    Hs = np.argmin(HT.texture_cost(ts, X, G), 1)                       # best valid heightmap for the texture
    x = RH.from_heights(ts, Hs, 128, G, "stone", "stone")
    keep = (ts.solid[X] == ts.solid[x]) & (np.arange(128) < 128 - G)[:, None]
    x = np.where(keep, X, x).astype(np.int32)
    ch = XC.ExChain(ts, x, G, E, T=0.9, w=W, seed=3000 + s)
    for _ in range(M):
        ch.sweep()
    vals = AV.pyramid(ts, ch.t, K, G, 64, lg)[0]
    recs.append([(lv.h, lv.vars[coord.name], vals[lv.h]) for lv in ctx.levels if lv.h >= K and lv.h in vals])

train = [x for rs in recs[:N // 2] for x in rs]
test = [x for rs in recs[N // 2:] for x in rs]
_, counts = AVV.fit_chi(ts, E, train, lg)
P = lg.P
print(f"texture-only source, {M} correction sweeps: held-out log-loss per sub-column value (nats)")
for h in sorted(counts):
    c = counts[h] + 1.0
    cond, base = c / c.sum(1, keepdims=True), c.sum(0) / c.sum()
    dist = np.exp(-np.abs(np.arange(P)[:, None] - np.arange(P)[None, :]))
    dist /= dist.sum(1, keepdims=True)
    pred = np.concatenate([AVV.window_fill(ts, E, h, lg)[S[..., 0] * m + S[..., 1]].ravel() for hh, S, v in test if hh == h])
    real = np.concatenate([lg.SUB[v].ravel() for hh, S, v in test if hh == h])
    ll = lambda p: float(-np.log(p).mean())
    print(f"  h{h:2d} ({len(real)} values): base rate {ll(base[real]):.3f}   distance chi {ll(dist[pred, real]):.3f}"
          f"   learned {ll(cond[pred, real]):.3f}   (predicted == realised {(pred == real).mean():.2f},"
          f" within 1 {(np.abs(pred - real) <= 1).mean():.2f})")

chi_all, _ = AVV.fit_chi(ts, E, [x for rs in recs for x in rs], lg)
chi_path = os.path.join(out, "texture_chi.npz")
np.savez(chi_path, **{f"h{h}": c for h, c in chi_all.items()})
for name, path in (("distance chi", None), ("learned chi", chi_path)):
    maps, edits, _ = CQ.generate(8, path, seed0=700)
    ex_c, pd, std = CQ.diversity(maps)
    print(f"{name:13s} tile edits {np.mean(edits):.3f}  height spread {std:.1f}  exemplar corr {ex_c:.2f}"
          f"  pairwise dist {pd:.1f}  unsupported {max(RH.violations(ts, t, G) for t in maps)}")
