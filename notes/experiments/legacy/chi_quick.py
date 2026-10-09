"""Quick look at learned chi (one round, no correction step).
1. generate 16 maps with the distance chi; record (coordinate, realised value)
2. fit learned chi on 8 maps; held-out log-loss of the realised sub-column
   value given the coordinate's predicted value, per level, for
   (a) the base rate only, (b) distance chi as a distribution, (c) learned
3. generate 8 maps with learned chi; compare tile edits and diversity
    .venv/bin/python notes/experiments/legacy/chi_quick.py OUTDIR"""
import copy
import glob
import os
import sys

import numpy as np

from castlegen.legacy import average_var as AVV
from castlegen.legacy import pipeline as PL
from castlegen.legacy import refheights as RH
from castlegen.legacy import tileset
from castlegen.legacy.quantities import average as AV

out = sys.argv[1]                                        # also when imported: the caller's OUTDIR
G, K, N = 8, 16, 16
ts = tileset.load("cliffs")
lg = AV.lang(K, G)
E = np.load(sorted(glob.glob("cache/exemplar-cliffs-*.npy"))[0])
m = E.shape[0]
Hex2 = np.tile(RH.heights(ts, E, G), 2)
rng = np.random.default_rng(0)
cpath = os.path.join(out, "quick_corpus.npy")
np.save(cpath, np.stack([RH.from_heights(ts, np.roll(Hex2, int(rng.integers(128))), 128, G, "stone", "stone")
                         for _ in range(N)]))
spec = copy.deepcopy(PL.load_spec("cliffs_envelope"))
spec["stages"][0].update(first_corrected=3, quantities=["average"], lam=1.0)
spec["stages"][1] = {"stage": "avgexheights", "w": {"4": 5.12, "8": 20.48, "16": 81.92}, "T": 0.9, "sweeps": 50}


def generate(n, chi_path=None, seed0=0):
    q = {"corpus_path": cpath}
    if chi_path:
        q["chi_path"] = chi_path
    spec["stages"][0]["average"] = q
    maps, edits, recs = [], [], []
    for s in range(n):
        ctx = []
        _, t, rep = PL.run(spec, seed=seed0 + s, log=lambda *a: None, ctx_out=ctx)
        maps.append(t)
        edits.append(rep[-1]["changed"])
        vals = AV.pyramid(ts, t, K, G, 64, lg)[0]
        recs.append([(lv.h, lv.vars["coord"], vals[lv.h]) for lv in ctx[0].cache["levels"]
                     if lv.h >= K and "coord" in lv.vars])
    return maps, edits, recs


def diversity(maps):
    Hs = [RH.heights(ts, t, G).astype(float) for t in maps]
    ref = Hex2.astype(float)

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return max(float((a * np.roll(b, k)).sum()) for k in range(len(a))) / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)
    ex_c = np.mean([corr(a, ref) for a in Hs])
    pd = np.mean([min(float(np.abs(Hs[i] - np.roll(Hs[j], k)).mean()) for k in range(128))
                  for i in range(len(Hs)) for j in range(i + 1, len(Hs))])
    std = np.mean([a.std() for a in Hs])
    return ex_c, pd, std


if __name__ == "__main__":
    maps, edits, recs = generate(N)
    train = [r for rs in recs[:N // 2] for r in rs]
    test = [r for rs in recs[N // 2:] for r in rs]
    chi, counts = AVV.fit_chi(ts, E, train, lg)
    P = lg.P
    print("held-out log-loss per sub-column value (nats; lower is better)")
    for h in sorted(counts):
        c = counts[h] + 1.0
        cond = c / c.sum(1, keepdims=True)
        base = c.sum(0) / c.sum()
        dist = np.exp(-np.abs(np.arange(P)[:, None] - np.arange(P)[None, :]))
        dist /= dist.sum(1, keepdims=True)
        pred = np.concatenate([AVV.window_fill(ts, E, h, lg)[S[..., 0] * m + S[..., 1]].ravel()
                               for hh, S, v in test if hh == h])
        real = np.concatenate([lg.SUB[v].ravel() for hh, S, v in test if hh == h])
        ll = lambda p: float(-np.log(p).mean())
        print(f"  h{h:2d} ({len(real)} values): base rate {ll(base[real]):.3f}   distance chi {ll(dist[pred, real]):.3f}"
              f"   learned {ll(cond[pred, real]):.3f}   (predicted == realised: {(pred == real).mean():.2f})")
    chi_path = os.path.join(out, "quick_chi.npz")
    chi_all, _ = AVV.fit_chi(ts, E, [r for rs in recs for r in rs], lg)
    np.savez(chi_path, **{f"h{h}": c for h, c in chi_all.items()})
    maps2, edits2, _ = generate(N // 2, chi_path, seed0=500)
    for name, mp, ed in (("distance chi", maps[:N // 2], edits[:N // 2]), ("learned chi", maps2, edits2)):
        ex_c, pd, std = diversity(mp)
        print(f"{name:13s} tile edits {np.mean(ed):.3f}  height spread {std:.1f}  exemplar corr {ex_c:.2f}  pairwise dist {pd:.1f}"
              f"  unsupported {max(RH.violations(ts, t, G) for t in mp)}")
