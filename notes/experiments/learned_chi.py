"""Ablation: distance chi vs learned chi in the corpus-free loop (average
promises, exemplar-defined target with patch terms at h = 4, 8, 16).
    .venv/bin/python notes/experiments/learned_chi.py CHI SEED OUTDIR [ROUNDS N M]
CHI: "distance" (window-value distance, lam 1) or "learned" (per-level PMI
between a block's predicted and realised value, refitted every round).

Per round, on N generated maps and their corrected versions:
  invariants   max unsupported cells, min satisfaction
  edits        fraction of cells the tile stage changes relative to the texture
  TV h16/h32   promise histograms, generated (first half) vs corrected (second
               half), and the noise floor corrected (first half) vs corrected
               (second half) - independent samples of equal size
  drift        corrected - generated: height mean, spread, |slope|
  dE           target energy per cell, generated - corrected (tile + exemplar)
  diversity    exemplar correlation (best-shift Pearson correlation of the
               height profile with the exemplar's) and pairwise distance
               (best-shift mean |H_a - H_b| between corrected maps)
"""
import copy
import glob
import json
import os
import sys
import time

import numpy as np

from castlegen import average_var as AVV
from castlegen import exchain as XC
from castlegen import exemplar as ex
from castlegen import pipeline as PL
from castlegen import refheights as RH
from castlegen import tileset
from castlegen.quantities import average as AV

CHI, SEED, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ROUNDS, N, M = (int(a) for a in (sys.argv[4:7] if len(sys.argv) > 6 else (5, 16, 200)))
G, K = 8, 16
W = {4: 5.12, 8: 20.48, 16: 81.92}
ts = tileset.load("cliffs")
lg = AV.lang(K, G)
E = np.load(sorted(glob.glob("cache/exemplar-cliffs-*.npy"))[0])
Hex = RH.heights(ts, E, G)
Hex2 = np.tile(Hex, 2).astype(float)
tag = f"chi-{CHI}_s{SEED}"
rng = np.random.default_rng(SEED)
corpus = np.stack([RH.from_heights(ts, np.roll(Hex2.astype(int), int(rng.integers(128))), 128, G, "stone", "stone")
                   for _ in range(N)])

spec = copy.deepcopy(PL.load_spec("cliffs_envelope"))
spec["stages"][0].update(first_corrected=3, quantities=["average"], lam=1.0)
spec["stages"][1] = {"stage": "avgexheights", "w": {str(h): x for h, x in W.items()}, "T": 0.9, "sweeps": 50}
cpath, xpath = os.path.join(out, f"{tag}_corpus.npy"), os.path.join(out, f"{tag}_chi.npz")


def H(t):
    return RH.heights(ts, t, G).astype(float)


def best_corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return max(float((a * np.roll(b, k)).sum() / den) for k in range(len(a)))


def best_dist(a, b):
    return min(float(np.abs(a - np.roll(b, k)).mean()) for k in range(len(a)))


def hist(maps, h):
    return np.bincount(np.concatenate([AV.pyramid(ts, m, K, G, 64, lg)[0][h].ravel() for m in maps]),
                       minlength=lg.V).astype(float)


def tv(a, b):
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def energy(t):
    ch = XC.ExChain(ts, t, G, E, T=0.9, w=W, seed=0)
    for h in ch.sc:
        ch.resample_S(h, T=1e-6)                         # best exemplar match per coarse cell
    pair, un = RH.energy_parts(ts, t)
    return (pair + un + ch.energy_ex()) / t.size


log = []
for r in range(ROUNDS + 1):
    t0 = time.time()
    np.save(cpath, corpus)
    q = {"corpus_path": cpath}
    if CHI == "learned" and r > 0:
        q["chi_path"] = xpath
    spec["stages"][0]["average"] = q
    PL._FIT_CACHE.clear()
    gen, reps, coords = [], [], []
    for s in range(N):
        ctx = []
        _, t, rep = PL.run(spec, seed=1000 * SEED + 100 * r + s, log=lambda *a: None, ctx_out=ctx)
        gen.append(t)
        reps.append(rep)
        coords.append({lv.h: lv.vars["coord"] for lv in ctx[0].cache["levels"] if lv.h >= K and "coord" in lv.vars})
    t1 = time.time()
    fixed = []
    for s, g in enumerate(gen):
        ch = XC.ExChain(ts, g, G, E, T=0.9, w=W, seed=7919 * SEED + 101 * r + s)    # exact MCMC on the target
        for _ in range(M):
            ch.sweep()
        fixed.append(ch.t.copy())
    t2 = time.time()
    half = N // 2
    row = dict(round=r, gen_s=round(t1 - t0), corr_s=round(t2 - t1),
               unsup=int(max(max(rp[-1]["unsupported"] for rp in reps), max(RH.violations(ts, f, G) for f in fixed))),
               sat=float(min(min(rp[-1]["sat"].values()) for rp in reps)),
               edits=float(np.mean([rp[-1]["changed"] for rp in reps])))
    for h in (16, 32):
        row[f"tv{h}"] = tv(hist(gen[:half], h), hist(fixed[half:], h))
        row[f"floor{h}"] = tv(hist(fixed[:half], h), hist(fixed[half:], h))
    Hg, Hf = [H(t) for t in gen], [H(t) for t in fixed]
    for name, fn in (("mean", np.mean), ("std", np.std), ("slope", lambda a: np.abs(np.roll(a, -1) - a).mean())):
        row[f"gen_{name}"] = float(np.mean([fn(a) for a in Hg]))
        row[f"cor_{name}"] = float(np.mean([fn(a) for a in Hf]))
    row["dE"] = float(np.mean([energy(a) - energy(b) for a, b in zip(gen[:4], fixed[:4])]))
    row["ex_corr"] = float(np.mean([best_corr(a, Hex2) for a in Hf]))
    row["pair_dist"] = float(np.mean([best_dist(Hf[i], Hf[j]) for i in range(half) for j in range(i + 1, half)]))
    log.append(row)
    print(json.dumps(row), flush=True)
    ex.to_png(ts, gen[0], os.path.join(out, f"{tag}_r{r}_gen.png"), 3)
    ex.to_png(ts, fixed[0], os.path.join(out, f"{tag}_r{r}_fixed.png"), 3)
    if CHI == "learned":                                   # refit chi on this round: coordinate -> realised value
        recs = [(h, S, AV.pyramid(ts, f, K, G, 64, lg)[0][h]) for cd, f in zip(coords, fixed) for h, S in cd.items()]
        chi, counts = AVV.fit_chi(ts, E, recs, lg)
        np.savez(xpath, **{f"h{h}": c for h, c in chi.items()})
        sharp = {h: float((c.max(1) / c.sum(1).clip(1)).mean()) for h, c in counts.items()}
        print(json.dumps({"round": r, "chi_levels": sorted(chi), "chi_sharpness": sharp}), flush=True)
    corpus = np.stack(fixed)
json.dump(log, open(os.path.join(out, f"{tag}.json"), "w"), indent=1)
