"""Corpus-free training loop (option 1 with the exemplar-defined target):
    corpus_0 = the exemplar's terrain (shifted copies)
    round r:  fit tables on corpus_r -> generate N maps with the hierarchy
              -> correct each with M sweeps of ExChain (exact MCMC for the
              exemplar-defined target) -> corpus_{r+1} = corrected maps.
Drift between generated and corrected statistics is the error signal.
    .venv/bin/python notes/experiments/legacy/exchain_loop.py LAM OUTDIR [ROUNDS N M]"""
import copy
import glob
import os
import sys
import time

import numpy as np

from castlegen.legacy import exchain as XC
from castlegen.legacy import exemplar as ex
from castlegen.legacy import pipeline as PL
from castlegen.legacy import refheights as RH
from castlegen.legacy import tileset
from castlegen.legacy.quantities import envelope as EN

lam, out = float(sys.argv[1]), sys.argv[2]
rounds, N, M = (int(a) for a in (sys.argv[3:6] if len(sys.argv) > 5 else (3, 8, 200)))
G, K = 8, 16
ts = tileset.load("cliffs")
lg = EN.lang()
E = np.load(sorted(glob.glob("cache/exemplar-cliffs-*.npy"))[0])
Hex = RH.heights(ts, E, G)
rng = np.random.default_rng(0)
corpus = np.stack([RH.from_heights(ts, np.roll(np.tile(Hex, 2), int(rng.integers(128))), 128, G, "stone", "stone")
                   for _ in range(N)])


def summary(maps):
    Hs = [RH.heights(ts, m, G) for m in maps]
    st = dict(H_mean=np.mean([H.mean() for H in Hs]), H_std=np.mean([H.std() for H in Hs]),
              slope=np.mean([np.abs(np.roll(H, -1) - H).mean() for H in Hs]))
    pyr = [EN.pyramid(ts, m, K, G, 64)[0] for m in maps]
    st["hist"] = {h: np.bincount(np.concatenate([p[h].ravel() for p in pyr]), minlength=lg.V) for h in (16, 32)}
    return st


def tv(a, b):
    return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()


spec = copy.deepcopy(PL.load_spec("cliffs_envelope"))
spec["stages"][1].update(T=0.9, w_tex=1.0, sweeps=20)
path = os.path.join(out, f"loop_corpus_lam{lam:g}.npy")
for r in range(rounds + 1):
    np.save(path, corpus)
    spec["stages"][0]["envelope"] = {"corpus_path": path}
    PL._FIT_CACHE.clear()
    t0 = time.time()
    gen = [PL.run(spec, seed=100 * r + s, log=lambda *a: None)[1] for s in range(N)]
    t1 = time.time()
    fixed = []
    for s, g in enumerate(gen):
        ch = XC.ExChain(ts, g, G, E, T=0.9, lam=lam, seed=1000 * r + s)
        e0 = ch.energy_ex()
        for _ in range(M):
            ch.sweep()
        fixed.append(ch.t.copy())
    t2 = time.time()
    a, b, c = summary(gen), summary(fixed), summary(corpus)
    print(f"round {r}: generated vs corrected ({t1 - t0:.0f}s gen, {t2 - t1:.0f}s correct)")
    for k in ("H_mean", "H_std", "slope"):
        print(f"   {k:7s} corpus {c[k]:6.2f}  generated {a[k]:6.2f}  corrected {b[k]:6.2f}  drift {b[k] - a[k]:+6.2f}")
    print("   promise TV generated vs corrected: " + "  ".join(f"h{h} {tv(a['hist'][h], b['hist'][h]):.3f}" for h in (16, 32)),
          flush=True)
    ex.to_png(ts, gen[0], os.path.join(out, f"loop_lam{lam:g}_r{r}_gen.png"), 3)
    ex.to_png(ts, fixed[0], os.path.join(out, f"loop_lam{lam:g}_r{r}_fixed.png"), 3)
    corpus = np.stack(fixed)
