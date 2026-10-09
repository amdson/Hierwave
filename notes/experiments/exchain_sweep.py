"""Sweep the weight of the coarsest (h = 16) exemplar patch term.  Per factor f
(h = 4, 8 at lam = 8; h = 16 at f * lam): a long target run from the
exemplar's terrain, then the training loop with coarse coordinate
correction off (texsyn's default first_corrected = 3), reporting the copy
score (share of columns within 2 rows of the exemplar terrain at the best
shift; unrelated terrain ~0.15).
    .venv/bin/python notes/experiments/exchain_sweep.py F OUTDIR"""
import copy
import glob
import os
import sys
import time

import numpy as np

from castlegen import exchain as XC
from castlegen import exemplar as ex
from castlegen import pipeline as PL
from castlegen import refheights as RH
from castlegen import tileset
from castlegen.quantities import average as AV
from castlegen.quantities import envelope as EN

f, out = float(sys.argv[1]), sys.argv[2]
TILE = sys.argv[3] if len(sys.argv) > 3 else "heights"          # "heights" (old), "exheights" or "avgexheights"
LANG = "average" if TILE == "avgexheights" else "envelope"
LAM, G, K, N, M, ROUNDS, TARGET_SWEEPS = 8.0, 8, 16, 8, 200, 5, 0
w = {h: LAM * h * h / 25.0 for h in (4, 8)}
if f > 0:
    w[16] = f * LAM * 256 / 25.0
ts = tileset.load("cliffs")
lg = EN.lang() if LANG == "envelope" else AV.lang(K, G)
pyramid = (lambda m: EN.pyramid(ts, m, K, G, 64)[0]) if LANG == "envelope" else (lambda m: AV.pyramid(ts, m, K, G, 64, lg)[0])
E = np.load(sorted(glob.glob("cache/exemplar-cliffs-*.npy"))[0])
Hex = RH.heights(ts, E, G)
Hex2 = np.tile(Hex, 2)


def hstats(maps):
    Hs = [RH.heights(ts, m, G) for m in maps]
    return (np.mean([H.mean() for H in Hs]), np.mean([H.std() for H in Hs]),
            np.mean([np.abs(np.roll(H, -1) - H).mean() for H in Hs]),
            np.mean([max((np.abs(H - np.roll(Hex2, k)) <= 2).mean() for k in range(128)) for H in Hs]))


def line(tag, maps):
    m, s, sl, cp = hstats(maps)
    return f"{tag:34s} H mean {m:5.1f} std {s:5.1f} |slope| {sl:4.2f} copy {cp:.2f}"


tag = f"f{f:g}_{TILE}"
print(f"== h16 weight factor {f:g}: w = {w}", flush=True)
if TARGET_SWEEPS:
    ch = XC.ExChain(ts, RH.from_heights(ts, Hex2, 128, G, "stone", "stone"), G, E, T=0.9, w=w, seed=1)
    for _ in range(TARGET_SWEEPS):
        ch.sweep()
    print(line(f"target, {TARGET_SWEEPS} sweeps from exemplar", [ch.t]), flush=True)
    ex.to_png(ts, ch.t, os.path.join(out, f"sweep_{tag}_target.png"), 3)

rng = np.random.default_rng(0)
corpus = np.stack([RH.from_heights(ts, np.roll(Hex2, int(rng.integers(128))), 128, G, "stone", "stone")
                   for _ in range(N)])
spec = copy.deepcopy(PL.load_spec("cliffs_envelope"))
spec["stages"][0]["first_corrected"] = 3
if LANG == "average":
    spec["stages"][0]["quantities"] = ["average"]
if TILE in ("exheights", "avgexheights"):
    spec["stages"][1] = {"stage": TILE, "w": {str(h): x for h, x in w.items()}, "T": 0.9, "sweeps": 50}
else:
    spec["stages"][1].update(T=0.9, w_tex=1.0, sweeps=20)
path = os.path.join(out, f"sweep_corpus_{tag}.npy")
for r in range(ROUNDS + 1):
    np.save(path, corpus)
    spec["stages"][0][LANG] = {"corpus_path": path}
    PL._FIT_CACHE.clear()
    gen = [PL.run(spec, seed=100 * r + s, log=lambda *a: None)[1] for s in range(N)]
    fixed = []
    for s, g in enumerate(gen):
        c = XC.ExChain(ts, g, G, E, T=0.9, w=w, seed=1000 * r + s)
        for _ in range(M):
            c.sweep()
        fixed.append(c.t.copy())
    pa = [pyramid(m) for m in gen]
    pb = [pyramid(m) for m in fixed]
    tvs = []
    for h in (16, 32):
        a = np.bincount(np.concatenate([p[h].ravel() for p in pa]), minlength=lg.V).astype(float)
        b = np.bincount(np.concatenate([p[h].ravel() for p in pb]), minlength=lg.V).astype(float)
        tvs.append(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())
    print(line(f"round {r} corpus", corpus))
    print(line(f"round {r} generated", gen))
    print(line(f"round {r} corrected", fixed) + f"   promise TV h16 {tvs[0]:.3f} h32 {tvs[1]:.3f}", flush=True)
    ex.to_png(ts, gen[0], os.path.join(out, f"sweep_{tag}_r{r}_gen.png"), 3)
    ex.to_png(ts, fixed[0], os.path.join(out, f"sweep_{tag}_r{r}_fixed.png"), 3)
    corpus = np.stack(fixed)
