"""Held-out column NLL of count vs linear promise tables for the average
language, on crops of the cliffs_big exemplar (args: CORPUS.npy)."""
import sys
import numpy as np
from castlegen import tileset, average_var as AVV, envpredict as EP
from castlegen.promise import Tables
from castlegen.quantities import average as AV

ts = tileset.load("cliffs")
G, K = 8, 16
maps = np.load(sys.argv[1])
n = maps.shape[1]
lg = AV.lang(K, G)
rng = np.random.default_rng(0)
idx = rng.permutation(len(maps))
tr, te = maps[idx[:12]], maps[idx[12:]]
gte = AVV.mine(ts, te, K, G, n, lg)
tabs = {"counts": AVV.tables(ts, tr, K, G, n, 0.5, lg)}
for l2 in (1e-3, 1e-2, 1e-1):
    tabs[f"linear l2={l2:g}"] = AVV.linear_tables(ts, tr, K, G, n, l2, lg, log=lambda *a: None)
print(f"maps {maps.shape}, train 12, test {len(te)}; held-out NLL per column (nats)")
for name, t in tabs.items():
    print(f"  {name:16s}" + "".join(f"  h{h}: {EP.column_nll(t, gte, h, lg, G):7.3f}" for h in sorted(gte)))
