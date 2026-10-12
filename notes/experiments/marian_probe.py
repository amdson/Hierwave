"""Where the wildcards sit and what the world is made of (scratch probe)."""
import collections
import sys

import numpy as np

from castlegen import marian
from castlegen.infinite import generate
sys.path.insert(0, "notes/experiments")
from marian_infinite import CONFIGS

ms = marian.load()
m = marian.tile_model(ms)
g = generate(m, CONFIGS[sys.argv[1] if len(sys.argv) > 1 else "B8Sb3K32"](), 0, 0, 0, 64, 64)
T = m.T
print("wild per layer", [(g[k] == T).mean().round(4) for k in range(m.L)])
print("air per layer", [(g[k] == ms.air()).mean().round(3) for k in range(m.L)])
c = collections.Counter()
for t in g[g < T].ravel():
    c[ms.names[t][:-3]] += 1
n = (g < T).sum()
print("top prototypes", [(k, round(v / n, 3)) for k, v in c.most_common(15)])
print("distinct prototypes", len(c))
# neighbours of wildcards: which prototypes surround them
nb = collections.Counter()
for k, i, j in zip(*np.nonzero(g == T)):
    for dk, di, dj in marian.STEP:
        kk, ii, jj = k + dk, i + di, j + dj
        if 0 <= kk < m.L and 0 <= ii < 64 and 0 <= jj < 64 and g[kk, ii, jj] < T:
            nb[ms.names[g[kk, ii, jj]][:-3]] += 1
print("around wildcards", nb.most_common(12))
