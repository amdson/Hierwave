"""Diagnostic for stage D of notes/induce_test.md: object presence on the
edge vs interior mid cells, oracle vs the forward model with the induced
tables (the induced mid_u is the interior one; an object whose ring spills
off the grid is cheaper in p*).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/induce_circles_edge.py
"""
import json
import numpy as np
from castlegen.channels.circles import Circles, Forward, Oracle

C = Circles(6, 6)
R = json.load(open("images/induce_circles.json"))
edge = np.zeros((C.nmy, C.nmx), bool)
edge[0] = edge[-1] = edge[:, 0] = edge[:, -1] = True


def acc(grids):
    p = np.mean([(g > 0) for g in grids], axis=0)
    return p[edge].mean(), p[~edge].mean(), p.mean()


orc = Oracle(C, seed=7)
orc.sweep(50)
og = []
for _ in range(400):
    orc.sweep(1)
    og.append(orc.mid.grid.copy())
th = {k: np.asarray(v) for k, v in R["D"]["exact-mid + top(exact-mid)"]["theta"].items()}
fw = Forward(C, th, seed=555)
fg = []
for _ in range(64):
    fw.run(30, 30, 20, fresh=True)
    fg.append(fw.mid.grid.copy())
for name, g in (("oracle", og), ("forward exact-mid + top", fg)):
    e, i, a = acc(g)
    print(f"{name}: present edge {e:.3f} interior {i:.3f} all {a:.3f}")
