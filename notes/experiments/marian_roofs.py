"""Roof-group upweight and the flat-top-under-air pair energy on marian
(K64, air weights [2, 3, 3, 3.5, 5, 8]): exposed flat tops per column, share
of columns whose top module is a roof, height, wildcards, prototypes.
Run with NUMBA_NUM_THREADS=1."""
import sys
import time

import numpy as np

from castlegen import marian
from castlegen.infinite import BlockSchedule, generate

ms = marian.load()
AIR = ms.air()
FLAT = marian.flat_top_modules(ms)
ROOF = marian.roof_modules(ms)
AW = [2, 3, 3, 3.5, 5, 8]
BS = BlockSchedule.anneal(B=8, Sb=3, K=64, shift=True, restart=False, beta0=1.0, gamma0=2.0)


def stats(m, g):
    T = m.T
    built = g != AIR
    L = g.shape[1]
    topk = L - 1 - np.argmax(built[:, ::-1], axis=1)                     # (s, n, n) top built layer
    has = built.any(1)
    topmod = np.take_along_axis(g, topk[:, None], 1)[:, 0]
    ok = has & (topmod < T)
    height = np.where(has, topk + 1, 0)
    ex = (np.isin(g[:, :-1], np.flatnonzero(FLAT)) & (g[:, 1:] == AIR)).sum() / (g.shape[0] * g.shape[2] * g.shape[3])
    nonair = g[(g != AIR) & (g < T)]
    return dict(exposed_flat=ex, top_roof=float(ROOF[topmod[ok]].mean()), ge4=float((height >= 4).mean()),
                mean_h=float(height.mean()), wild=float((g == T).mean()),
                protos=len(set(ms.prototype[nonair].tolist())), roof_cells=float(ROOF[nonair].mean()))


def run(label, roof=1.0, flat_air=0.0, aw=AW, n=40, seeds=(0, 1)):
    m = marian.tile_model(ms, air_weights=aw, group_weights={"flat_top": 1.0, "roof": roof}, flat_air=flat_air)
    t = time.time()
    g = np.stack([generate(m, BS, sd, 0, 0, n, n) for sd in seeds])
    s = stats(m, g)
    print(f"{label:28s} exposed flat {s['exposed_flat']:.2f}  top=roof {s['top_roof']:.2f}  roof cells "
          f"{s['roof_cells']:.3f}  >=4 {s['ge4']:.2f}  mean h {s['mean_h']:.2f}  wild {s['wild']:.4f}  "
          f"protos {s['protos']}  {(time.time() - t) / len(seeds):.0f}s", flush=True)
    return m


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        kw = eval(f"dict({spec})")
        run(spec or "base", **kw)
