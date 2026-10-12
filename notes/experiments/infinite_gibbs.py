"""Infinite worlds by annealed checkerboard Gibbs (castlegen/infinite.py):
exactness across requests, cost, defects versus sweeps, images.
Run with NUMBA_NUM_THREADS=1.  Images go to images/infinite_*.png."""
import os
import time

import numpy as np
from PIL import Image

from castlegen.infinite import Schedule, World, biomes, generate, render_roads, roads

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images")
os.makedirs(OUT, exist_ok=True)


def save(img, name):
    Image.fromarray(img).save(os.path.join(OUT, name))
    print("  wrote images/" + name)


def chunk_order(model, sched, seed, C=32, n=6):
    """A world filled chunk by chunk in random order equals one batch."""
    rng = np.random.default_rng(0)
    w = World(model, sched, seed, C)
    keys = [(cy, cx) for cy in range(-n // 2, n // 2) for cx in range(-n // 2, n // 2)]
    for k in rng.permutation(len(keys)):
        w.chunk(*keys[k])
    a = w.region(-n // 2 * C, -n // 2 * C, n * C, n * C)
    b = generate(model, sched, seed, -n // 2 * C, -n // 2 * C, n * C, n * C)
    return np.array_equal(a, b), w.stats


def pad_sensitivity(model, sched, seed, H=96):
    exact = generate(model, sched, seed, 0, 0, H, H)
    out = []
    for f in (0.0, 0.25, 0.5, 0.75, 0.95):
        p = int(f * sched.pad)
        out.append((p, int((generate(model, sched, seed, 0, 0, H, H, pad=p) != exact).sum())))
    return out


def defects(model, seeds, Ss, H=128, **kw):
    rows = []
    for S in Ss:
        sch = Schedule.anneal(S=S, **kw)
        v, e, t = [], [], 0.0
        for sd in seeds:
            t0 = time.time()
            g = generate(model, sch, sd, 0, 0, H, H)
            t += time.time() - t0
            E, V = model.energy(g)
            v.append(V / g.size)
            e.append(E / g.size)
        rows.append((S, np.mean(v), np.mean(e), t / len(seeds)))
    return rows


def main():
    generate(roads(), Schedule.anneal(S=8), 0, 0, 0, 8, 8)       # compile
    sets = {"roads": (roads(), render_roads), "biomes": (biomes(), None)}
    for name, (m, rend) in sets.items():
        print(f"== {name}: T = {m.T}")
        sch = Schedule.anneal(S=64)
        ok, st = chunk_order(m, sch, 3)
        print(f"  random chunk order == one batch: {ok};  updates per stored cell {st['updates'] / st['cells']:.0f}"
              f" (S = 64: ideal 64)")
        print("  pad sensitivity (pad, differing cells of 96^2):", pad_sensitivity(m, sch, 3))
        print("  S, violations / cell, soft energy / cell, s per 128^2")
        for S, v, e, t in defects(m, range(4), (8, 16, 32, 64, 128)):
            print(f"  {S:4d}  {v:.4f}  {e:+.3f}  {t:.2f}")
        # a 384^2 window assembled from 32^2 chunks requested in two batches
        w = World(m, sch, 11, 32)
        left = w.region(-192, -192, 384, 192)
        full = w.region(-192, -192, 384, 384)
        assert np.array_equal(full[:, :192], left)
        E, V = m.energy(full)
        print(f"  384^2 world in two batches: violations {V} ({V / full.size:.4f}/cell), "
              f"updates per stored cell {w.stats['updates'] / w.stats['cells']:.0f}")
        img = rend(m, full[:192, :192]) if rend else m.render(full, scale=2)
        save(img, f"infinite_{name}.png")


if __name__ == "__main__":
    main()
