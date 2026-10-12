"""The city tile set (castlegen/citytiles.py, 3907 dual-grid Wang tiles) in
an infinite world (castlegen/infinite.py SocketModel).  Defects versus
schedule on bulk runs, then an exact 256^2 world from 32^2 chunks in two
batches, checked against one batch.  Run with NUMBA_NUM_THREADS=1.
Images: images/infinite_city.png, images/infinite_city_hot.png,
images/infinite_city_bulk256.png."""
import os
import time

import numpy as np
from PIL import Image

from castlegen import citytiles as ct
from castlegen.infinite import Schedule, SocketModel, World, bulk, generate

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images")
os.makedirs(OUT, exist_ok=True)


def main():
    ts = ct.TileSet()
    m = SocketModel(-np.log(ts.w), ts.sN, ts.sE, ts.sS, ts.sW)
    bulk(m, Schedule.anneal(S=4, quench=1), 0, 8, 8)
    print(f"T = {m.T}")
    cls = np.array([ct.zclass(q) for q in ct.ZONES])

    def mix(g):
        z = np.bincount(ts.corners[g].ravel(), minlength=len(ct.ZONES))
        return {k: round(float(z[cls == k].sum() / z.sum()), 3) for k in "PRCDIWA" if z[cls == k].sum()}

    print("bulk 128^2 (centre 96^2): S, beta0, violations / cell, corner zones")
    for S, b0 in ((32, 0.1), (32, 0.3), (32, 1.0), (64, 0.1), (64, 0.3), (128, 0.1), (128, 0.3), (256, 0.1)):
        g = bulk(m, Schedule.anneal(S=S, beta0=b0), 3, 128, 128)[16:-16, 16:-16]
        print(f"  {S:4d} {b0:.1f}  {m.energy(g)[1] / g.size:.4f}  {mix(g)}", flush=True)
        if S == 256:
            Image.fromarray(ct.render(ts, g, px=6)).save(os.path.join(OUT, "infinite_city_bulk256.png"))
            print("  wrote images/infinite_city_bulk256.png")
    for name, sch in (("city", Schedule.anneal(S=32, beta0=1.0)), ("city_hot", Schedule.anneal(S=32))):
        w = World(m, sch, seed=5, C=32)
        t = time.time()
        top = w.region(-128, -128, 128, 256)
        full = w.region(-128, -128, 256, 256)
        dt = time.time() - t
        assert np.array_equal(full[:128], top)
        same = np.array_equal(full, generate(m, sch, 5, -128, -128, 256, 256))
        E, V = m.energy(full)
        frac = mix(full)
        print(f"{name}: 256^2 in two batches {dt:.0f}s, equals one batch: {same}, violations {V} "
              f"({V / full.size:.4f}/cell), updates per stored cell {w.stats['updates'] / w.stats['cells']:.0f}")
        print("  corner zones:", frac)
        Image.fromarray(ct.render(ts, full[:128, :128], px=8)).save(os.path.join(OUT, f"infinite_{name}.png"))
        print(f"  wrote images/infinite_{name}.png", flush=True)


if __name__ == "__main__":
    main()
