"""Roots alone on the generic channel kernel (notes/channels_demo.md, step 3):
a flat surface at row R0 (sky clamped above it), the root set's factors
and certificate on port tiles, trunks asked for by the designed `trees`
channel.  Writes images/chan_roots.png (top: tiles; bottom: the certificate
d, dark = small, white = INF).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/chan_roots.py
"""
import os, time
import numpy as np
from PIL import Image

from castlegen.channels import Channel, Factor, Kinds, Model, roots

H, W = int(os.environ.get("H", 64)), int(os.environ.get("W", 192))
R0 = int(os.environ.get("R0", 12))
SWEEPS = int(os.environ.get("SWEEPS", 1000))
SEED = int(os.environ.get("SEED", 0))
BETA, DMAX = float(os.environ.get("BETA", 1.0)), int(os.environ.get("DMAX", 4096))
DELTA = float(os.environ.get("DELTA", 0.05))
DANGLE, GROW = float(os.environ.get("DANGLE", 4.0)), float(os.environ.get("GROW", 1.0))
OUT = os.environ.get("OUT", "images/chan_roots.png")

SKY_SOIL = Kinds(["sky", "soil"], [frozenset({"sky"}), frozenset({"solid", "earth"})],
                 [(169, 212, 240), (138, 98, 66)])
kinds = Kinds.concat(SKY_SOIL, roots.KINDS)
tile = roots.tile_views(Channel("tile", 1, len(kinds)), kinds)
cert, trees = roots.cert_channel(DMAX), roots.trees_channel()
flat_ground = Factor.unary((tile.name, "root"), np.array([6.0, 0.0, 0.0, 0.0]), name="flat_ground")   # the ground set's stand-in: sky is costly below the line
m = Model(H, W, [tile, cert, trees], roots.factors(kinds, beta=BETA, dangle=DANGLE, grow=GROW) + [flat_ground],
          [roots.certificate(Dmax=DMAX, delta=DELTA)])
print(m.describe("tile"))
print(f"{len(kinds)} kinds")

tile.grid[:] = kinds.index("soil")
tile.grid[:R0] = kinds.index("sky")
tile.fixed[:R0] = True
cert.grid[:] = DMAX + 1
roots.sample_trees(trees, lambda j: R0, SEED, spacing=4, p=0.7)
print("trees wanted:", int(trees.grid.sum()))

t0 = time.time()
m.sweep("tile", 1, seed=SEED)
print(f"compile+1 sweep {time.time() - t0:.1f}s")
t0 = time.time()
done = 1
for k in (100, 250, 500, 750, SWEEPS):
    bad = m.sweep("tile", k - done, seed=SEED + k)
    done = k
    met = roots.metrics(tile, cert, trees)
    tot, viol = m.energy("tile")
    print(f"sweep {k:3d}  no-candidate {bad}  " + "  ".join(f"{a} {b}" for a, b in met.items())
          + f"  E {tot:.0f} viol {viol}")
print(f"{(time.time() - t0) / (SWEEPS - 1) * 1000:.1f} ms per sweep at {H}x{W}")

px = 6
img = roots.render(kinds, tile, px)
d = cert.grid.astype(float)
dimg = np.where(d > DMAX, 255, 40 + 200 * d / DMAX).astype(np.uint8)
dimg = np.repeat(np.repeat(np.stack([dimg] * 3, -1), px, 0), px, 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(np.concatenate([img, np.full((8, W * px, 3), 255, np.uint8), dimg], 0)).save(OUT)
print("wrote", OUT)
