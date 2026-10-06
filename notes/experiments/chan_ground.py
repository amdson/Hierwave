"""Ground alone on the generic channel kernel (notes/channels_demo.md, step 2):
a value-noise surface channel at level 8, support hard, count honour soft,
ground pair tables soft.  Prints metrics per checkpoint and writes
images/chan_ground.png (the tiles; chunk borders faint; left strip: surf).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/chan_ground.py
"""
import os, sys, time
import numpy as np
from PIL import Image

from castlegen.channels import Model, ground

H, W = int(os.environ.get("H", 128)), int(os.environ.get("W", 256))
SWEEPS = int(os.environ.get("SWEEPS", 60))
SEED = int(os.environ.get("SEED", 0))
KAPPA = float(os.environ.get("KAPPA", 0.5))
OUT = os.environ.get("OUT", "images/chan_ground.png")

kinds = ground.KINDS
tile, surf = ground.tile_channel(kinds), ground.surf_channel()
m = Model(H, W, [tile, surf], ground.factors(kappa=KAPPA))
print(m.describe("tile"))
ys = ground.sample_surface(surf, H, W, seed=SEED)
tile.grid[:] = 0                                                   # all sky: a supported start
t0 = time.time()
m.sweep("tile", 1, seed=SEED)                                      # compile
print(f"compile+1 sweep {time.time() - t0:.1f}s")
t0 = time.time()
done = 1
for k in (5, 10, 20, 40, SWEEPS):
    bad = m.sweep("tile", k - done, seed=SEED + k)
    done = k
    met = ground.metrics(tile, surf)
    tot, viol = m.energy("tile")
    print(f"sweep {k:3d}  no-candidate sites {bad}  unsupported {met['unsupported']}  "
          f"count MAE {met['count_mae']:.2f}  chunks exact {met['chunks_exact']:.2f}  "
          f"solid {met['solid_frac']:.3f}  E {tot:.0f} viol {viol}")
print(f"{(time.time() - t0) / (SWEEPS - 1) * 1000:.1f} ms per sweep at {H}x{W}")

px = 4
img = ground.render(kinds, tile, px)
for i in range(0, H * px, ground.CH * px):
    img[i, :] = (img[i, :] * 0.85).astype(np.uint8)
for j in range(0, W * px, ground.CH * px):
    img[:, j] = (img[:, j] * 0.85).astype(np.uint8)
for j in range(W // ground.CH):                                    # the noise surface, red, per chunk column
    r = int(np.clip(round(ys[j] * px), 0, H * px - 1))
    img[r, j * ground.CH * px:(j + 1) * ground.CH * px] = (220, 40, 40)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(img).save(OUT)
print("wrote", OUT)
