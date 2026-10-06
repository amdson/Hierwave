"""Ground + roots on one tile kernel (notes/channels_demo.md, step 4): the
union of the two channel sets' factors on one tile domain (the union of
their kinds), one certificate, the designed surf and trees channels.  No
code beyond this script: the ground set sees root kinds through its
`solid` / `ground` views, the root set sees soil and stone as earth.
Writes images/chan_combined.png (top: tiles; bottom: certificate d).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/chan_combined.py
"""
import os, time
import numpy as np
from PIL import Image

from castlegen.channels import Channel, Kinds, Model, ground, roots

H, W = int(os.environ.get("H", 96)), int(os.environ.get("W", 256))
SWEEPS = int(os.environ.get("SWEEPS", 1500))
SEED = int(os.environ.get("SEED", 0))
KAPPA, BETA_G = float(os.environ.get("KAPPA", 0.5)), float(os.environ.get("BETA_G", 1.0))
GROW, DANGLE = float(os.environ.get("GROW", 1.5)), float(os.environ.get("DANGLE", 4.0))   # GROW 1.0 alone; +0.5 beside the ground texture
SPACING, PTREE = int(os.environ.get("SPACING", 3)), float(os.environ.get("PTREE", 0.6))
OUT = os.environ.get("OUT", "images/chan_combined.png")

kinds = Kinds.concat(ground.KINDS, roots.KINDS)
tile = roots.tile_views(ground.tile_channel(kinds), kinds)          # both sets' views on one domain
surf, trees, cert = ground.surf_channel(), roots.trees_channel(), roots.cert_channel(roots.certificate().Dmax)
factors = ground.factors(kappa=KAPPA, beta=BETA_G) + roots.factors(kinds, grow=GROW, dangle=DANGLE)
m = Model(H, W, [tile, surf, trees, cert], factors, [roots.certificate()])
print(m.describe("tile"))
print(f"{len(kinds)} kinds, {len(factors)} factors declared")

# designed channels, top-down
ys = ground.sample_surface(surf, H, W, seed=SEED, mean_depth=0.4)
roots.sample_trees(trees, lambda j: int(round(ys[j])), SEED, spacing=SPACING, p=PTREE)
tile.grid[:] = kinds.index("sky")
cert.grid[:] = cert.D - 1
print("trees wanted:", int(trees.grid.sum()))

t0 = time.time()
m.sweep("tile", 1, seed=SEED)
print(f"compile+1 sweep {time.time() - t0:.1f}s")
t0 = time.time()
done = 1
for k in (50, 100, 250, 500, 1000, SWEEPS):
    bad = m.sweep("tile", k - done, seed=SEED + k)
    done = k
    g, r = ground.metrics(tile, surf), roots.metrics(tile, cert, trees)
    tot, viol = m.energy("tile")
    print(f"sweep {k:4d}  no-candidate {bad}  viol {viol}  "
          + "  ".join(f"{a} {b}" for a, b in g.items() if a != "solid_frac") + "  |  "
          + "  ".join(f"{a} {b}" for a, b in r.items() if a not in ("mean_d",)))
print(f"{(time.time() - t0) / (SWEEPS - 1) * 1000:.1f} ms per sweep at {H}x{W}")

px = 6
img = roots.render(kinds, tile, px)
for j in range(W // ground.CH):
    r = int(np.clip(round(ys[j] * px), 0, H * px - 1))
    img[r, j * ground.CH * px:(j + 1) * ground.CH * px] = (220, 40, 40)
d = cert.grid.astype(float)
dm = d[d < cert.D - 1].max() if (d < cert.D - 1).any() else 1
dimg = np.where(d >= cert.D - 1, 255, 40 + 200 * d / dm).astype(np.uint8)
dimg = np.repeat(np.repeat(np.stack([dimg] * 3, -1), px, 0), px, 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(np.concatenate([img, np.full((8, W * px, 3), 255, np.uint8), dimg], 0)).save(OUT)
print("wrote", OUT)
