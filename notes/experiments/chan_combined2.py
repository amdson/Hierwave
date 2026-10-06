"""Ground + masked-exemplar roots on one tile kernel (coord.py): the ground
set's factors, the root set's structural rules and certificate, and the
coordinate channel's alpha coupling, drawn by the joint (u, t, d) kernel
over the tile model.  Writes images/chan_combined2.png (top: tiles with
the noise surface in red; middle: coordinates, grey = FREE; bottom:
certificate d).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/chan_combined2.py
"""
import os, time
import numpy as np
from PIL import Image

from castlegen.channels import Kinds, Model, ground, roots, coord

H, W = int(os.environ.get("H", 96)), int(os.environ.get("W", 256))
SWEEPS = int(os.environ.get("SWEEPS", 400))
SEED = int(os.environ.get("SEED", 0))
KAPPA, BETA_G = float(os.environ.get("KAPPA", 0.5)), float(os.environ.get("BETA_G", 1.0))
NU, LAM = float(os.environ.get("NU", 6.0)), float(os.environ.get("LAM", 1.0))
DANGLE = float(os.environ.get("DANGLE", 1.0))
KR, KT = int(os.environ.get("KR", 1)), int(os.environ.get("KT", 1))
TREE = bool(int(os.environ.get("TREE", 1)))
SPACING, PTREE = int(os.environ.get("SPACING", 4)), float(os.environ.get("PTREE", 0.6))
OUT = os.environ.get("OUT", "images/chan_combined2.png")

kinds = Kinds.concat(ground.KINDS, roots.KINDS)
tile = roots.tile_views(ground.tile_channel(kinds), kinds)
alpha, g, mask, _ = coord.parse_exemplar(kinds, tree=TREE)
assert coord.check_exemplar(kinds, alpha, g) == 0
u = coord.coord_channel(alpha)
surf, trees, cert = ground.surf_channel(), roots.trees_channel(), roots.cert_channel(roots.certificate().Dmax)
factors = ground.factors(kappa=KAPPA, beta=BETA_G) + roots.factors(kinds, dangle=DANGLE, counted=False)
m = Model(H, W, [tile, u, surf, trees, cert], factors, [roots.certificate(tree=TREE)])
_, coup = coord.coupling(kinds, tile, u, nu=NU)
ck = coord.CoordKernel(u, tile, alpha, lam=LAM, nu=NU, K=KR, Kt=KT)
print(m.describe("tile"))
print(f"{len(kinds)} kinds, {len(factors)} factors + coupling")

ys = ground.sample_surface(surf, H, W, seed=SEED, mean_depth=0.4)
ground.init_tiles(tile, surf, kinds)
ck.init_free()
cert.grid[:] = cert.D - 1
top = ground.surface_rows(tile)
roots.sample_trees(trees, lambda j: int(top[j * ground.CH + ground.CH // 2]), SEED, spacing=SPACING, p=PTREE)
print("trees wanted:", int(trees.grid.sum()))

t0 = time.time()
ck.sweep_joint(m, coup, 1, SEED)
print(f"compile+1 sweep {time.time() - t0:.1f}s")
t0 = time.time()
done = 1
for k in (50, 100, 200, SWEEPS):
    bad = ck.sweep_joint(m, coup, k - done, SEED + k)
    done = k
    gm, r, c = ground.metrics(tile, surf), roots.metrics(tile, cert, trees), coord.metrics(u, tile, alpha)
    tot, viol = m.energy("tile")
    print(f"sweep {k:4d}  no-candidate {bad}  viol {viol}  "
          + "  ".join(f"{a} {b}" for a, b in gm.items() if a != "solid_frac") + "  |  "
          + "  ".join(f"{a} {b}" for a, b in r.items() if a not in ("mean_d", "sky_contacts")) + "  |  "
          + "  ".join(f"{a} {b}" for a, b in c.items()))
print(f"{(time.time() - t0) / (SWEEPS - 1) * 1000:.1f} ms per joint sweep at {H}x{W}")

px = 5
img = roots.render(kinds, tile, px)
for j in range(W // ground.CH):
    r_ = int(np.clip(round(ys[j] * px), 0, H * px - 1))
    img[r_, j * ground.CH * px:(j + 1) * ground.CH * px] = (220, 40, 40)
cimg = coord.render_coords(u, alpha, px)
d = cert.grid.astype(float)
dm = d[d < cert.D - 1].max() if (d < cert.D - 1).any() else 1
dimg = np.where(d >= cert.D - 1, 255, 40 + 200 * d / dm).astype(np.uint8)
dimg = np.repeat(np.repeat(np.stack([dimg] * 3, -1), px, 0), px, 1)
gap = np.full((8, W * px, 3), 255, np.uint8)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(np.concatenate([img, gap, cimg, gap, dimg], 0)).save(OUT)
print("wrote", OUT)
