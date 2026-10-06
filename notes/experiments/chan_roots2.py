"""Roots from masked exemplar coordinates, alone on flat ground
(castlegen/channels/coord.py).  Tile kernel: structural root rules + the
alpha coupling + certificate; coordinate kernel: coherence + masked
patch + coupling.  Alternating sweeps.  Writes images/chan_roots2.png
(top: tiles; middle: coordinates, grey = FREE; bottom: certificate d).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/chan_roots2.py
"""
import os, time
import numpy as np
from PIL import Image

from castlegen.channels import Channel, Factor, Kinds, Model, roots, coord

H, W = int(os.environ.get("H", 64)), int(os.environ.get("W", 192))
R0 = int(os.environ.get("R0", 12))
SWEEPS = int(os.environ.get("SWEEPS", 300))
SEED = int(os.environ.get("SEED", 0))
NU, LAM, WP = float(os.environ.get("NU", 6.0)), float(os.environ.get("LAM", 1.0)), float(os.environ.get("WP", 0.0))
DANGLE = float(os.environ.get("DANGLE", 1.0))
KR, KT = int(os.environ.get("KR", 1)), int(os.environ.get("KT", 1))
OUT = os.environ.get("OUT", "images/chan_roots2.png")

SKY_SOIL = Kinds(["sky", "soil"], [frozenset({"sky"}), frozenset({"solid", "earth"})],
                 [(169, 212, 240), (138, 98, 66)])
kinds = Kinds.concat(SKY_SOIL, roots.KINDS)
tile = roots.tile_views(Channel("tile", 1, len(kinds)), kinds)
alpha, g, mask, trunk_yx = coord.parse_exemplar(kinds)
assert coord.check_exemplar(kinds, alpha, g) == 0, "exemplar roots not all attached"
print(f"exemplar {alpha.shape}, {int((alpha >= 2).sum())} root cells, {int((alpha == coord.EARTH).sum())} ring cells, "
      f"{int((alpha == coord.FREE).sum())} free")
u = coord.coord_channel(alpha)
cert, trees = roots.cert_channel(roots.certificate().Dmax), roots.trees_channel()
flat_ground = Factor.unary((tile.name, "root"), np.array([8.0, 0.0, 0.0, 0.0]), name="flat_ground")
factors = roots.factors(kinds, dangle=DANGLE, counted=False) + [flat_ground]
m = Model(H, W, [tile, u, cert, trees], factors, [roots.certificate()])
coupling, coup = coord.coupling(kinds, tile, u, nu=NU)              # applied inside the joint kernel
print(m.describe("tile"))
ck = coord.CoordKernel(u, tile, alpha, lam=LAM, w=WP, nu=NU, K=KR, Kt=KT)

tile.grid[:] = kinds.index("soil")
tile.grid[:R0] = kinds.index("sky")
tile.fixed[:R0] = True
u.fixed[:R0] = True
ck.init_free()
cert.grid[:] = cert.D - 1
roots.sample_trees(trees, lambda j: R0, SEED, spacing=5, p=0.7)
print("trees wanted:", int(trees.grid.sum()))

t0 = time.time()
ck.sweep_joint(m, coup, 1, SEED)
print(f"compile+1 sweep {time.time() - t0:.1f}s")
t0 = time.time()
done = 1
for k in (25, 50, 100, 200, SWEEPS):
    bad = ck.sweep_joint(m, coup, k - done, SEED + k)
    done = k
    r, c = roots.metrics(tile, cert, trees), coord.metrics(u, tile, alpha)
    print(f"sweep {k:4d}  " + "  ".join(f"{a} {b}" for a, b in r.items() if a not in ("mean_d", "sky_contacts"))
          + "  |  " + "  ".join(f"{a} {b}" for a, b in c.items()))
print(f"{(time.time() - t0) / (SWEEPS - 1) * 1000:.1f} ms per joint sweep at {H}x{W}")

px = 6
img = roots.render(kinds, tile, px)
cimg = coord.render_coords(u, alpha, px)
d = cert.grid.astype(float)
dm = d[d < cert.D - 1].max() if (d < cert.D - 1).any() else 1
dimg = np.where(d >= cert.D - 1, 255, 40 + 200 * d / dm).astype(np.uint8)
dimg = np.repeat(np.repeat(np.stack([dimg] * 3, -1), px, 0), px, 1)
gap = np.full((8, W * px, 3), 255, np.uint8)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(np.concatenate([img, gap, cimg, gap, dimg], 0)).save(OUT)
print("wrote", OUT)
