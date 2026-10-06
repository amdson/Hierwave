"""Ground + roots with a coarse coordinate channel (coord.Coarse): one
exemplar window per chunk, sampled top-down by the generic kernel, then a
consistent refinement of fine coordinates, tiles and certificate depth,
then a few joint sweeps.  No sequential growth: the schedule is a fixed
small number of sweeps at each level.  Writes images/chan_combined3.png
(top: tiles; middle: fine coordinates, grey = FREE; bottom: certificate d).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/chan_combined3.py
"""
import os, time
import numpy as np
from PIL import Image

from castlegen.channels import Kinds, Model, ground, roots, coord

H, W = int(os.environ.get("H", 128)), int(os.environ.get("W", 512))
SEED = int(os.environ.get("SEED", 0))
EX = os.environ.get("EX", "big")
S8, SF = int(os.environ.get("S8", 30)), int(os.environ.get("SF", 30))         # coarse sweeps, fine joint sweeps
LAM8, F8 = float(os.environ.get("LAM8", "inf")), float(os.environ.get("F8", 0.1))
BONUS, WBONUS = float(os.environ.get("BONUS", 4.0)), float(os.environ.get("WBONUS", 2.0))
NU, LAM, MU = float(os.environ.get("NU", 6.0)), float(os.environ.get("LAM", 1.0)), float(os.environ.get("MU", 2.0))
DANGLE = float(os.environ.get("DANGLE", 1.0))
KR, KT = int(os.environ.get("KR", 1)), int(os.environ.get("KT", 1))
OUT = os.environ.get("OUT", "images/chan_combined3.png")

kinds = Kinds.concat(ground.KINDS, roots.KINDS)
tile = roots.tile_views(ground.tile_channel(kinds), kinds)
alpha, g, mask, _ = coord.parse_exemplar(kinds, rows=coord.EXEMPLARS[EX], tree=True)
assert coord.check_exemplar(kinds, alpha, g) == 0
depth = coord.exemplar_depth(kinds, alpha)
u = coord.coord_channel(alpha)
surf, cert = ground.surf_channel(), roots.cert_channel(roots.certificate().Dmax)
coarse = coord.Coarse(kinds, alpha, ground.CH)
u8, d8 = coarse.channel(), coarse.cert_channel()
trees = roots.trees_channel()
trees.grid = np.zeros((H // ground.CH, W // ground.CH), np.int32)
print(f"exemplar {alpha.shape}: {int((alpha >= 2).sum())} root cells; {coarse.D - 1} windows + FREE")

# level 8: surf designed, u8 by the generic kernel (no trees channel: the trunk window is the tree)
m8 = Model(H, W, [u8, surf, d8], coarse.factors(lam8=LAM8, f=F8, bonus=BONUS, win_bonus=WBONUS), [coarse.certificate()])
ys = ground.sample_surface(surf, H, W, seed=SEED, mean_depth=0.4)
print(m8.describe("u8"))
t0 = time.time()
u8.grid[:] = 0
d8.grid[:] = d8.D - 1
bad8 = m8.sweep("u8", S8, seed=SEED)
e8, v8 = m8.energy("u8")
nwin, ntrunk = int((u8.grid > 0).sum()), int(coarse.has_trunk[u8.grid].sum())
print(f"level 8: {S8} sweeps {time.time() - t0:.1f}s  windows {nwin}  trunk windows {ntrunk}  no-candidate {bad8}  viol {v8}")

# level 1: refinement, then joint sweeps
roots_structural = roots.factors(kinds, dangle=DANGLE, counted=False)
roots_structural = [f for f in roots_structural if f.name != "trunk_count"]
m = Model(H, W, [tile, u, surf, cert], ground.factors() + roots_structural, [roots.certificate(tree=True)])
_, coup = coord.coupling(kinds, tile, u, nu=NU)
ck = coord.CoordKernel(u, tile, alpha, lam=LAM, nu=NU, K=KR, Kt=KT)
ground.init_tiles(tile, surf, kinds)
uref = coarse.refine(u8, u, tile, cert, kinds, alpha, depth)
tot, viol = m.energy("tile")
r = roots.metrics(tile, cert, trees)
print(f"after refinement: root cells {r['root_cells']}  pruned {coarse.pruned}  rule violations {r['rule_violations']}  tile viol {viol}")
t0 = time.time()
done = 0
for k in (5, 15, SF):
    bad = ck.sweep_joint(m, coup, k - done, SEED + k, uref=uref, mu=MU)
    done = k
    gm, r, c = ground.metrics(tile, surf), roots.metrics(tile, cert, trees), coord.metrics(u, tile, alpha)
    tot, viol = m.energy("tile")
    solid = tile.views["solid"][tile.grid].sum()
    print(f"fine sweep {k:3d}  no-candidate {bad}  viol {viol}  unsupported {gm['unsupported']}  count_mae {gm['count_mae']:.2f}  | "
          f"root cells {r['root_cells']}  coverage {r['root_cells'] / solid:.3f}  rule_violations {r['rule_violations']}  "
          f"extra_joins {r['extra_joins']}  dangling {r['dangling_ports']}  trunks {r['trunks']}  mass {r['mass_hist']}  | "
          f"mismatched {c['mismatched']}")
print(f"{(time.time() - t0) / SF * 1000:.1f} ms per fine joint sweep at {H}x{W}")

px = 4
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
