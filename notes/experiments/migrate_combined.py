"""chan_combined3.py through the one interface (sampler.generate): level 8
= [Sampler(u8)] (table kernel with the level-8 certificate), level 1 =
[CoordSampler(joint)] ((u, t, d) per site), painters for the designed
surface and the refinement.  Same seeds as chan_combined3 (S given as
seeded chunks), so the image must equal images/chan_combined3_sheet.png
(EX=sheet, the defaults).  Writes images/migrate_combined.png.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/migrate_combined.py
"""
import os, time
import numpy as np
from PIL import Image

from castlegen.channels import Kinds, Model, ground, roots, coord
from castlegen.channels.sampler import Sampler, generate

H, W = int(os.environ.get("H", 128)), int(os.environ.get("W", 512))
SEED, EX = int(os.environ.get("SEED", 0)), os.environ.get("EX", "sheet")
S8, SF = 30, 30
OUT = os.environ.get("OUT", "images/migrate_combined.png")
REF = os.environ.get("REF", "/Users/amdson/dev/Hierwave/images/chan_combined3_sheet.png")


def build(H, W, seed, ex="sheet", lam8=1.0, f8=0.1, cut=2.0, bonus=4.0, wbonus=2.0, nu=6.0, lam=1.0, mu=2.0,
          dangle=1.0, kr=1, kt=1):
    """chan_combined3's channel sets as (levels, painters, state)."""
    kinds = Kinds.concat(ground.KINDS, roots.KINDS)
    tile = roots.tile_views(ground.tile_channel(kinds), kinds)
    alpha, g, mask, _ = coord.parse_exemplar(kinds, rows=coord.EXEMPLARS[ex], tree=True)
    depth = coord.exemplar_depth(kinds, alpha)
    u = coord.coord_channel(alpha)
    surf, cert = ground.surf_channel(), roots.cert_channel(roots.certificate().Dmax)
    coarse = coord.Coarse(kinds, alpha, ground.CH)
    u8, d8 = coarse.channel(), coarse.cert_channel()
    m8 = Model(H, W, [u8, surf, d8], coarse.factors(lam8=lam8, f=f8, bonus=bonus, win_bonus=wbonus, cut=cut),
               [coarse.certificate()])
    rs = [f for f in roots.factors(kinds, dangle=dangle, counted=False) if f.name != "trunk_count"]
    m = Model(H, W, [tile, u, surf, cert], ground.factors() + rs, [roots.certificate(tree=True)])
    _, coup = coord.coupling(kinds, tile, u, nu=nu)
    cs = coord.CoordSampler(coord.CoordKernel(u, tile, alpha, lam=lam, nu=nu, K=kr, Kt=kt), joint=(m, coup), mu=mu)
    trees = roots.trees_channel()
    trees.grid = np.zeros((H // ground.CH, W // ground.CH), np.int32)
    st = dict(trees=trees, kinds=kinds, tile=tile, u=u, cert=cert, alpha=alpha, surf=surf, u8=u8, d8=d8, coarse=coarse)

    def paint8():
        st["ys"] = ground.sample_surface(surf, H, W, seed=seed, mean_depth=0.4)
        u8.grid[:] = 0
        d8.grid[:] = d8.D - 1

    def paint1():
        ground.init_tiles(tile, surf, kinds)
        cs.uref = coarse.refine(u8, u, tile, cert, kinds, alpha, depth)

    return [[Sampler(m8, "u8")], [cs]], [paint8, paint1], st


def schedule(seed, S8=30, SF=30):
    """chan_combined3's sweeps and seeds as generate's seeded chunks."""
    return [[(S8, seed)], [(5, seed + 5), (10, seed + 15), (SF - 15, seed + SF)]]


if __name__ == "__main__":
    levels, painters, st = build(H, W, SEED, EX)
    t0 = time.time()
    dorm = generate(levels, schedule(SEED, S8, SF), painters=painters)
    tile, u, cert, alpha = st["tile"], st["u"], st["cert"], st["alpha"]
    gm, r = ground.metrics(tile, st["surf"]), roots.metrics(tile, cert, st["trees"])
    c = coord.metrics(u, tile, alpha)
    print(f"generate {time.time() - t0:.1f}s  dormant {dorm}  windows {int((st['u8'].grid > 0).sum())}  "
          f"unsupported {gm['unsupported']}  count_mae {gm['count_mae']:.2f}  root cells {r['root_cells']}  "
          f"rule_violations {r['rule_violations']}  trunks {r['trunks']}  mass {r['mass_hist']}  "
          f"mismatched {c['mismatched']}")
    px = 4                                                      # chan_combined3's render
    img = roots.render(st["kinds"], tile, px)
    for j in range(W // ground.CH):
        r_ = int(np.clip(round(st["ys"][j] * px), 0, H * px - 1))
        img[r_, j * ground.CH * px:(j + 1) * ground.CH * px] = (220, 40, 40)
    cimg = coord.render_coords(u, alpha, px)
    d = cert.grid.astype(float)
    dm = d[d < cert.D - 1].max() if (d < cert.D - 1).any() else 1
    dimg = np.where(d >= cert.D - 1, 255, 40 + 200 * d / dm).astype(np.uint8)
    dimg = np.repeat(np.repeat(np.stack([dimg] * 3, -1), px, 0), px, 1)
    gap = np.full((8, W * px, 3), 255, np.uint8)
    out = np.concatenate([img, gap, cimg, gap, dimg], 0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    Image.fromarray(out).save(OUT)
    print("wrote", OUT)
    if os.path.exists(REF):
        ref = np.array(Image.open(REF))
        print(f"identical to {REF}: {ref.shape == out.shape and np.array_equal(ref, out)}")
