"""Colour a saved macro_objects.py hier map by connected component.

Reads cache/<OUT>_hier.npz (written by macro_objects.py) and writes
images/<OUT>_components.png at 1 px per cell, each component in its own
colour (walls darker), components of fewer than SMALL rooms grey; plus
images/<OUT>_components_preview.png scaled to PREVIEW px.  Prints component
size statistics.

The macro_dense_conn map at 5x the side (1000 x 1000 blocks, 8000^2 cells):

    TILES=macro_dense OUT=macro_dense_conn_big VARIANTS=hier N=1000 CROP=384 \\
        python notes/experiments/legacy/macro_objects.py
    TILES=macro_dense OUT=macro_dense_conn_big python notes/experiments/legacy/macro_components.py
"""
import os
import numpy as np
from PIL import Image
from castlegen.legacy import macroobj as MO

TILES = os.environ.get("TILES", "macro_dense")
OUT = os.environ.get("OUT", "macro_dense_conn")
SMALL = int(os.environ.get("SMALL", 20))
PREVIEW = int(os.environ.get("PREVIEW", 2000))
IMG = os.environ.get("IMG", "/Users/amdson/dev/Hierwave/images")
Image.MAX_IMAGE_PIXELS = None

C = MO.load(TILES)
d = np.load(os.path.join("cache", OUT + "_hier.npz"))
S = MO.State(int(d["BY"]), int(d["BX"]))
for o in np.nonzero(d["lab"] >= 0)[0]:
    MO._set(o, d["lab"][o], d["phy"][o], d["phx"][o], C.tb, S.st)
par = np.empty(S.lab.size, np.int64)
nobj, bad = MO.stats(C.tb, S.st, par)
on = S.lab >= 0
roots, inv, sizes = np.unique(par[on], return_inverse=True, return_counts=True)
print(f"{S.BY} x {S.BX} blocks: rooms {nobj}, components {len(sizes)}, unmatched doors {bad}")
for lo, hi in [(1, 1), (2, 4), (5, 19), (20, 99), (100, 999), (1000, 10 ** 12)]:
    m = (sizes >= lo) & (sizes <= hi)
    print(f"  size {lo}-{hi if hi < 10 ** 12 else 'inf'}: {m.sum():6d} components, {sizes[m].sum() / nobj:.1%} of rooms")
print("  largest:", np.sort(sizes)[::-1][:10].tolist())

pal = np.random.default_rng(0).integers(60, 255, (len(sizes), 3)).astype(np.uint8)
pal[sizes < SMALL] = (110, 110, 110)
dark = (pal * 0.45).astype(np.uint8)
H, W = S.BY * MO.BS, S.BX * MO.BS
img = np.full((H, W, 3), (25, 30, 25), np.uint8)
comp = np.full(S.lab.size, -1)
comp[on] = inv
cells = {}                                                # per variant: its non-void cells, wall flags
for o in np.nonzero(on)[0]:
    k = S.lab[o]
    if k not in cells:
        kc = C.KC[k, :C.KH[k], :C.KW[k]]
        yy, xx = np.nonzero(kc != MO.VOID)
        cells[k] = (yy, xx, kc[yy, xx] == MO.WALL)
    yy, xx, wall = cells[k]
    ay, ax = MO._anchor(o, C.tb, S.st)
    img[(ay + yy) % H, (ax + xx) % W] = np.where(wall[:, None], dark[comp[o]], pal[comp[o]])
im = Image.fromarray(img)
im.save(os.path.join(IMG, OUT + "_components.png"))
if max(H, W) > PREVIEW:
    im.resize((PREVIEW * W // max(H, W), PREVIEW * H // max(H, W)), Image.LANCZOS).save(
        os.path.join(IMG, OUT + "_components_preview.png"))
print("saved", os.path.join(IMG, OUT + "_components.png"))
