"""Isometric previews of the prefab templates (images/prefab_*.png).

  PYTHONPATH=. NUMBA_NUM_THREADS=1 uv run --extra dev python notes/experiments/prefab_preview.py

Renders: one city template, a 6x6 city region sampled with the contract weights,
a 6x6 biome region, and a contact sheet of distinct city templates.  Missing
prefab files are replaced by placeholder blocks (listed on stdout).
"""
import os

import numpy as np
from PIL import Image, ImageDraw

from castlegen import citytiles as ct
from castlegen import prefabs as pf
from castlegen.infinite import Schedule, SocketModel, TileModel, biomes, generate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images")

COL = {
    "stone": "#808080", "dirt": "#7a5a3a", "grass": "#6fae4a", "sand": "#e0d08e", "water": "#3f78c8",
    "gravel": "#9a9690", "snow": "#f4f6fa", "asphalt": "#3a3a3e", "road_line": "#e8e8d0",
    "concrete": "#b4b4b0", "paving": "#c2b9a6", "curb": "#9c9c98", "rail": "#6b4a2e",
    "ballast": "#7a7268", "fence": "#8a6a40", "tarmac": "#52565c", "runway_mark": "#f0f0f0",
    "deck": "#c8c8c0", "pillar": "#8e8e8a", "trunk": "#5a3d22", "leaves": "#2f7a35",
    "hedge": "#3e8a40", "house_wall": "#e6d2b0", "house_roof": "#b04a3a", "shop_wall": "#e0a050",
    "glass": "#a8d8e8", "tower_wall": "#c0524a", "tower_top": "#7a2e2a", "shed_wall": "#8c84a6",
    "shed_roof": "#5e5878", "chimney": "#6a4a42", "tank": "#b8bcc0", "plaza": "#d8cba8",
    "fountain": "#5ab0d8", "lamp": "#f8e070", "rock": "#6e6a66", "deep_water": "#1f4a90",
    "dark_grass": "#4c8a3a", "window": "#8cc8e8",
}
NAMES = list(pf.PAL)
RGB = np.zeros((len(NAMES), 3), np.float64)
for _i, _n in enumerate(NAMES):
    if _n in COL:
        RGB[_i] = [int(COL[_n][k:k + 2], 16) for k in (1, 3, 5)]


def render_iso(vox, s=8, kmin=0, bg=(246, 246, 244)):
    """vox: (L, H, W) palette ids (0 air).  Viewer at the south-east above: shows
    top faces and the south (+i) and east (+j) faces.  Returns a PIL image."""
    Lh, H, W = vox.shape
    w, h = s, s // 2           # half-width and half-height of a top diamond
    zh = s * 0.9               # pixel height of one layer
    width = (H + W) * w + 4
    height = int((H + W) * h + Lh * zh) + 4
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    solid = vox != 0
    pad = np.pad(solid, ((0, 1), (0, 1), (0, 1)))
    # draw back to front: i + j ascending, then k ascending
    order = sorted(zip(*np.nonzero(solid)), key=lambda p: (p[1] + p[2], p[0]))
    for k, i, j in order:
        if k < kmin:
            continue
        top_hidden = pad[k + 1, i, j]
        south_hidden = pad[k, i + 1, j]
        east_hidden = pad[k, i, j + 1]
        if top_hidden and south_hidden and east_hidden:
            continue
        c = RGB[vox[k, i, j]]
        # screen position of the top-left... of the diamond top at height k+1
        cx = (j - i + H) * w + 2
        cy = (i + j) * h + (Lh - k - 1) * zh + 2
        top = [(cx, cy), (cx + w, cy + h), (cx, cy + 2 * h), (cx - w, cy + h)]
        if not top_hidden:
            d.polygon(top, fill=tuple(int(x) for x in c))
        if not south_hidden:
            d.polygon([top[3], top[2], (top[2][0], top[2][1] + zh), (top[3][0], top[3][1] + zh)],
                      fill=tuple(int(x * 0.72) for x in c))
        if not east_hidden:
            d.polygon([top[2], top[1], (top[1][0], top[1][1] + zh), (top[2][0], top[2][1] + zh)],
                      fill=tuple(int(x * 0.5) for x in c))
    return img


def tile_grid(templates, grid):
    """(L, 8 R, 8 C) voxels from a grid of template ids."""
    R, C = grid.shape
    out = np.zeros((pf.L, R * 8, C * 8), np.uint8)
    for r in range(R):
        for c in range(C):
            out[:, r * 8:r * 8 + 8, c * 8:c * 8 + 8] = templates[grid[r, c]]
    return out


def crop_top(vox):
    """Drop empty top layers so images are not mostly bg."""
    nz = np.nonzero(vox.reshape(vox.shape[0], -1).any(axis=1))[0]
    return vox[: nz.max() + 1] if len(nz) else vox


def contract_city():
    old = (ct.BOUNDARY_W, dict(ct.ZONE_W))
    ct.BOUNDARY_W = 0.5
    ct.ZONE_W.update({"P": 0.4, "R": 1.0, "C": 1.0, "D": 1.0, "I": 1.0, "W": 0.8, "A": 0.6})
    try:
        ts = ct.TileSet()
    finally:
        ct.BOUNDARY_W, _ = old[0], ct.ZONE_W.update(old[1])
    return ts


def save(img, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.abspath(os.path.join(OUT, name))
    img.save(path)
    print("wrote", path)


def main():
    cdir, bdir = os.path.join(pf.PREFAB_DIR, "city"), os.path.join(pf.PREFAB_DIR, "biomes")
    miss = pf.missing_files(cdir, pf.city_files())
    bmiss = pf.missing_files(bdir, pf.biome_files(biomes().names))
    print("city prefabs missing (placeholders used):", miss or "none")
    print("biome prefabs missing (placeholders used):", bmiss or "none")

    ts = contract_city()
    city = pf.assemble_city(ts, cdir, allow_missing=True)
    bio = pf.assemble_biomes(bdir, allow_missing=True)

    # (a1) a single template: an intersection with a street on every side if present
    feats = ts.feats
    cand = np.nonzero((feats == 1).all(axis=1) & (ts.corners == ct.Z["R"]).all(axis=1))[0]
    if len(cand) == 0:
        cand = np.nonzero((feats > 0).sum(axis=1) >= 2)[0]
    t = int(cand[0])
    save(render_iso(crop_top(city[t]), s=40), "prefab_template.png")

    # (a2) city region: a short sampler run (S = 32, contract weights, wildcard fix pass)
    m = SocketModel(-np.log(ts.w), ts.sN, ts.sE, ts.sS, ts.sW, wild_u=20.0)
    sched = Schedule.anneal(S=32, beta0=1.0, gamma0=2.0)
    grid = generate(m, sched, 5, 0, 0, 6, 6)
    save(render_iso(crop_top(tile_grid(city, grid)), s=9), "prefab_city_region.png")

    # (a3) biome region
    bm = biomes()
    g = generate(bm, Schedule.anneal(S=32), 3, 0, 0, 6, 6)
    save(render_iso(crop_top(tile_grid(bio, g)), s=9), "prefab_biome_region.png")

    # (b) contact sheet: distinct templates (one per zone-class corner pattern), a few dozen
    rng = np.random.RandomState(0)
    seen, picks = set(), []
    for t in rng.permutation(ts.T):
        key = (tuple(ct.zclass(ct.ZONES[c]) for c in ts.corners[t]), tuple(int(f) for f in ts.feats[t]))
        zkey = (tuple(sorted(set(key[0]))), tuple(sorted(set(key[1]))))
        if zkey in seen:
            continue
        seen.add(zkey)
        picks.append(int(t))
        if len(picks) == 36:
            break
    cells = [render_iso(crop_top(city[t])[:14], s=14) for t in picks]
    cw, ch = max(c.width for c in cells), max(c.height for c in cells)
    cols = 6
    sheet = Image.new("RGB", (cols * cw, ((len(cells) + cols - 1) // cols) * ch), (246, 246, 244))
    for n, c in enumerate(cells):
        sheet.paste(c, ((n % cols) * cw, (n // cols) * ch))
    save(sheet, "prefab_contact_sheet.png")


if __name__ == "__main__":
    main()
