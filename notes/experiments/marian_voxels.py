"""Marian block voxels: neighbour check and contact sheets.

    PYTHONPATH=. NUMBA_NUM_THREADS=1 uv run --extra dev python notes/experiments/marian_voxels.py check
    PYTHONPATH=. NUMBA_NUM_THREADS=1 uv run --extra dev python notes/experiments/marian_voxels.py sheets

check: for every (module A, direction d, module B in A's PossibleNeighbors[d])
compare A's boundary layer in direction d with B's opposite boundary layer
(occupancy agreement, as a fraction of cells), for each combination of the
conventions (x sign, prototype turn sign, module rotation sign), and against
random pairs.  The right combination should agree much better than chance;
this is how X_SIGN / TURN_SIGN / ROT_SIGN in castlegen/marian_meshes.py were
chosen.

sheets: images/marian_vox_sheet.png (prototypes), images/marian_vox_rotations.png
(a few modules at rotations 0-3), isometric voxel views.
"""

import itertools
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from castlegen import marian_meshes as mm

ROOT = mm.ROOT
IMG = os.path.join(ROOT, "images")


def colours():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_textures", os.path.join(ROOT, "luanti", "tools", "make_textures.py"))
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    names = open(mm.PALETTE_FILE).read().split()
    out = {0: None}
    for i, n in enumerate(names):
        if n != "air":
            out[i] = mt.COLOURS[n][:3]
    return out


def face(t, d):
    """Boundary layer of template t [y][i][j] in Unity direction d."""
    k = t.shape[0]
    if d == 0:
        return t[:, :, 0] > 0        # LEFT  -x  : j = 0
    if d == 3:
        return t[:, :, k - 1] > 0    # RIGHT +x  : j = k-1
    if d == 2:
        return t[:, k - 1, :] > 0    # BACK  -z  : i = k-1 (south)
    if d == 5:
        return t[:, 0, :] > 0        # FORWARD +z: i = 0 (north)
    if d == 1:
        return t[0] > 0              # DOWN
    return t[k - 1] > 0              # UP


def agree(fa, fb):
    """Jaccard agreement of two boundary layers (1.0 when both are empty)."""
    u = (fa | fb).sum()
    return 1.0 if u == 0 else float((fa & fb).sum() / u)


def check(k=8):
    protos, modules = mm.load_modules()
    nb = np.stack([m["neighbors"] for m in modules])   # (454, 6, 454)
    rng = np.random.default_rng(0)
    res = []
    for xs, ts, rs in itertools.product((-1.0, 1.0), (0,), (-1, 1)):
        mm.X_SIGN, mm.TURN_SIGN, mm.ROT_SIGN = xs, ts, rs  # TURN_SIGN 0: ignore prefab rotation; 2: double
        tem = mm.module_templates(k, use_cache=False)
        hor, ver, rnd, odd = [], [], [], []
        oddp = np.array([protos[m['proto']]['turns'] % 2 == 1 for m in modules])
        for a in range(0, 454, 3):
            for d in range(6):
                bs = np.nonzero(nb[a, d])[0]
                if not len(bs):
                    continue
                bs = rng.choice(bs, min(len(bs), 6), replace=False)
                fa = face(tem[a], d)
                for b in bs:
                    s = agree(fa, face(tem[b], (d + 3) % 6))
                    (ver if d in (1, 4) else hor).append(s)
                    if d not in (1, 4) and (oddp[a] or oddp[b]):
                        odd.append(s)
                    r = rng.integers(454)
                    rnd.append(agree(fa, face(tem[r], (d + 3) % 6)))
        res.append((xs, ts, rs, np.mean(hor), np.mean(ver), np.mean(rnd), np.mean(odd)))
        print("x_sign %+d turn %+d rot %+d  horiz %.4f  vert %.4f  random %.4f  odd-turn %.4f"
              % ((xs, ts, rs) + res[-1][3:]), flush=True)
    best = max(res, key=lambda r: r[3])
    print("best:", best[:3])


# ---------------------------------------------------------------- rendering

def iso_cube_image(tpl, col, cell=10):
    """Isometric view of template tpl [y][i][j] (i north->south, j west->east).
    View from the south-east above: depth increases with j (east is toward the
    viewer's right-front?)  We draw with screen x = (j - i)*c, screen y =
    (i + j)*c/2 - y*c*1.0: north-west cell is at the back."""
    k = tpl.shape[0]
    c = cell
    W = int((2 * k) * c * 1.0) + 8
    H = int(k * c * 1.0 + k * c) + 8
    img = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(img)
    ox, oy = W // 2, k * c + 4
    order = sorted(itertools.product(range(k), range(k), range(k)),
                   key=lambda t: (t[0], t[1] + t[2]))
    # back to front: low y first; within a layer small (i + j) first
    order = sorted(order, key=lambda t: (t[0], t[1] + t[2]))
    for y, i, j in order:
        m = tpl[y, i, j]
        if m == 0:
            continue
        r, g, b = col[m]
        sx = ox + (j - i) * c
        sy = oy + (i + j) * c / 2 - y * c
        top = [(sx, sy - c), (sx + c, sy - c / 2), (sx, sy), (sx - c, sy - c / 2)]
        left = [(sx - c, sy - c / 2), (sx, sy), (sx, sy + c), (sx - c, sy + c / 2)]
        right = [(sx, sy), (sx + c, sy - c / 2), (sx + c, sy + c / 2), (sx, sy + c)]
        dr.polygon(top, fill=(r, g, b))
        dr.polygon(left, fill=(int(r * .7), int(g * .7), int(b * .7)))
        dr.polygon(right, fill=(int(r * .5), int(g * .5), int(b * .5)))
    return img


def label_image(img, text):
    out = Image.new("RGB", (img.width, img.height + 12), (245, 245, 245))
    out.paste(img, (0, 12))
    ImageDraw.Draw(out).text((2, 0), text, fill=(0, 0, 0))
    return out


def sheet(tiles, ncol, path):
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    nrow = (len(tiles) + ncol - 1) // ncol
    img = Image.new("RGB", (ncol * w, nrow * h), (245, 245, 245))
    for n, t in enumerate(tiles):
        img.paste(t, ((n % ncol) * w, (n // ncol) * h))
    os.makedirs(IMG, exist_ok=True)
    img.save(path)
    print("wrote", path, img.size)


def sheets(k=8):
    col = colours()
    protos, modules = mm.load_modules()
    tem = mm.module_templates(k)
    first = {}
    for mi, m in enumerate(modules):
        first.setdefault(m["proto"], mi)
    pick = list(range(len(protos)))
    pick = pick[:: max(1, len(pick) // 40)][:42]
    tiles = [label_image(iso_cube_image(tem[first[p]], col, 9),
                         "%d %s" % (first[p], protos[p]["name"][:22])) for p in pick]
    sheet(tiles, 7, os.path.join(IMG, "marian_vox_sheet.png"))
    names = ["Stairs R0", "Wall_Corner R0", "Roof_2 R0", "Spiral_Stairs R0",
             "Wall_Window R0", "High_Wall_Corner R0"]
    rows = []
    for nm in names:
        mis = [i for i, m in enumerate(modules)
               if m["name"].replace(" ", "_").startswith(nm.replace(" ", "_").split("_R")[0] + "_R")
               or m["name"] == nm.replace("_", " ")]
        base = nm.rsplit(" R", 1)[0].replace("_", " ")
        mis = [i for i, m in enumerate(modules)
               if m["name"].rsplit(" R", 1)[0].replace("_", " ") == base]
        if not mis:
            continue
        for mi in mis[:4]:
            rows.append(label_image(iso_cube_image(tem[mi], col, 9), modules[mi]["name"]))
        rows += [Image.new("RGB", rows[-1].size, (245, 245, 245))] * (4 - len(mis[:4]))
    sheet(rows, 4, os.path.join(IMG, "marian_vox_rotations.png"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheets"
    {"check": check, "sheets": sheets}[cmd]()
