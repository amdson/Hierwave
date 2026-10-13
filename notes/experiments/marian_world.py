"""Isometric voxel render of a marian world region: modules from the 3D
sampler (castlegen.infinite, marian.tile_model) expanded into their 8^3
templates (castlegen.marian_meshes).  Usage:
    python notes/experiments/marian_world.py OUT.png "[air weights]" [n_cols] [seed]
Images go to images/."""
import itertools
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from castlegen import marian, marian_meshes as mm
from castlegen.infinite import BlockSchedule, generate

sys.path.insert(0, os.path.dirname(__file__))
from marian_voxels import colours  # noqa: E402


def voxels(g, tpl):
    """(L, h, w) module ids -> (L k, h k, w k) palette ids; wildcard = last template."""
    L, h, w = g.shape
    k = tpl.shape[1]
    t = tpl[np.minimum(g, tpl.shape[0] - 1)]                   # (L, h, w, k, k, k) [y][i][j]
    return t.transpose(0, 3, 1, 4, 2, 5).reshape(L * k, h * k, w * k)


def iso(V, col, c=3):
    Y, I, J = V.shape
    W = (I + J) * c + 8
    H = (I + J) * c // 2 + Y * c + 8
    img = Image.new("RGB", (W, H), (235, 238, 242))
    dr = ImageDraw.Draw(img)
    ox, oy = I * c + 4, Y * c + 4
    solid = V > 0
    pad = np.pad(solid, 1)
    # exposed: top (y+1), south face (i+1), east face (j+1) are visible from the south-east
    exp_top = solid & ~pad[2:, 1:-1, 1:-1]
    exp_s = solid & ~pad[1:-1, 2:, 1:-1]
    exp_e = solid & ~pad[1:-1, 1:-1, 2:]
    vis = exp_top | exp_s | exp_e
    ys, is_, js = np.nonzero(vis)
    order = np.lexsort((ys, is_ + js))                         # back to front, low to high
    for n in order:
        y, i, j = ys[n], is_[n], js[n]
        r, g_, b = col[int(V[y, i, j])]
        sx = ox + (j - i) * c
        sy = oy + (i + j) * c // 2 - y * c
        if exp_top[y, i, j]:
            dr.polygon([(sx, sy - c), (sx + c, sy - c // 2), (sx, sy), (sx - c, sy - c // 2)], fill=(r, g_, b))
        if exp_s[y, i, j]:
            dr.polygon([(sx - c, sy - c // 2), (sx, sy), (sx, sy + c), (sx - c, sy + c // 2)],
                       fill=(int(r * .72), int(g_ * .72), int(b * .72)))
        if exp_e[y, i, j]:
            dr.polygon([(sx, sy), (sx + c, sy - c // 2), (sx + c, sy + c // 2), (sx, sy + c)],
                       fill=(int(r * .52), int(g_ * .52), int(b * .52)))
    return img


def main():
    out = sys.argv[1]
    prof = eval(sys.argv[2]) if len(sys.argv) > 2 else None
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ms = marian.load()
    m = marian.tile_model(ms, air_weights=prof,
                          group_weights={"flat_top": float(os.environ.get("FLAT_W", "1")),
                                         "roof": float(os.environ.get("ROOF_W", "1"))},
                          flat_air=float(os.environ.get("FLAT_AIR", "0")))
    solid_w = float(os.environ.get("SOLID_W", "1"))
    if solid_w != 1:                                           # Solid's weight on layers >= 1
        t = ms.names.index("Solid R0")
        lu = m.lu if m.lu is not None else np.zeros((m.L, m.T))
        lu[1:, t] += -np.log(solid_w)
        m.lu = lu
    bs = BlockSchedule.anneal(B=8, Sb=3, K=int(os.environ.get("K", "32")), shift=True, restart=False,
                              beta0=1.0, gamma0=2.0)
    g = generate(m, bs, seed, 0, 0, n, n)
    tpl = mm.module_templates(8)
    V = voxels(g, tpl)
    col = colours()
    col[len(col)] = (255, 0, 255)
    wild_id = max(col)
    V = V.astype(np.int32)
    V[voxels((g == m.T).astype(np.int32) * 0 + (g == m.T), np.stack([np.zeros((8, 8, 8), np.uint8),
                                                                    np.ones((8, 8, 8), np.uint8)])) > 0] = wild_id
    iso(V, col).save(out)
    names = np.array([n.rsplit(" ", 1)[0] for n in ms.names] + ["WILD"])
    top = {}
    for nm in names[g.ravel()]:
        top[nm] = top.get(nm, 0) + 1
    print("wrote", out, "wildcards", int((g == m.T).sum()), "of", g.size,
          "top modules", sorted(top.items(), key=lambda x: -x[1])[:8])


if __name__ == "__main__":
    main()
