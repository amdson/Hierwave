"""No-promise texture synthesis of a procedural tree (branches that split,
leaf clumps only at the tips): the plain coordinate MRF (use_v=False, no loc)
through the joint sampler's levels, as joint_multi.py but promise-free.
Exemplar m x m (bounds="edge": sky above, soil below); output n x n.
-> images/tree_exemplar.png, images/tree_texsyn.png"""
import os, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, tileset

m, n = int(os.environ.get("M", 64)), int(os.environ.get("N", 128))
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1,2,3").split(","))
PLAN = [(16, 16), (8, 8), (4, 4), (2, 2), (1, 4)]
SKY, BARK, LEAF, DARK, GRASS, SOIL = range(6)
ts = tileset.load("tree")


def tree(m, seed=0):
    rng = np.random.default_rng(seed)
    A = np.full((m, m), SKY, np.int32)
    g = int(m * 0.86)
    A[g:] = SOIL
    A[g:g + 2] = GRASS
    yy, xx = np.mgrid[:m, :m]

    def seg(y0, x0, y1, x1, w):
        d = np.array([y1 - y0, x1 - x0]); L2 = max(d @ d, 1e-9)
        t = np.clip(((yy - y0) * d[0] + (xx - x0) * d[1]) / L2, 0, 1)
        A[(yy - y0 - t * d[0]) ** 2 + (xx - x0 - t * d[1]) ** 2 <= (w / 2) ** 2] = BARK

    tips = []

    def grow(y, x, ang, L, w, depth):
        y1, x1 = y - L * np.cos(ang), x + L * np.sin(ang)
        seg(y, x, y1, x1, w)
        if depth == 0:
            tips.append((y1, x1)); return
        for s in (-1, 1):
            grow(y1, x1, ang + s * rng.uniform(0.35, 0.6), L * rng.uniform(0.66, 0.78), max(w * 0.62, 1.0), depth - 1)

    grow(g + 1, m / 2, 0.0, m * 0.2, m / 14, int(os.environ.get("DEPTH", 5)))
    for y, x in tips:                                                  # bushy clumps at the tips only
        r = m / 40 * rng.uniform(0.9, 1.2)
        for _ in range(4):
            cy, cx, rr = y + rng.normal(0, r / 2), x + rng.normal(0, r / 2), r * rng.uniform(0.5, 0.8)
            A[(yy - cy) ** 2 + (xx - cx) ** 2 <= rr ** 2] = LEAF
        dk = ((yy - y - r * 0.3) ** 2 + (xx - x) ** 2 <= (0.6 * r) ** 2) & (A == LEAF)
        A[dk] = DARK
    return A


E = tree(m)
tmp = tempfile.mkdtemp()
p = os.path.join(tmp, "x.png")
ex.to_png(ts, E, p, 4)
Image.open(p).save("images/tree_exemplar.png")
solid = np.asarray(ts.solid, bool)
t0 = time.time()
an = GN.analysis(ts, E, [h for h, _ in PLAN])
print(f"analysis {time.time() - t0:.0f}s", flush=True)
SKYR = int(os.environ.get("SKYR", 0))                               # mask sky farther than SKYR from non-sky
keep = an.E != SKY
for _ in range(SKYR):
    k = keep.copy()
    k[1:] |= keep[:-1]; k[:-1] |= keep[1:]
    keep = k | np.roll(k, 1, 1) | np.roll(k, -1, 1)
bad = np.where(keep.ravel(), 0.0, 1e3) if SKYR else None
print(f"masked coords {0 if bad is None else (bad > 0).mean():.2f}")
out, Up = {sd: None for sd in SEEDS}, {sd: None for sd in SEEDS}
for h, sweeps in PLAN:
    lvl = GN.JointLevel(an, solid, h, n, lam_c=16, r=0.25, knn=16, use_v=False,
                        sigma=None if h == PLAN[0][0] else h / 2, fast=True)
    if bad is not None:
        lvl.extra = lambda py, px, allc: bad[allc]
    for sd in SEEDS:
        lvl.rng = np.random.default_rng(sd * 100 + h)
        lvl.PX = lvl.rng.integers(0, lvl.m, lvl.PX.shape)
        if Up[sd] is None:
            lvl.init(np.random.default_rng(sd).integers(0, an.my * an.m, (lvl.R, lvl.C)))
            lvl.run(sweeps, T=1.0, T_hot=10.0)
        else:
            lvl.init(lvl.prolong(Up[sd]), parent=Up[sd])
            lvl.run(sweeps, T=1.0)
        Up[sd] = lvl.U.copy()
        out[sd] = lvl.tiles()
        print(f"h {h} seed {sd}: {lvl.energy_parts()}", flush=True)
    del lvl
px, w = 3, n * 3
img = Image.new("RGB", (len(SEEDS) * (w + 8), w), "white")
for i, sd in enumerate(SEEDS):
    ex.to_png(ts, out[sd], p, px)
    img.paste(Image.open(p).convert("RGB"), (i * (w + 8), 0))
img.save("images/tree_texsyn.png")
