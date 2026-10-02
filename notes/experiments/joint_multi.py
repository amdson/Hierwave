"""Full multi-level run of the joint sampler (notes/joint_sampler.md): h = 16
from a uniform start (annealed), then h = 8, 4, 2, 1, each initialised at the
parent's prolongation and sampled with E_par (sigma = h / 2 tiles) toward it;
h = 1 adds E_loc (the tile set's energy) and its promises are the solid bits,
so the output satisfies the support rule exactly.  Batched kNN MH (K = 16;
FAST=0: the single-cell reference; FAST=numba: numba E_c), lam_c = 16, r = 0.25.  Rows: seeds;
columns: the pasted windows at each level -> images/joint_multi.png.  Also
counts support violations of each level's pasted windows."""
import os, sys, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen import exemplar as ex, generic as GN, texsyn, tileset

n, px = 128, 2
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1,2").split(","))
PLAN = [(16, 16), (8, 8), (4, 4), (2, 2), (1, 2)]                                  # (h, sweeps)
ts = tileset.load("cliffs")
solid = np.asarray(ts.solid, bool)
E = np.load("castlegen/exemplars/cliffs_big.npy").astype(np.int32)
t0 = time.time()
an = GN.analysis(ts, E, [h for h, _ in PLAN])
print(f"analysis {time.time() - t0:.0f}s", flush=True)
out = {sd: [] for sd in SEEDS}
Up = {sd: None for sd in SEEDS}
for h, sweeps in PLAN:
    t0 = time.time()
    lvl = GN.JointLevel(an, solid, h, n, lam_c=16, r=0.25, knn=16, sigma=None if h == PLAN[0][0] else h / 2,
                        loc=ts, fast={"0": False, "1": True}.get(os.environ.get("FAST", "1"), os.environ.get("FAST")))
    tb = time.time() - t0
    for sd in SEEDS:
        t1 = time.time()
        lvl.rng = np.random.default_rng(sd * 100 + h)
        lvl.PX = lvl.rng.integers(0, lvl.m, lvl.PX.shape)
        if Up[sd] is None:
            lvl.init(np.random.default_rng(sd).integers(0, an.my * an.m, (lvl.R, lvl.C)))
            lvl.run(sweeps, T=1.0, T_hot=10.0)
        else:
            lvl.init(lvl.prolong(Up[sd]), parent=Up[sd])
            lvl.run(sweeps, T=1.0)
        Up[sd] = lvl.U.copy()
        x = lvl.tiles()
        s = solid[x]
        bad = int((s[:-1] & ~s[1:]).sum())                                  # solid over non-solid
        ep = lvl.energy_parts()
        print(f"h {h:2d} seed {sd}: build {tb:.0f}s  run {time.time() - t1:.0f}s  E {sum(ep.values()):.0f} "
              f"(lam E_c {ep['c']:.0f}, E_par {ep['par']:.0f}, E_loc {ep['loc']:.0f}, cost {ep['cost']:.0f})  "
              f"valid {lvl.valid()}  support violations {bad}", flush=True)
        out[sd].append(x)
    del lvl
p, w = os.path.join(tempfile.mkdtemp(), "x.png"), n * px
img = Image.new("RGB", (60 + len(PLAN) * (w + 8), 20 + len(SEEDS) * (w + 8)), "white")
d = ImageDraw.Draw(img)
for j, (h, k) in enumerate(PLAN):
    d.text((60 + j * (w + 8) + w // 2 - 30, 4), f"h {h}, {k} sweeps", fill="black")
for i, sd in enumerate(SEEDS):
    d.text((4, 20 + i * (w + 8) + w // 2), f"seed {sd}", fill="black")
    for j, x in enumerate(out[sd]):
        ex.to_png(ts, x, p, px)
        img.paste(Image.open(p).convert("RGB"), (60 + j * (w + 8), 20 + i * (w + 8)))
img.save("images/joint_multi.png")
