"""Top level (h = 16, 8 x 8 cells on 128^2) after 1, 4, 8, 16 sweeps: separate
runs from the same uniform start, each annealed T 10 -> 1 over its first
half (JointLevel.run); promises on, kNN MH (K = 16), lam_c = 16, r = 0.25.
Rows: seeds; columns: sweeps -> images/joint_sweeps.png."""
import os, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, texsyn, tileset

h, n, px, SEEDS, SWEEPS = 16, 128, 2, (1, 2, 3, 4), (1, 4, 8, 16)
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
an = texsyn.Analysis(ts, E, bounds="edge", pad=2 * h)
lvl = GN.JointLevel(an, np.asarray(ts.solid, bool), h, n, lam_c=16, r=0.25, knn=16)
p, w = os.path.join(tempfile.mkdtemp(), "x.png"), n * px
img = Image.new("RGB", (60 + len(SWEEPS) * (w + 8), 20 + len(SEEDS) * (w + 8)), "white")
d = ImageDraw.Draw(img)
for j, k in enumerate(SWEEPS):
    d.text((60 + j * (w + 8) + w // 2 - 20, 4), f"{k} sweeps", fill="black")
for i, sd in enumerate(SEEDS):
    d.text((4, 20 + i * (w + 8) + w // 2), f"seed {sd}", fill="black")
    U0 = np.random.default_rng(sd).integers(0, an.my * an.m, (lvl.R, lvl.C))
    for j, k in enumerate(SWEEPS):
        lvl.rng = np.random.default_rng(sd)
        lvl.PX = lvl.rng.integers(0, lvl.m, lvl.PX.shape)
        lvl.init(U0.copy())
        lvl.run(k, T=1.0, T_hot=10.0)
        print(f"seed {sd} sweeps {k:2d}  E {lvl.energy():.0f}  valid {lvl.valid()}", flush=True)
        ex.to_png(ts, lvl.tiles(), p, px)
        img.paste(Image.open(p).convert("RGB"), (60 + j * (w + 8), 20 + i * (w + 8)))
img.save("images/joint_sweeps.png")
