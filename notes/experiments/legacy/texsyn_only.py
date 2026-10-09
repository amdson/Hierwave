"""Unconstrained texture synthesis (args: [edge] -> sky / ground boundary
conditions, corrected from the top cell down) (the hier coordinate sampler with no promises) from the
cliffs_big exemplar, 6 seeds -> images/texsyn_only.png."""
import os, tempfile
import numpy as np
from PIL import Image
from castlegen.legacy import exemplar as ex, pipeline as PL, tileset

ts, tmp = tileset.load("cliffs"), tempfile.mkdtemp()
spec = {"name": "texsyn_only", "tileset": "cliffs", "exemplar": "cliffs_big", "size": 128, "seed": 0,
        "stages": [{"stage": "hier", "K": 16, "h_top": 64, "tables": "flat", "quantities": [], "lam": 0.0,
                    "T": 1.0, "sweeps": 3, "top_sweeps": 10, "kappa": 4.0, "r": [1, 1, 0, 0, 0, 0, 0],
                    "first_corrected": 3}]}
import sys
if sys.argv[1:] == ["edge"]:
    spec["stages"][0].update(bounds="edge", first_corrected=1)
out = "images/texsyn_edge.png" if sys.argv[1:] == ["edge"] else "images/texsyn_only.png"
ims = []
for sd in range(1, 7):
    t = PL.run(spec, seed=sd, log=lambda *a: None)[1]
    p = os.path.join(tmp, f"{sd}.png")
    ex.to_png(ts, np.asarray(t, np.int32), p, 2)
    ims.append(Image.open(p).convert("RGB"))
w, h = ims[0].size
img = Image.new("RGB", (3 * (w + 8), 2 * (h + 8)), "white")
for i, im in enumerate(ims):
    img.paste(im, ((i % 3) * (w + 8), (i // 3) * (h + 8)))
img.save(out)
