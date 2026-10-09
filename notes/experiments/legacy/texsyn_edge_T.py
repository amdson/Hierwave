"""Bounded (sky / ground) texture synthesis with sampled coordinates: rows =
coord_T values, columns = seeds -> images/texsyn_edge_T.png."""
import os, sys, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, pipeline as PL, tileset

ts, tmp = tileset.load("cliffs"), tempfile.mkdtemp()
TS, SEEDS = [float(a) for a in sys.argv[1:]] or [0.1, 1.0, 10.0], (1, 2, 3)
rows = []
for T in TS:
    ims = []
    for sd in SEEDS:
        spec = {"name": "texsyn_edge", "tileset": "cliffs", "exemplar": "cliffs_big", "size": 128, "seed": 0,
                "stages": [{"stage": "hier", "K": 16, "h_top": 64, "tables": "flat", "quantities": [], "lam": 0.0,
                            "T": 1.0, "sweeps": 3, "top_sweeps": 10, "kappa": 4.0, "r": [1, 1, 0, 0, 0, 0, 0],
                            "first_corrected": 1, "bounds": "edge", "coord_T": T}]}
        t0 = time.time()
        t = PL.run(spec, seed=sd, log=lambda *a: None)[1]
        print(f"T {T} seed {sd}: {time.time() - t0:.0f}s", flush=True)
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, np.asarray(t, np.int32), p, 2)
        ims.append(Image.open(p).convert("RGB"))
    rows.append(ims)
w, h = rows[0][0].size
img = Image.new("RGB", (60 + len(SEEDS) * (w + 8), len(TS) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (T, ims) in enumerate(zip(TS, rows)):
    d.text((4, r * (h + 8) + h // 2), f"T={T:g}", fill="black")
    for c, im in enumerate(ims):
        img.paste(im, (60 + c * (w + 8), r * (h + 8)))
img.save("images/texsyn_edge_T.png")
