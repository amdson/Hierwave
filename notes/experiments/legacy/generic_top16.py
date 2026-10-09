"""Texture synthesis started at 16 px cells (8 x 8 independent uniform exemplar
coordinates) vs the map-sized top cell, and the full generic pipeline on the
16 px textures -> images/generic_top16.png.  args: CORPUS.npy"""
import itertools, os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, texsyn, tileset

corpus = np.load(sys.argv[1])
SEEDS, LEVELS, n, tmp = (1, 2, 3, 4), [16, 32, 64], 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
con = GN.AverageFill(ts, 16)
tab = GN.fit_tables(con, corpus, LEVELS)
an = texsyn.Analysis(ts, E, bounds="edge")
rows = {"texture, top 128": [GN.texture(an, n, sd) for sd in SEEDS]}
rows["texture, top 16"] = [GN.texture(an, n, sd, h_top=16) for sd in SEEDS]
rows["generic, top 16"] = [GN.generate(ts, E, con, tab, LEVELS, an, n, sd, x_tex=t, sweeps=50)["x"]
                           for sd, t in zip(SEEDS, rows["texture, top 16"])]
dis = lambda a, b: float((ts.solid[a] != ts.solid[b]).mean())
for k, xs in rows.items():
    print(f"{k:18s} pairwise solidity disagreement {np.mean([dis(a, b) for a, b in itertools.combinations(xs, 2)]):.3f}"
          f"   solid {np.mean([ts.solid[x].mean() for x in xs]):.2f}", flush=True)
ims = {}
for k, xs in rows.items():
    ims[k] = []
    for x in xs:
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, x, p, 2)
        ims[k].append(Image.open(p).convert("RGB"))
w, h = ims[k][0].size
img = Image.new("RGB", (110 + len(SEEDS) * (w + 8), len(ims) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (k, row) in enumerate(ims.items()):
    d.text((4, r * (h + 8) + h // 2), k.replace(", ", "\n"), fill="black")
    for c, im in enumerate(row):
        img.paste(im, (110 + c * (w + 8), r * (h + 8)))
img.save("images/generic_top16.png")
