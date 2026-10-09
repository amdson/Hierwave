"""castlegen.legacy.generic end to end on cliffs_big, three seeds -> images/generic_guided.png
(rows: texture, witness start, tile chain).  args: CORPUS.npy [sweeps] [coord_T]."""
import os, sys, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, pipeline as PL, texsyn, tileset

corpus = np.load(sys.argv[1])
SWEEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
SEEDS, LEVELS, tmp = (1, 2, 3), [16, 32, 64], tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
con = GN.AverageFill(ts, 16)
t0 = time.time()
an = texsyn.Analysis(ts, E, bounds="edge")
tab = GN.fit_tables(con, corpus, LEVELS)
print(f"analysis + tables {time.time() - t0:.0f}s", flush=True)


def stats(x):
    solid = ts.solid[x]
    return f"solid {solid.mean():.2f}  rows with matter {solid.any(1).sum()}"


rows = {"texture": [], "witness": [], "generic": []}
for sd in SEEDS:
    t0 = time.time()
    out = GN.generate(ts, E, con, tab, LEVELS, an, 128, sd, coord_T=CT, sweeps=SWEEPS)
    print(f"seed {sd} ({time.time() - t0:.0f}s): texture {stats(out['tex'])} | final {stats(out['x'])}", flush=True)
    for k, key in (("texture", "tex"), ("witness", "start"), ("generic", "x")):
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, out[key], p, 2)
        rows[k].append(Image.open(p).convert("RGB"))
w, h = rows["texture"][0].size
img = Image.new("RGB", (90 + len(SEEDS) * (w + 8), len(rows) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (name, ims) in enumerate(rows.items()):
    d.text((4, r * (h + 8) + h // 2), name, fill="black")
    for c, im in enumerate(ims):
        img.paste(im, (90 + c * (w + 8), r * (h + 8)))
img.save("images/generic_guided.png")
