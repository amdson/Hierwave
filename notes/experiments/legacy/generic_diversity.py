"""Diversity of castlegen.legacy.generic: per texture seed, several stage 2-3 seeds
-> images/generic_diversity.png (rows: texture seed; columns: texture, then
one output per sub-seed).  Prints mean pairwise solidity disagreement within a
texture (stages 2-3 alone) and across textures.  args: CORPUS.npy CACHE_DIR."""
import itertools, os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, texsyn, tileset

corpus, cache = np.load(sys.argv[1]), sys.argv[2]
TEX, SUB, LEVELS, n, tmp = (1, 2, 3), (10, 11, 12), [16, 32, 64], 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
con = GN.AverageFill(ts, 16)
tab = GN.fit_tables(con, corpus, LEVELS)
an = None
out = {}
for sd in TEX:
    p = os.path.join(cache, f"tex_{sd}.npy")
    if not os.path.exists(p):
        an = an or texsyn.Analysis(ts, E, bounds="edge")
        np.save(p, GN.texture(an, n, sd))
    tex = np.load(p)
    out[sd] = [tex] + [GN.generate(ts, E, con, tab, LEVELS, None, n, sd, x_tex=tex, sub_seed=ss, sweeps=50)["x"]
                       for ss in SUB]
    print(f"texture {sd} done", flush=True)
dis = lambda a, b: float((ts.solid[a] != ts.solid[b]).mean())
within = [dis(a, b) for sd in TEX for a, b in itertools.combinations(out[sd][1:], 2)]
across = [dis(out[a][1], out[b][1]) for a, b in itertools.combinations(TEX, 2)]
tex_final = [dis(out[sd][0], x) for sd in TEX for x in out[sd][1:]]
print(f"solidity disagreement: same texture {np.mean(within):.3f}  different textures {np.mean(across):.3f}  "
      f"output vs its texture {np.mean(tex_final):.3f}")
ims = []
for sd in TEX:
    row = []
    for x in out[sd]:
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, x, p, 2)
        row.append(Image.open(p).convert("RGB"))
    ims.append(row)
w, h = ims[0][0].size
img = Image.new("RGB", (len(ims[0]) * (w + 8), 20 + len(ims) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for c, lab in enumerate(["texture"] + [f"sub-seed {s}" for s in SUB]):
    d.text((c * (w + 8) + 4, 4), lab, fill="black")
for r, row in enumerate(ims):
    for c, im in enumerate(row):
        img.paste(im, (c * (w + 8), 20 + r * (h + 8)))
img.save("images/generic_diversity.png")
