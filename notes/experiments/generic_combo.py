"""32 px-start textures + pseudo-likelihood promise tables, for a sweep of the
texture pull lam -> images/generic_combo.png (rows: texture, then the output
per lam).  args: CORPUS.npy"""
import itertools, os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen import exemplar as ex, generic as GN, texsyn, tileset

corpus = np.load(sys.argv[1])
SEEDS, LEVELS, LAMS, n, tmp = (1, 2, 3, 4), [16, 32, 64], (0.25, 0.5, 1.0, 4.0), 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/exemplars/cliffs_big.npy").astype(np.int32)
con = GN.AverageFill(ts, 16)
m = E.shape[1]
top = GN.boundary_values(con, E[0, np.arange(n) % m])
bot = GN.boundary_values(con, E[-1, np.arange(n) % m])
tab = GN.fit_tables_pl(con, corpus, LEVELS, top, bot, l2=1.0)
an = texsyn.Analysis(ts, E, bounds="edge")
rows = {"texture, top 32": [GN.texture(an, n, sd, h_top=32) for sd in SEEDS]}
for lam in LAMS:
    rows[f"lam {lam:g}"] = [GN.generate(ts, E, con, tab, LEVELS, an, n, sd, lam=lam, x_tex=t, sweeps=50)["x"]
                            for sd, t in zip(SEEDS, rows["texture, top 32"])]
dis = lambda a, b: float((ts.solid[a] != ts.solid[b]).mean())
for k, xs in rows.items():
    print(f"{k:16s} solid {np.mean([ts.solid[x].mean() for x in xs]):.2f}   pairwise disagreement "
          f"{np.mean([dis(a, b) for a, b in itertools.combinations(xs, 2)]):.3f}   vs texture "
          f"{np.mean([dis(a, b) for a, b in zip(xs, rows['texture, top 32'])]):.3f}", flush=True)
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
img.save("images/generic_combo.png")
