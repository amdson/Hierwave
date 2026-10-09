"""Promise grids realised by the plain witness vs guided_witness driven by the
learned local energy alone (no hint): corpus promises and pseudo-likelihood
table samples (lam = 0) -> images/generic_render.png.  args: CORPUS.npy"""
import os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, tileset

corpus = np.load(sys.argv[1])
LEVELS, n, tmp = [16, 32, 64], 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
con = GN.AverageFill(ts, 16)
m = E.shape[1]
top_row, bot_row = E[0, np.arange(n) % m], E[-1, np.arange(n) % m]
top, bot = GN.boundary_values(con, top_row), GN.boundary_values(con, bot_row)
loc, rule = GN.fit_local(E, ts.n_sig), GN.support_rule(ts)
tab = GN.fit_tables_pl(con, corpus, LEVELS, top, bot, l2=1.0)
R = n // 16
zero = [np.zeros((R >> i, R >> i), np.int64) for i in range(len(LEVELS))]
grids = {"corpus": [con.leaf(x.reshape(R, 16, R, 16).transpose(0, 2, 1, 3)) for x in corpus[:4]],
         "PL tables": [GN.PromiseSampler(con, tab, LEVELS, zero, top, bot, lam=0.0, seed=sd).run(200) for sd in range(4)]}
rows = {}
for k, Ps in grids.items():
    rows[f"{k}, plain"] = [con.witness(P, bot_row, top_row) for P in Ps]
    rows[f"{k}, E_loc"] = [GN.guided_witness(con, P, None, rule, bot_row, top_row, np.random.default_rng(i), sweeps=6,
                                             local=loc) for i, P in enumerate(Ps)]
    print(k, "done", flush=True)
ims = {}
for k, xs in rows.items():
    ims[k] = []
    for x in xs:
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, x, p, 2)
        ims[k].append(Image.open(p).convert("RGB"))
w, h = ims[k][0].size
img = Image.new("RGB", (110 + 4 * (w + 8), len(ims) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (k, row) in enumerate(ims.items()):
    d.text((4, r * (h + 8) + h // 2), k.replace(", ", "\n"), fill="black")
    for c, im in enumerate(row):
        img.paste(im, (110 + c * (w + 8), r * (h + 8)))
img.save("images/generic_render.png")
