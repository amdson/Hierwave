"""Promise tables: counts (fit_tables) vs pseudo-likelihood (fit_tables_pl).
Held-out pseudo-NLL per block (12 train / 4 test corpus maps), and promise
grids sampled from the tables alone (lam = 0) vs the corpus: mean fill and
mean top height -> images/generic_tables.png (rows: corpus, counts, PL).
args: CORPUS.npy"""
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
train, test = corpus[:12], corpus[12:]
cnt = GN.fit_tables(con, train, LEVELS)
print(f"counts: held-out pseudo-NLL/block {GN.pl_nll(con, cnt, test, LEVELS, top, bot):.3f}", flush=True)
best = None
for l2 in (0.3, 1.0, 3.0, 10.0):
    t = GN.fit_tables_pl(con, train, LEVELS, top, bot, l2=l2, log=print)
    v = GN.pl_nll(con, t, test, LEVELS, top, bot)
    print(f"PL l2 {l2:g}: held-out pseudo-NLL/block {v:.3f}", flush=True)
    if best is None or v < best[0]:
        best = (v, l2)
l2 = best[1]
tabs = {"counts": GN.fit_tables(con, corpus, LEVELS), f"PL l2={l2:g}": GN.fit_tables_pl(con, corpus, LEVELS, top, bot, l2=l2)}
R = n // 16
leafs = [con.leaf(x.reshape(R, 16, R, 16).transpose(0, 2, 1, 3)) for x in corpus]


def top_h(P):
    has = con.SUB[P].sum(-1) > 0
    return float(np.mean([(P.shape[0] - np.argmax(c)) * 16 if c.any() else 0 for c in has.T]))


fill = lambda P: con.SUB[P].sum() / (2 * con.w * P.size)
print(f"corpus: fill {np.mean([fill(P) for P in leafs]):.2f} ± {np.std([fill(P) for P in leafs]):.2f}   "
      f"top {np.mean([top_h(P) for P in leafs]):5.1f} ± {np.std([top_h(P) for P in leafs]):4.1f}")
rows = {"corpus": [con.witness(P, bot_row, top_row) for P in leafs[:4]]}
zero = [np.zeros((R >> i, R >> i), np.int64) for i in range(len(LEVELS))]
for k, t in tabs.items():
    Ps = [GN.PromiseSampler(con, t, LEVELS, zero, top, bot, lam=0.0, seed=sd).run(200) for sd in range(8)]
    print(f"{k:12s}: fill {np.mean([fill(P) for P in Ps]):.2f} ± {np.std([fill(P) for P in Ps]):.2f}   "
          f"top {np.mean([top_h(P) for P in Ps]):5.1f} ± {np.std([top_h(P) for P in Ps]):4.1f}", flush=True)
    rows[k] = [con.witness(P, bot_row, top_row) for P in Ps[:4]]
ims = {}
for k, xs in rows.items():
    ims[k] = []
    for x in xs:
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, x, p, 1)
        ims[k].append(Image.open(p).convert("RGB"))
w, h = ims["corpus"][0].size
img = Image.new("RGB", (90 + 4 * (w + 6), len(ims) * (h + 6)), "white")
d = ImageDraw.Draw(img)
for r, (k, row) in enumerate(ims.items()):
    d.text((4, r * (h + 6) + h // 2), k, fill="black")
    for c, im in enumerate(row):
        img.paste(im, (90 + c * (w + 6), r * (h + 6)))
img.save("images/generic_tables.png")
