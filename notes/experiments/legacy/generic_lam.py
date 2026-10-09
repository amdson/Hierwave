"""Stage 2 of castlegen.legacy.generic alone: promise grids for a sweep of the texture
coupling lam -> images/generic_lam.png (rows: texture, then witness(P) per lam).
args: CORPUS.npy CACHE_DIR (textures are cached there)."""
import os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, texsyn, tileset

corpus, cache = np.load(sys.argv[1]), sys.argv[2]
SEEDS, LEVELS, LAMS, n, tmp = (1, 2, 3), [16, 32, 64], (0.0, 1.0, 4.0, 16.0, 64.0), 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
con = GN.AverageFill(ts, 16)
tab = GN.fit_tables(con, corpus, LEVELS)
m = E.shape[1]
top_row, bot_row = E[0, np.arange(n) % m], E[-1, np.arange(n) % m]
top, bot = GN.boundary_values(con, top_row), GN.boundary_values(con, bot_row)
an = None
tex = {}
for sd in SEEDS:
    p = os.path.join(cache, f"tex_{sd}.npy")
    if not os.path.exists(p):
        an = an or texsyn.Analysis(ts, E, bounds="edge")
        np.save(p, GN.texture(an, n, sd))
    tex[sd] = np.load(p)


def top_block(P):
    """Highest block row (from the bottom, in cells) holding matter, per block column, averaged."""
    R = P.shape[0]
    has = con.SUB[P].sum(-1) > 0
    return float(np.mean([(R - np.argmax(c)) * 16 if c.any() else 0 for c in has.T]))


rows = {"texture": [tex[sd] for sd in SEEDS]}
for lam in LAMS:
    rows[f"lam {lam:g}"] = []
    for sd in SEEDS:
        R = n // 16
        tgt = GN.pyramid(con, con.leaf(tex[sd].reshape(R, 16, R, 16).transpose(0, 2, 1, 3)), LEVELS)
        P = GN.PromiseSampler(con, tab, LEVELS, tgt, top, bot, lam=lam, seed=sd).run(100)
        d = float(con.dist(P, tgt[0]).mean())
        print(f"lam {lam:5g} seed {sd}: L1 to texture per block {d:5.2f}   mean top {top_block(P):5.1f} "
              f"(texture {top_block(tgt[0]):5.1f})   fill {con.SUB[P].sum() / con.SUB[tgt[0]].sum():.2f}", flush=True)
        rows[f"lam {lam:g}"].append(con.witness(P, bot_row, top_row))
ims = {}
for k, xs in rows.items():
    ims[k] = []
    for x in xs:
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, x, p, 1)
        ims[k].append(Image.open(p).convert("RGB"))
w, h = ims["texture"][0].size
img = Image.new("RGB", (70 + len(SEEDS) * (w + 6), len(ims) * (h + 6)), "white")
dr = ImageDraw.Draw(img)
for r, (k, xs) in enumerate(ims.items()):
    dr.text((4, r * (h + 6) + h // 2), k, fill="black")
    for c, im in enumerate(xs):
        img.paste(im, (70 + c * (w + 6), r * (h + 6)))
img.save("images/generic_lam.png")
