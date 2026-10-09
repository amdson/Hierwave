"""Texture synthesis with the seam rule penalty (RuleCoordVar, gamma) vs
without, 32 px start, and the full generic pipeline on the penalised
textures (pseudo-likelihood tables) -> images/generic_rule.png.
args: CORPUS.npy [gamma]"""
import itertools, os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, texsyn, tileset

corpus = np.load(sys.argv[1])
G = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
SEEDS, LEVELS, LAMS, n, tmp = (1, 2, 3, 4), [16, 32, 64], (4.0,), 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
con, rule = GN.AverageFill(ts, 16), GN.support_rule(ts)
m = E.shape[1]
top = GN.boundary_values(con, E[0, np.arange(n) % m])
bot = GN.boundary_values(con, E[-1, np.arange(n) % m])
tab = GN.fit_tables_pl(con, corpus, LEVELS, top, bot, l2=1.0)
an = texsyn.Analysis(ts, E, bounds="edge")
rows = {"texture, gamma 0": [GN.texture(an, n, sd, h_top=32) for sd in SEEDS],
        f"texture, gamma {G:g}": [GN.texture(an, n, sd, h_top=32, rule=rule, gamma=G) for sd in SEEDS]}
for lam in LAMS:
    rows[f"lam {lam:g}"] = [GN.generate(ts, E, con, tab, LEVELS, an, n, sd, lam=lam, x_tex=t, sweeps=50)["x"]
                            for sd, t in zip(SEEDS, rows[f"texture, gamma {G:g}"])]


def needles(x):
    H = ts.solid[x].sum(0)
    return int(((H - np.roll(H, 1) > 8) & (H - np.roll(H, -1) > 8)).sum())


dis = lambda a, b: float((ts.solid[a] != ts.solid[b]).mean())
for k, xs in rows.items():
    print(f"{k:18s} solid {np.mean([ts.solid[x].mean() for x in xs]):.2f}   pairwise disagreement "
          f"{np.mean([dis(a, b) for a, b in itertools.combinations(xs, 2)]):.3f}   needles {[needles(x) for x in xs]}",
          flush=True)
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
img.save("images/generic_rule.png")
