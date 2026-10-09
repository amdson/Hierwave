"""The current recipe (32 px-start texture with the seam penalty, gamma = 100;
pseudo-likelihood tables, lam = 4; guided witness) with snapshots of the
tile chain after several sweep counts -> images/generic_sweeps.png (rows:
texture, then the map after each sweep count, 0 = the guided witness).
args: CORPUS.npy"""
import os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen import exemplar as ex, generic as GN, texsyn, tileset

corpus = np.load(sys.argv[1])
SEEDS, LEVELS, SNAP, n, tmp = (1, 2, 3, 4, 5, 6), [16, 32, 64], (0, 5, 20, 50, 200), 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/exemplars/cliffs_big.npy").astype(np.int32)
con, rule = GN.AverageFill(ts, 16), GN.support_rule(ts)
m = E.shape[1]
top_row, bot_row = E[0, np.arange(n) % m], E[-1, np.arange(n) % m]
top, bot = GN.boundary_values(con, top_row), GN.boundary_values(con, bot_row)
tab = GN.fit_tables_pl(con, corpus, LEVELS, top, bot, l2=1.0)
loc = GN.fit_local(E, ts.n_sig)
an = texsyn.Analysis(ts, E, bounds="edge")
rows = {"texture": []} | {f"{s} sweeps": [] for s in SNAP}
for sd in SEEDS:
    tex = GN.texture(an, n, sd, h_top=32, rule=rule, gamma=100.0)
    R = n // 16
    tgt = GN.pyramid(con, con.leaf(tex.reshape(R, 16, R, 16).transpose(0, 2, 1, 3)), LEVELS)
    P = GN.PromiseSampler(con, tab, LEVELS, tgt, top, bot, lam=4.0, seed=sd).run(100)
    x0 = GN.guided_witness(con, P, tex, rule, bot_row, top_row, np.random.default_rng(sd), 0.1, local=loc)
    ch = GN.TileChain(ts, x0, E, [con], [P], rule, {4: 5.12, 8: 20.48, 16: 81.92}, 0.9, seed=sd, local=loc)
    rows["texture"].append(tex)
    for s in range(max(SNAP) + 1):
        if s in SNAP:
            rows[f"{s} sweeps"].append(ch.t.copy())
        if s < max(SNAP):
            ch.sweep()
    assert ch.valid()
    print(f"seed {sd} done", flush=True)
ims = {}
for k, xs in rows.items():
    ims[k] = []
    for x in xs:
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, x, p, 2)
        ims[k].append(Image.open(p).convert("RGB"))
w, h = ims["texture"][0].size
img = Image.new("RGB", (90 + len(SEEDS) * (w + 8), len(ims) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (k, row) in enumerate(ims.items()):
    d.text((4, r * (h + 8) + h // 2), k, fill="black")
    for c, im in enumerate(row):
        img.paste(im, (90 + c * (w + 8), r * (h + 8)))
img.save("images/generic_sweeps.png")
