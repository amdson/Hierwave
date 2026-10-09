"""Sanity check of the surface-class promise (SurfaceClass: E / S / P / F per
sub-column) against AverageFill: from the same rule-aware texture, P =
leaf(texture), TileChain for 0 and 10 sweeps -> images/generic_surface.png
(rows: texture, then 10 sweeps under each promise)."""
import os, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, texsyn, tileset

SEEDS, SW, n, tmp = (1, 2, 3, 4), 10, 128, tempfile.mkdtemp()
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
rule, loc = GN.support_rule(ts), GN.fit_local(E, ts.n_sig)
an = texsyn.Analysis(ts, E, bounds="edge")
empty = E[0, np.arange(n) % E.shape[1]]
solid = np.asarray(ts.solid, bool)
cons = {"AverageFill": GN.AverageFill(ts, 16), "SurfaceClass": GN.SurfaceClass(ts, 16)}
rows = {"texture": []} | {f"{k} {SW} sw": [] for k in cons}
R = n // 16
for sd in SEEDS:
    x = GN.texture(an, n, sd, h_top=32, rule=rule, gamma=100.0)
    for y in range(n - 2, -1, -1):                                       # drop any unsupported matter
        bad = solid[x[y]] & ~solid[x[y + 1]]
        x[y, bad] = empty[bad]
    rows["texture"].append(x)
    for k, con in cons.items():
        P = con.leaf(x.reshape(R, 16, R, 16).transpose(0, 2, 1, 3))
        ch = GN.TileChain(ts, x, E, [con], [P], rule, {4: 5.12, 8: 20.48, 16: 81.92}, 0.9, seed=sd, local=loc)
        e0 = ch.energy_ex() + ch.energy_local()
        for _ in range(SW):
            ch.sweep()
        assert ch.valid()
        print(f"seed {sd} {k:12s} E {e0:9.1f} -> {ch.energy_ex() + ch.energy_local():9.1f}   "
              f"changed {(ch.t != x).mean():.3f}   solid changed {(solid[ch.t] != solid[x]).mean():.3f}", flush=True)
        rows[f"{k} {SW} sw"].append(ch.t.copy())
ims = {}
for k, xs in rows.items():
    ims[k] = []
    for x in xs:
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, x, p, 2)
        ims[k].append(Image.open(p).convert("RGB"))
w, h = ims["texture"][0].size
img = Image.new("RGB", (130 + len(SEEDS) * (w + 8), len(ims) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (k, row) in enumerate(ims.items()):
    d.text((4, r * (h + 8) + h // 2), k, fill="black")
    for c, im in enumerate(row):
        img.paste(im, (130 + c * (w + 8), r * (h + 8)))
img.save("images/generic_surface.png")
