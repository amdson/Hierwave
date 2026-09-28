"""Bounded (sky / ground) texture synthesis at coord_T, alone vs with the
average promises (linear tables) and the tile stage; same seeds ->
images/edge_promises[_FEAT].png (args: CORPUS.npy [coord_T] [feat: solid | tiles])."""
import copy, os, sys, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen import exemplar as ex, pipeline as PL, refheights as RH, tileset
from castlegen.quantities import envelope as EN

corpus = sys.argv[1]
CT = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
FEAT = sys.argv[3] if len(sys.argv) > 3 else "solid"
ts, G, SEEDS, tmp = tileset.load("cliffs"), 8, (1, 2, 3), tempfile.mkdtemp()
hier = {"stage": "hier", "K": 16, "h_top": 64, "lam": 1.0, "T": 1.0, "sweeps": 3, "top_sweeps": 10, "kappa": 4.0,
        "r": [1, 1, 0, 0, 0, 0, 0], "first_corrected": 1, "bounds": "edge", "coord_T": CT}
specs = {
    "texsyn only": {"name": "e", "tileset": "cliffs", "exemplar": "cliffs_big", "size": 128, "seed": 0,
                    "stages": [dict(hier, tables="flat", quantities=[], lam=0.0)]},
    "promises": {"name": "e", "tileset": "cliffs", "exemplar": "cliffs_big", "size": 128, "seed": 0,
                 "stages": [dict(hier, tables="linear", quantities=["average"], average={"corpus_path": corpus}),
                            {"stage": "avgexheights", "w": {"4": 5.12, "8": 20.48, "16": 81.92}, "T": 0.9, "sweeps": 20, "feat": FEAT}]},
}


def stats(t):
    H = RH.heights(ts, t, G).astype(int)
    unsupported = int(EN.effective_solid(ts, t, G).sum() - H.sum())      # solid cells not stacked on the ground
    needles = int(((H - np.roll(H, 1) > 12) & (H - np.roll(H, -1) > 12)).sum())
    return f"H mean {H.mean():5.1f} std {H.std():5.1f}  unsupported cells {unsupported:5d}  needles {needles}"


rows = []
for name, spec in specs.items():
    ims = []
    for sd in SEEDS:
        t0 = time.time()
        t = np.asarray(PL.run(copy.deepcopy(spec), seed=sd, log=lambda *a: None)[1], np.int32)
        print(f"{name:12s} seed {sd} ({time.time() - t0:.0f}s): {stats(t)}", flush=True)
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, t, p, 2)
        ims.append(Image.open(p).convert("RGB"))
    rows.append((name, ims))
w, h = rows[0][1][0].size
img = Image.new("RGB", (90 + len(SEEDS) * (w + 8), len(rows) * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (name, ims) in enumerate(rows):
    d.text((4, r * (h + 8) + h // 2), f"{name}\nT={CT:g}", fill="black")
    for c, im in enumerate(ims):
        img.paste(im, (90 + c * (w + 8), r * (h + 8)))
img.save("images/edge_promises.png" if FEAT == "solid" else f"images/edge_promises_{FEAT}.png")
