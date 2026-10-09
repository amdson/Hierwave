"""Maps from the average-promise pipeline with count vs linear tables, same
seeds -> OUT.png (args: CORPUS.npy OUT.png)."""
import copy, os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen import exemplar as ex, pipeline as PL, refheights as RH, tileset

corpus, out = sys.argv[1], sys.argv[2]
ts, G, SEEDS = tileset.load("cliffs"), 8, (1, 2, 3)
tmp = tempfile.mkdtemp()


def spec(tables):
    s = copy.deepcopy(PL.load_spec("cliffs_envelope"))
    s["exemplar"] = "cliffs_big"
    s["stages"][0].update(first_corrected=3, quantities=["average"], tables=tables, average={"corpus_path": corpus})
    s["stages"][1] = {"stage": "avgexheights", "w": {"4": 5.12, "8": 20.48, "16": 81.92}, "T": 0.9, "sweeps": 20}
    return s


def needles(t):
    H = RH.heights(ts, t, G).astype(int)
    return int(((H - np.roll(H, 1) > 12) & (H - np.roll(H, -1) > 12)).sum())


rows = []
for tables in ("reference", "linear"):
    ims = []
    for sd in SEEDS:
        t = PL.run(spec(tables), seed=sd, log=lambda *a: None)[1]
        H = RH.heights(ts, t, G)
        print(f"{'counts' if tables == 'reference' else 'linear':7s} seed {sd}: H mean {H.mean():5.1f} std {H.std():5.1f} "
              f"|slope| {np.abs(np.diff(H)).mean():4.2f} needles {needles(t)}", flush=True)
        p = os.path.join(tmp, "x.png")
        ex.to_png(ts, np.asarray(t, np.int32), p, 2)
        ims.append(Image.open(p).convert("RGB"))
    rows.append(ims)
w, h = rows[0][0].size
img = Image.new("RGB", (len(SEEDS) * (w + 8) + 70, 2 * (h + 8)), "white")
d = ImageDraw.Draw(img)
for r, (name, ims) in enumerate(zip(("counts", "linear"), rows)):
    d.text((4, r * (h + 8) + h // 2), name, fill="black")
    for c, im in enumerate(ims):
        img.paste(im, (70 + c * (w + 8), r * (h + 8)))
img.save(out)
