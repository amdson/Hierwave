"""Demo figure for the promise-hierarchy support experiments -> images/demo.png.
    .venv/bin/python notes/experiments/demo_image.py [SCRATCH]
SCRATCH: folder holding sweep_corpus_f{0,1}_avgexheights.npy from
notes/experiments/exchain_sweep.py (the corpus-free loop's last round)."""
import copy
import glob
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from castlegen import exemplar as ex
from castlegen import pipeline as PL
from castlegen import refheights as RH
from castlegen import tileset
from castlegen.quantities import envelope as EN

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "."
PX, SEED, G = 3, 1, 8
ts = tileset.load("cliffs")
tmp = tempfile.mkdtemp()


def render(t):
    p = os.path.join(tmp, "x.png")
    ex.to_png(ts, np.asarray(t, np.int32), p, PX)
    return Image.open(p).convert("RGB")


def run(spec):
    _, t, rep = PL.run(spec, seed=SEED, log=lambda *a: None)
    return t, rep[-1]


def env(tables="reference", lam=1.0, w_tex=1.0, sweeps=10, corpus=None, clamps=None):
    spec = copy.deepcopy(PL.load_spec("cliffs_envelope"))
    h, t = spec["stages"]
    h.update(tables=tables, lam=lam)
    q = {}
    if corpus:
        q["corpus_path"] = corpus
    if clamps:
        q["clamps"] = [list(k) + [v] for k, v in clamps.items()]
    if q:
        h["envelope"] = q
    t.update(T=0.9, w_tex=w_tex, sweeps=sweeps)
    return spec


def unsup(t):
    return RH.violations(ts, t, G)


E = np.load(sorted(glob.glob("cache/exemplar-cliffs-*.npy"))[0])
lg = EN.lang()
full = int(lg.encode([lg.F, lg.F], [lg.F, lg.F]))
panels = []

panels.append(("Exemplar (64 x 64)", "the texture source", E))
t, r = run(PL.load_spec("cliffs_promise"))
panels.append(("Old pipeline", f"repair edits 28% of cells; {unsup(t)} unsupported", t))
t, r = run(env())
panels.append(("Envelope promises", f"flat target; no repair; {unsup(t)} unsupported", t))
t, r = run(env(lam=20.0))
panels.append(("Envelope, strong texture coupling", f"6% of cells edited; {unsup(t)} unsupported", t))

hills = "cache/hills_corpus.npy"
t, r = run(env(w_tex=0.0, sweeps=100, corpus=hills))
panels.append(("Tables from a hills corpus", f"rolling terrain; {unsup(t)} unsupported", t))
t, r = run(env(tables="linear", w_tex=0.0, sweeps=100, corpus=hills, clamps={(32, 1, 1): full}))
panels.append(("... plus a clamped mountain", f"clamp honoured exactly; {unsup(t)} unsupported", t))
for f, title, sub in (("1", "Corpus-free training loop", "average promises, exemplar energy"),
                      ("0", "... without the coarse patch term", "converges to low hills")):
    path = os.path.join(SCRATCH, f"sweep_corpus_f{f}_avgexheights.npy")
    if os.path.exists(path):
        t = np.load(path)[0]
        panels.append((title, f"{sub}; {unsup(t)} unsupported", t))

ims = [(a, b, render(t)) for a, b, t in panels]
W = max(im.width for *_, im in ims)
H = max(im.height for *_, im in ims)
cols, pad, cap, top = 4, 16, 44, 56
rows = (len(ims) + cols - 1) // cols
canvas = Image.new("RGB", (cols * (W + pad) + pad, top + rows * (H + cap + pad)), "white")
d = ImageDraw.Draw(canvas)
try:
    f_title, f_head, f_sub = (ImageFont.load_default(size=s) for s in (22, 15, 12))
except TypeError:
    f_title = f_head = f_sub = ImageFont.load_default()
d.text((pad, 16), "Support guaranteed by a promise hierarchy (side view, 128 x 128, seed 1)", fill="black", font=f_title)
for k, (a, b, im) in enumerate(ims):
    x = pad + (k % cols) * (W + pad)
    y = top + (k // cols) * (H + cap + pad)
    d.text((x, y), a, fill="black", font=f_head)
    d.text((x, y + 20), b, fill=(90, 90, 90), font=f_sub)
    canvas.paste(im, (x + (W - im.width) // 2, y + cap + (H - im.height)))
os.makedirs("images", exist_ok=True)
canvas.save("images/demo.png")
print("wrote images/demo.png", canvas.size)
