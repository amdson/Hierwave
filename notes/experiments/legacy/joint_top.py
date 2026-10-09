"""Test 1 of notes/history/joint_sampler.md: the top level (h = 32, 4 x 4 cells on
128^2) alone, uniform start, annealed exact joint (u, v) sweeps.  Rows: with
promises (cost + compat), without (use_v=False); columns: seeds.  Each cell
shows its exemplar window, grid lines, and v (upper-left letters: sampled
sub-column classes; lower: the window's own classes) ->
images/<OUT>.png.  args: [LAM_C | ex] [SWEEPS] [R] [OUT] [H]; LAM_C = ex: generic.ex_weight(h)
with E_ex's features (w_sig = 0.5, no sockets)."""
import os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import exemplar as ex, generic as GN, texsyn, tileset

arg = sys.argv[1] if len(sys.argv) > 1 else "1.0"
SWEEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
RAD = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
OUT = sys.argv[4] if len(sys.argv) > 4 else "joint_top"
h = int(sys.argv[5]) if len(sys.argv) > 5 else 32
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1,2,3,4").split(","))
n, px, tmp = 128, 2, tempfile.mkdtemp()
LAM_C, feat = (GN.ex_weight(h), dict(w_sig=0.5, w_sock=0.0)) if arg == "ex" else (float(arg), {})
CL = "ESPF"
ts = tileset.load("cliffs")
E = np.load("castlegen/legacy/exemplars/cliffs_big.npy").astype(np.int32)
RAISE = int(os.environ.get("RAISE", 0))                                    # lift the ground: drop top rows, repeat the bottom row
E = np.concatenate([E[RAISE:], np.repeat(E[-1:], RAISE, 0)], 0)
an = texsyn.Analysis(ts, E, bounds="edge", pad=2 * h, **feat)
solid = np.asarray(ts.solid, bool)
rows = {}
for use_v in (True, False):
    key = "promises" if use_v else "no promises"
    rows[key] = []
    lvl = GN.JointLevel(an, solid, h, n, lam_c=LAM_C, r=RAD, use_v=use_v,
                         knn=int(os.environ["KNN"]) if "KNN" in os.environ else None)
    for sd in SEEDS:
        lvl.rng = np.random.default_rng(sd)
        lvl.init(lvl.rng.integers(0, an.my * an.m, (lvl.R, lvl.C)))
        trace = []
        lvl.run(SWEEPS, T=1.0, T_hot=10.0, log=lambda s, t, L: trace.append(L.energy()) if s % 5 == 4 else None)
        agree = (lvl.V == lvl.vals[lvl.U]).mean()
        print(f"{key:12s} seed {sd}  E trace {' '.join(f'{e:.0f}' for e in trace)}  "
              f"v = window class {agree:.2f}  valid {lvl.valid()}", flush=True)
        rows[key].append((lvl.tiles(), lvl.V.copy(), lvl.vals[lvl.U]))
p = os.path.join(tmp, "x.png")
ims = {}
for k, row in rows.items():
    ims[k] = []
    for x, Vs, Vw in row:
        ex.to_png(ts, x, p, px)
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        c = h * px
        for i in range(1, n // h):
            d.line([(i * c, 0), (i * c, n * px)], fill=(255, 0, 255))
            d.line([(0, i * c), (n * px, i * c)], fill=(255, 0, 255))
        for y in range(n // h):
            for x_ in range(n // h):
                for j, (vv, yo) in enumerate(((Vs, 2), (Vw, c - 12)) if k == "promises" else ((Vw, c - 12),)):
                    t = CL[vv[y, x_] // 4] + CL[vv[y, x_] % 4]
                    d.text((x_ * c + 3, y * c + yo), t, fill=(255, 255, 0) if vv is Vs else (0, 255, 255))
        ims[k].append(im)
w, hh = ims["promises"][0].size
img = Image.new("RGB", (90 + len(SEEDS) * (w + 8), len(ims) * (hh + 8)), "white")
d = ImageDraw.Draw(img)
for r, (k, row) in enumerate(ims.items()):
    d.text((4, r * (hh + 8) + hh // 2), k, fill="black")
    for c_, im in enumerate(row):
        img.paste(im, (90 + c_ * (w + 8), r * (hh + 8)))
img.save(f"images/{OUT}.png")
