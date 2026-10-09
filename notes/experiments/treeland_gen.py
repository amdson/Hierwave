"""Stage B of notes/trees_test.md, generation: trees with limbs under the
certificate on a sampled terrain (castlegen/channels/treeland.py), RUNS
forward runs.  Validity (rule violations, attached fraction, dangling
ports), the limb statistics against the local relief and slope (how the
hard rules make the trees react to the terrain), and a sheet of renders.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/treeland_gen.py

Env: N (24), RUNS (64), SEED (0), GROW (3.0), DANGLE (4.0), S_TILE (150).
Outputs images/treeland_gen.json, images/treeland_sheet.png."""
import os, json, time
import numpy as np
from PIL import Image

from castlegen.channels import limbs
from castlegen.channels.treeland import CH, Forward, Treeland

E = os.environ.get
N, RUNS, SEED = int(E("N", 24)), int(E("RUNS", 64)), int(E("SEED", 0))
GROW, DANGLE, S_TILE = float(E("GROW", 3.0)), float(E("DANGLE", 4.0)), int(E("S_TILE", 150))
os.makedirs("images", exist_ok=True)
T = Treeland(N, margin=2, limb_kw=dict(grow=GROW, dangle=DANGLE))
fw = Forward(T, seed=SEED)

acc = {}
relief_cells = {0: [], 1: [], 2: []}          # limb cells per tree by the relief at its slot
slope_asym = []                               # (slope, right - left limb cells)
valid = dict(rule_violations=0, dangling_ports=0, limb_cells=0, attached=0, trunks=0, multi_parent=0)
sheet = []
t0 = time.time()
for r in range(RUNS):
    st = fw.run(30, 20, S_TILE, fresh=True)
    for k, v in st.items():
        if k != "h_hist" and np.isfinite(v):
            acc.setdefault(k, []).append(v)
    m = limbs.metrics(fw.c["tile"], fw.c["cert"])
    for k in valid:
        valid[k] += m[k] if k != "attached" else m["attached"]
    c = fw.c
    g = c["tree"].grid[T.brow]
    h = T.heights(c["surf"])
    rv = c["tile"].views["root"][c["tile"].grid]
    limb = rv == limbs.LIMB
    own = limbs.per_tree(c["tile"])                                 # the limbs attached to each trunk
    for J in range(T.J0, T.J1):
        if g[J] == 0:
            continue
        rel = max(abs(h[J] - h[J - 1]), abs(h[J] - h[J + 1]))
        y, x = T.R0 + T.rec_r[g[J]], J * CH + T.rec_c[g[J]]
        cells, left, right = own.get((y, x), (0, 0, 0))
        relief_cells[min(rel, 2)].append(cells)
        slope_asym.append((int(h[J + 1] - h[J - 1]), right - left))
    if r < 3:
        sheet.append(T.render(fw.chans, px=5))
secs = time.time() - t0
sa = np.array(slope_asym)
OUT = dict(config=dict(N=N, RUNS=RUNS, SEED=SEED, GROW=GROW, DANGLE=DANGLE, S_TILE=S_TILE, S=(30, 20, S_TILE)),
           seconds=secs, per_run=secs / RUNS,
           stats={k: dict(mean=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v)))) for k, v in acc.items()},
           validity=valid,
           limb_cells_by_relief={r: dict(n=len(v), mean=float(np.mean(v)) if v else None) for r, v in relief_cells.items()},
           asym_by_slope={s: dict(n=int((sa[:, 0] == s).sum()), mean=float(sa[sa[:, 0] == s, 1].mean()))
                          for s in sorted(set(sa[:, 0].tolist())) if (sa[:, 0] == s).sum() >= 10})
with open("images/treeland_gen.json", "w") as f:
    json.dump(OUT, f, indent=1)
Image.fromarray(np.concatenate(sheet, 0)).save("images/treeland_sheet.png")
print(f"{RUNS} runs, {secs / RUNS:.2f}s each")
print("validity:", valid, f"attached {valid['attached'] / max(valid['limb_cells'], 1):.4f}")
for k, v in OUT["stats"].items():
    print(f"  {k}: {v['mean']:.3f} +- {v['se']:.3f}")
print("limb cells per tree by relief:", OUT["limb_cells_by_relief"])
print("right - left limb cells by slope (h_right - h_left):", OUT["asym_by_slope"])
