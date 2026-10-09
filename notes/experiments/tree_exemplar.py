"""The procedural tree exemplar copied with guarantees
(castlegen/channels/treecopy.py): the coarse window channel seeds one copy
per trunk, the refinement places it, SF joint sweeps finish it under the
certificate; against sequential growth alone (SEQ sweeps).  Prints the
guarantees (certificate violations, attached fraction, trunks, ground
integrity, leaves on tips, coordinate mismatches) and writes
images/tree_exemplar_{exemplar,coarse,seq}.png.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/tree_exemplar.py

Env: EX (sheet: four procedural trees and their mirror images on one ground
line; single: the one tree), ROT (0: upright only; 1: quarter turns too,
floating patches that only attach to standing trees), SEEDS (0,1,2,3), W (320), S8 (100), SF (60), SEQ (600), MU (2.0),
NU (6.0), LAM (1.0), LAM8 (inf; finite = recombination at hard seams), SEED (0)."""
import os, json, time
import numpy as np
from PIL import Image

from castlegen.channels.treecopy import TreeCopy, procedural_tree, sheet

E = os.environ.get
W, S8, SF, SEQ = int(E("W", 320)), int(E("S8", 100)), int(E("SF", 60)), int(E("SEQ", 600))
EX = E("EX", "sheet")
ROT, SEEDS = int(E("ROT", 0)), tuple(int(x) for x in E("SEEDS", "0,1,2,3").split(","))
MU, NU, LAM, SEED = float(E("MU", 2.0)), float(E("NU", 6.0)), float(E("LAM", 1.0)), int(E("SEED", 0))
LAM8 = float(E("LAM8", "inf"))
TRUNKS = [x for x in range(40, W, 64)]
os.makedirs("images", exist_ok=True)
OUT = dict(config=dict(EX=EX, ROT=ROT, SEEDS=SEEDS, W=W, S8=S8, SF=SF, SEQ=SEQ, MU=MU, NU=NU, LAM=LAM, LAM8=LAM8, SEED=SEED, trunks=TRUNKS))
A = sheet(seeds=SEEDS, rotations=(0, 1) if ROT else (0,)) if EX == "sheet" else procedural_tree()


def show(tag, m, t):
    keys = ["rule_violations", "multi_parent", "attached_frac", "trunks", "in_ground", "tree_cells", "leaves",
            "leaves_on_tips", "dangling_ports", "mismatched", "windows"]
    print(f"{tag} ({t:.0f}s): " + "  ".join(f"{k} {m[k]:.3f}" if isinstance(m[k], float) else f"{k} {m[k]}" for k in keys), flush=True)


tc = TreeCopy(W, TRUNKS, A=A, seed=SEED, nu=NU, lam=LAM, lam8=LAM8)
print(f"exemplar {EX}: {len(tc.trunks)} trees, {int((tc.mass > 0).sum())} tree cells, mass hist {np.bincount(tc.mass.ravel())[1:].tolist()}, "
      f"{tc.coarse.D - 1} windows, {len(tc.trunk_win)} trunk windows; world {tc.H} x {W}, {len(TRUNKS)} trunks")
Image.fromarray(tc.render_exemplar(2 if EX == "sheet" else 4)).save("images/tree_exemplar_exemplar.png")

# coarse seeding + refinement + joint sweeps
t0 = time.time()
nwin, pruned, bad8 = tc.seed_coarse(S8)
m = tc.metrics()
print(f"level 8: {nwin} windows placed, {pruned} cells pruned, no-candidate {bad8}, recombined seams "
      f"{tc.coarse.recombined_seams(tc.u8)}")
show("after refinement", m, time.time() - t0)
OUT["refined"] = m
tc.sweep(SF, seed=SEED + 1, mu=MU)
m = tc.metrics()
show(f"coarse + {SF} joint sweeps", m, time.time() - t0)
OUT["coarse"] = m
Image.fromarray(tc.render(4)).save("images/tree_exemplar_coarse.png")

# sequential growth alone
tc2 = TreeCopy(W, TRUNKS, A=A, seed=SEED, nu=NU, lam=LAM)
t0 = time.time()
tc2.sweep(SEQ, seed=SEED + 2)
m2 = tc2.metrics()
show(f"sequential {SEQ} sweeps", m2, time.time() - t0)
OUT["sequential"] = m2
Image.fromarray(tc2.render(4)).save("images/tree_exemplar_seq.png")
with open("images/tree_exemplar.json", "w") as f:
    json.dump(OUT, f, indent=1)
