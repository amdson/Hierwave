"""Towers on labelled exemplar windows (castlegen/channels/tower.py).

INIT=blank: sky and ground only, towers must nucleate and grow under
single-site window moves.  INIT=copy: two sheet towers copied verbatim
(a consistent refinement of an object level), then the same sweeps:
how much recombination the hard seams allow.  Reports seam violations,
complete towers with their floor / room / stair counts (geometry only),
novelty against the sheet, and writes images/tower_<INIT>.png.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/tower_windows.py
"""
import os, time
import numpy as np
from PIL import Image

from castlegen.channels import Model
from castlegen.channels import tower as TW

H, W = int(os.environ.get("H", 128)), int(os.environ.get("W", 256))
SEED = int(os.environ.get("SEED", 0))
NT = int(os.environ.get("NT", 6))
INIT = os.environ.get("INIT", "blank")
S8 = int(os.environ.get("S8", 60))
LAM8, CUT, BONUS = float(os.environ.get("LAM8", 1.0)), float(os.environ.get("CUT", 1.0)), float(os.environ.get("BONUS", 1.0))
GY = H - 12                                                    # first ground row in the world
OUT = os.environ.get("OUT", f"images/tower_{INIT}.png")
print(f"INIT={INIT} LAM8={LAM8} CUT={CUT} BONUS={BONUS} S8={S8}")

codes, table, placed = TW.make_sheet(NT, seed=SEED)
rules = TW.pair_rules(codes, table, CUT)
hard, cut = TW.pair_violations(codes, rules)
geo = TW.towers_geometry(codes, table)
sheet_keys = {TW.tower_key(codes, table, g["bbox"]) for g in geo if g["complete"]}
print(f"sheet {codes.shape}, {len(table)} codes, pair violations {hard}/{cut}; towers "
      + ", ".join(f"{g['floors']}f rooms {set(g['rooms'])} stairs {g['stairs'][:-1].count(1)}/{g['floors'] - 1}+top {g['stairs'][-1]}"
                  for g in geo if g["complete"]))
win = TW.Windows(codes, table)
u8 = win.channel()
gnd = TW.gnd_channel(H, W, GY)
CERT = int(os.environ.get("CERT", 0))
if CERT:
    from castlegen.channels import Certificate, Channel
    t0 = time.time()
    J, Dmax, delta = win.certificate(rules)
    u8.add_view("mass", win.mass, 2)
    u8.add_view("root", win.root, 2)
    d8 = Channel("d8", TW.B, Dmax + 2).add_view("d", np.arange(Dmax + 2))
    m8 = Model(H, W, [u8, gnd, d8], win.factors(rules, lam8=LAM8, bonus=BONUS),
               [Certificate("u8", "mass", "root", "d8", Dmax, delta, joins=J)])
    d8.grid[:] = Dmax + 1
    print(f"certificate joins {int(J.sum())} ({time.time() - t0:.1f}s), root windows {int(win.root.sum())}")
else:
    m8 = Model(H, W, [u8, gnd], win.factors(rules, lam8=LAM8, bonus=BONUS))
print(f"{win.D} windows; {int(win.tower.sum())} with tower codes")

gy_sheet = codes.shape[0] - 10
if INIT == "blank":
    win.blank(u8, gnd)
else:
    win.blank(u8, gnd)
    xw = 8
    for x0, floors in placed[:2]:
        span = sum(floors[0][0]) + TW.NROOM - 1 + 2
        win.copy_tower(u8, x0, xw, gy_sheet, GY, span)
        xw += span + 24


def report(tag):
    e, v = m8.energy("u8")
    T = win.tiles(u8)
    hard, cut = TW.pair_violations(T, rules)
    geo = TW.towers_geometry(T, table)
    comp = [g for g in geo if g["complete"]]
    novel = sum(TW.tower_key(T, table, g["bbox"]) not in sheet_keys for g in comp)
    ok = sum(10 <= g["floors"] <= 15 and set(g["rooms"]) == {5} and all(s == 1 for s in g["stairs"][:-1])
             and g["stairs"][-1] == 0 and g["holes_ok"] for g in comp)
    print(f"{tag}: energy {e:8.1f}  hard {v}  | tile pairs: hard {hard} cut {cut} | tower windows "
          f"{int(win.tower[u8.grid].sum())}  components {len(geo)}  complete {len(comp)}  valid {ok}  novel {novel}  "
          f"recombined seams {win.recombined_seams(u8)}")
    for g in comp:
        print(f"    complete tower at {g['bbox']}: {g['floors']} floors, rooms {set(g['rooms'])}, stairs {g['stairs']}")
    return T


T = report("init")
t0 = time.time()
done = 0
for k in (5, 20, S8):
    bad = m8.sweep("u8", k - done, seed=SEED + k)
    done = k
    T = report(f"sweep {k:3d} ({time.time() - t0:5.1f}s, no-candidate {bad})")

img = TW.render(T, table, 3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(img).save(OUT)
Image.fromarray(TW.render(codes, table, 2)).save("images/tower_sheet.png")
print("wrote", OUT)
