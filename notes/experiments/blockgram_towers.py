"""Towers from a split grammar compiled to a block pyramid
(castlegen/blockgram).  Renders world columns [X, X + W) whole, then again
chunk by chunk (64 columns, caches cleared before each chunk) and checks
the two agree tile for tile; checks every tower from tile types alone
(floors, rooms per floor, stairs per floor, hole above each top step);
reports slot occupancy per level.  Writes images/blockgram_towers.png and
a zoomed crop images/blockgram_towers_zoom.png.

    PYTHONPATH=. python notes/experiments/blockgram_towers.py
"""
import os, time
import numpy as np
from PIL import Image
from scipy import ndimage

from castlegen.blockgram import towers as TG
from castlegen.blockgram.core import K

X, W = int(os.environ.get("X", 0)), int(os.environ.get("W", 960))
CHUNK = 64
OUT = os.environ.get("OUT", "images/blockgram_towers.png")


def check(T):
    body = np.isin(T, TG.BODY)
    lab, n = ndimage.label(body)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if x1 - x0 < 10 or x0 == 0 or x1 == T.shape[1] - 1:           # clipped by the window
            continue
        sub = np.where(lab[y0:y1 + 1, x0:x1 + 1] == k, T[y0:y1 + 1, x0:x1 + 1], -1)
        inner = sub[:, 1:-1]
        slabs = [y for y in range(inner.shape[0]) if np.isin(inner[y], (TG.SLAB, TG.HOLE, TG.ROOF)).all()]
        floors, rooms, stairs, holes = [], [], [], True
        for a, b in zip(slabs[:-1], slabs[1:]):
            rows = inner[a + 1:b]
            floors.append(len(rows))
            rooms.append(int(np.isin(rows, (TG.WALL, TG.DOOR)).all(0).sum()) + 1)
            stairs.append(int((rows[-1] == TG.STEP).sum()))
            tops = np.flatnonzero(rows[0] == TG.STEP)
            holes &= bool((inner[a, tops] == TG.HOLE).all()) and int((inner[a] == TG.HOLE).sum()) == len(tops)
        closed = (sub[:, 0] >= 0).all() and (sub[:, -1] >= 0).all() and slabs and slabs[0] == 0 \
            and slabs[-1] == inner.shape[0] - 1 and (inner[0] != TG.SLAB).all()
        out.append(dict(x=int(x0), floors=len(floors), heights=set(floors), rooms=set(rooms), stairs=set(stairs),
                        holes=holes, closed=bool(closed)))
    return out


g = TG.Towers()
stats = {}
t0 = time.time()
T, C = g.render(X, X + W, stats)
t_full = time.time() - t0
t0 = time.time()
Tc = np.concatenate([g.clear() or g.render(x, x + CHUNK)[0] for x in range(X, X + W, CHUNK)], 1)
t_chunk = time.time() - t0
print(f"render {W} columns: {t_full:.1f}s whole, {t_chunk:.1f}s in {W // CHUNK} chunks from empty caches; "
      f"chunks == whole: {np.array_equal(T, Tc)}; max slots per block {stats} (K = {K})")

res = check(T)
for t in res:
    print(f"  tower at x={t['x']:5d}: {t['floors']:2d} floors (heights {t['heights']}), rooms/floor {t['rooms']}, "
          f"stairs/floor {t['stairs']}, holes ok {t['holes']}, closed {t['closed']}")
ok = all(10 <= t["floors"] <= 15 and t["rooms"] == {5} and t["stairs"] == {1} and t["holes"] and t["closed"] for t in res)
print(f"{len(res)} towers, all valid: {ok}")
st = TG.owner_states(X // TG.RW, (X + W - 1) // TG.RW)
print("owner states (n, bridge story):", st)

far = 10 ** 6
Tf, Cf = g.render(far, far + 320)
rf = check(Tf)
print(f"far window x={far}: {len(rf)} towers, valid {all(10 <= t['floors'] <= 15 and t['rooms'] == {5} and t['stairs'] == {1} and t['holes'] and t['closed'] for t in rf)}")

px = 3
img = np.repeat(np.repeat(C, px, 0), px, 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(img).save(OUT)
zx = res[len(res) // 2]["x"] - X - 8 if res else 0
crop = C[40:, max(zx, 0):max(zx, 0) + 140]
Image.fromarray(np.repeat(np.repeat(crop, 6, 0), 6, 1)).save(OUT.replace(".png", "_zoom.png"))
print("wrote", OUT)
