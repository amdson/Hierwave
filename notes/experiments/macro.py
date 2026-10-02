"""Rigid objects in the maze (castlegen/objects.py): maze.py's pipeline (dist
method) on castlegen/exemplars/castle_macro.json, 128^2 torus, tile set
"macro": the castle plus 5 pyramids (17 x 17: six stone kinds, a burial hall,
a corridor out) and 3 temples (21 x 21: a courtyard, a shrine with altar and
braziers, pillars, a fountain, trees), each an object with a template
(exemplar "objects"), no tile kind per cell.

Promise: JointLevel at h = 64, 32, 16 as maze.py, plus (variant "coarse") the object
term as lvl.extra: LRC per object rule check broken across a cell's seams
(objects.coarse_term).  Tiles: the h = 16 windows pasted, each cell's object
code from its exemplar coordinate; map_sweep draws (tile, code) jointly with
LO (default 1000: hard) per broken check; tiles that only occur inside
objects in the exemplar are not on offer without a code.  Variants
VARIANTS (default "coarse,none": the object term at the coarse levels or
not; the tile stage keeps the rule either way).

Reports per seed and variant: objects complete / object cells outside a
complete copy / broken checks, after pasting and after the tile stage, plus
maze.py's connectivity line.  images/macro.png: exemplar, then per seed the
pasted and final maps of each variant; complete objects boxed white, object
cells outside one tinted red."""
import os, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen import blockconn as BC, connmetrics as CM, exemplar as ex, generic as GN, objects as OB, tileset

N, K = int(os.environ.get("N", 128)), 16
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1,2,3").split(","))
VARIANTS = os.environ.get("VARIANTS", "coarse,none").split(",")
LAMD = float(os.environ.get("LAMD", 16))
SW, HOLD = int(os.environ.get("SW", 100)), int(os.environ.get("HOLD", 100))
LAMAX = float(os.environ.get("LAMAX", 32))
BETA, R = float(os.environ.get("BETA", 1)), int(os.environ.get("R", 2))
SEAMW = float(os.environ.get("SEAMW", 0.5))
LRC, LO = float(os.environ.get("LRC", 50)), float(os.environ.get("LO", 1000))
PX = int(os.environ.get("PX", 4))
PLAN = [(64, 16), (32, 8), (16, 8)]

lay = ex.load_layout("castle_macro")
ts = tileset.load("macro")
if not os.path.exists("cache/castle_macro_ex.npy"):
    np.save("cache/castle_macro_ex.npy", np.asarray(ex.build(lay)[1]))
E = np.load("cache/castle_macro_ex.npy").astype(np.int32)
otabs = OB.tables(ts, lay)
names, _, _, TMPL, NBR = otabs
Oex = OB.channel(ts, lay, otabs)
m, S = E.shape[0], ts.n_sig
solid = np.asarray(ts.solid, bool)
an = GN.analysis(ts, E, [h for h, _ in PLAN], torus=True)
tabs = BC.tables(ts)
t = ts.np_tables
Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
allowed = np.zeros(S, bool)
allowed[np.unique(E)] = True
soft = np.zeros(S, bool)
soft[np.unique(E[Oex < 0])] = True                                         # tiles seen outside objects
PC = BC.parent_cost(E.astype(np.int64), S, R)
GRP = np.ones(BC.MLAB + 1, np.int64)
W4 = np.ones(4)


def promise(seed, coarse):
    Up = Vp = Kp = None
    for h, sweeps in PLAN:
        lvl = GN.JointLevel(an, solid, h, N, lam_c=16.0, r=0.25, knn=16, sigma=None if Up is None else h / 2,
                            use_v=True, loc=ts, fast="numba", conn=ts, lam_d=LAMD, rk=4)
        if coarse:
            lvl.extra = OB.coarse_term(lvl, Oex, NBR, LRC)
        lvl.rng = np.random.default_rng(seed * 100 + h)
        if Up is None:
            lvl.init(np.random.default_rng(seed).integers(0, an.my * an.m, (lvl.R, lvl.C)))
            lvl.run(sweeps, T=1.0, T_hot=10.0)
        else:
            lvl.init(lvl.prolong(Up), parent=Up, vparent=Vp, kparent=Kp)
            lvl.run(sweeps, T=1.0)
        Up, Vp, Kp = lvl.U.copy(), lvl.V.copy(), lvl.keys()
    return lvl


def coords(U):
    """(N, N) flat exemplar coordinate of every tile: its block's window."""
    uy, ux = np.divmod(np.repeat(np.repeat(U, K, 0), K, 1), m)
    ly, lx = np.mgrid[:N, :N] % K
    return ((uy - K // 2 + ly) % m) * m + (ux - K // 2 + lx) % m


def tile_phase(T0, O0, labs, pc, rng):
    T, O = T0.copy(), O0.copy()
    for s in range(SW + HOLD):
        lam = LAMAX * min(1.0, (s + 1) / SW)
        BC.map_sweep(T, K, rng.permutation(N * N), rng.gumbel(size=(N * N, S + 5)), Eh, Ev, logz, allowed, pc,
                     labs, GRP, *tabs, BC.MODES["dist"], W4, SEAMW, lam, ts.WALL, O, TMPL, NBR, soft, LO)
    return T, O


def objline(T, O):
    c = OB.census(O, T, otabs)
    return "  ".join(f"{n} {c[n][0]} (+{c[n][1]} cells)" for n in names) + f"  broken {OB.broken(O, NBR)}"


def stats(T, labs):
    r = BC.map_report(T, K, labs, GRP, tabs)
    g = CM.global_stats(ts, T, torus=False)
    return (f"blocks broken {r['blocks']:2d}  seams {r['seams']:3d}  components {g['components']:3d}  "
            f"outside largest {g['unreached_cells']:.3f}")


print("exemplar:", objline(E, Oex), flush=True)
rows = []
for sd in SEEDS:
    row = []
    for var in VARIANTS:
        t0 = time.time()
        lvl = promise(sd, var == "coarse")
        O = lvl.V & 15
        labs = BC.block_labels(O, lvl._lowbits(O, lvl.keys()))
        T0 = lvl.tiles().astype(np.int64)
        c = coords(lvl.U)
        O0 = Oex.ravel()[c]
        pc = np.ascontiguousarray(BETA * PC[c])
        print(f"seed {sd} {var:6s} promise {time.time() - t0:.0f}s  pasted: {objline(T0, O0)}", flush=True)
        t0 = time.time()
        Tg, Og = tile_phase(T0, O0, labs, pc, np.random.default_rng(sd))
        print(f"    tiles {time.time() - t0:.0f}s: {objline(Tg, Og)}  |  {stats(Tg, labs)}", flush=True)
        row += [(T0, O0), (Tg, Og)]
    rows.append(row)


# ---- figure
def panel(T, O):
    p = os.path.join(tempfile.mkdtemp(), "x.png")
    ex.to_png(ts, T, p, PX)
    im = np.asarray(Image.open(p).convert("RGB")).astype(float)
    H, W = O.shape
    cover = np.zeros((H, W), bool)
    boxes = []
    for k, n in enumerate(names):
        h, w = otabs[1][k].shape
        codes = otabs[2][k] + np.arange(h * w).reshape(h, w)
        for y, x in zip(*np.nonzero(O == otabs[2][k])):
            if y + h <= H and x + w <= W and np.array_equal(O[y:y + h, x:x + w], codes):
                cover[y:y + h, x:x + w] = True
                boxes.append((x, y, x + w, y + h))
    bad = np.repeat(np.repeat((O >= 0) & ~cover, PX, 0), PX, 1)
    im[bad] = 0.4 * im[bad] + 0.6 * np.array([230, 30, 30])
    img = Image.fromarray(im.astype(np.uint8))
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:
        d.rectangle((x0 * PX - 1, y0 * PX - 1, x1 * PX, y1 * PX), outline=(255, 255, 255), width=2)
    return img


w, gap = N * PX, 10
cols = [f"{v}: {s}" for v in VARIANTS for s in ("pasted", "tiles")]
img = Image.new("RGB", (60 + (len(cols) + 1) * (w + gap), 20 + len(rows) * (w + gap)), "white")
d = ImageDraw.Draw(img)
d.text((64, 4), "exemplar", fill="black")
img.paste(panel(E, Oex), (60, 20))
for j, name in enumerate(cols):
    d.text((60 + (j + 1) * (w + gap) + 4, 4), name, fill="black")
for i, row in enumerate(rows):
    d.text((4, 20 + i * (w + gap) + w // 2), f"seed {SEEDS[i]}", fill="black")
    for j, (T, O) in enumerate(row):
        img.paste(panel(T, O), (60 + (j + 1) * (w + gap), 20 + i * (w + gap)))
img.save("images/macro.png")
print("wrote images/macro.png")
