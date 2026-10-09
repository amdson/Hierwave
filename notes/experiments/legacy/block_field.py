"""Quick test: local auxiliary fields (castlegen/legacy/blockfield.py, fully parallel
updates) against the sequential dist score (blockconn.sweep, mode dist) on one
16 x 16 block, maze-style patterns (as maze.py's blocks): a side is open where
the parent window has an opening, 1 or 2 random open sides are downhill (class
A, both halves), the other open sides OPEN.

Castle, NPAR parent coordinates x SEEDS seeds; the chain as block_ports.py
(T = 1, E_pair - logz + BETA -log p(t | parent window), closed ports facing
wall, from the parent window), lam ramped 0 -> LAMAX over SW sweeps then HOLD.
Success: blockconn's exact score 0 at the end.  Field knobs: G, D, MU, LV, EPS,
DELTA (blockfield docstring).  images/block_field.png: parent window, dist,
field per case."""
import os, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import blockconn as BC, blockfield as BF, exemplar as ex, tileset

K = 16
NPAR, SEEDS = int(os.environ.get("NPAR", 9)), int(os.environ.get("SEEDS", 3))
SW, HOLD = int(os.environ.get("SW", 80)), int(os.environ.get("HOLD", 40))
LAMAX = float(os.environ.get("LAMAX", 8))
BETA, R = float(os.environ.get("BETA", 1)), int(os.environ.get("R", 2))
G, D = int(os.environ.get("G", 8)), int(os.environ.get("D", 96))
MU, LV = float(os.environ.get("MU", 1.0)), float(os.environ.get("LV", 1000.0))
EPS, DELTA = float(os.environ.get("EPS", 0.1)), float(os.environ.get("DELTA", 0.05))
METHODS = os.environ.get("METHODS", "dist,field").split(",")

ts = tileset.load("demo")
E = np.load("cache/castle_ex.npy").astype(np.int64)
m, S = E.shape[0], ts.n_sig
tabs = BC.tables(ts)
t = ts.np_tables
Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
allowed = np.zeros(S, bool)
allowed[np.unique(E)] = True
PC = BC.parent_cost(E, S, R)
GRP = np.ones(BC.MLAB + 1, np.int64)
W4 = np.ones(4)
yy, xx = np.mgrid[:K, :K]
lam_of = lambda s: LAMAX * min(1.0, (s + 1) / SW)

rng0 = np.random.default_rng(0)
cases = []
while len(cases) < NPAR:
    c = int(rng0.integers(m * m))
    Y, X = (c // m - K // 2 + yy) % m, (c % m - K // 2 + xx) % m
    Wt = np.ascontiguousarray(E[Y, X])
    sides = np.flatnonzero((BC.pattern(Wt, tabs).reshape(4, 2) > 0).any(1))
    if len(sides) < 2:
        continue
    down = rng0.choice(sides, int(rng0.integers(1, 3)), replace=False)
    lab = np.zeros(8, np.int64)
    for d in sides:
        lab[2 * d:2 * d + 2] = 1 if d in down else BC.OPEN
    pc = np.ascontiguousarray(BETA * PC[Y * m + X]) if BETA else np.zeros((K, K, S))
    cases.append(dict(c=c, lab=lab, W=Wt, pc=pc))


def ok(T, lab):
    return BC.parts(T, lab, GRP, *tabs, 0).sum() == 0


def run(method, cs, rng):
    T, lab, pc = cs["W"].copy(), cs["lab"], cs["pc"]
    first = None
    if method == "field":
        Gf, Df = BF.init_fields(T, lab, tabs, G, D)
    for s in range(SW + HOLD):
        if method == "dist":
            BC.sweep(T, rng.permutation(K * K), rng.gumbel(size=(K * K, S)), Eh, Ev, logz, allowed, pc, lab, GRP,
                     *tabs, 3, W4, lam_of(s), ts.WALL)
        else:
            BF.sweep(T, Gf, Df, int(rng.integers(2 ** 31)), Eh, Ev, logz, allowed, pc, lab, *tabs, lam_of(s),
                     MU, LV, EPS, DELTA, G, D, ts.WALL)
        if first is None and ok(T, lab):
            first = s
    extra = BF.field_energy_ok(T, Gf, Df, lab, tabs, G) if method == "field" else None
    return T, first, extra


res = {md: [] for md in METHODS}
tsw = {}
figs = []
for ci, cs in enumerate(cases):
    line = f"case {ci} c {cs['c']:4d} [{BC.fmt(cs['lab'])}]"
    row = [cs["W"]]
    for md in METHODS:
        marks = ""
        for sd in range(SEEDS):
            t0 = time.time()
            T, first, extra = run(md, cs, np.random.default_rng(100 * ci + sd))
            tsw[md] = (time.time() - t0) / (SW + HOLD)
            good = ok(T, cs["lab"])
            res[md].append(dict(ok=good, first=first, left=BC.parts(T, cs["lab"], GRP, *tabs, 0),
                                E=BC.energy(T, cs["lab"], Eh, Ev, logz, cs["pc"], ts.WALL)[0], extra=extra))
            marks += "+" if good else "-"
            if sd == 0:
                row.append(T)
        line += f"  {md} {marks}"
        if md == "field":
            line += f" (invalid cells, nodes g > 0: {[r['extra'] for r in res[md][-SEEDS:]]})"
    print(line, flush=True)
    figs.append((row, cs["lab"]))

print("ms per sweep: " + "  ".join(f"{k} {v * 1e3:.0f}" for k, v in tsw.items()))
print(f"{'method':6s} {'success':>8s} {'first sweep':>12s} {'E/cell':>7s}   broken parts (closed open merge exit)")
for md in METHODS:
    rr = res[md]
    fs = [r["first"] for r in rr if r["first"] is not None]
    eg = [r["E"] for r in rr if r["ok"]]
    print(f"{md:6s} {np.mean([r['ok'] for r in rr]):8.3f} {np.median(fs) if fs else float('nan'):12.0f} "
          f"{(np.mean(eg) / K ** 2) if eg else float('nan'):7.3f}   "
          + " ".join(f"{v:.2f}" for v in np.mean([r["left"] > 0 for r in rr], 0)))

px, gap = 8, 12
w = K * px
COL = {1: (220, 50, 50), BC.OPEN: (40, 120, 220), 0: (150, 150, 150)}
cols = ["parent window"] + METHODS
img = Image.new("RGB", (len(cols) * (w + 2 * gap) + gap, len(figs) * (w + 2 * gap) + 20), "white")
d = ImageDraw.Draw(img)
p = os.path.join(tempfile.mkdtemp(), "x.png")
for j, name in enumerate(cols):
    d.text((gap + j * (w + 2 * gap) + 4, 4), name, fill="black")
for i, (row, lab) in enumerate(figs):
    for j, T in enumerate(row):
        x0, y0 = gap + j * (w + 2 * gap), 20 + gap + i * (w + 2 * gap)
        ex.to_png(ts, T, p, px)
        img.paste(Image.open(p).convert("RGB"), (x0, y0))
        for q in range(8):
            dd, h = divmod(q, 2)
            a, b = h * w // 2 + 2, (h + 1) * w // 2 - 2
            box = ((x0 + a, y0 - 6, x0 + b, y0 - 2), (x0 + w + 2, y0 + a, x0 + w + 6, y0 + b),
                   (x0 + a, y0 + w + 2, x0 + b, y0 + w + 6), (x0 - 6, y0 + a, x0 - 2, y0 + b))[dd]
            d.rectangle(box, fill=COL[int(lab[q])])
        if j and not ok(T, lab):
            d.text((x0 + 2, y0 + 2), "x", fill="red")
img.save("images/block_field.png")
