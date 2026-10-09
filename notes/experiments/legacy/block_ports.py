"""One 16 x 16 block under a labelled port pattern (castlegen/legacy/blockconn.py):
can the tiles be sampled to satisfy it in a reasonable number of steps?

Castle (demo tile set, cache/castle_ex.npy).  Per case: a parent coordinate c
(uniform on the exemplar torus; the only use of hierarchy) and a pattern, a
class per open port (ports of one class are joined outside the block):
  own    the parent's 16 x 16 window's open / closed ports, each open port a
         class in 1..NC uniformly
  rand   each port closed w.p. 0.3, else a class in 1..NC uniformly
Groups GRP (one per class A..D, default all 1): the classes of a group must
be joined inside the block; all in one group = the block plus the classes'
virtual nodes is one component (blockconn docstring).
The chain (blockconn.sweep): tile heat-bath at T = 1 on E_pair - logz + BETA
-log p(t | exemplar tiles within R of the site's coordinate in c's window),
closed ports facing wall, from c's window.  Methods, per case and seed:
  reject     lam = 0: BURN sweeps, then RS sweeps; acceptance = the fraction
             of those sweeps ending on a satisfying block
  <mode>     lam ramped 0 -> LAMAX over SW sweeps, then HOLD at LAMAX
             (score mode binary | comps | nodes, weights 1); success = the
             block satisfies at the end; first = the first satisfying sweep
Reports per pattern source and method, and the pair energy per cell of the
satisfying ends against the lam = 0 chain's.  images/block_ports.png: rows
of cases (ports drawn as bars, colour = label, grey = closed), columns
parent window, the lam = 0 chain's last block, each mode's end."""
import os, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import blockconn as BC, exemplar as ex, tileset

K = 16
NPAR = int(os.environ.get("NPAR", 6))
SEEDS = int(os.environ.get("SEEDS", 3))
PATS = os.environ.get("PATS", "own,rand").split(",")
NC = int(os.environ.get("NC", 3))
GRP = np.array([1] + [int(g) for g in os.environ.get("GRP", "1,1,1,1").split(",")], np.int64)
MODES = os.environ.get("MODES", "binary,comps,nodes").split(",")
SW, HOLD = int(os.environ.get("SW", 40)), int(os.environ.get("HOLD", 20))
LAMAX = float(os.environ.get("LAMAX", 8))
BETA, R = float(os.environ.get("BETA", 1)), int(os.environ.get("R", 2))
BURN, RS = int(os.environ.get("BURN", 20)), int(os.environ.get("RS", 100))
NFIG = int(os.environ.get("NFIG", 6))

ts = tileset.load("demo")
E = np.load("cache/castle_ex.npy").astype(np.int64)
m = E.shape[0]
S = ts.n_sig
tabs = BC.tables(ts)
t = ts.np_tables
Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
allowed = np.zeros(S, bool)
allowed[np.unique(E)] = True                                               # the exemplar's tiles (no gate)
PC = BC.parent_cost(E, S, R)
yy, xx = np.mgrid[:K, :K]
W8 = np.ones(4)


def window(c):
    cy, cx = divmod(int(c), m)
    return (cy - K // 2 + yy) % m, (cx - K // 2 + xx) % m


def classes(open_, rng):
    return np.where(open_, rng.integers(1, NC + 1, 8), 0)


def ok(T, lab):
    return BC.parts(T, lab, GRP, *tabs, 0).sum() == 0


def run(T, lab, pc, rng, lam_of, n, mode="comps", on_sweep=None):
    T = T.copy()
    for s in range(n):
        BC.sweep(T, rng.permutation(K * K), rng.gumbel(size=(K * K, S)), Eh, Ev, logz, allowed, pc, lab, GRP,
                 *tabs, BC.MODES[mode], W8, lam_of(s), ts.WALL)
        if on_sweep:
            on_sweep(s, T)
    return T


rng0 = np.random.default_rng(0)
cases = []
for i in range(NPAR):
    c = int(rng0.integers(m * m))
    Y, X = window(c)
    Wt = np.ascontiguousarray(E[Y, X])
    pc = np.ascontiguousarray(BETA * PC[Y * m + X]) if BETA else np.zeros((K, K, S))
    for src in PATS:
        lab = classes(BC.pattern(Wt, tabs) > 0 if src == "own" else rng0.random(8) >= 0.3, rng0)
        cases.append(dict(c=c, src=src, lab=lab, W=Wt, pc=pc))

res = {(src, meth): [] for src in PATS for meth in ["reject"] + MODES}
figs = []
t_sw = {}
for ci, cs in enumerate(cases):
    lab, Wt, pc = cs["lab"], cs["W"], cs["pc"]
    wall = BC.parts(Wt, lab, GRP, *tabs, 1)
    row = dict(W=Wt, lab=lab, ends={})
    line = f"case {ci:2d} c {cs['c']:4d} {cs['src']:5s} [{BC.fmt(lab)}]  window parts {wall.astype(int)}"
    for sd in range(SEEDS):
        rng = np.random.default_rng(1000 * ci + sd)
        hits, E0, P0 = [], [], np.zeros(4)

        def rec(s, T):
            if s >= BURN:
                p = BC.parts(T, lab, GRP, *tabs, 0)
                hits.append(p.sum() == 0)
                P0[:] += p > 0
                E0.append(BC.energy(T, lab, Eh, Ev, logz, pc, ts.WALL)[0])
        t0 = time.time()
        Tr = run(Wt, lab, pc, rng, lambda s: 0.0, BURN + RS, on_sweep=rec)
        t_sw["reject"] = (time.time() - t0) / (BURN + RS)
        res[(cs["src"], "reject")].append(dict(acc=np.mean(hits), E=np.mean(E0), viol=P0 / RS))
        if sd == 0:
            row["ends"]["lam 0"] = Tr
        for md in MODES:
            first = [None]

            def rec2(s, T):
                if first[0] is None and ok(T, lab):
                    first[0] = s
            t0 = time.time()
            Ta = run(Wt, lab, pc, rng, lambda s: LAMAX * min(1.0, (s + 1) / SW), SW + HOLD, md, rec2)
            t_sw[md] = (time.time() - t0) / (SW + HOLD)
            good = ok(Ta, lab)
            res[(cs["src"], md)].append(dict(ok=good, first=first[0], E=BC.energy(Ta, lab, Eh, Ev, logz, pc, ts.WALL)[0],
                                             left=BC.parts(Ta, lab, GRP, *tabs, 0)))
            if sd == 0:
                row["ends"][md] = Ta
    print(line + "  acc " + " ".join(f"{r['acc']:.2f}" for r in res[(cs['src'], 'reject')][-SEEDS:])
          + "  | " + " ".join(f"{md} " + "".join("+" if r["ok"] else "-" for r in res[(cs['src'], md)][-SEEDS:])
                             for md in MODES), flush=True)
    figs.append(row)

print(f"\nseconds per sweep: " + "  ".join(f"{k} {v * 1e3:.0f} ms" for k, v in t_sw.items()))
print(f"{'pattern':7s} {'method':7s} {'success':>8s} {'first sweep':>12s} {'E/cell':>7s}   violated parts (closed open merge exit)")
for src in PATS:
    rj = res[(src, "reject")]
    e0 = np.mean([r["E"] for r in rj]) / K ** 2
    print(f"{src:7s} {'reject':7s} {np.mean([r['acc'] for r in rj]):8.3f} {'':>12s} {e0:7.3f}   "
          + " ".join(f"{v:.2f}" for v in np.mean([r["viol"] for r in rj], 0)))
    for md in MODES:
        rr = res[(src, md)]
        fs = [r["first"] for r in rr if r["first"] is not None]
        eg = [r["E"] for r in rr if r["ok"]]
        print(f"{src:7s} {md:7s} {np.mean([r['ok'] for r in rr]):8.3f} {np.median(fs) if fs else float('nan'):12.0f} "
              f"{(np.mean(eg) / K ** 2) if eg else float('nan'):7.3f}   "
              + " ".join(f"{v:.2f}" for v in np.mean([r["left"] > 0 for r in rr], 0)))

# ---- figure
px, gap = 8, 12
w = K * px
COL = {1: (220, 50, 50), 2: (40, 120, 220), 3: (40, 170, 70), 4: (230, 160, 20), 0: (150, 150, 150)}
cols = ["parent window", "lam 0"] + MODES
sel = [figs[i] for i in np.linspace(0, len(figs) - 1, min(NFIG, len(figs))).astype(int)]
img = Image.new("RGB", (len(cols) * (w + 2 * gap) + gap, len(sel) * (w + 2 * gap) + 20), "white")
d = ImageDraw.Draw(img)
p = os.path.join(tempfile.mkdtemp(), "x.png")
for j, name in enumerate(cols):
    d.text((gap + j * (w + 2 * gap) + 4, 4), name, fill="black")
for i, row in enumerate(sel):
    for j, name in enumerate(cols):
        T = row["W"] if j == 0 else row["ends"][name]
        x0, y0 = gap + j * (w + 2 * gap), 20 + gap + i * (w + 2 * gap)
        ex.to_png(ts, T, p, px)
        img.paste(Image.open(p).convert("RGB"), (x0, y0))
        for q in range(8):
            dd, h = divmod(q, 2)
            a, b = h * w // 2 + 2, (h + 1) * w // 2 - 2
            box = ((x0 + a, y0 - 6, x0 + b, y0 - 2), (x0 + w + 2, y0 + a, x0 + w + 6, y0 + b),
                   (x0 + a, y0 + w + 2, x0 + b, y0 + w + 6), (x0 - 6, y0 + a, x0 - 2, y0 + b))[dd]
            d.rectangle(box, fill=COL[int(row["lab"][q])])
        if j and not ok(T, row["lab"]):
            d.text((x0 + 2, y0 + 2), "x", fill="red")
img.save("images/block_ports.png")
