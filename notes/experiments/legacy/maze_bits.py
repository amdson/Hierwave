"""A full maze with the bit-sliced stack: maze.py's promise (JointLevel at h =
64, 32, 16, connectivity promise, block port patterns, sink blocks' root at
the centre), then the tile stage by castlegen/bitgibbs.sweep_field (C, single
core): per site a joint (tile, g) | d draw bit-sliced over the demo tile
set, then (g, d) | tile exactly; validity hard; real pairs across block
seams, LSEAM... see the C source.  Energies on a grid of UNIT (default ln 2:
dyadic weights), 4-bit factors (a term of 16 units or more is forbidden).
lam ramps 0 -> LAMAX (default 8) over SW sweeps, then HOLD; the seam rule
costs SEAMW lam.  Reports maze.py's line per seed; images/maze_bits.png:
pasted windows and the result per seed (red: outside the largest component;
block seams drawn)."""
import os, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen import bitgibbs as BG
from castlegen.legacy import blockconn as BC, blockfield as BF, connmetrics as CM, exemplar as ex
from castlegen.legacy import generic as GN, tileset

N, K = int(os.environ.get("N", 128)), 16
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1,2,3").split(","))
SW, HOLD = int(os.environ.get("SW", 100)), int(os.environ.get("HOLD", 100))
LAMAX, SEAMW = float(os.environ.get("LAMAX", 8)), float(os.environ.get("SEAMW", 1.0))
UNIT = float(os.environ.get("UNIT", np.log(2)))
G, D, MU, EPS, DELTA = 8, 96, 1.0, 0.1, 0.05
PX = int(os.environ.get("PX", 6))
PLAN = [(64, 16), (32, 8), (16, 8)]
FB = 4

ts = tileset.load("demo")
E = np.load("cache/castle_ex.npy").astype(np.int32)
m, S = E.shape[0], ts.n_sig
solid = np.asarray(ts.solid, bool)
an = GN.analysis(ts, E, [h for h, _ in PLAN], torus=True)
tabs = BC.tables(ts)
node = tabs[0]
t = ts.np_tables
Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
allowed = np.zeros(S, bool)
allowed[np.unique(E)] = True
PC = BC.parent_cost(E.astype(np.int64), S, 2)
GRP = np.ones(BC.MLAB + 1, np.int64)
pair = BG.pair_tables(Eh, Ev, UNIT, FB)
J, openout, all_, seambad, nodem = BG.field_tables(ts, tabs)
cw = BG.weights(UNIT)
legal = BG.masks(np.broadcast_to(allowed, (N * N, S)).copy())


def promise(seed):
    Up = Vp = Kp = None
    for h, sweeps in PLAN:
        lvl = GN.JointLevel(an, solid, h, N, lam_c=16.0, r=0.25, knn=16, sigma=None if Up is None else h / 2,
                            use_v=True, loc=ts, fast="numba", conn=ts, lam_d=16.0, rk=4)
        lvl.rng = np.random.default_rng(seed * 100 + h)
        if Up is None:
            lvl.init(np.random.default_rng(seed).integers(0, an.my * an.m, (lvl.R, lvl.C)))
            lvl.run(sweeps, T=1.0, T_hot=10.0)
        else:
            lvl.init(lvl.prolong(Up), parent=Up, vparent=Vp, kparent=Kp)
            lvl.run(sweeps, T=1.0)
        Up, Vp, Kp = lvl.U.copy(), lvl.V.copy(), lvl.keys()
    return lvl


def gcost(lam):
    g = np.arange(G + 2)[:, None]
    return BG.stacks(BG.quantize(np.where(node[None], lam * MU * g, lam * MU * EPS * g), UNIT, FB), FB)


def stats(T, labs):
    r = BC.map_report(T, K, labs, GRP, tabs)
    g = CM.global_stats(ts, T, torus=False)
    return (f"blocks broken {r['blocks']:2d}/{labs.shape[0] * labs.shape[1]}  seams {r['seams']:3d}  "
            f"components {g['components']:3d}  outside largest {g['unreached_cells']:.3f}")


def outside_largest(T):
    lab, n = CM.labels(ts, T, torus=False)
    if n == 0:
        return np.zeros(T.shape, bool)
    big = np.argmax(np.bincount(lab[lab >= 0], minlength=n))
    return (lab >= 0) & (lab != big)


rows = []
for sd in SEEDS:
    t0 = time.time()
    lvl = promise(sd)
    O = lvl.V & 15
    labs = BC.block_labels(O, lvl._lowbits(O, lvl.keys()))
    labs9 = np.concatenate([labs, np.full(labs.shape[:2] + (1,), -1)], 2)
    sink = ~(labs == 1).any(2) & (labs > 0).any(2)
    labs9[sink, 8] = (K // 2) * K + K // 2
    T0 = lvl.tiles().astype(np.int64)
    uy, ux = np.divmod(np.repeat(np.repeat(lvl.U, K, 0), K, 1), m)
    ly, lx = np.mgrid[:N, :N] % K
    pc = PC[((uy - K // 2 + ly) % m) * m + (ux - K // 2 + lx) % m]
    print(f"seed {sd}: promise {time.time() - t0:.0f}s", flush=True)
    print(f"  pasted   {stats(T0, labs)}", flush=True)
    t0 = time.time()
    unary = BG.stacks(BG.quantize((-logz[None, None] + pc).reshape(N * N, S), UNIT, FB), FB)
    exits = BG.site_exits(labs9, tabs, K)
    T = T0.astype(np.int32)
    Gf, Df = np.zeros((N, N), np.int64), np.zeros((N, N), np.int64)
    for by in range(N // K):
        for bx in range(N // K):
            sl = (slice(by * K, (by + 1) * K), slice(bx * K, (bx + 1) * K))
            Gf[sl], Df[sl] = BF.init_fields(np.ascontiguousarray(T0[sl]), labs9[by, bx], tabs, G, D)
    Gf, Df = Gf.astype(np.int32), Df.astype(np.int32)
    prep = time.time() - t0
    t0 = time.time()
    for s in range(SW + HOLD):
        lam = LAMAX * min(1.0, (s + 1) / SW)
        BG.sweep_field(T, Gf, Df, K, pair, unary, gcost(lam), J, legal, exits, openout, all_, seambad, labs9,
                       int(round(lam / UNIT)), int(round(lam * SEAMW / UNIT)), nodem, ts.WALL, G, D, lam * MU,
                       lam * MU * EPS, DELTA, cw, sd * 100003 + s, 1)
    dt = time.time() - t0
    T = T.astype(np.int64)
    print(f"  bits     {stats(T, labs)}  ({dt:.1f}s tiles, {dt / (SW + HOLD) / N ** 2 * 1e9:.0f} ns/site; "
          f"tables {prep:.1f}s)", flush=True)
    rows.append(([T0, T], O))

w, gap = N * PX, 10
cols = ["pasted h = 16 windows", f"bit-sliced field stack (lam -> {LAMAX:g})"]
img = Image.new("RGB", (60 + len(cols) * (w + gap), 20 + len(rows) * (w + gap)), "white")
d = ImageDraw.Draw(img)
p = os.path.join(tempfile.mkdtemp(), "x.png")
for j, name in enumerate(cols):
    d.text((60 + j * (w + gap) + 4, 4), name, fill="black")
for i, (row, O) in enumerate(rows):
    d.text((4, 20 + i * (w + gap) + w // 2), f"seed {SEEDS[i]}", fill="black")
    for j, T in enumerate(row):
        ex.to_png(ts, T, p, PX)
        im = np.asarray(Image.open(p).convert("RGB")).astype(float)
        bad = np.repeat(np.repeat(outside_largest(T), PX, 0), PX, 1)
        im[bad] = 0.45 * im[bad] + 0.55 * np.array([230, 30, 30])
        x0, y0 = 60 + j * (w + gap), 20 + i * (w + gap)
        img.paste(Image.fromarray(im.astype(np.uint8)), (x0, y0))
        for by in range(N // K):
            for bx in range(N // K):
                for dd, (a0, a1, b0, b1) in ((1, ((bx + 1) * K, by * K, (bx + 1) * K, (by + 1) * K)),
                                             (2, (bx * K, (by + 1) * K, (bx + 1) * K, (by + 1) * K))):
                    if (dd == 1 and bx + 1 == N // K) or (dd == 2 and by + 1 == N // K):
                        continue
                    op = (O[by, bx] >> dd) & 1
                    d.line((x0 + a0 * PX, y0 + a1 * PX, x0 + b0 * PX, y0 + b1 * PX),
                           fill=(40, 90, 220) if op else (0, 0, 0), width=1 if op else 3)
img.save("images/maze_bits.png")
print("wrote images/maze_bits.png")
