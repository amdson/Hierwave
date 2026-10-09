"""A connected castle: generic's connectivity promise down to h = 16, then a
tile Gibbs per 16 x 16 block guided by its port pattern (castlegen/legacy/blockconn.py).

Promise: JointLevel(conn=ts) at h = 64, 32, 16 on an N x N map (default 128;
castle_multi's settings, lam_d = LAMD, default 16: at 4 a few blocks end with
no descent).  Each 16-block takes from it its exemplar coordinate u, its open
sides O and its downhill sides (open into a lower key); block_labels turns
these into a port pattern (downhill halves class A, uphill halves OPEN, the
root a sink).

Tiles: from the h = 16 windows pasted, map_sweep at T = 1 on E_pair - logz +
BETA -log p(t | exemplar tiles within R of the site's coordinate in its
block's window) (blockconn.parent_cost), plus lam times the block score (mode
MODE, default dist) and SEAMW (default 0.5) per broken seam rule; lam ramps 0 -> LAMAX over SW sweeps,
then HOLD sweeps at LAMAX.  Baseline: the same sweeps at lam = 0.

Reports per seed: blocks breaking a rule, broken seams, the map's components
(connmetrics, map not wrapped), the share of node cells outside the largest,
bad cells and E_pair per cell.  images/maze.png: per seed, the pasted h = 16
windows, the lam = 0 baseline and the guided map, cells outside the largest
component tinted red, block seams drawn (open: thin, closed: thick).

GW=1: the demo tile set plus, for each entrance kind (courtyard_gateway,
great_hall_entrance: a door and open sockets elsewhere), a variant with a
wall on each non-door side (the demo has none, so an entrance on a closed
side cannot close it without losing its door).  The exemplar is remapped by tile name; the tile
chain may then use every tile but the gate, and the parent term is mixed
with the uniform (weight FLOOR, default 1e-3) so a tile the exemplar never
holds costs finite.

REDRAW=k (default 0: off): afterwards, each block that breaks a rule or
touches a broken seam is redrawn up to k times: its parent window pasted
back, then the same lam schedule over its own tiles only, every neighbour
fixed (a block's seams are scored on its side, so a redraw cannot break a
neighbour's own score).  Reports the tries each block needed; the figure
gains a column.

METHOD=field: the tile stage uses the local auxiliary fields instead of the
dist score (castlegen/legacy/blockfield.map_sweep: fully local, the 5 classes of
(x + 2 y) mod 5 exact in parallel; a sink block's root cell is its centre).
The same lam schedule (block_field.py's LAMAX 8 leaves ~4x the broken
blocks on a map: the seam rule is too weak), block_field.py's G, D, MU, LV,
EPS, DELTA.  Reports also the fields' own check (invalid cells, nodes with
g > 0).  The lam = 0 baseline is the same plain tile chain either way."""
import json, os, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import blockconn as BC, blockfield as BF, connmetrics as CM, exemplar as ex, generic as GN, tileset

N = int(os.environ.get("N", 128))
K = 16
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1,2,3").split(","))
LAMD = float(os.environ.get("LAMD", 16))
METHOD = os.environ.get("METHOD", "dist")
FIELD = METHOD == "field"
SW, HOLD = int(os.environ.get("SW", 100)), int(os.environ.get("HOLD", 100))
LAMAX = float(os.environ.get("LAMAX", 32))
FG, FD = int(os.environ.get("G", 8)), int(os.environ.get("D", 96))
MU, LV = float(os.environ.get("MU", 1.0)), float(os.environ.get("LV", 1000.0))
EPS, DELTA = float(os.environ.get("EPS", 0.1)), float(os.environ.get("DELTA", 0.05))
BETA, R = float(os.environ.get("BETA", 1)), int(os.environ.get("R", 2))
MODE = os.environ.get("MODE", "dist")
SEAMW = float(os.environ.get("SEAMW", 0.5))
PX = int(os.environ.get("PX", 6))
REDRAW = int(os.environ.get("REDRAW", 0))
PLAN = [(64, 16), (32, 8), (16, 8)]                                       # (h, sweeps)

GW = os.environ.get("GW", "0") == "1"
FLOOR = float(os.environ.get("FLOOR", 1e-3)) if GW else 0.0
ts = tileset.load("demo")
E = np.load("cache/castle_ex.npy").astype(np.int32)
if GW:
    spec = json.load(open(os.path.join(tileset.TILESET_DIR, "demo.json")))
    for base in ("courtyard_gateway", "great_hall_entrance"):            # door N, the rest open: wall one of them
        kd = next(k for k in spec["kinds"] if k["name"] == base)
        for side in "ESW":
            sides = dict(kd["sides"], **{side: "wall"})
            spec["kinds"].insert(spec["kinds"].index(kd) + 1, dict(kd, name=f"{base}_w{side}", sides=sides))
    spec["name"] = "demo_gw"
    ts0, ts = ts, tileset.compile_spec(spec, base_dir=tileset.TILESET_DIR)
    ids = {ts.sig_name(s): s for s in range(ts.n_sig)}
    E = np.array([ids[ts0.sig_name(s)] for s in range(ts0.n_sig)], np.int32)[E]
solid = np.asarray(ts.solid, bool)
m = E.shape[0]
an = GN.analysis(ts, E, [h for h, _ in PLAN], torus=True)
S = ts.n_sig
tabs = BC.tables(ts)
t = ts.np_tables
Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
allowed = np.zeros(S, bool)
allowed[np.unique(E)] = True                                               # the exemplar's tiles (no gate)
if GW:
    allowed[:] = True
    allowed[ts.GATE] = False
PC = BC.parent_cost(E.astype(np.int64), S, R, floor=FLOOR)
GRP = np.ones(BC.MLAB + 1, np.int64)
W4 = np.ones(4)
NOOBJ = (np.full((N, N), -1, np.int64), np.zeros(0, np.int64), np.full((1, 4), -1, np.int64), allowed, 0.0)


def promise(seed):
    Up = Vp = Kp = None
    for h, sweeps in PLAN:
        lvl = GN.JointLevel(an, solid, h, N, lam_c=16.0, r=0.25, knn=16, sigma=None if Up is None else h / 2,
                            use_v=True, loc=ts, fast="numba", conn=ts, lam_d=LAMD, rk=4)
        lvl.rng = np.random.default_rng(seed * 100 + h)
        if Up is None:
            lvl.init(np.random.default_rng(seed).integers(0, an.my * an.m, (lvl.R, lvl.C)))
            lvl.run(sweeps, T=1.0, T_hot=10.0)
        else:
            lvl.init(lvl.prolong(Up), parent=Up, vparent=Vp, kparent=Kp)
            lvl.run(sweeps, T=1.0)
        Up, Vp, Kp = lvl.U.copy(), lvl.V.copy(), lvl.keys()
    return lvl


def parent_term(U):
    """(N, N, S) BETA -log p(t | ...) for each tile from its block's coordinate."""
    uy, ux = np.divmod(np.repeat(np.repeat(U, K, 0), K, 1), m)
    ly, lx = np.mgrid[:N, :N] % K
    c = ((uy - K // 2 + ly) % m) * m + (ux - K // 2 + lx) % m
    return np.ascontiguousarray(BETA * PC[c]) if BETA else np.zeros((N, N, S))


def field_phase(T0, labs9, pc, rng):
    """METHOD=field: map_sweep with the fields started per block at their least valid values."""
    T = T0.copy()
    Gf, Df = np.zeros((N, N), np.int64), np.zeros((N, N), np.int64)
    for by in range(N // K):
        for bx in range(N // K):
            sl = (slice(by * K, (by + 1) * K), slice(bx * K, (bx + 1) * K))
            Gf[sl], Df[sl] = BF.init_fields(np.ascontiguousarray(T[sl]), labs9[by, bx], tabs, FG, FD)
    for s in range(SW + HOLD):
        lam = LAMAX * min(1.0, (s + 1) / SW)
        BF.map_sweep(T, Gf, Df, K, labs9, int(rng.integers(2 ** 31)), Eh, Ev, logz, allowed, pc, *tabs, lam,
                     MU, LV, EPS, DELTA, FG, FD, ts.WALL, SEAMW)
    bad = gpos = 0
    for by in range(N // K):
        for bx in range(N // K):
            sl = (slice(by * K, (by + 1) * K), slice(bx * K, (bx + 1) * K))
            b, g = BF.field_energy_ok(np.ascontiguousarray(T[sl]), np.ascontiguousarray(Gf[sl]),
                                      np.ascontiguousarray(Df[sl]), labs9[by, bx], tabs, FG)
            bad, gpos = bad + b, gpos + g
    return T, bad, gpos


def tile_phase(T0, labs, pc, rng, lamax):
    T = T0.copy()
    for s in range(SW + HOLD):
        lam = lamax * min(1.0, (s + 1) / SW)
        BC.map_sweep(T, K, rng.permutation(N * N), rng.gumbel(size=(N * N, S)), Eh, Ev, logz, allowed, pc,
                     labs, GRP, *tabs, BC.MODES[MODE], W4, SEAMW, lam, ts.WALL, *NOOBJ)
    return T


def block_bad(T, labs, by, bx):
    """Block (by, bx) breaks a rule or one of its seams."""
    node, Dh, Dv, sock = tabs
    B = np.ascontiguousarray(T[by * K:(by + 1) * K, bx * K:(bx + 1) * K])
    if BC.parts(B, labs[by, bx], GRP, *tabs, 0).sum() > 0:
        return True
    op = lambda t, d: node[t] & sock[d, t]
    y0, x0, s = by * K, bx * K, np.arange(K)
    for a, b, D, da, db in ((T[y0 - 1, x0 + s], T[y0, x0 + s], Dv, 2, 0) if by else (None,) * 5,
                            (T[y0 + K - 1, x0 + s], T[y0 + K, x0 + s], Dv, 2, 0) if y0 + K < N else (None,) * 5,
                            (T[y0 + s, x0 - 1], T[y0 + s, x0], Dh, 1, 3) if bx else (None,) * 5,
                            (T[y0 + s, x0 + K - 1], T[y0 + s, x0 + K], Dh, 1, 3) if x0 + K < N else (None,) * 5):
        if a is not None and ((op(a, da) | op(b, db)) & ~D[a, b]).any():
            return True
    return False


def redraw(T, T0, labs, pc, rng):
    """REDRAW stage (module docstring): (map, {block: tries, -1 if never clean})."""
    T = T.copy()
    out = {}
    for by in range(N // K):
        for bx in range(N // K):
            if not block_bad(T, labs, by, bx):
                continue
            sites = ((by * K + np.arange(K))[:, None] * N + bx * K + np.arange(K)[None]).ravel()
            out[(by, bx)] = -1
            for k in range(1, REDRAW + 1):
                T[by * K:(by + 1) * K, bx * K:(bx + 1) * K] = T0[by * K:(by + 1) * K, bx * K:(bx + 1) * K]
                for s in range(SW + HOLD):
                    lam = LAMAX * min(1.0, (s + 1) / SW)
                    BC.map_sweep(T, K, rng.permutation(sites), rng.gumbel(size=(K * K, S)), Eh, Ev, logz, allowed,
                                 pc, labs, GRP, *tabs, BC.MODES[MODE], W4, SEAMW, lam, ts.WALL, *NOOBJ)
                if not block_bad(T, labs, by, bx):
                    out[(by, bx)] = k
                    break
    return T, out


def stats(T, labs):
    r = BC.map_report(T, K, labs, GRP, tabs)
    g = CM.global_stats(ts, T, torus=False)
    ep = float(BC.energy(T, np.ones(8, np.int64), Eh, Ev, logz, np.zeros((N, N, S)), ts.WALL)[0]) / N ** 2
    return (f"blocks broken {r['blocks']:2d}/{labs.shape[0] * labs.shape[1]} (closed open merge exit {r['parts']})  "
            f"seams {r['seams']:3d}  components {g['components']:3d}  outside largest {g['unreached_cells']:.3f}  "
            f"bad cells {ex.bad_cells(ts, T, torus=False).mean():.3f}  E/cell {ep:.3f}")


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
    low = lvl._lowbits(O, lvl.keys())
    root = np.zeros(O.shape, bool)
    root[lvl.root] = True
    sinks = int(((O != 0) & ~low.any(2) & ~root).sum())
    labs = BC.block_labels(O, low)
    cv = lvl.conn_violations()
    print(f"seed {sd}: promise {time.time() - t0:.0f}s  seams {cv['seams']} parent {cv['parent']} descent {cv['descent']}"
          f"  block components {cv['comps']}  extra sinks {sinks}  open sides {int(sum(((O >> d) & 1).sum() for d in range(4)))}",
          flush=True)
    T0 = lvl.tiles().astype(np.int64)
    pc = parent_term(lvl.U)
    print(f"  pasted   {stats(T0, labs)}", flush=True)
    t0 = time.time()
    Tb = tile_phase(T0, labs, pc, np.random.default_rng(sd), 0.0)
    print(f"  lam 0    {stats(Tb, labs)}  ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    if FIELD:
        labs9 = np.concatenate([labs, np.full(labs.shape[:2] + (1,), -1)], 2)
        sink = ~(labs == 1).any(2) & (labs > 0).any(2)
        labs9[sink, 8] = (K // 2) * K + K // 2
        Tg, fbad, fpos = field_phase(T0, labs9, pc, np.random.default_rng(sd))
        print(f"  guided   {stats(Tg, labs)}  ({time.time() - t0:.0f}s)  fields: invalid {fbad}, nodes g > 0 {fpos}",
              flush=True)
    else:
        Tg = tile_phase(T0, labs, pc, np.random.default_rng(sd), LAMAX)
        print(f"  guided   {stats(Tg, labs)}  ({time.time() - t0:.0f}s)", flush=True)
    row = [T0, Tb, Tg]
    if REDRAW:
        t0 = time.time()
        Tr, tries = redraw(Tg, T0, labs, pc, np.random.default_rng(sd + 7))
        print(f"  redrawn  {stats(Tr, labs)}  ({time.time() - t0:.0f}s)", flush=True)
        print("    tries per block (-1: never clean): " + "  ".join(f"{b}:{k}" for b, k in tries.items()), flush=True)
        row.append(Tr)
    rows.append((row, O))

# ---- figure
w, gap = N * PX, 10
cols = ["pasted h = 16 windows", "lam = 0", f"guided ({'field' if FIELD else MODE}, lam -> {LAMAX:g})"] + ([f"redrawn (<= {REDRAW} tries)"] if REDRAW else [])
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
img.save(f"images/maze{'_gw' if GW else ''}{'_field' if FIELD else ''}.png")
