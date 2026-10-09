"""A fixed distance map (castle_dist's recursive D) and purely local tile
Gibbs over the hall vocabulary only: hall_straight, hall_corner, hall_tee,
hall_cross and chamber (a dead end), every nonempty side set once, all one
socket type.  Energy per map: E_loc (the tile set's pair energies) plus
  MODE=strict: HARD per node but the root without a joined lower neighbour;
  MODE=half:   HARD per node but the root without an open side into a lower
               key, LAMM per such door the tile across does not join.
Heat bath over the 15 tiles, 25 colours (cells 5 apart share no term), from
a uniform start at T = 1.  Reports violations per sweep and components;
images/halls_dist_<MODE>.png (D, tiles)."""
import os, tempfile
import numpy as np
from PIL import Image
from castlegen import connmetrics as CM, exemplar as ex, generic as GN, tileset

n, px = int(os.environ.get("N", 64)), 8
SEED = int(os.environ.get("SEED", 1))
SW = int(os.environ.get("SW", 30))
MODE = os.environ.get("MODE", "half")
HARD = float(os.environ.get("HARD", 1e3))
LAMM = float(os.environ.get("LAMM", 100))
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPP = (2, 3, 0, 1)
VOC = np.arange(15)                                                         # halls and chambers
SAMPLE_D = os.environ.get("SAMPLE_D", "0") == "1"                         # keys 2 D_parent + r, r sampled
RK = int(os.environ.get("RK", 4))
FREEZE = int(os.environ.get("FREEZE", 10 ** 9))                           # keys fixed from this sweep on
KEYSW = int(os.environ.get("KEYSW", 0))                                   # a keys-only pass first, then frozen


def refine(D, root, rng):
    R, C = D.shape
    K = np.zeros((2 * R, 2 * C), np.int64)
    for i in range(R):
        for j in range(C):
            low = [d for d, (dy, dx) in enumerate(DIRS)
                   if 0 <= i + dy < R and 0 <= j + dx < C and D[i + dy, j + dx] < D[i, j]]
            e = np.zeros((2, 2), bool)
            if not low:
                a, b = divmod(int(rng.integers(4)), 2)
                e[a, b] = True
                root = (2 * i + a, 2 * j + b)
            else:
                keep = [d for d in low if rng.random() < 0.5] or [low[rng.integers(len(low))]]
                for d in keep:
                    e[(0, slice(None), 1, slice(None))[d], (slice(None), 1, slice(None), 0)[d]] = True
            r = np.where(e, 0, 2)
            r[~e & (np.roll(e, 1, 0) | np.roll(e, 1, 1))] = 1
            K[2 * i:2 * i + 2, 2 * j:2 * j + 2] = 3 * D[i, j] + r
    return K, root


rng = np.random.default_rng(SEED)
D, root = np.zeros((1, 1), np.int64), (0, 0)
while D.shape[0] < (n // 2 if SAMPLE_D else n):
    D, root = refine(D, root, rng)
if SAMPLE_D:                                                               # the parent keys stay fixed
    PK = 2 * np.repeat(np.repeat(D, 2, 0), 2, 1)
    root = (2 * root[0], 2 * root[1])
    R_ = rng.integers(0, RK, (n, n))
    D = PK + R_
BIG = np.iinfo(np.int64).max // 4
Dp = np.pad(D, 1, constant_values=BIG)
ry, rx = root

ts = tileset.load("demo")
node, Dh, Dv, sock = GN.conn_tables(ts)
Eh, Ev, logz = (np.asarray(ts.np_tables[k], np.float64) for k in ("Eh", "Ev", "logz"))


def joins(d, t, u):
    return (Dv[u, t], Dh[t, u], Dv[t, u], Dh[u, t])[d]


def nb(X, dy, dx):
    return X[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx]


def measure(x):
    """(strict violations, no aim, unmet doors)."""
    X = np.pad(x, 1, constant_values=54)                                    # off map: a wall
    low = np.stack([nb(Dp, dy, dx) < D for dy, dx in DIRS], -1)
    J = np.stack([joins(d, x, nb(X, dy, dx)) & node[nb(X, dy, dx)] for d, (dy, dx) in enumerate(DIRS)], -1)
    aim = low & sock[:, x].transpose(1, 2, 0)
    strict, noaim = ~(low & J).any(-1), ~aim.any(-1)
    strict[ry, rx] = noaim[ry, rx] = False
    return int(strict.sum()), int(noaim.sum()), int((aim & ~J).sum())


def cell_energy(x, py, px):
    """(k, 15) the terms of cells (py, px) for every tile of VOC."""
    X = np.pad(x, 1, constant_values=54)
    t = VOC[None]
    e = -logz[t] + np.zeros((len(py), 1))
    dc = D[py, px][:, None]
    aim = np.zeros(e.shape, bool)
    ok = np.zeros(e.shape, bool)
    for d, (dy, dx) in enumerate(DIRS):
        ny, nx = py + dy, px + dx
        tn = X[ny + 1, nx + 1][:, None]
        inm = ((ny >= 0) & (ny < n) & (nx >= 0) & (nx < n))[:, None]
        e = e + np.where(inm, (Ev[tn, t], Eh[t, tn], Ev[t, tn], Eh[tn, t])[d], 0.0)
        dn = Dp[ny + 1, nx + 1][:, None]
        J = joins(d, t, tn) & node[tn]
        o = OPP[d]
        if MODE == "half":
            a = (dn < dc) & sock[d, t]
            aim |= a
            e = e + LAMM * (a & ~J)
            e = e + LAMM * ((dc < dn) & node[tn] & sock[o, tn] & ~joins(o, tn, t))
        else:
            ok |= (dn < dc) & J
            # the neighbour's own condition, with c's tile t
            other = np.zeros(len(py), bool)
            for d2, (ey, ex_) in enumerate(DIRS):
                if d2 == o:
                    continue
                my, mx = ny + ey, nx + ex_
                tm = X[np.clip(my + 1, 0, n + 1), np.clip(mx + 1, 0, n + 1)]
                dm = Dp[np.clip(my + 1, 0, n + 1), np.clip(mx + 1, 0, n + 1)]
                other |= (dm < dn[:, 0]) & joins(d2, tn[:, 0], tm) & node[tm]
            isroot = ((ny == ry) & (nx == rx))[:, None]
            e = e + HARD * (inm & node[tn] & ~isroot & ~other[:, None] & ~((dc < dn) & joins(o, tn, t)))
    isroot = ((py == ry) & (px == rx))[:, None]
    e = e + HARD * (~(aim if MODE == "half" else ok) & ~isroot)
    return e


def joint_energy(x, py, px):
    """(k, 15 * RK) the terms of cells (py, px) for every (tile, r), half
    connections: c's aim and doors, and the neighbours' aims and doors, which
    depend on c's key."""
    X = np.pad(x, 1, constant_values=54)
    k = len(py)
    t = np.repeat(VOC, RK)[None]                                           # (1, S) tile of each state
    kc = PK[py, px][:, None] + np.tile(np.arange(RK), len(VOC))[None]       # (k, S) key of each state
    e = -logz[t] + np.zeros((k, 1))
    aim = np.zeros((k, t.shape[1]), bool)
    for d, (dy, dx) in enumerate(DIRS):
        ny, nx = py + dy, px + dx
        inm = ((ny >= 0) & (ny < n) & (nx >= 0) & (nx < n))[:, None]
        tn = X[ny + 1, nx + 1][:, None]
        dn = Dp[ny + 1, nx + 1][:, None]
        e = e + np.where(inm, (Ev[tn, t], Eh[t, tn], Ev[t, tn], Eh[tn, t])[d], 0.0)
        J = joins(d, t, tn) & node[tn]
        o = OPP[d]
        a = (dn < kc) & sock[d, t]
        aim |= a
        e = e + LAMM * (a & ~J)
        e = e + LAMM * ((kc < dn) & node[tn] & sock[o, tn] & ~joins(o, tn, t))
        # the neighbour's aim: its other sides (fixed) or its side o into c
        other = np.zeros(k, bool)
        for d2, (ey, ex_) in enumerate(DIRS):
            if d2 == o:
                continue
            my, mx = np.clip(ny + ey + 1, 0, n + 1), np.clip(nx + ex_ + 1, 0, n + 1)
            other |= (Dp[my, mx] < dn[:, 0]) & sock[d2, tn[:, 0]]
        isroot = ((ny == ry) & (nx == rx))[:, None]
        n_aim = other[:, None] | ((kc < dn) & sock[o, tn])
        e = e + HARD * (inm & node[tn] & ~isroot & ~n_aim)
    isroot = ((py == ry) & (px == rx))[:, None]
    return e + HARD * (~aim & ~isroot)


def key_energy(py, px):
    """(k, RK) HARD per cell but the root without a lower neighbour: c's own
    and its neighbours', for each r of c."""
    kc = PK[py, px][:, None] + np.arange(RK)[None]
    e = np.zeros(kc.shape)
    has = np.zeros(kc.shape, bool)
    for d, (dy, dx) in enumerate(DIRS):
        ny, nx = py + dy, px + dx
        inm = ((ny >= 0) & (ny < n) & (nx >= 0) & (nx < n))[:, None]
        dn = Dp[ny + 1, nx + 1][:, None]
        has |= dn < kc
        o = OPP[d]
        other = np.zeros(len(py), bool)
        for d2, (ey, ex_) in enumerate(DIRS):
            if d2 != o:
                other |= Dp[np.clip(ny + ey + 1, 0, n + 1), np.clip(nx + ex_ + 1, 0, n + 1)] < dn[:, 0]
        isroot = ((ny == ry) & (nx == rx))[:, None]
        e = e + HARD * (inm & ~isroot & ~other[:, None] & ~(kc < dn))
    isroot = ((py == ry) & (px == rx))[:, None]
    return e + HARD * (~has & ~isroot)


def key_minima():
    low = np.stack([Dp[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx] < D for dy, dx in DIRS], -1).any(-1)
    low[ry, rx] = True
    return int((~low).sum())


x = rng.choice(VOC, (n, n))
Y, Xc = np.divmod(np.arange(n * n), n)
if SAMPLE_D and KEYSW:
    print(f"keys start: minima {key_minima()}", flush=True)
    for s in range(KEYSW):
        for c in rng.permutation(25):
            sel = (Y % 5 == c // 5) & (Xc % 5 == c % 5)
            py, px_ = Y[sel], Xc[sel]
            g = -key_energy(py, px_)
            D[py, px_] = PK[py, px_] + np.argmax(g + rng.gumbel(size=g.shape), 1)
            Dp[py + 1, px_ + 1] = D[py, px_]
        if s % 5 == 4 or s == KEYSW - 1:
            print(f"  key sweep {s}: minima {key_minima()}", flush=True)
    FREEZE = 0
Y, Xc = np.divmod(np.arange(n * n), n)
print(f"MODE {MODE}  start (strict, no aim, unmet) {measure(x)}", flush=True)
for s in range(SW):
    for c in rng.permutation(25):
        sel = (Y % 5 == c // 5) & (Xc % 5 == c % 5)
        py, px_ = Y[sel], Xc[sel]
        if SAMPLE_D:
            g = -joint_energy(x, py, px_)
            if s >= FREEZE:                                                 # only the current key allowed
                g = np.where(np.arange(g.shape[1])[None] % RK == (D[py, px_] - PK[py, px_])[:, None], g, -np.inf)
            j = np.argmax(g + rng.gumbel(size=g.shape), 1)
            x[py, px_] = VOC[j // RK]
            D[py, px_] = PK[py, px_] + j % RK
            Dp[py + 1, px_ + 1] = D[py, px_]
        else:
            g = -cell_energy(x, py, px_)
            x[py, px_] = VOC[np.argmax(g + rng.gumbel(size=g.shape), 1)]
    if s % 5 == 4 or s == SW - 1:
        print(f"  sweep {s}: (strict, no aim, unmet) {measure(x)}  "
              f"components {CM.global_stats(ts, x, torus=False)['components']}", flush=True)
p = os.path.join(tempfile.mkdtemp(), "x.png")
ex.to_png(ts, x, p, px)
tiles = Image.open(p).convert("RGB")
rank = np.argsort(np.argsort(D.ravel())).reshape(D.shape) / D.size
dimg = Image.fromarray((255 * rank).astype(np.uint8)).resize(tiles.size, Image.NEAREST).convert("RGB")
out = Image.new("RGB", (2 * tiles.size[0] + 8, tiles.size[1]), "white")
out.paste(dimg, (0, 0))
out.paste(tiles, (tiles.size[0] + 8, 0))
out.save(f"images/halls_dist_{MODE}{'_sampled' if SAMPLE_D else ''}.png")
