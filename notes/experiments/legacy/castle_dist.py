"""A fixed distance map, then a hard descent constraint only at h = 1.

D is built recursively from one root cell, independent of the terrain: each
level splits every cell 2 x 2, keeps a random nonempty subset of the parent's
sides into a lower parent, and gives each child key 3 D_parent + r, r its
distance inside the block from the children on those sides (the root parent:
from one random child, the new root).  Then every cell but the root has a
neighbour with a lower key.  The joint sampler runs with no promise at h = 8,
4, 2 (as castle_multi); at h = 1 every node tile but the root pays HARD unless
it joins a node tile with a lower key (the cell's own condition and its
neighbours', through the JointLevel.extra hook).  Reports violations and
components; images/castle_dist.png (D, tiles)."""
import os, tempfile, time
import numpy as np
from PIL import Image
from castlegen.legacy import connmetrics as CM, exemplar as ex, generic as GN, tileset

n, px = 128, 8
SEED = int(os.environ.get("SEED", 1))
SW1 = int(os.environ.get("SW1", 10))
HARD = float(os.environ.get("HARD", 1e3))
HALF = os.environ.get("HALF", "0") == "1"                                # half connections
LAMM = float(os.environ.get("LAMM", 100))                                 # per unmet descent door
BLOCK = int(os.environ.get("BLOCK", 0))                                   # 2 x 2 block moves: candidates per block
PLAN = [(8, 16), (4, 8), (2, 4), (1, SW1)]
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPP = (2, 3, 0, 1)


def refine(D, root, rng):
    R, C = D.shape
    K = np.zeros((2 * R, 2 * C), np.int64)
    for i in range(R):
        for j in range(C):
            low = [d for d, (dy, dx) in enumerate(DIRS)
                   if 0 <= i + dy < R and 0 <= j + dx < C and D[i + dy, j + dx] < D[i, j]]
            ex_ = np.zeros((2, 2), bool)
            if not low:                                                    # the root parent
                a, b = divmod(int(rng.integers(4)), 2)
                ex_[a, b] = True
                root = (2 * i + a, 2 * j + b)
            else:
                keep = [d for d in low if rng.random() < 0.5] or [low[rng.integers(len(low))]]
                for d in keep:
                    ex_[(0, slice(None), 1, slice(None))[d], (slice(None), 1, slice(None), 0)[d]] = True
            r = np.where(ex_, 0, 2)
            r[~ex_ & (np.roll(ex_, 1, 0) | np.roll(ex_, 1, 1))] = 1            # 2 x 2: roll = the other one
            K[2 * i:2 * i + 2, 2 * j:2 * j + 2] = 3 * D[i, j] + r
    return K, root


rng = np.random.default_rng(SEED)
D, root = np.zeros((1, 1), np.int64), (0, 0)
while D.shape[0] < n:
    D, root = refine(D, root, rng)
Dp = np.pad(D, 1, constant_values=np.iinfo(np.int64).max)                 # off map: never lower
ry, rx = root
_low = np.stack([Dp[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx] < D for dy, dx in DIRS]).any(0)
assert _low.sum() == n * n - 1 and not _low[root], "every cell but the root descends"

ts = tileset.load("demo")
solid = np.asarray(ts.solid, bool)
node, Dh, Dv, sock = GN.conn_tables(ts)
E = np.load("cache/castle_ex.npy").astype(np.int32)
an = GN.analysis(ts, E, [h for h, _ in PLAN], torus=True)


def joins(d, t, u):
    """Tile t joins tile u on t's side d."""
    return (Dv[u, t], Dh[t, u], Dv[t, u], Dh[u, t])[d]


class Descent:
    def __init__(self, lvl):
        self.lvl = lvl

    def state(self):
        T = np.pad(self.lvl.Et[self.lvl.U], 1)                             # padding: never a lower key
        c = T[1:-1, 1:-1]
        desc = np.stack([node[c] & (Dp[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx] < D)
                         & joins(d, c, T[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx]) for d, (dy, dx) in enumerate(DIRS)], -1)
        return T, desc

    def viol(self):
        T, desc = self.state()
        v = node[T[1:-1, 1:-1]] & ~desc.any(-1)
        v[ry, rx] = False
        return v

    def __call__(self, py, px, allc):
        T, desc = self.state()
        t = self.lvl.Et[allc]                                               # (k, M)
        dc = D[py, px][:, None]
        own = np.zeros(t.shape, bool)
        pen = np.zeros(t.shape)
        for d, (dy, dx) in enumerate(DIRS):
            ny, nx = py + dy, px + dx
            inm = (ny >= 0) & (ny < n) & (nx >= 0) & (nx < n)
            tn = T[ny + 1, nx + 1][:, None]
            dn = Dp[ny + 1, nx + 1][:, None]
            own |= (dn < dc) & node[tn] & joins(d, t, tn)
            o = OPP[d]
            other = desc[np.clip(ny, 0, n - 1), np.clip(nx, 0, n - 1)].copy()
            other[:, o] = False
            via = (dc < dn) & node[t] & joins(o, tn, t)
            isroot = ((ny == ry) & (nx == rx))[:, None]
            pen += inm[:, None] & node[tn] & ~isroot & ~other.any(1)[:, None] & ~via
        isroot = ((py == ry) & (px == rx))[:, None]
        pen += node[t] & ~own & ~isroot
        return HARD * pen


class HalfDescent(Descent):
    """Half connections: every node tile but the root opens a door socket on
    a side into a lower key (HARD; the tile alone), and each such door that
    the tile across does not join costs LAMM (c's own and its neighbours'
    into c)."""

    def aims(self):
        T = np.pad(self.lvl.Et[self.lvl.U], 1)
        c = T[1:-1, 1:-1]
        low = np.stack([Dp[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx] < D for dy, dx in DIRS], -1)
        aim = node[c][..., None] & low & sock[:, c].transpose(1, 2, 0)
        bad = node[c] & ~aim.any(-1)
        bad[ry, rx] = False
        unmet = aim & ~np.stack([node[T[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx]]
                                 & joins(d, c, T[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx]) for d, (dy, dx) in enumerate(DIRS)], -1)
        return int(bad.sum()), int(unmet.sum())

    def __call__(self, py, px, allc):
        T = np.pad(self.lvl.Et[self.lvl.U], 1)
        t = self.lvl.Et[allc]                                               # (k, M)
        dc = D[py, px][:, None]
        aim = np.zeros(t.shape, bool)
        pen = np.zeros(t.shape)
        for d, (dy, dx) in enumerate(DIRS):
            tn = T[py + dy + 1, px + dx + 1][:, None]
            dn = Dp[py + dy + 1, px + dx + 1][:, None]
            a = (dn < dc) & node[t] & sock[d, t]
            aim |= a
            pen += LAMM * (a & ~(node[tn] & joins(d, t, tn)))
            o = OPP[d]                                                      # the neighbour's door into c
            pen += LAMM * ((dc < dn) & node[tn] & sock[o, tn] & ~(node[t] & joins(o, tn, t)))
        isroot = ((py == ry) & (px == rx))[:, None]
        return pen + HARD * (node[t] & ~aim & ~isroot)


def e_cell(lvl, py, px, cand):
    """(k, M) every term of cell (py, px) at u = cand, the rest of U as is
    (up to a per-cell constant)."""
    e = lvl.lam_c * lvl._cond_batch(py + lvl.B, px, cand, lvl._ext(lvl.U))
    ty, tx = np.divmod(lvl.target[py, px], lvl.m)
    dy = lvl._dy(lvl.cy[cand] - ty[:, None])
    dx = (lvl.cx[cand] - tx[:, None] + lvl.m // 2) % lvl.m - lvl.m // 2
    return e + (dy ** 2 + dx ** 2) / (2.0 * lvl.sigma ** 2) + lvl._loc_batch(py, px, cand) + lvl.extra(py, px, cand)


def block_sweep(lvl, K, rng):
    """2 x 2 block moves: per block, a heat bath over the current block and K
    exemplar 2 x 2 windows (half at the kNN of the top-left cell's parent
    target, half uniform), the energy change of each by telescoping the four
    cells' conditional terms.  Anchors 6 apart share no term (36 colours).
    Approximate: the candidate set includes the current block."""
    R, C, m = lvl.R, lvl.C, lvl.m
    AY, AX = np.divmod(np.arange((R - 1) * (C - 1)), C - 1)
    OFF = ((0, 0), (0, 1), (1, 0), (1, 1))
    for c in rng.permutation(36):
        sel = (AY % 6 == c // 6) & (AX % 6 == c % 6)
        ay, ax = AY[sel], AX[sel]
        k = len(ay)
        w0 = np.concatenate([lvl.knn[lvl.target[ay, ax]][:, :K // 2],
                             rng.integers(0, len(lvl.NE), (k, K - K // 2))], 1)        # (k, K) window anchors
        wy, wx = np.divmod(w0, m)
        cand = [lvl._wy(wy + oy) * m + (wx + ox) % m for oy, ox in OFF]               # per cell (k, K)
        old = [lvl.U[ay + oy, ax + ox].copy() for oy, ox in OFF]
        dE = np.zeros((k, K))
        for j in range(K):
            for i, (oy, ox) in enumerate(OFF):
                py, px = ay + oy, ax + ox
                e = e_cell(lvl, py, px, np.stack([lvl.U[py, px], cand[i][:, j]], 1))
                dE[:, j] += e[:, 1] - e[:, 0]
                lvl.U[py, px] = cand[i][:, j]
            for i, (oy, ox) in enumerate(OFF):
                lvl.U[ay + oy, ax + ox] = old[i]
        g = -np.concatenate([np.zeros((k, 1)), dE], 1)
        g = np.where(np.isfinite(g), g, -np.inf)
        pick = np.argmax(g + rng.gumbel(size=g.shape), 1)
        mv = pick > 0
        for i, (oy, ox) in enumerate(OFF):
            lvl.U[ay[mv] + oy, ax[mv] + ox] = cand[i][mv, pick[mv] - 1]
        yield int(mv.sum())


Up = None
for h, sweeps in PLAN:
    t0 = time.time()
    lvl = GN.JointLevel(an, solid, h, n, lam_c=16.0, r=0.25, knn=16, sigma=None if Up is None else h / 2,
                        use_v=False, loc=ts, fast="numba")
    lvl.rng = np.random.default_rng(SEED * 100 + h)
    if Up is None:
        lvl.init(np.random.default_rng(SEED).integers(0, an.my * an.m, (lvl.R, lvl.C)))
        lvl.run(sweeps, T=1.0, T_hot=10.0)
    else:
        lvl.init(lvl.prolong(Up), parent=Up)
        if h == 1:
            lvl.extra = HalfDescent(lvl) if HALF else Descent(lvl)
            aims = (lambda: "  (no aim, unmet doors) %s" % (lvl.extra.aims(),)) if HALF else (lambda: "")
            print(f"h 1 start: violations {lvl.extra.viol().sum()}{aims()}", flush=True)
            for s in range(sweeps):
                lvl.sweep(1.0)
                moved = sum(block_sweep(lvl, BLOCK, lvl.rng)) if BLOCK else 0
                print(f"  sweep {s}: blocks moved {moved}  violations {lvl.extra.viol().sum()}{aims()}", flush=True)
        else:
            lvl.run(sweeps, T=1.0)
    Up = lvl.U.copy()
    x = lvl.tiles()
    ep = lvl.energy_parts()
    print(f"h {h}: {time.time() - t0:.0f}s  E_c {ep['c']:.0f}  E_par {ep['par']:.0f}  "
          f"bad cells {ex.bad_cells(ts, x, torus=False).mean():.3f}  "
          f"components {CM.global_stats(ts, x, torus=False)['components']}", flush=True)
p = os.path.join(tempfile.mkdtemp(), "x.png")
ex.to_png(ts, x, p, px)
tiles = Image.open(p).convert("RGB")
rank = np.argsort(np.argsort(D.ravel())).reshape(D.shape) / D.size
dimg = Image.fromarray((255 * rank).astype(np.uint8)).resize(tiles.size, Image.NEAREST).convert("RGB")
out = Image.new("RGB", (2 * tiles.size[0] + 8, tiles.size[1]), "white")
out.paste(dimg, (0, 0))
out.paste(tiles, (tiles.size[0] + 8, 0))
out.save("images/castle_dist.png")
