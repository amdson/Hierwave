"""Full-scale macro generation with connectivity promises
(castlegen/macroobj.py, tile set TILES, default macro_dense).

Map: NWIN x NWIN windows (torus) of WB x WB blocks of 8 x 8 cells (default
8 x 8 windows of 16 x 16 blocks: 1024^2 cells, ~15k rooms).

Promises (high level): a uniform spanning tree of the window grid (Wilson);
each window but the root has a ROOT side, toward its tree parent.
Window contract: every room of a window is joined, by attachments inside
the window, to a room with an attachment across the root side (into the
parent window); the root window's rooms are one component.  If every window
keeps its contract the map is one structure (induction from the root: each
window's rooms reach a parent room, which reaches the root).  (Allowing
pieces to join "through" a promised side, by the attachments of both
windows across it, is not sound: two windows can each be split in two with
the halves paired across their seam.)

Coarse: exemplar coordinates per window (as macro_objects.py's hier,
windows of WB blocks), pasted.  Fine: SW block sweeps at T = 1, lam 2 ->
HARD, NU per block off the pasted labels, with the spanning-forest heuristic
in window mode (macroobj.conn_state): per-window components, promised sides
as virtual nodes (root side rooted), BETA / GAMMA / GSMALL.  Baseline
(METHODS plain): the same sweeps without the heuristic.

Reports per method: unmatched doors, windows breaking (a) / (b), the map's
components and largest, time.  images/macro_full_<method>.png (1 px per
cell, promised edges drawn between window centres, white: tree, grey:
extra), images/macro_full_<method>_components.png (coloured by component)."""
import os, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen import macroobj as MO

TILES = os.environ.get("TILES", "macro_dense")
NWIN, WB = int(os.environ.get("NWIN", 8)), int(os.environ.get("WB", 16))
EB = int(os.environ.get("EB", 96))
ROOTMIN = int(os.environ.get("ROOTMIN", 20))
GRAMP = float(os.environ.get("GRAMP", 0.5))
LOWF, LAMLOW = float(os.environ.get("LOWF", 0.0)), float(os.environ.get("LAMLOW", 2))
CSW, SW = int(os.environ.get("CSW", 8)), int(os.environ.get("SW", 80))
HARD, NU, KAPPA = float(os.environ.get("HARD", 1000)), float(os.environ.get("NU", 1.0)), 0.3
BETA, GAMMA, GSMALL = (float(os.environ.get(k, v)) for k, v in (("BETA", 8), ("GAMMA", 8), ("GSMALL", 30)))
METHODS = os.environ.get("METHODS", "forest,plain").split(",")
SEED = int(os.environ.get("SEED", 1))
IMG = os.environ.get("IMG", "/Users/amdson/dev/Hierwave/images")
N = NWIN * WB
NW = NWIN * NWIN
OPP = (2, 3, 0, 1)

t0 = time.time()
C = MO.load(TILES)
F, fam = len(C.fams), C.fam
X = MO.State(EB, EB)
MO.grow(int(np.nonzero((fam == 0) & C.FULL)[0][0]), EB * EB, C.weight.astype(np.float64), fam, C.FULL, C.tb,
        X.st, SEED, 2000000)
MO.close(C.SEAL, C.tb, X.st, SEED, 50)
full_ports = np.array([C.PN[(fam == f) & C.FULL].max() for f in range(F)])
grp = fam * (C.MAXP + 1) + full_ports[fam] - C.PN
G = F * (C.MAXP + 1)
gfreq = lambda S: np.bincount(grp[S.lab[S.lab >= 0]], minlength=G) / S.lab.size
xg = gfreq(X)
mu = np.full(G, 2.0)
Fs = MO.State(32, 32)
for it in range(24):
    MO.sweeps(np.full(6, 8.0), np.ones(6), mu, grp, C.tb, Fs.st, SEED + it)
    mu += 0.7 * (np.log(gfreq(Fs) + 1e-4) - np.log(xg + 1e-4))
print(f"exemplar and tables {time.time() - t0:.1f}s")

# ---- promises: Wilson's uniform spanning tree on the window torus, extra edges, root sides
rng = np.random.default_rng(SEED)
nbr = lambda w, d: (((w // NWIN + (-1, 0, 1, 0)[d]) % NWIN) * NWIN + (w % NWIN + (0, 1, 0, -1)[d]) % NWIN)
intree = np.zeros(NW, bool)
intree[0] = True
nxt = np.full(NW, -1)
for w0 in rng.permutation(NW):
    w = w0
    while not intree[w]:                                   # loop-erased random walk
        nxt[w] = rng.integers(4)
        w = nbr(w, nxt[w])
    w = w0
    while not intree[w]:
        intree[w] = True
        w = nbr(w, nxt[w])
prom = np.zeros((NW, 4), bool)
tree_e = []
for w in range(1, NW):
    d = nxt[w]
    prom[w, d] = prom[nbr(w, d), OPP[d]] = True
    tree_e.append((w, d))
extra_e = []
vroot = np.zeros((NW, 4), bool)
for w in range(1, NW):
    vroot[w, nxt[w]] = True                                # toward the tree parent (window 0 the root)
wsmin = np.full(NW, MO.VBIG)
wsmin[0] = ROOTMIN                                         # the root window: pieces under ROOTMIN rooms
by, bx = np.mgrid[:N, :N]
win = ((by // WB) * NWIN + bx // WB).ravel()
print(f"promises: {len(tree_e)} tree edges; tree degree per window {np.bincount(prom.sum(1), minlength=5).tolist()}"
      " (0..4)")


def contract(S):
    """-> (unmatched doors, windows with no attachment into the parent,
    windows breaking the contract, map components, largest, rooms)."""
    NB = S.lab.size
    gpar, wpar = np.arange(NB), np.arange(NB + NW)
    def find(par, a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    up = np.zeros(NW, bool)
    bad = 0
    rooms = np.nonzero(S.lab >= 0)[0]
    for o in rooms:
        k = S.lab[o]
        ay, ax = MO._anchor(o, C.tb, S.st)
        for j in range(C.PN[k]):
            q = MO._match(k, ay, ax, j, C.tb, S.st)
            if q < 0:
                bad += 1
                continue
            gpar[find(gpar, o)] = find(gpar, q)
            w = win[o]
            if win[q] == w:
                wpar[find(wpar, o)] = find(wpar, q)
            elif w > 0 and nbr(w, nxt[w]) == win[q]:      # across the root side into the parent
                up[w] = True
                wpar[find(wpar, o)] = find(wpar, NB + w)
    miss = int((~up[1:]).sum())
    broken = 0
    for w in range(NW):
        nodes = [o for o in rooms if win[o] == w] + ([NB + w] if w > 0 else [])
        broken += len({find(wpar, a) for a in nodes}) > 1
    roots, sizes = np.unique([find(gpar, o) for o in rooms], return_counts=True)
    return bad, miss, broken, len(sizes), int(sizes.max()), len(rooms)


def render(S, path):
    img = Image.fromarray(MO.render(C, S))
    dr = ImageDraw.Draw(img)
    c = lambda w: ((w % NWIN) * WB * MO.BS + WB * MO.BS // 2, (w // NWIN) * WB * MO.BS + WB * MO.BS // 2)
    for es, col in ((extra_e, (150, 150, 150)), (tree_e, (255, 255, 255))):
        for w, d in es:
            (x0, y0), (x1, y1) = c(w), c(nbr(w, d))
            if abs(x1 - x0) > WB * MO.BS or abs(y1 - y0) > WB * MO.BS:     # wraps the torus: a stub
                x1, y1 = x0 + (0, 1, 0, -1)[d] * WB * MO.BS // 2, y0 + (-1, 0, 1, 0)[d] * WB * MO.BS // 2
            dr.line((x0, y0, x1, y1), fill=col, width=3)
    img.save(path)


def render_comps(S, path):
    NB = S.lab.size
    par = np.empty(NB, np.int64)
    MO.stats(C.tb, S.st, par)
    on = S.lab >= 0
    roots, inv = np.unique(par[on], return_inverse=True)
    pal = np.random.default_rng(0).integers(60, 255, (len(roots), 3)).astype(np.uint8)
    H = N * MO.BS
    img = np.full((H, H, 3), (25, 30, 25), np.uint8)
    comp = np.full(NB, -1)
    comp[on] = inv
    for o in np.nonzero(on)[0]:
        k = S.lab[o]
        ay, ax = MO._anchor(o, C.tb, S.st)
        kc = C.KC[k, :C.KH[k], :C.KW[k]]
        yy, xx = np.nonzero(kc != MO.VOID)
        col = np.where((kc[yy, xx] == MO.WALL)[:, None], (pal[comp[o]] * 0.45).astype(np.uint8), pal[comp[o]])
        img[(ay + yy) % H, (ax + xx) % H] = col
    Image.fromarray(img).save(path)


# ---- coarse: exemplar coordinates per window, pasted
t0 = time.time()
S0 = MO.State(N, N)
U = rng.integers(0, EB, (NWIN, NWIN, 2)).astype(np.int64)
MO.paste_all(U, WB, X.st, EB, C.tb, S0.st)
for s in range(CSW):
    lc = 2 + 6 * s / max(CSW - 1, 1)
    MO.coarse_sweeps(U, WB, np.array([lc]), np.array([2 * 0.15 ** (s / max(CSW - 1, 1))]), 24,
                     np.full(G, -KAPPA * lc), grp, X.st, EB, C.tb, S0.st, SEED + s)
print(f"coarse {time.time() - t0:.1f}s: rooms {(S0.lab >= 0).sum()}")
pa = (S0.lab.copy(), S0.phy.copy(), S0.phx.copy())

for method in METHODS:
    S = S0.copy()
    a = int(LOWF * SW)                                     # lam stays low while orphans dissolve
    lams = np.concatenate([np.geomspace(2, LAMLOW, a), np.geomspace(LAMLOW, HARD, SW - a)])
    t0 = time.time()
    hist = []
    for s0 in range(0, SW, 10):
        g = GSMALL * min(1.0, s0 / (GRAMP * SW))           # the orphan penalty ramps in like lam
        cn = MO.conn_state(S.lab.size, C.MAXP, BETA, GAMMA, g, MO.VBIG, on=method == "forest",
                           win=win, nwin=(NWIN, NWIN), prom=vroot, vroot=vroot, wsmin=wsmin)
        MO.sweeps(lams[s0:s0 + 10], np.ones(10), mu, grp, C.tb, S.st, SEED * 100 + s0, pa, NU, cn)
        hist.append(contract(S))
    dt = time.time() - t0
    bad, miss, split, ncomp, big, nrooms = hist[-1]
    ok = bad == 0 and miss == 0 and split == 0
    print(f"{method:7s} {dt:6.1f}s  rooms {nrooms}  unmatched {bad}  windows not reaching their parent {miss}"
          f"  windows breaking the contract {split}  -> map components {ncomp}, largest {big} ({big / nrooms:.1%})"
          f"  {'CONTRACT KEPT' if ok else ''}")
    print("         per 10 sweeps (not reaching parent, breaking, components):", [(h[1], h[2], h[3]) for h in hist])
    render(S, os.path.join(IMG, f"macro_full_{method}.png"))
    render_comps(S, os.path.join(IMG, f"macro_full_{method}_components.png"))
