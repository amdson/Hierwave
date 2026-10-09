"""One 16 x 16-block region of macro rooms (castlegen/legacy/macroobj.py, tile set
TILES, default macro_dense) under a labelled port pattern: can the rooms be
sampled so that, with the outside's connectivity assumed, everything is one
component?  The macro analogue of block_ports.py.

Region: W x W blocks of 8 x 8 cells (default 16: 128^2 cells) inside a
margin of outside cells; rooms must lie inside.  Ports: the region's 8
half-sides q = 2 d + i (d = N, E, S, W; i = 0 the top / left half), each
closed (0) or a class 1..: a door facing out across an open half-side is an
OPENING (attached to that class's virtual node; half-sides of one class are
joined outside), across a closed one it is unmatched.
Success (checked exactly): no unmatched door, every open half-side has an
opening, and the rooms plus the virtual nodes (classes merged) are one
component.
Patterns: one (every half-side class 1), two (N class 1, S class 2), four
(a class per side), eight (a class per half-side), sink (only N's left half
open), corner (N left half class 1, W top half class 2), and NRAND random
ones (each half closed w.p. 0.3, else a class in 1..3, at least one open).
Per pattern and seed: a parent exemplar window (uniform on the exemplar
torus) pasted, then SW block sweeps at T = 1, lam 2 -> HARD, NU off the
parent, with (forest) or without (plain) the spanning-forest heuristic rooted
at the exits (BETA merge / first-opening bonus, GAMMA island, GSMALL joining
no structure that reaches an exit), or (count) the pure component count:
energy BETA x (components of rooms + exit classes, + unopened open
half-sides), i.e. GAMMA = BETA, GSMALL = 0, splits still barred by the
forest's protection.  Reports per method: success rate, the
first successful sweep (median), components left on failures.
images/macro_ports.png: per pattern (rows) the parent window, then METHODS;
exits drawn as bars coloured by class (grey: closed)."""
import os, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import macroobj as MO

TILES = os.environ.get("TILES", "macro_dense")
W = int(os.environ.get("W", 16))
EB = int(os.environ.get("EB", 64))
SEEDS = int(os.environ.get("SEEDS", 4))
NRAND = int(os.environ.get("NRAND", 8))
SW, HARD, NU = int(os.environ.get("SW", 60)), float(os.environ.get("HARD", 1000)), float(os.environ.get("NU", 1.0))
LOWF, LAMLOW = float(os.environ.get("LOWF", 0.0)), float(os.environ.get("LAMLOW", 2.0))
BETA, GAMMA, GSMALL = (float(os.environ.get(k, v)) for k, v in (("BETA", 8), ("GAMMA", 8), ("GSMALL", 30)))
FIT, LAMFIT = 32, 8.0
METHODS = os.environ.get("METHODS", "plain,forest").split(",")
IMG = os.environ.get("IMG", "/Users/amdson/dev/Hierwave/images")
MARG = 2
NBX = W + 2 * MARG
Y0 = MARG * MO.BS

C = MO.load(TILES)
F, fam = len(C.fams), C.fam

# exemplar and tables (as macro_objects.py)
X = MO.State(EB, EB)
MO.grow(int(np.nonzero((fam == 0) & C.FULL)[0][0]), EB * EB, C.weight.astype(np.float64), fam, C.FULL, C.tb,
        X.st, 1, 2000000)
MO.close(C.SEAL, C.tb, X.st, 1, 50)
full_ports = np.array([C.PN[(fam == f) & C.FULL].max() for f in range(F)])
grp = fam * (C.MAXP + 1) + full_ports[fam] - C.PN
G = F * (C.MAXP + 1)
gfreq = lambda S: np.bincount(grp[S.lab[S.lab >= 0]], minlength=G) / S.lab.size
xg = gfreq(X)
mu = np.full(G, 2.0)
Fs = MO.State(FIT, FIT)
for it in range(24):
    MO.sweeps(np.full(6, LAMFIT), np.ones(6), mu, grp, C.tb, Fs.st, 1 + it)
    mu += 0.7 * (np.log(gfreq(Fs) + 1e-4) - np.log(xg + 1e-4))

rng = np.random.default_rng(0)
PATS = {"one": [1] * 8, "two": [1, 1, 0, 0, 2, 2, 0, 0], "four": [1, 1, 2, 2, 3, 3, 4, 4],
        "eight": list(range(1, 9)), "sink": [1, 0, 0, 0, 0, 0, 0, 0], "corner": [1, 0, 0, 0, 0, 0, 0, 2]}
for r in range(NRAND):
    while True:
        p = np.where(rng.random(8) < 0.3, 0, rng.integers(1, 4, 8))
        if p.any():
            break
    PATS[f"rand{r}"] = p.tolist()


def check(S, lab8):
    """-> (ok, unmatched, closed-or-empty open half-sides, components)."""
    NB = S.lab.size
    par = np.arange(NB + 9)
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for q in range(8):
        for q2 in range(q):
            if lab8[q] > 0 and lab8[q] == lab8[q2]:
                par[find(NB + 1 + q)] = find(NB + 1 + q2)
    bad, opened = 0, np.zeros(8, bool)
    rooms = np.nonzero(S.lab >= 0)[0]
    for o in rooms:
        k = S.lab[o]
        ay, ax = MO._anchor(o, C.tb, S.st)
        for j in range(C.PN[k]):
            q = MO._match(k, ay, ax, j, C.tb, S.st)
            if q < 0:
                bad += 1
                continue
            if q > NB:
                opened[q - NB - 1] = True
            par[find(o)] = find(q)
    nodes = list(rooms) + [NB + 1 + q for q in range(8) if lab8[q] > 0]
    comps = len({find(a) for a in nodes})
    missing = int(sum(lab8[q] > 0 and not opened[q] for q in range(8)))
    return bad == 0 and missing == 0 and comps == 1 and len(rooms) > 0, bad, missing, comps


def region_state(lab8, c):
    S = MO.State(NBX, NBX)
    S.region(C.tb, Y0, Y0, W * MO.BS, W * MO.BS, lab8)
    MO._paste(MARG, MARG, W, c[0], c[1], X.st, EB, C.tb, S.st)
    return S


def run(lab8, c, method, seed):
    S = region_state(lab8, c)
    pa = (S.lab.copy(), S.phy.copy(), S.phx.copy())
    if method == "count":                                  # pure count: beta per component / unopened half-side
        g, gs = BETA, 0.0
    else:
        g, gs = GAMMA, GSMALL
    cn = MO.conn_state(S.lab.size, C.MAXP, BETA, g, gs, MO.VBIG, on=method != "plain", vcls=lab8,
                       root=lab8[lab8 > 0][0])                # one root class: the rest must reach it
    a = int(LOWF * SW)                                     # lam held at LAMLOW, then -> HARD
    lams = np.concatenate([np.full(a, LAMLOW), np.geomspace(LAMLOW, HARD, SW - a)])
    first = None
    for s in range(SW):
        MO.sweeps(lams[s:s + 1], np.ones(1), mu, grp, C.tb, S.st, seed * 1000 + s, pa, NU, cn)
        if first is None and s >= SW // 2 and s % 5 == 4 and check(S, lab8)[0]:
            first = s
    return S, check(S, lab8), first


def draw(S, lab8):
    img = MO.render(C, S, Y0 - 4, Y0 - 4, W * MO.BS + 8, W * MO.BS + 8, 2)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    cols = [(150, 150, 150), (230, 60, 60), (60, 120, 230), (240, 200, 40), (60, 200, 90), (200, 80, 220),
            (40, 210, 210), (250, 140, 30), (255, 255, 255)]
    n, o = W * MO.BS * 2, 8
    for q in range(8):
        d, i = divmod(q, 2)
        a, b = o + i * n // 2, o + (i + 1) * n // 2
        box = ((a, 0, b, 5), (o + n + 2, a, o + n + 7, b), (a, o + n + 2, b, o + n + 7), (0, a, 5, b))[d]
        dr.rectangle(box, fill=cols[lab8[q] % len(cols)])
    return np.asarray(im)


rows, res = [], {m: [] for m in METHODS}
t0 = time.time()
for name, lab8 in PATS.items():
    lab8 = np.array(lab8, np.int64)
    line = f"{name:7s} {''.join(str(v) for v in lab8)}"
    figs = []
    for seed in range(SEEDS):
        c = np.random.default_rng(100 + seed).integers(0, EB, 2)
        if seed == 0:
            figs.append(draw(region_state(lab8, c), lab8))
        for m in METHODS:
            S, (ok, bad, miss, comps), first = run(lab8, c, m, seed)
            res[m].append((ok, first, comps))
            if seed == 0:
                figs.append(draw(S, lab8))
            line += f"  {m[0]}:{'ok' if ok else f'x(u{bad} m{miss} c{comps})'}"
    print(line)
    gap = np.full((figs[0].shape[0], 8, 3), 255, np.uint8)
    rows.append(np.concatenate(sum([[f, gap] for f in figs], [])[:-1], 1))
for m in METHODS:
    ok = np.array([r[0] for r in res[m]])
    firsts = [r[1] for r in res[m] if r[1] is not None]
    comps = [r[2] for r in res[m] if not r[0]]
    print(f"{m:7s} success {ok.sum()}/{len(ok)} ({ok.mean():.0%})"
          f"  first satisfying sweep median {np.median(firsts) if firsts else float('nan'):.0f}"
          f"  failures' components {comps}")
print(f"{time.time() - t0:.1f}s")
hgap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
Image.fromarray(np.concatenate(sum([[r, hgap] for r in rows], [])[:-1], 0)).save(
    os.path.join(IMG, os.environ.get("OUT", "macro_ports") + ".png"))
