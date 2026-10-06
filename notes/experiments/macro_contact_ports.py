"""macro_ports.py with CONTACT connectivity (castlegen/macrocontact.py):
rooms are shapes, two rooms are connected when >= MINC slot cells of
door-capable walls face each other flush (exits likewise), doors are chosen
at rendering.  Same patterns, seeds and success test (on the contact graph).

Tables: the macroobj exemplar (TILES) with every room reopened to its full
variant; mu per family and the contact reward lam moment-matched on a FIT^2
torus to the exemplar's family frequencies and contacts per block.
Per pattern and seed: the parent exemplar window pasted, then SW exact block
heat-bath sweeps at T = 1, NU off the parent, with no heuristic (plain), the
forest heuristic (forest: BETA / GAMMA / GSMALL, rooted at the first class),
or the pure component count (count: GAMMA = BETA, GSMALL = 0).
images/<OUT>.png: per pattern the parent window then METHODS (seed 0), doors
cut along a random spanning forest of the contact graph (+ PEXTRA of the
other contacts)."""
import os, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen import macroobj as MO, macrocontact as MC

TILES = os.environ.get("TILES", "macro_dense")
W = int(os.environ.get("W", 16))
EB = int(os.environ.get("EB", 64))
SEEDS = int(os.environ.get("SEEDS", 4))
NRAND = int(os.environ.get("NRAND", 8))
SW, NU, MINC = int(os.environ.get("SW", 60)), float(os.environ.get("NU", 1.0)), int(os.environ.get("MINC", 3))
BETA, GAMMA, GSMALL = (float(os.environ.get(k, v)) for k, v in (("BETA", 8), ("GAMMA", 8), ("GSMALL", 30)))
PEXTRA = float(os.environ.get("PEXTRA", 0.15))
FIT = 24
METHODS = os.environ.get("METHODS", "plain,forest,count").split(",")
IMG = os.environ.get("IMG", "/Users/amdson/dev/Hierwave/images")
OUT = os.environ.get("OUT", "macro_contact_ports")
MARG = 2
NBX = W + 2 * MARG
Y0 = MARG * MO.BS

t0 = time.time()
C = MO.load(TILES)
ct = MC.tables(C)
fo = MC.fullof(C)
F, fam = len(C.fams), C.fam
X = MO.State(EB, EB)
MO.grow(int(np.nonzero((fam == 0) & C.FULL)[0][0]), EB * EB, C.weight.astype(np.float64), fam, C.FULL, C.tb,
        X.st, 1, 2000000)
MO.close(C.SEAL, C.tb, X.st, 1, 50)
MC.to_full(X, C, fo)
ffreq = lambda S: np.bincount(fam[S.lab[S.lab >= 0]], minlength=F) / S.lab.size
cpb = lambda S: len(MC.contact_graph(ct, S, MINC)) / S.lab.size
xf, xc = ffreq(X), cpb(X)
mu, lam = np.zeros(F), 2.0
Fs = MO.State(FIT, FIT)
for it in range(30):
    MC.sweeps(np.full(3, lam), np.ones(3), mu, fam, ct, Fs.st, 1 + it)
    mu += 0.7 * (np.log(ffreq(Fs) + 1e-4) - np.log(xf + 1e-4))
    lam = max(0.0, lam + 2.0 * (xc - cpb(Fs)))
print(f"tables {time.time() - t0:.1f}s: lam {lam:.2f}, contacts/block exemplar {xc:.3f} fit {cpb(Fs):.3f}; "
      "family freq exemplar/fit " + " ".join(f"{n}={a:.3f}/{b:.3f}" for n, a, b in zip(C.fams, xf, ffreq(Fs))))

rng = np.random.default_rng(0)
PATS = {"one": [1] * 8, "two": [1, 1, 0, 0, 2, 2, 0, 0], "four": [1, 1, 2, 2, 3, 3, 4, 4],
        "eight": list(range(1, 9)), "sink": [1, 0, 0, 0, 0, 0, 0, 0], "corner": [1, 0, 0, 0, 0, 0, 0, 2]}
for r in range(NRAND):
    while True:
        p = np.where(rng.random(8) < 0.3, 0, rng.integers(1, 4, 8))
        if p.any():
            break
    PATS[f"rand{r}"] = p.tolist()


def region_state(lab8, c):
    S = MO.State(NBX, NBX)
    S.region(C.tb, Y0, Y0, W * MO.BS, W * MO.BS, lab8)
    MO._paste(MARG, MARG, W, c[0], c[1], X.st, EB, C.tb, S.st)
    return S


def run(lab8, c, method, seed):
    S = region_state(lab8, c)
    MC.to_full(S, C, fo)
    pa = (S.lab.copy(), S.phy.copy(), S.phx.copy())
    g, gs = (BETA, 0.0) if method == "count" else (GAMMA, GSMALL)
    cn = MC.conn_state(S.lab.size, BETA, g, gs, MO.VBIG, MINC, on=method != "plain", vcls=lab8,
                       root=lab8[lab8 > 0][0])
    first = None
    for s in range(SW):
        MC.sweeps(np.full(1, lam), np.ones(1), mu, fam, ct, S.st, seed * 1000 + s, pa, NU, cn)
        if first is None and s % 5 == 4 and MC.check(ct, S, lab8, MINC)[0]:
            first = s
    return S, MC.check(ct, S, lab8, MINC), first


def draw(S, lab8, seed=0):
    dl = MC.doors(ct, S, np.random.default_rng(seed), lab8, MINC, PEXTRA)
    img = MC.render(C, ct, S, dl, Y0 - 4, Y0 - 4, W * MO.BS + 8, W * MO.BS + 8, 2)
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
            S0 = region_state(lab8, c)
            MC.to_full(S0, C, fo)
            figs.append(draw(S0, lab8))
        for m in METHODS:
            S, (ok, comps, miss), first = run(lab8, c, m, seed)
            res[m].append((ok, first, comps))
            if seed == 0:
                figs.append(draw(S, lab8))
            line += f"  {m[0]}:{'ok' if ok else f'x(c{comps} m{miss})'}"
    print(line, flush=True)
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
Image.fromarray(np.concatenate(sum([[r, hgap] for r in rows], [])[:-1], 0)).save(os.path.join(IMG, OUT + ".png"))
