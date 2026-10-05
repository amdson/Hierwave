"""Large structures from macro objects (castlegen/macroobj.py): rooms and
corridors (tilesets/macro_rooms.json) with A/B doorway ports, one object per
8 x 8 block at any phase, every attachment hard at the end.

Exemplar: kinetic growth on an EB x EB block torus (default 96: 768^2
cells), then closed (no unmatched port).  Tables: one chemical potential mu
per (family, ports walled up), moment-matched so a FIT x FIT rules-only chain
at T = 1, lam = LAMFIT holds the exemplar's per-block frequencies.

Variants on an N x N block torus (default 200: 1600^2 cells):
  rules   block Gibbs only, from empty: SW sweeps, T 3 -> 1 and lam 0.5 -> 4
          over the first 2/3, then lam 4 -> HARD at T = 1.
  hier    coarse exemplar coordinates per R x R block window (R = 8): random
          U pasted, CSW coarse sweeps on lam (unmatched - KAPPA objects), lam
          2 -> 8, T 2 -> 0.3; then FSW block sweeps at T = 1, lam 2 -> HARD,
          plus NU per block off the pasted labels (the parent term).
Reports per variant: seconds, objects, unmatched ports, density, components,
the largest, the share of objects in components of >= 20, the share of
doors walled up, family frequencies vs the exemplar; for hier the share of
coherent window seams and where the block stage edited.
images/macro_objects.png: exemplar | rules | hier, the same CROP^2-cell crop
at PX px per cell; images/macro_objects_full.png: the whole hier map."""
import os, time
import numpy as np
from PIL import Image
from castlegen import macroobj as MO

N = int(os.environ.get("N", 200))
EB = int(os.environ.get("EB", 96))
R = int(os.environ.get("R", 8))
SW, CSW, FSW = int(os.environ.get("SW", 40)), int(os.environ.get("CSW", 8)), int(os.environ.get("FSW", 20))
HARD = float(os.environ.get("HARD", 1000))
FIT, LAMFIT = int(os.environ.get("FIT", 32)), float(os.environ.get("LAMFIT", 8))
CROP, PX = int(os.environ.get("CROP", 384)), int(os.environ.get("PX", 2))
KAPPA, NU = float(os.environ.get("KAPPA", 0.3)), float(os.environ.get("NU", 3.0))
SEED = int(os.environ.get("SEED", 1))
VARIANTS = os.environ.get("VARIANTS", "rules,hier").split(",")
IMG = os.environ.get("IMG", "/Users/amdson/dev/Hierwave/images")
TILES = os.environ.get("TILES", "macro_rooms")
OUT = os.environ.get("OUT", "macro_objects")

C = MO.load(TILES)
F = len(C.fams)
print(f"{C.K} oriented components ({C.FULL.sum()} with every port) in {F} families, max {C.MAXP} ports")


def show(name, m, dt=None):
    t = "" if dt is None else f"{dt:6.2f}s "
    print(f"{name:8s} {t}objects {m['objects']:5d} unmatched {m['unmatched']:4d} density {m['density']:.2f} "
          f"comps {m['comps']:4d} largest {m['largest']:5d} in>=20 {m['in_big']:.2f}")


# exemplar
t0 = time.time()
X = MO.State(EB, EB)
fam = C.fam
k0 = int(np.nonzero((fam == 0) & C.FULL)[0][0])        # the first family seeds growth
MO.grow(k0, EB * EB, C.weight.astype(np.float64), fam, C.FULL, C.tb, X.st, SEED, 2000000)
left = MO.close(C.SEAL, C.tb, X.st, SEED, 50)
mx = MO.measure(C, X)
show("exemplar", mx, time.time() - t0)
xfreq = np.bincount(fam[X.lab[X.lab >= 0]], minlength=F) / (EB * EB)

# tables: mu per group (family, sealed ports) by moment matching
full_ports = np.array([C.PN[(fam == f) & C.FULL].max() for f in range(F)])
nseal = full_ports[fam] - C.PN
grp = fam * (C.MAXP + 1) + nseal
G = F * (C.MAXP + 1)
gfreq = lambda S: np.bincount(grp[S.lab[S.lab >= 0]], minlength=G) / S.lab.size
sealed = lambda S: nseal[S.lab[S.lab >= 0]].sum() / full_ports[fam[S.lab[S.lab >= 0]]].sum()
xg = gfreq(X)
t0 = time.time()
mu = np.full(G, 2.0)
Fs = MO.State(FIT, FIT)
for it in range(24):
    MO.sweeps(np.full(6, LAMFIT), np.ones(6), mu, grp, C.tb, Fs.st, SEED + it)
    mu += 0.7 * (np.log(gfreq(Fs) + 1e-4) - np.log(xg + 1e-4))
print(f"mu fit {time.time() - t0:.1f}s; sealed door share: exemplar {sealed(X):.3f}, fit chain {sealed(Fs):.3f}")
print("  mu (full variant):", " ".join(f"{n}={mu[f * (C.MAXP + 1)]:.1f}" for f, n in enumerate(C.fams)))

panels, full = [MO.render(C, X, 0, 0, CROP, CROP, PX)], None
for v in VARIANTS:
    S = MO.State(N, N)
    t0 = time.time()
    if v == "rules":
        a = 2 * SW // 3
        lams = np.concatenate([np.linspace(0.5, 4, a), np.geomspace(4, HARD, SW - a)])
        Ts = np.concatenate([np.geomspace(3, 1, a), np.ones(SW - a)])
        MO.sweeps(lams, Ts, mu, grp, C.tb, S.st, SEED)
    else:
        rng = np.random.default_rng(SEED)
        U = rng.integers(0, EB, (N // R, N // R, 2)).astype(np.int64)
        MO.paste_all(U, R, X.st, EB, C.tb, S.st)
        show("pasted", MO.measure(C, S))
        for s in range(CSW):                       # coarse energy: lam (unmatched - KAPPA objects)
            lc = 2 + 6 * s / max(CSW - 1, 1)
            MO.coarse_sweeps(U, R, np.array([lc]), np.array([2 * 0.15 ** (s / max(CSW - 1, 1))]), 24,
                             np.full(G, -KAPPA * lc), grp, X.st, EB, C.tb, S.st, SEED + s)
        show("coarse", MO.measure(C, S), time.time() - t0)
        cy = len(U)
        coh = np.mean([((U[(y + dy) % cy, (x + dx) % cy] - U[y, x] - R * np.array([dy, dx])) % EB == 0).all()
                       for y in range(cy) for x in range(cy) for dy, dx in ((0, 1), (1, 0))])
        print(f"         coherent window seams {coh:.2f}, distinct exemplar windows {len(np.unique(U.reshape(-1, 2), axis=0))}"
              f" of {U.shape[0] * U.shape[1]}")
        pa = (S.lab.copy(), S.phy.copy(), S.phx.copy())
        MO.sweeps(np.geomspace(2, HARD, FSW), np.ones(FSW), mu, grp, C.tb, S.st, SEED, pa, NU)
        diff = ((S.lab != pa[0]) | (S.phy != pa[1]) | (S.phx != pa[2])).reshape(N, N)
        by, bx = np.mgrid[:N, :N]
        near = (np.minimum(by % R, R - 1 - by % R) < 1) | (np.minimum(bx % R, R - 1 - bx % R) < 1)
        print(f"         block edits by the fine stage: {diff[near].mean():.3f} of blocks on window seams, "
              f"{diff[~near].mean():.3f} inside windows")
        full = MO.render(C, S)
        np.savez(os.path.join("cache", OUT + "_hier.npz"), lab=S.lab, phy=S.phy, phx=S.phx, BY=S.BY, BX=S.BX)
    m = MO.measure(C, S)
    show(v, m, time.time() - t0)
    print(f"         sealed door share {sealed(S):.3f} (exemplar {sealed(X):.3f})")
    print("         fam freq/block", " ".join(f"{n}={a:.3f}/{b:.3f}" for n, a, b in
                                         zip(C.fams, m["famfreq"] * m["objects"] / (N * N), xfreq)))
    print("         top component sizes", m["sizes"][:8].tolist(), "| exemplar", mx["sizes"][:4].tolist())
    panels.append(MO.render(C, S, 0, 0, CROP, CROP, PX))

gap = np.full((CROP * PX, 12, 3), 255, np.uint8)
row = np.concatenate(sum([[p, gap] for p in panels], [])[:-1], 1)
os.makedirs(IMG, exist_ok=True)
Image.fromarray(row).save(os.path.join(IMG, OUT + ".png"))
if full is not None:
    Image.fromarray(full).save(os.path.join(IMG, OUT + "_full.png"))
print("saved", os.path.join(IMG, OUT + ".png"))
