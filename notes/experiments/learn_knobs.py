"""Fit the hand-tuned knobs of chan_combined3.py by moment matching: can
the fit replace the tuning session?

Every soft knob multiplies a statistic of the output linearly (the table in
notes/channels_system.tex, "Every knob is a multiplier on a statistic").
The model being fitted is the budgeted procedure itself (S8 coarse sweeps,
refinement, SF joint sweeps), not an equilibrium.  Each step runs the
procedure once and moves every knob by its statistic's gap,

    c_k += eta_t (s_k(model) - s_k(target)) / scale_k     (too much of a statistic: raise its cost)

Level 8 (lam8, f, cut, bonus, win_bonus): s_k = E(c + e_k) - E(c), the
finite energy of u8 with the knob raised by one, so the statistic is
exactly the one the knob multiplies (the factors are linear in each knob).
Level 1 (nu, lam, mu, dangle): the joint kernel's own terms, counted:
tile-coordinate mismatches, incoherent coordinate pairs (not both FREE),
cells off their parent's refinement, ports facing a non-port.

Targets: the hand-tuned procedure's averages over NT seeds (a recovery
test: the tuning by eye is what the fit should reproduce).  Start: every
knob at 1 (no tuning).  Levels are fitted coarse to fine: level 8 first,
then level 1 with level 8 at its fitted values.  Held-out (not fitted):
windows, trunk windows, recombined seams, root cells, coverage, pruned,
rule violations, no-candidate sites, mass histogram.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/learn_knobs.py
"""
import os, time, json
import numpy as np
from PIL import Image

from castlegen.channels import Kinds, Model, ground, roots, coord

H, W = int(os.environ.get("H", 128)), int(os.environ.get("W", 512))
EX = os.environ.get("EX", "sheet")
S8, SF = int(os.environ.get("S8", 30)), int(os.environ.get("SF", 30))
KR, KT = 1, 1
NT, NE = int(os.environ.get("NT", 6)), int(os.environ.get("NE", 4))     # target seeds, evaluation seeds
STEPS8, STEPS1 = int(os.environ.get("STEPS8", 150)), int(os.environ.get("STEPS1", 100))
ETA = float(os.environ.get("ETA", 0.3))
OUT = os.environ.get("OUT", "images/learn_knobs.json")

K8 = ["lam8", "f", "cut", "bonus", "win_bonus"]
K1 = ["nu", "lam", "mu", "dangle"]
HAND = dict(lam8=1.0, f=0.1, cut=2.0, bonus=4.0, win_bonus=2.0, nu=6.0, lam=1.0, mu=2.0, dangle=1.0)
INIT = {k: 1.0 for k in HAND}

kinds = Kinds.concat(ground.KINDS, roots.KINDS)
tile = roots.tile_views(ground.tile_channel(kinds), kinds)
alpha, g, mask, _ = coord.parse_exemplar(kinds, rows=coord.EXEMPLARS[EX], tree=True)
depth = coord.exemplar_depth(kinds, alpha)
u = coord.coord_channel(alpha)
surf, cert = ground.surf_channel(), roots.cert_channel(roots.certificate().Dmax)
coarse = coord.Coarse(kinds, alpha, ground.CH)
u8, d8 = coarse.channel(), coarse.cert_channel()
trees = roots.trees_channel()
trees.grid = np.zeros((H // ground.CH, W // ground.CH), np.int32)
Model(H, W, [u8, surf, d8, tile, u, cert])                              # allocates the grids
my, mx = alpha.shape
afl = alpha.ravel()


def model8(c):
    return Model(H, W, [u8, surf, d8], coarse.factors(lam8=c["lam8"], f=c["f"], bonus=c["bonus"],
                                                      win_bonus=c["win_bonus"], cut=c["cut"]), [coarse.certificate()])


def run8(c, seed):
    """Level 8 from scratch; returns its statistics (the knobs' multipliers)."""
    ground.sample_surface(surf, H, W, seed=seed, mean_depth=0.4)
    u8.grid[:] = 0
    d8.grid[:] = d8.D - 1
    m8 = model8(c)
    m8.sweep("u8", S8, seed=seed)
    e0 = m8.energy("u8")[0]
    s = {}
    for k in K8:
        c2 = dict(c); c2[k] += 1.0
        s[k] = model8(c2).energy("u8")[0] - e0
    return s


def incoherent_pairs():
    U = u.grid
    n = 0
    for dy, dx in ((0, 1), (1, 0)):
        up, uq = U[:H - dy, :W - dx], U[dy:, dx:]
        coh = (uq // mx - dy == up // mx) & (uq % mx - dx == up % mx)
        free2 = (afl[up] == coord.FREE) & (afl[uq] == coord.FREE)
        n += int((~coh & ~free2).sum())
    return n


def run1(c, seed):
    """Refinement and joint sweeps on the current level 8; level-1 statistics and held-out metrics."""
    rs = [f for f in roots.factors(kinds, dangle=c["dangle"], counted=False) if f.name != "trunk_count"]
    m = Model(H, W, [tile, u, surf, cert], ground.factors() + rs, [roots.certificate(tree=True)])
    _, coup = coord.coupling(kinds, tile, u, nu=c["nu"])
    ck = coord.CoordKernel(u, tile, alpha, lam=c["lam"], nu=c["nu"], K=KR, Kt=KT)
    ground.init_tiles(tile, surf, kinds)
    uref = coarse.refine(u8, u, tile, cert, kinds, alpha, depth)
    pruned = coarse.pruned
    bad = ck.sweep_joint(m, coup, SF, seed, uref=uref, mu=c["mu"])
    r, cm = roots.metrics(tile, cert, trees), coord.metrics(u, tile, alpha)
    s = dict(nu=cm["mismatched"], lam=incoherent_pairs(), mu=int(((uref >= 0) & (u.grid != uref)).sum()),
             dangle=r["dangling_ports"])
    solid = tile.views["solid"][tile.grid].sum()
    ho = dict(windows=int((u8.grid > 0).sum()), trunk_windows=int(coarse.has_trunk[u8.grid].sum()),
              seams=coarse.recombined_seams(u8), root_cells=r["root_cells"], coverage=r["root_cells"] / solid,
              trunks=r["trunks"], pruned=pruned, rule_violations=r["rule_violations"], no_candidate=int(bad),
              viol=int(m.energy("tile")[1]), mass1=r["mass_hist"][0], mass2=r["mass_hist"][1], mass3=r["mass_hist"][2])
    return s, ho


def full(c, seed):
    s8 = run8(c, seed)
    s1, ho = run1(c, seed)
    return {**s8, **s1}, ho


def average(c, seeds):
    S, HO = [], []
    for sd in seeds:
        s, ho = full(c, sd)
        S.append(s); HO.append(ho)
    return ({k: float(np.mean([x[k] for x in S])) for k in S[0]},
            {k: float(np.mean([x[k] for x in HO])) for k in HO[0]})


def fit(c, keys, target, steps, runner, seed0, label):
    """Robbins-Monro on the gap; returns the average of the second half."""
    c = dict(c)
    scale = {k: max(abs(target[k]), 5.0) for k in keys}
    hist = []
    t0 = time.time()
    for t in range(steps):
        s = runner(c, seed0 + t)
        eta = ETA / np.sqrt(1.0 + t / 10.0)
        for k in keys:
            step = np.clip(eta * (s[k] - target[k]) / scale[k], -0.5, 0.5)
            c[k] = max(c[k] + step, 0.0)
        hist.append(dict(c))
        if (t + 1) % 10 == 0:
            gaps = "  ".join(f"{k} {c[k]:.2f} ({s[k]:.0f}/{target[k]:.0f})" for k in keys)
            print(f"  {label} step {t + 1:4d} {time.time() - t0:5.0f}s  {gaps}", flush=True)
    avg = {k: float(np.mean([h[k] for h in hist[steps // 2:]])) for k in keys}
    return {**c, **avg}, hist


# ---- targets: the hand-tuned procedure
t0 = time.time()
target, ho_hand = average(HAND, range(1000, 1000 + NT))
print(f"targets ({NT} seeds, {time.time() - t0:.0f}s):", {k: round(v, 1) for k, v in target.items()})

# ---- level 8, then level 1 on the fitted level 8
c = dict(INIT)
c, h8 = fit(c, K8, target, STEPS8, run8, 10000, "level 8")
print("level 8 fitted:", {k: round(c[k], 2) for k in K8})


def run_both(c_, seed):
    run8(c_, seed)
    return run1(c_, seed)[0]


c, h1 = fit(c, K1, target, STEPS1, run_both, 20000, "level 1")
print("level 1 fitted:", {k: round(c[k], 2) for k in K1})

# ---- evaluation on fresh seeds
seeds = range(3000, 3000 + NE)
rows = {}
for name, cc in (("hand-tuned", HAND), ("untrained (all 1)", INIT), ("fitted", c)):
    s, ho = average(cc, seeds)
    rows[name] = (cc, s, ho)
print("\nknobs:")
print(f"{'':18s}" + "".join(f"{k:>10s}" for k in HAND))
for name, (cc, _, _) in rows.items():
    print(f"{name:18s}" + "".join(f"{cc[k]:10.2f}" for k in HAND))
print("\nfitted statistics (evaluation seeds):")
print(f"{'':18s}" + "".join(f"{k:>10s}" for k in HAND))
for name, (_, s, _) in rows.items():
    print(f"{name:18s}" + "".join(f"{s[k]:10.1f}" for k in HAND))
print("\nheld-out:")
keys = list(ho_hand)
print(f"{'':18s}" + "".join(f"{k[:9]:>10s}" for k in keys))
for name, (_, _, ho) in rows.items():
    print(f"{name:18s}" + "".join(f"{ho[k]:10.2f}" for k in keys))

# ---- picture: the three procedures at one seed (tiles), stacked
px, panels = 2, []
for name, (cc, _, _) in rows.items():
    full(cc, 3000)
    panels.append(roots.render(kinds, tile, px))
    panels.append(np.full((6, W * px, 3), 255, np.uint8))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(np.concatenate(panels[:-1], 0)).save(OUT.replace(".json", ".png"))
json.dump(dict(target=target, rows={k: dict(knobs=v[0], stats=v[1], held_out=v[2]) for k, v in rows.items()},
               hist8=h8, hist1=h1), open(OUT, "w"))
print("wrote", OUT)
