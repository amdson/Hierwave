"""Port-partition connectivity promise (castlegen/ports.py), penalty version,
on the castle: the joint sampler's u at h = 8, 4, 2, 1 (as castle_multi), plus
sigma at h >= 2; every rule a penalty lam per violation, lam ramped linearly
to LAMAX over each level's sweeps (u and sigma alternate).  Reports the rule
violations per level and the tile map's components; images/castle_ports.png."""
import os, tempfile, time
import numpy as np
from PIL import Image
from castlegen import connmetrics as CM, exemplar as ex, generic as GN, ports as PT, tileset

n, px = 128, 8
SEED = int(os.environ.get("SEED", 1))
LAMAX = float(os.environ.get("LAMAX", 20))
SW = int(os.environ.get("SW", 10))
PLAN = [(8, 16), (4, SW), (2, SW), (1, SW)]
ts = tileset.load("demo")
solid = np.asarray(ts.solid, bool)
E = np.load("cache/castle_ex.npy").astype(np.int32)
an = GN.analysis(ts, E, [h for h, _ in PLAN], torus=True)
rng = np.random.default_rng(SEED)
Up, Pp = None, None
for h, sweeps in PLAN:
    t0 = time.time()
    lvl = GN.JointLevel(an, solid, h, n, lam_c=1.0 if h == 1 else 16.0, r=0.25, knn=16,
                        sigma=None if Up is None else h / 2, use_v=False, loc=ts, fast="numba")
    lvl.rng = rng
    if Up is None:
        lvl.init(np.random.default_rng(SEED).integers(0, an.my * an.m, (lvl.R, lvl.C)))
    else:
        lvl.init(lvl.prolong(Up), parent=Up)
    rules = PT.PortLevel(ts, E, lvl, h) if h >= 2 else PT.TileRules(ts, lvl, Pp)
    if h >= 2:
        rules.init(Pp, rng)
    for s in range(sweeps):
        lam = LAMAX * (s + 1) / sweeps
        T = 10.0 ** (1 - s / (sweeps // 2)) if Up is None and s < sweeps // 2 else 1.0
        if h >= 2:
            lvl.sweep(T)
            rules.sweep(lam, rng=rng)
        else:
            rules.lam = lam
            lvl.sweep(T)
    if h >= 2:
        rules.compute_exits()
    Up, Pp = lvl.U.copy(), rules
    x = lvl.tiles()
    ep = lvl.energy_parts()
    extra = f"  values {len(rules.codes)}" if h >= 2 else ""
    print(f"h {h}: {time.time() - t0:.0f}s  violations {rules.violations():.0f}{extra}  E_par {ep['par']:.0f}  "
          f"bad cells {ex.bad_cells(ts, x, torus=False).mean():.3f}  "
          f"components {CM.global_stats(ts, x, torus=False)['components']}", flush=True)
p = os.path.join(tempfile.mkdtemp(), "x.png")
ex.to_png(ts, x, p, px)
Image.open(p).save("images/castle_ports.png")
