"""The joint sampler on the castle (top-down, demo tile set): exemplar
castlegen/legacy/exemplars/castle.json (64^2 torus, Analysis(bounds=None): no padding
rows, all four map sides free), h = 8 from a uniform start (annealed), then
4, 2, 1 with E_par (sigma = h / 2) and E_loc at h = 1.  No promise
(use_v=False): connectivity has no joint promise yet, so it is only measured
(connmetrics, map not wrapped).  n = 128, lam_c = LAM (default 16), K = 16,
numba.  Rows: exemplar, then seeds; columns: levels -> images/castle_multi.png (PX px per tile, default 8: smaller hides the hallways under the tile walls)."""
import os, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
from castlegen.legacy import connmetrics as CM, exemplar as ex, generic as GN, tileset

n, px = 128, int(os.environ.get("PX", 8))
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1,2,3").split(","))
LAM = float(os.environ.get("LAM", 16))
CONN = os.environ.get("CONN", "0") == "1"                                   # the connectivity promise
LAM1 = float(os.environ.get("LAM1", LAM))                                 # lam_c at h = 1
PLAN = [(8, 16), (4, 8), (2, 4), (1, int(os.environ.get("SW1", 4)))]                                    # (h, sweeps)
ts = tileset.load("demo")
solid = np.asarray(ts.solid, bool)
path = "cache/castle_ex.npy"
if not os.path.exists(path):
    np.save(path, np.asarray(ex.build(ex.load_layout("castle"))[1]))
E = np.load(path).astype(np.int32)
an = GN.analysis(ts, E, [h for h, _ in PLAN], torus=True)
print("exemplar:", CM.global_stats(ts, E, torus=True), tileset.structure_census(ts, E), f"bad cells {ex.bad_cells(ts, E).mean():.3f}", flush=True)
out = {sd: [] for sd in SEEDS}
Up = {sd: None for sd in SEEDS}
Vp = {sd: None for sd in SEEDS}
Kp = {sd: None for sd in SEEDS}
for h, sweeps in PLAN:
    lvl = GN.JointLevel(an, solid, h, n, lam_c=LAM1 if h == 1 else LAM, r=0.25, knn=16, sigma=None if h == PLAN[0][0] else h / 2,
                        use_v=CONN, loc=ts, fast="numba", conn=ts if CONN else None,
                        lam_d=float(os.environ.get("LAMD", 4)), rk=int(os.environ.get("RK", 4)))
    for sd in SEEDS:
        t1 = time.time()
        lvl.rng = np.random.default_rng(sd * 100 + h)
        if Up[sd] is None:
            lvl.init(np.random.default_rng(sd).integers(0, an.my * an.m, (lvl.R, lvl.C)))
            lvl.run(sweeps, T=1.0, T_hot=10.0)
        else:
            lvl.init(lvl.prolong(Up[sd]), parent=Up[sd], vparent=Vp[sd], kparent=Kp[sd])
            lvl.run(sweeps, T=1.0)
        Up[sd], Vp[sd] = lvl.U.copy(), lvl.V.copy()
        Kp[sd] = lvl.keys() if CONN else None
        x = lvl.tiles()
        ep = lvl.energy_parts()
        print(f"h {h} seed {sd}: {time.time() - t1:.1f}s  lam E_c {ep['c']:.0f}  E_par {ep['par']:.0f}  "
              f"E_loc {ep['loc']:.0f}  bad cells {ex.bad_cells(ts, x, torus=False).mean():.3f}  "
              f"greenhouses (complete, orphan cells) {tileset.structure_census(ts, x)['greenhouse']}  "
              f"components {CM.global_stats(ts, x, torus=False)['components']}"
              + (f"  promise violations {lvl.conn_violations()}  v = 0: {(lvl.V == 0).mean():.2f}" if CONN else ""), flush=True)
        out[sd].append(x)
RELAX, RT = int(os.environ.get("RELAX", 0)), float(os.environ.get("RT", 1.0))
RT_HOT = float(os.environ.get("RT_HOT", RT))                             # anneal RT_HOT -> RT over the first half
if RELAX:                                                                  # relax toward p: tile Gibbs on E_loc
    for sd in SEEDS:
        x = ex.gibbs(ts, out[sd][-1], RELAX, seed=sd, T=RT, T_hot=RT_HOT, torus=False)
        print(f"relax {RELAX} sweeps T {RT} seed {sd}: bad cells {ex.bad_cells(ts, x, torus=False).mean():.3f}  "
              f"greenhouses (complete, orphan cells) {tileset.structure_census(ts, x)['greenhouse']}  "
              f"components {CM.global_stats(ts, x, torus=False)['components']}", flush=True)
        out[sd].append(x)
    PLAN = PLAN + [(f"relax T {RT_HOT}->{RT}", RELAX)]
p, w = os.path.join(tempfile.mkdtemp(), "x.png"), n * px
img = Image.new("RGB", (60 + len(PLAN) * (w + 8), 20 + (len(SEEDS) + 1) * (w + 8)), "white")
d = ImageDraw.Draw(img)
ex.to_png(ts, E, p, px)
img.paste(Image.open(p).convert("RGB"), (60, 20))
d.text((4, 20 + w // 4), "exemplar", fill="black")
for j, (h, k) in enumerate(PLAN):
    d.text((60 + j * (w + 8) + w // 2 - 30, 4), f"{h if isinstance(h, str) else f'h {h}'}, {k} sweeps", fill="black")
for i, sd in enumerate(SEEDS):
    d.text((4, 20 + (i + 1) * (w + 8) + w // 2), f"seed {sd}", fill="black")
    for j, x in enumerate(out[sd]):
        ex.to_png(ts, x, p, px)
        img.paste(Image.open(p).convert("RGB"), (60 + j * (w + 8), 20 + (i + 1) * (w + 8)))
img.save("images/castle_multi.png")
fin = Image.new("RGB", (len(SEEDS) * (w + 8) + (n // 2) * px + 8, w), "white")
ex.to_png(ts, E, p, px)
fin.paste(Image.open(p).convert("RGB"), (0, 0))
for i, sd in enumerate(SEEDS):
    ex.to_png(ts, out[sd][-1], p, px)
    fin.paste(Image.open(p).convert("RGB"), ((n // 2) * px + 8 + i * (w + 8), 0))
fin.save(f"images/castle_final{os.environ.get('TAG', '')}.png")
