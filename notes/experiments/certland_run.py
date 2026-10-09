"""Certified terrain (notes/certland_test.md): the untrained model (Phi = 0)
against the model trained by the C2 bootstrap with AIS targets, on held-out
noise fields.  Usage: python certland_run.py [m] [tag] [small | admissible]"""
import json
import os
import sys
import time

import numpy as np

from castlegen.channels import certland as cl, certland_fit as cf

IMG = os.environ.get("CERTLAND_IMG", "/Users/amdson/dev/Hierwave/images")
m = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tag = sys.argv[2] if len(sys.argv) > 2 else f"m{m}"
small = len(sys.argv) > 3 and sys.argv[3] == "small"
cf.FREE = not (len(sys.argv) > 3 and sys.argv[3] == "admissible")
H, W = (64, 64) if small else (128, 256)
ITERS, NSITES = (1, 10) if small else (3, 100)
EVAL = [101, 102, 103, 104]
logf = open(f"{IMG}/certland_{tag}_log.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def evaluate(world, name):
    rows, imgs = [], []
    for s in EVAL:
        cf.set_noise(world, s)
        t = time.time()
        bad = cl.generate(world, seed=s)
        mk = cl.metrics(world)
        mk["stuck"] = bad
        mk["time"] = time.time() - t
        rows.append(mk)
        imgs.append(cl.render(world))
    raw = [cl.overhangs((cl.density_field(H, W, s) > 0).astype(np.int64)) for s in EVAL]
    log(f"--- {name}: overhangs {[r['overhangs'] for r in rows]} (raw noise threshold {raw}), stuck {[r['stuck'] for r in rows]}")
    log(f"    fill_err {np.mean([r['fill_err'] for r in rows]):.3f}  gen time {np.mean([r['time'] for r in rows]):.1f}s")
    for h in cl.HS[:-1]:
        g = lambda k: np.nanmean([r[f'h{h}'][k] for r in rows])
        log(f"    h={h:2d}: |tiles - rho|/h^2 {g('rho_err'):.3f}  A kept {g('A_kept'):.2f} B kept {g('B_kept'):.2f}  "
            f"A gap {g('A_gap'):+.2f} B gap {g('B_gap'):+.2f}  corr(A+B, rho) {g('cert_vs_rho'):+.2f}")
    return rows, imgs


def sheet(imgs_by_row, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(imgs_by_row), len(imgs_by_row[0][1])
    fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 2.4 * nr), squeeze=False)
    for r, (name, ims) in enumerate(imgs_by_row):
        for c, im in enumerate(ims):
            ax[r, c].imshow(im, interpolation="nearest")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def noise_images():
    out = []
    for s in EVAL:
        d = cl.density_field(H, W, s)
        t = (d > 0).astype(np.int64)
        w = cl.World(H, W)
        w.rho[cl.NL - 1][:] = t
        out.append(cl.render(w))
    return out


w = cl.World(H, W)
log(f"certland m={m} free={cf.FREE} H={H} W={W} iters={ITERS} sites={NSITES} kappa={w.kappa} lam={w.lam} beta={w.beta}")
res = {}
res["phi0"], im0 = evaluate(w, "Phi = 0 (no learned coupling)")
t = time.time()
diag = cf.train(w, noise_seeds=list(range(1, 40)), iters=ITERS, n_sites=NSITES, m=m, log=log)
log(f"training {time.time() - t:.0f}s")
res["train"] = diag
res["learned"], im1 = evaluate(w, "learned")
# AIS repeatability at the top: same site, same candidate, 4 seeds
cf.set_noise(w, EVAL[0]); cl.generate(w, seed=EVAL[0])
rng = np.random.default_rng(5)
sds = []
for _ in range(10):
    y, x = int(rng.integers(w.rho[0].shape[0])), int(rng.integers(w.rho[0].shape[1]))
    sds.append(np.std([cf.child_logz(w, 0, y, x, 64, 8, s) for s in range(4)]))
log(f"AIS sd of log Z at the top (h=8 children), 10 sites x 4 seeds: mean {np.mean(sds):.2f} max {np.max(sds):.2f}")
np.savez(f"{IMG}/certland_{tag}_params.npz", **{f"th{i}": w.params[i][0] for i in range(4)},
         m=np.array([w.params[i][1] for i in range(4)]))
json.dump(res, open(f"{IMG}/certland_{tag}.json", "w"), indent=1, default=float)
sheet([("raw noise > 0", noise_images()), ("Phi = 0", im0), ("learned", im1)], f"{IMG}/certland_{tag}.png")
log("wrote", f"{IMG}/certland_{tag}.png")
