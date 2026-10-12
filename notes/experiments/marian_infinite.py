"""Marian's 3D WFC module set under the infinite annealed Gibbs sampler.

Usage: python notes/experiments/marian_infinite.py [config ...]
Each config is a name from CONFIGS; prints wildcards per cell, per column,
the fraction of columns holding a wildcard, misfits, time per 64^2 region and
per lone 32^2 chunk.  --render NAME writes images/marian_topdown.png and
images/marian_slice.png from that config (seed 0, 96^2 columns);
--render NAME SUFFIX appends SUFFIX to both file names."""
import sys
import time

import numpy as np

from castlegen import marian
from castlegen.infinite import BlockSchedule, Schedule, generate

CONFIGS = {
    "S32_b1_g2": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=2.0),
    "S32_b1_g.5": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=0.5),
    "S64_b1_g2": lambda: Schedule.anneal(S=64, beta0=1.0, gamma0=2.0),
    "S64_b1_g.5": lambda: Schedule.anneal(S=64, beta0=1.0, gamma0=0.5),
    "S64_hot": lambda: Schedule.anneal(S=64, beta0=0.1, gamma0=0.5),
    "S32_q16": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=2.0, quench=16),
    "S32_g32": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=2.0, gamma1=32.0),
    "S32_g.1": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=0.1, gamma1=8.0),
    "S32_b.5": lambda: Schedule.anneal(S=32, beta0=0.5, gamma0=2.0),
    "S32_q32": lambda: Schedule(np.ones(32), np.full(32, np.inf)),
    "S32_q0_g4": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=2.0, gamma1=4.0, quench=0),
    "S32_q0_g8": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=2.0, gamma1=8.0, quench=0),
    "S32_q0_g16": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=2.0, gamma1=16.0, quench=0),
    "S32_q1_g4": lambda: Schedule.anneal(S=32, beta0=1.0, gamma0=1.0, gamma1=4.0, quench=1),
    "S64_q0_g4": lambda: Schedule.anneal(S=64, beta0=1.0, gamma0=2.0, gamma1=4.0, quench=0),
    "S64_q0_g1_4": lambda: Schedule.anneal(S=64, beta0=1.0, gamma0=1.0, gamma1=4.0, quench=0),
    "B8Sb3K32_q0_g4": lambda: BlockSchedule.anneal(B=8, Sb=3, K=32, restart=False, shift=True, beta0=1.0,
                                                   gamma0=2.0, gamma1=4.0, quench=0),
    "B8Sb2K32_q0_g4": lambda: BlockSchedule.anneal(B=8, Sb=2, K=32, restart=False, shift=True, beta0=1.0,
                                                   gamma0=2.0, gamma1=4.0, quench=0),
    "B8Sb3K32_g4": lambda: BlockSchedule.anneal(B=8, Sb=3, K=32, restart=False, shift=True, beta0=1.0,
                                                gamma0=4.0, gamma1=16.0),
    "B8Sb3K64": lambda: BlockSchedule.anneal(B=8, Sb=3, K=64, restart=False, shift=True, beta0=1.0, gamma0=2.0),
    "B16Sb2K32": lambda: BlockSchedule.anneal(B=16, Sb=2, K=32, restart=False, shift=True, beta0=1.0, gamma0=2.0),
    "B8Sb4K32": lambda: BlockSchedule.anneal(B=8, Sb=4, K=32, restart=False, shift=True, beta0=1.0, gamma0=2.0),
    "S128_b1_g.5": lambda: Schedule.anneal(S=128, beta0=1.0, gamma0=0.5),
    "B8Sb2K16": lambda: BlockSchedule.anneal(B=8, Sb=2, K=16, restart=False, shift=True, beta0=1.0, gamma0=2.0),
    "B8Sb2K32": lambda: BlockSchedule.anneal(B=8, Sb=2, K=32, restart=False, shift=True, beta0=1.0, gamma0=2.0),
    "B8Sb3K32": lambda: BlockSchedule.anneal(B=8, Sb=3, K=32, restart=False, shift=True, beta0=1.0, gamma0=2.0),
    "B8Sb3K32_g.5": lambda: BlockSchedule.anneal(B=8, Sb=3, K=32, restart=False, shift=True, beta0=1.0, gamma0=0.5),
    "B8Sb3K32_hot": lambda: BlockSchedule.anneal(B=8, Sb=3, K=32, restart=False, shift=True, beta0=0.1, gamma0=0.5),
}


def measure(model, name, seeds=(0, 1, 2), N=64, domains="ac"):
    sched = CONFIGS[name]()
    T = model.T
    wc, frac, mis, dts = [], [], [], []
    for s in seeds:
        t0 = time.time()
        g = generate(model, sched, s, 0, 0, N, N)
        dts.append(time.time() - t0)
        w = g == T
        wc.append(w.mean())
        frac.append(w.any(0).mean())
        mis.append(sum(model.violations(g)))
    t0 = time.time()
    generate(model, sched, 99, 0, 0, 32, 32)
    tc = time.time() - t0
    print(f"| {name} | {domains} | {np.mean(wc):.4f} | {np.mean(wc) * model.L:.3f} | {np.mean(frac):.3f} | "
          f"{max(mis)} | {np.mean(dts):.1f} s | {tc:.1f} s |", flush=True)
    return g


def render(ms, model, g, path_top, path_slice, scale=4):
    from PIL import Image
    T = model.T
    air = ms.air()
    col = np.vstack([model.colors, [[255, 0, 255]]]).astype(np.uint8)
    col[air] = (190, 210, 235)
    L, H, W = g.shape
    top = np.full((H, W), air, np.int64)
    height = np.zeros((H, W), np.int64)
    for k in range(L):
        m = g[k] != air
        top[m] = g[k][m]
        height[m] = k + 1
    img = col[top].astype(float) * (0.55 + 0.45 * height / L)[..., None]
    img[top == T] = (255, 0, 255)
    img = np.repeat(np.repeat(img.astype(np.uint8), scale, 0), scale, 1)
    Image.fromarray(img).save(path_top)
    rows = [H // 4, H // 2, 3 * H // 4]
    sl = []
    for r in rows:
        s = col[g[::-1, r, :]]
        sl.append(s)
        sl.append(np.zeros((1, W, 3), np.uint8))
    sl = np.concatenate(sl[:-1], 0)
    sl = np.repeat(np.repeat(sl, scale * 2, 0), scale * 2, 1)
    Image.fromarray(sl).save(path_slice)


def main(argv):
    ms = marian.load()
    domains = "ac"
    if "--raw" in argv:
        domains = "raw"
        argv = [a for a in argv if a != "--raw"]
    model = marian.tile_model(ms, domains=domains)
    if argv and argv[0] == "--render":
        sched = CONFIGS[argv[1]]()
        g = generate(model, sched, 0, 0, 0, 96, 96)
        print("render wild/cell", (g == model.T).mean(), "violations", model.violations(g))
        suf = argv[2] if len(argv) > 2 else ""
        render(ms, model, g, f"images/marian_topdown{suf}.png", f"images/marian_slice{suf}.png")
        return
    print("| schedule | domains | wildcards / cell | / column | columns with one | misfits | 64^2 region | lone 32^2 chunk |")
    print("|---|---|---|---|---|---|---|---|")
    for name in argv or list(CONFIGS):
        measure(model, name, domains=domains)


if __name__ == "__main__":
    main(sys.argv[1:])
