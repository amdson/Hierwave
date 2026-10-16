"""Smoke cfitpen_ml2_stage: train 30 s (one seed), then generate 32 periodic W128 worlds + the open-window support check."""
import time, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.energy import energy
p = get("ground"); V = load_variants()
v = V["cfitpen_ml2_stage"](p, 0, "/tmp/combo_smoke")
t0 = time.time(); v.train(30, lambda d: None); print(f"train {time.time()-t0:.1f}s updates {v.updates} info {v.info().get('stage')}", flush=True)
t0 = time.time(); viol = np.zeros(5, int); chg = []
for i in range(32):
    s = EVAL_SEED0 + i; g = v.generate(s, Region(128))
    x = _with_designed(p, g["final"], s, Region(128))
    viol += np.array([energy(p, x, Region(128)).viol[l] for l in range(5)]); chg.append((g["raw"]["t"] != g["final"]["t"]).mean())
print(f"periodic 32 worlds {time.time()-t0:.1f}s viol {viol.tolist()} tiles changed {np.mean(chg):.4f}", flush=True)
W = p.W_eval; half = (W // 2) >> p.T << p.T; seed = EVAL_SEED0 + 999; t0 = time.time()
ga = v.generate(seed, Region(W, periodic=False, x0=0)); gb = v.generate(seed, Region(W, periodic=False, x0=half))
m = 0
for kind in ("raw", "final"):
    for c in p.sampled():
        s = half >> c.level; a = np.asarray(ga[kind][c.name])[:, s:]; b = np.asarray(gb[kind][c.name])[:, :a.shape[1]]
        m += int((a != b).sum())
print(f"support check {time.time()-t0:.1f}s mismatches {m}", flush=True)
