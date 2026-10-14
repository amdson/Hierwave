"""line_match (ground.stats) on the eval's saved t (first 8 rows per condition)."""
import sys, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems.ground import stats
p = get_problem("ground"); reg = Region(p.W_train)
seeds = np.arange(3_000_000, 3_000_010)
fill = {i: p.designed_values(int(s), reg)["fill"] for i, s in enumerate(seeds)}
for r in sys.argv[1:]:
    z = np.load(f"reportables/2026-10-14_ccur/runs/{r}/ground_raw.npz")
    for m in ("untrained", "trained"):
        for c in ("own", "probe_L4", "probe_L3"):
            t = z[f"{m}__{c}__t"]
            si = list(range(8)) if c == "own" else [i // 4 for i in range(8)]
            lm = [stats({"t": t[i], "fill": fill[si[i]], "rho4": np.zeros_like(fill[si[i]])})["line_match"] for i in range(len(t))]
            print(f"{r:12s} {m} {c:9s} line_match {np.mean(lm):.3f}  dirt {t.mean():.3f}")
