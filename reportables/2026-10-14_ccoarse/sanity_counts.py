"""ccoarse sanity on count-only coarse channels: columns and toy_support/toy_modes.
Restricted prior vs lifted probes (exact target samples there), 64 worlds."""
import json, time
import numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _coarse_prior as cpr

seeds = np.arange(3_000_000, 3_000_064)
res = {}
for name, Ls in (("columns", (4, 3, 2)), ("toy_support", (2, 1)), ("toy_modes", (2, 1))):
    p = get(name); reg = Region(p.W_train)
    prb = [p.probes(int(s), reg)[i % 4] for i, s in enumerate(seeds)]
    for L in Ls:
        t = time.time()
        x = cpr.sample_coarse_batch(p, reg, L, seeds, sweeps=2)
        dt = (time.time() - t) / len(seeds)
        v = int(cpr.hard_violations(p, x, reg, L).sum())
        for c in p.chans:
            if c.level < L or c.designed is not None:
                continue
            a = x[c.name] / (c.D - 1)
            b = np.stack([q[c.name] for q in prb]) / (c.D - 1)
            res[f"{name} L={L} {c.name}"] = {
                "prior mean": float(a.mean()), "probe mean": float(b.mean()),
                "prior sd": float(a.std()), "probe sd": float(b.std()),
                "prior P(0)": float((a == 0).mean()), "probe P(0)": float((b == 0).mean()),
                "hard viol (all worlds)": v, "s/world": dt}
            print(name, L, c.name, {k: round(u, 4) for k, u in res[f"{name} L={L} {c.name}"].items()}, flush=True)
json.dump(res, open("reportables/2026-10-14_ccoarse/counts_metrics.json", "w"), indent=1)
