"""Quick ctrunc training check (not a bench): python quick.py <problem> <budget_s> [cfg json]."""
import json
import sys
import time

from castlegen.bench.problems import get
from castlegen.variants import ctrunc as m

prob = sys.argv[1]
budget = float(sys.argv[2])
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
p = get(prob)
v = m.CTrunc(p, 0, "/tmp", **cfg)
logs = []
t = time.time()
v.train(budget, logs.append)
print("train wall", round(time.time() - t, 1), "updates", v.updates, "B", v.B)
ups = [d for d in logs if "update" in d]
print("first update at", ups[0]["t"] if ups else None)
for d in ups[:: max(1, len(ups) // 8)] + ups[-1:]:
    print(round(d["t"], 1), d["update"], "cost", round(d["cost_per_world"], 1),
          "viol", [f"{x:.1e}" for x in d["raw_viol"]], "vl", [round(x, 3) for x in d["value_loss"]],
          "lam", [round(x, 2) for x in d["lam"]])

import numpy as np  # noqa: E402

half = ups[len(ups) // 2:]
cells = p.H * p.W_train
for l in sorted({k for d in half for k in d["vcheck"]}, key=int):
    pr = np.concatenate([d["vcheck"][l][0] for d in half])
    re = np.concatenate([d["vcheck"][l][1] for d in half])
    print(f"V below level {l}: pred {pr.mean() / cells:+.3f}/cell real {re.mean() / cells:+.3f}/cell "
          f"bias {(pr - re).mean() / cells:+.3f}/cell rmse {np.sqrt(((pr - re) ** 2).mean()) / cells:.3f}/cell "
          f"sd(real) {re.std() / cells:.3f}/cell "
          f"corr {np.corrcoef(pr, re)[0, 1] if re.std() > 0 else float('nan'):.2f}")
