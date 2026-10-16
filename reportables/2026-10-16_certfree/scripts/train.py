"""train ccertfree for <secs> on ground (-> ground_cf inside), save params.  usage: train.py <secs> <outdir> [cfg json]"""
import json, pickle, sys, time
import jax, numpy as np
from castlegen.bench.problems import get
from castlegen.bench.variant import load_variants
secs, out = float(sys.argv[1]), sys.argv[2]
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
import os; os.makedirs(out, exist_ok=True)
V = load_variants(); p = get("ground")
v = V["ccertfree"](p, 0, out, **cfg)
f = open(f"{out}/loop.jsonl", "w")
t0 = time.time()
v.train(secs, lambda d: f.write(json.dumps(d, default=float) + "\n"))
f.close()
tt = time.time() - t0
setup = v.timing["setup"]
pickle.dump({"params": jax.tree_util.tree_map(np.asarray, v.params), "lam": np.asarray(v.lam, float)}, open(f"{out}/params.pkl", "wb"))
r = {"train_wall_s": tt, "setup_s": setup, "updates": v.updates, "rkl_updates": v.rkl_updates, "iters": v.timing["iters"],
     "updates_per_s": v.updates / max(tt - setup, 1e-9), "lam": np.asarray(v.lam).tolist()}
json.dump(r, open(f"{out}/train_summary.json", "w"), indent=1); print(r); print("DONE")
