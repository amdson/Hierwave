"""usage: train_arm.py <variant> <budget_s> <max_iters|0> <seed> <outdir>  -> <outdir>/params_s<seed>.pkl, updates_s<seed>.json
Continuation of big300 params_s<seed> (l3cap variants load them).  max_iters 0 = time budget only."""
import json, os, pickle, sys, time
import jax, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
variant, budget, mi, seed, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.makedirs(out, exist_ok=True)
p = get("ground"); v = load_variants()[variant](p, seed, out)
if mi: v.cfg["max_iters"] = mi
lf = open(f"{out}/train_s{seed}.jsonl", "a"); t0 = time.time(); nit = [0]
def log(rec):
    if "iter" in rec:
        nit[0] = rec["iter"]
        lf.write(json.dumps({k: rec[k] for k in ("iter", "update", "t", "t_train") if k in rec}, default=float) + "\n"); lf.flush()
v.train(budget, log)
print("trained", variant, seed, "updates", v.updates, "iters", nit[0], flush=True)
json.dump({"variant": variant, "seed": seed, "budget": budget, "max_iters": mi, "updates": int(v.updates), "iters": nit[0], "train_wall": time.time() - t0}, open(f"{out}/updates_s{seed}.json", "w"))
pickle.dump({"params": jax.device_get(v.params), "lam": np.asarray(v.lam), "updates": v.updates}, open(f"{out}/params_s{seed}.pkl", "wb"))
