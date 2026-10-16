"""usage: train.py <variant> <budget_s> <seed> <outdir>  -> <outdir>/params_s<seed>.pkl (ML-only fit, same as longfit minus periodic evals)"""
import json, os, pickle, sys, time
import jax, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
variant, budget, seed, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
p = get("ground"); v = load_variants()[variant](p, seed, out)
lf = open(f"{out}/train_s{seed}.jsonl", "a"); t0 = time.time()
def log(rec):
    if "iter" in rec:
        lf.write(json.dumps({k: rec[k] for k in ("iter", "update", "t", "own8", "wall") if k in rec} | {"wall": time.time() - t0}, default=float) + "\n"); lf.flush()
v.train(budget, log)
print("trained", variant, seed, "updates", v.updates, flush=True)
json.dump({"variant": variant, "seed": seed, "budget": budget, "updates": int(v.updates)}, open(f"{out}/updates_s{seed}.json", "w"))
pickle.dump({"params": jax.device_get(v.params), "lam": np.asarray(v.lam), "updates": v.updates}, open(f"{out}/params_s{seed}.pkl", "wb"))
