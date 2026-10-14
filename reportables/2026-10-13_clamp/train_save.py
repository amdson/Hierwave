"""Diagnostic: train cfast (default config) on one problem for a budget and
save the untrained and trained parameters (pickle).  No code changes."""
import json, pickle, sys, time
import jax, numpy as np
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants

prob, budget, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
V = load_variants()["cfast"]
p = get_problem(prob)
tw = time.time()
V(p, seed=99, workdir=out).train(1.0, lambda d: None)     # warm-up: compiles (shared per process)
print("warm-up compile", time.time() - tw, flush=True)
v = V(p, seed=0, workdir=out)
p0 = jax.device_get(v.params); lam0 = v.lam.copy()
logf = open(f"{out}/train_{prob}.jsonl", "w")
t0 = time.time()
v.train(budget, lambda d: logf.write(json.dumps(d, default=float) + "\n"))
logf.close()
pickle.dump({"untrained": {"params": p0, "lam": lam0.tolist()},
             "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist()},
             "updates": v.updates, "cfg": v.cfg, "train_s": time.time() - t0,
             "devices": str(jax.devices())}, open(f"{out}/params_{prob}.pkl", "wb"))
print("done", prob, v.updates, "updates", v.lam.tolist(), flush=True)
