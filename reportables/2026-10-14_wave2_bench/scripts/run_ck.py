"""Instrumented ckernel training run (no bench): logs every iteration to
<out>/loop.jsonl, saves params_ground.pkl in the cstart2/kernel-a format (for gate.py).
usage: run_ck.py <budget_s> <outdir> [key=value cfg overrides, python literals]"""
import ast, json, os, pickle, sys, time
import jax, numpy as np
from castlegen.bench.problems import get
from castlegen.variants.ckernel import CKernel

budget, out = float(sys.argv[1]), sys.argv[2]
cfg = {}
for a in sys.argv[3:]:
    k, v = a.split("=", 1)
    cfg[k] = ast.literal_eval(v)
os.makedirs(out, exist_ok=True)
p = get("ground")
V = CKernel(p, 0, out, **cfg)
p0 = jax.device_get(V.params)
f = open(f"{out}/loop.jsonl", "w")
def log(d):
    f.write(json.dumps(d) + "\n"); f.flush()
    if d.get("setup_done") or d.get("iter", 0) % 5 == 1:
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in d.items()}), flush=True)
t0 = time.time()
V.train(budget, log)
print("train done", round(time.time() - t0, 1), "updates", V.updates, V.timing, flush=True)
pickle.dump({"untrained": {"params": p0, "lam": V.lam.tolist()}, "trained": {"params": jax.device_get(V.params), "lam": V.lam.tolist()},
             "updates": V.updates, "cfg": {**V.cfg, "start": V.start}, "train_s": V.timing["total"], "devices": str(jax.devices())},
            open(f"{out}/params_ground.pkl", "wb"))
print("saved", flush=True)
