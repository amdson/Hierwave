"""Instrumented training of a registered ckernel variant, imported the way the bench imports it
(load_variants first, so the JAX compile-cache dir is the bench's: warms the bench's cache).
usage: run_ck2.py <variant> <budget_s> <outdir> [key=value cfg overrides]"""
import ast, json, os, pickle, sys, time
import jax
from castlegen.bench.problems import get
from castlegen.bench.variant import load_variants

V_ = load_variants()
name, budget, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[4:])}
os.makedirs(out, exist_ok=True)
print("cache dir", jax.config.jax_compilation_cache_dir, flush=True)
V = V_[name](get("ground"), 0, out, **cfg)
f = open(f"{out}/loop.jsonl", "w")
def log(d):
    f.write(json.dumps(d) + "\n"); f.flush()
t0 = time.time()
V.train(budget, log)
print("train done", round(time.time() - t0, 1), "updates", V.updates, "rkl", getattr(V, "rkl_updates", 0), V.timing, flush=True)
pickle.dump({"trained": {"params": jax.device_get(V.params), "lam": V.lam.tolist()}, "updates": V.updates,
             "cfg": {**V.cfg, "start": V.start}, "variant": name}, open(f"{out}/params_ground.pkl", "wb"))
print("saved", flush=True)
