"""Gate-2 training: a variant (cfine / cfast / ...) on a problem for a budget,
with csat's saturation diagnostics (per-level entropy of q over non-forced
draws, dirt fraction) every N updates; saves untrained/trained params for the
clamp gate (same format as csat's train_save.py) and the train log.
usage: train_gate.py <problem> <budget_s> <outdir> <variant> [key=value ...]"""
import ast, json, os, pickle, sys, time
import jax
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants
from castlegen.variants.csat import CSat

prob, budget, out, vname = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
over = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[5:])}
os.makedirs(out, exist_ok=True)
Base = load_variants()[vname]
V = type("Sat_" + vname, (CSat, Base), {"name": vname + "_sat"}) if Base is not CSat else CSat
p = get_problem(prob)
tw = time.time()
w = V(p, seed=99, workdir=out, **over)
w.train(1.0, lambda d: None)                       # warm-up: compiles the update (shared per process)
if prob == "ground":
    w.diag(tau=2.0)
print("warm-up compile", time.time() - tw, flush=True)
v = V(p, seed=0, workdir=out, **over)
if prob != "ground":
    v.cfg["diag_every"] = 10 ** 9                  # dirt diagnostics are ground-specific
p0 = jax.device_get(v.params); lam0 = v.lam.copy()
logf = open(f"{out}/train_{prob}.jsonl", "w")
t0 = time.time()
v.train(budget, lambda d: (logf.write(json.dumps(d, default=float) + "\n"), logf.flush()))
logf.close()
pickle.dump({"untrained": {"params": p0, "lam": lam0.tolist()},
             "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist()},
             "updates": v.updates, "cfg": v.cfg, "train_s": time.time() - t0,
             "diag_time": getattr(v, "_dtime", 0.0), "devices": str(jax.devices())},
            open(f"{out}/params_{prob}.pkl", "wb"))
print("done", prob, vname, v.updates, "updates", v.lam.tolist(), flush=True)
