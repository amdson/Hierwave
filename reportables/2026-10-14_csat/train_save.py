"""Train a csat variant (cfast + saturation diagnostics) on ground for a budget;
save untrained/trained params (for the clamp gate) and the train log."""
import json, pickle, sys, time
import jax
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants

prob, budget, out, vname = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
V = load_variants()[vname]
p = get_problem(prob)
tw = time.time()
w = V(p, seed=99, workdir=out)
w.train(1.0, lambda d: None)                       # warm-up: compiles the update (shared per process)
w.diag(tau=2.0)                                    # and the diagnostic pass
print("warm-up compile", time.time() - tw, flush=True)
v = V(p, seed=0, workdir=out)
p0 = jax.device_get(v.params); lam0 = v.lam.copy()
logf = open(f"{out}/train_{prob}.jsonl", "w")
t0 = time.time()
v.train(budget, lambda d: (logf.write(json.dumps(d, default=float) + "\n"), logf.flush()))
logf.close()
pickle.dump({"untrained": {"params": p0, "lam": lam0.tolist()},
             "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist()},
             "updates": v.updates, "cfg": v.cfg, "train_s": time.time() - t0,
             "diag_time": v._dtime, "devices": str(jax.devices())}, open(f"{out}/params_{prob}.pkl", "wb"))
print("done", prob, vname, v.updates, "updates", v.lam.tolist(), flush=True)
