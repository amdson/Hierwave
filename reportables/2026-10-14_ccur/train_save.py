"""Train a ccur variant on ground for a budget; save untrained / release-time /
trained params (for the clamp gate) and the train log.
Writes <out>/params_ground.pkl (untrained vs trained) and, for ccur,
<out>_release/params_ground.pkl (untrained vs the params at the first release)."""
import json, os, pickle, sys, time
import jax
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants

prob, budget, out, vname = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
warm = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
V = load_variants()[vname]
p = get_problem(prob)
os.makedirs(out, exist_ok=True)
tw = time.time()
w = V(p, seed=99, workdir=out)
w.train(warm, lambda d: None)                      # warm-up: compiles the update (shared per process)
w.diag(tau=2.0)
print("warm-up compile", time.time() - tw, flush=True)
if budget <= 0:
    sys.exit(0)
v = V(p, seed=0, workdir=out)
p0 = jax.device_get(v.params); lam0 = v.lam.copy()
logf = open(f"{out}/train_{prob}.jsonl", "w")
t0 = time.time()
v.train(budget, lambda d: (logf.write(json.dumps(d, default=float) + "\n"), logf.flush()))
logf.close()
base = {"untrained": {"params": p0, "lam": lam0.tolist()}, "cfg": v.cfg, "train_s": time.time() - t0,
        "diag_time": v._dtime, "devices": str(jax.devices())}
pickle.dump({**base, "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist()}, "updates": v.updates},
            open(f"{out}/params_{prob}.pkl", "wb"))
if getattr(v, "snap_release", None):
    os.makedirs(out + "_release", exist_ok=True)
    s = v.snap_release
    pickle.dump({**base, "trained": {"params": s["params"], "lam": s["lam"]}, "updates": s["update"]},
                open(f"{out}_release/params_{prob}.pkl", "wb"))
print("done", prob, vname, v.updates, "updates", v.lam.tolist(), flush=True)
