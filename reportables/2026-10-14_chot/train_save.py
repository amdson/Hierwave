"""Train a chot-family variant on ground for a budget (after a warm-up compile
in the same process); save untrained/trained params (clamp gate) and the log."""
import json, os, pickle, sys, time
import jax
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants

prob, budget, out, vname = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
V = load_variants()[vname]
p = get_problem(prob)
tw = time.time()
w = V(p, seed=99, workdir=out)
w.train(1.0, lambda d: None)                       # warm-up: compiles the baseline update (shared per process)
import numpy as np, jax.numpy as jnp
from castlegen.bench.problem import Region
reg = Region(w.cfg["W_train"] or p.W_train)
sd = np.arange(w.cfg["B"])
w.fn["fused_hot"](w.params, w.cparams, w.opt, w.copt, w.gs, jnp.asarray(sd, jnp.int32), w._batch(sd, reg),
                  jnp.float32(1.0), jnp.asarray(w.lam, jnp.float32), 0.0, 0.0, 10.0, jax.random.PRNGKey(0),
                  jnp.asarray(w.eps_vec))                    # and the hot update
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
             "devices": str(jax.devices())}, open(f"{out}/params_{prob}.pkl", "wb"))
print("done", prob, vname, v.updates, "updates", v.lam.tolist(), flush=True)
