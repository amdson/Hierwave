"""Train cfast on dimers for N seconds (default cfast config, as cdim/cbound) and save params + lambda."""
import json, pickle, sys, time
import jax, numpy as np
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants
budget, out = float(sys.argv[1]), sys.argv[2]
p = get_problem("dimers")
v = load_variants()["cfast"](p, seed=0, workdir="/tmp/dk_train60")
p0 = jax.device_get(v.params); lam0 = v.lam.copy()
v.train(budget, lambda d: None)
pickle.dump({"untrained": {"params": p0, "lam": lam0.tolist()}, "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist()},
             "updates": v.updates}, open(out, "wb"))
print("trained", v.updates, "updates lam", v.lam.tolist(), flush=True)
