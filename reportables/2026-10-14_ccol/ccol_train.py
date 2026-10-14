"""cfast on columns with csat-style per-level instrumentation (subclass of CSat, adds
density / empty-xcol fraction from the diagnostic worlds).  Diagnostics only."""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problem import Region
import csat_cols as cs
from castlegen.variants._cfast_core import train_pass
from castlegen.variants import _cpass_core as core

class CCol(cs.CSat):
    name = "ccol"
    def diag(self, tau=None, lam=None):
        res = super().diag(tau, lam)
        region = Region(self.cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        if not hasattr(self, "_sf"):
            self._sf = jax.jit(jax.vmap(lambda p, s, d, t, l: train_pass(lay, p, s, d, t, l)[0]["k"], in_axes=(None, 0, 0, None, None)))
        seeds = np.arange(900_000, 900_008)
        des = self._batch(seeds, region)
        k = np.asarray(self._sf(self.params, jnp.asarray(seeds, jnp.int32), des,
                                jnp.float32(self.tau if tau is None else tau),
                                jnp.asarray(self.lam if lam is None else lam, jnp.float32)))
        res["density"] = float((k > 0).mean())
        res["empty_xcol"] = float((~(k > 0).any(1)).mean())
        res["bottom"] = float((k == 1).mean())
        return res

prob, budget, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
p = get_problem(prob)
tw = time.time()
w = CCol(p, seed=99, workdir=out); w.train(1.0, lambda d: None); w.diag(tau=2.0)
print("warm", time.time() - tw, flush=True)
v = CCol(p, seed=0, workdir=out)
p0 = jax.device_get(v.params); lam0 = v.lam.copy()
logf = open(f"{out}/train_{prob}.jsonl", "w")
v.train(budget, lambda d: (logf.write(json.dumps(d, default=float) + "\n"), logf.flush()))
logf.close()
pickle.dump({"untrained": {"params": p0, "lam": lam0.tolist()},
             "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist()},
             "updates": v.updates, "cfg": v.cfg}, open(f"{out}/params_{prob}.pkl", "wb"))
print("done", v.updates, v.lam.tolist(), flush=True)
