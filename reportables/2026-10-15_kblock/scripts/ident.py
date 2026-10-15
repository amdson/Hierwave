"""kblock identity check: one iteration of ckpop_hard_pen vs ckpop_hard_pen_kb0 (same seed) must give
bit-identical params and population; then 2 iterations of ckpop_hard_pen_kb to warm the compile."""
import sys, time, json
import numpy as np, jax
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
V = load_variants(); prob = get("ground")
res = {}
for name in (() if "--warm" in sys.argv else ("ckpop_hard_pen", "ckpop_hard_pen_kb0")):
    t0 = time.time(); logs = []
    v = V[name](prob, seed=0, workdir="/tmp/kblock_ident", max_iters=1)
    v.train(1e9, logs.append)
    res[name] = (jax.tree_util.tree_leaves(jax.device_get(v.params)), {k: np.asarray(x) for k, x in v.pop.items()}, logs[-1])
    print(name, f"{time.time()-t0:.1f} s", flush=True)
if res:
  a, b = res["ckpop_hard_pen"], res["ckpop_hard_pen_kb0"]
  pe = all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
  po = all(np.array_equal(a[1][k], b[1][k]) for k in a[1])
  print(json.dumps({"params_identical": pe, "pop_identical": po, "n_leaves": len(a[0]),
                  "pop_a": a[2]["pop"], "pop_b": b[2]["pop"]}, default=float), flush=True)
t0 = time.time(); logs = []
v = V["ckpop_hard_pen_kb"](prob, seed=99, workdir="/tmp/kblock_ident", max_iters=6)
v.train(1e9, logs.append)
print("kb warm", f"{time.time()-t0:.1f} s", [(r.get("time_kb_ms"), r.get("kb_chg_t"), r.get("time_refine")) for r in logs if "iter" in r], flush=True)
