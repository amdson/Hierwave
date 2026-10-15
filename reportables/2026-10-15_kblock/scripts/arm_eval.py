"""wave5-pop instrumented arm: warm-up instance (memoised build), then train(<train_s>) seed <k>
with the loop log, then the bench's own eval_seed (frozen harness code, imported, not edited):
final/raw violations per level, F, stats, v2 block check (rho, starvation).  Nothing is written
to results/bench.jsonl.  Own draws only, inside the frozen constitution."""
import sys, json, time, pickle
import numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.run import eval_seed
from pathlib import Path
name, train_s, k, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
V = load_variants(); prob = get("ground")
t0 = time.time()
w = V[name](prob, seed=99, workdir="/tmp/kblock_warm", max_iters=3); w.train(1e9, lambda d: None)
print(f"warm-up {time.time() - t0:.1f} s", flush=True); del w
v = V[name](prob, seed=k, workdir=str(out))
f = open(out / "loop.jsonl", "w")
t0 = time.time()
v.train(train_s, lambda d: (f.write(json.dumps(d, default=float) + "\n"), f.flush()))
t_train = time.time() - t0
pickle.dump({"params": v.params, "lam": v.lam}, open(out / "params.pkl", "wb"))
m = eval_seed(prob, v, k, 16, out / ("eval_s%d.png" % k))
m["train_s"] = t_train; m["updates"] = v.updates; m["rkl_updates"] = v.rkl_updates; m["timing"] = v.timing
json.dump(m, open(out / "eval.json", "w"), default=float, indent=1)
pickle.dump({"params": v.params, "lam": v.lam}, open(out / "params.pkl", "wb"))
b = m["blocks"]
print(json.dumps({"train_s": t_train, "setup": v.timing["setup"], "updates": v.updates, "rkl": v.rkl_updates,
                  "final": m["viol"]["final"], "raw": m["viol"]["raw"], "F": m["free_energy"],
                  "line_match": m["stats"]["final"].get("line_match"), "dirt": m["stats"]["final"].get("dirt_frac"),
                  "n_valid_raw": m["n_valid_raw"], "probes": m["probes"],
                  "blocks": {kk: b[kk] for kk in b if kk in ("rho", "starved", "rho_pool", "starve", "per_level")}},
                 default=float), flush=True)
