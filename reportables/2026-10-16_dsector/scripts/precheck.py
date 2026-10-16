"""usage: precheck.py <variant> <train_s> <out.json> [key=val ...]  -- train, then own-draw h/v/d, viol/cell, % valid
of raw and (if staged) final draws over NS held-out seeds (periodic eval region)."""
import sys, json, time, pickle
import numpy as np
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench import energy as en
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.variant import load_variants
name, train_s, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = {}
for a in sys.argv[4:]:
    k, v = a.split("="); cfg[k] = float(v) if "." in v else int(v)
pr = get("dimers"); region = Region(pr.W_eval); cells = pr.H * pr.W_eval
v = load_variants()[name](pr, seed=0, workdir="/tmp/dsec_pre_" + name, **cfg)
logs = []
t0 = time.time(); v.train(train_s, logs.append); print("trained", time.time() - t0, "updates", v.updates, flush=True)
its = [r for r in logs if "iter" in r]
res = {"name": name, "cfg": cfg, "updates": v.updates, "lam": its[-1].get("lam"), "rkl_raw_viol_last": its[-1].get("rkl_raw_viol"),
       "pop_last": its[-1]["pop"], "kinds": {}}
NS = 16
rows = {}
t1 = time.time()
for i in range(NS):
    seed = EVAL_SEED0 + 100 + i
    g = v.generate(seed, region)
    for k in ("raw", "final"):
        if k not in g: continue
        x = _with_designed(pr, g[k], seed, region); s = pr.stats(x)
        E = en.energy(pr, x, region); nv = sum(E.viol.values())
        rows.setdefault(k, []).append([s["horizontal"], s["vertical"], s["diagonal"], nv / cells, float(nv == 0)])
for k, r in rows.items():
    m = np.mean(r, 0)
    res["kinds"][k] = {"hvd": m[:3].round(4).tolist(), "viol_per_cell": float(m[3]), "valid_frac": float(m[4]),
                       "hvd_std": np.std(r, 0)[:3].round(3).tolist()}
    print(k, res["kinds"][k], flush=True)
print("gen time", time.time() - t1)
json.dump(res, open(out, "w"), indent=1)
pickle.dump({"trained": {"params": v.params, "lam": list(map(float, v.lam))}}, open(out.replace(".json", ".pkl"), "wb"))
print("DONE", flush=True)
