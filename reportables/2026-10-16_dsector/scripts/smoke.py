import sys, json, time
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
name, t = sys.argv[1], float(sys.argv[2])
p = get("dimers"); V = load_variants()
logs = []
v = V[name](p, seed=0, workdir="/tmp/dsec_smoke")
v.train(t, logs.append)
its = [r for r in logs if "iter" in r]
print("updates", v.updates, "iters", len(its), "timing", v.timing)
for r in its[::max(1, len(its)//5)] + its[-1:]:
    print(r["iter"], r["t"], r["pop"].get("valid"), r["pop"].get("viol_per_cell"), r.get("rkl_raw_viol"), r.get("lam"), r.get("worm_repaired"))
