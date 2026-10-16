import sys, time, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench import energy as en
from castlegen.bench.variant import load_variants
budget = float(sys.argv[1]); R = int(sys.argv[2]); F = int(sys.argv[3])
p = get("ground"); V = load_variants()
v = V["ccertfree"](p, 0, "/tmp/claude-0/-workspace/bc078e85-9881-4790-a6e3-899916358fb2/scratchpad/cf_smoke", stage_R=R, stage_F=F)
t0 = time.time(); v.train(budget, lambda d: None); print("train", time.time()-t0, "updates", v.updates, flush=True)
reg = Region(128)
for R_, F_ in [(0, 0), (R, F)]:
    v.cfg["stage_R"], v.cfg["stage_F"] = R_, F_
    t0 = time.time(); g = v.generate(5000, reg); print("gen", R_, F_, time.time()-t0)
    for k in ("raw", "final"):
        x = dict(g[k]); x["fill"] = p.designed_values(5000, reg)["fill"]
        E = en.energy(p, x, reg); print(R_, F_, k, E.viol, p.stats(x))
