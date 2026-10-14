"""(adapted from cbound/dimers.py) dimers: orientation fractions, violations per cell (raw / final) and the
path-KL bound F_ub through a training run of a variant; score-less variants (crevise) log the bound instead of
the entropy.  usage: python dimers_run.py <variant> <train_s> <every> [NS=4] [out.json]"""
import json
import sys
import time

import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.variant import load_variants

name, train_s, every = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
NSf = int(sys.argv[4]) if len(sys.argv) > 4 else 4
out = sys.argv[5] if len(sys.argv) > 5 else f"reportables/2026-10-14_crevise/runs/dimers_{name}.json"
pr = get_problem("dimers")
region = Region(pr.W_eval)
cells = pr.H * pr.W_eval
V = load_variants()
c = V[name](pr, seed=0, workdir="/tmp/crevise_" + name)


def measure(NS, kinds=("raw",)):
    o = {k: [] for k in kinds}
    lq = []
    vi = {k: [] for k in kinds}
    for i in range(NS):
        seed = EVAL_SEED0 + 100 + i
        g = c.generate(seed, region)
        for k in kinds:
            x = _with_designed(pr, g[k], seed, region)
            s = pr.stats(x)
            o[k].append([s["horizontal"], s["vertical"], s["diagonal"]])
            E = en.energy(pr, x, region)
            vi[k].append(sum(E.viol.values()) / max(sum(E.nhard.values()), 1))
            if k == "raw":
                q = c.score(x, region)
                if q is not None:
                    lq.append(-q / cells)
    r = {"ent": float(np.mean(lq)) if lq else None}
    for k in kinds:
        r[k] = np.mean(o[k], 0).round(4).tolist()
        r[k + "_viol"] = float(np.mean(vi[k]))
    b = c.bound_report(8) if hasattr(c, "bound_report") else None
    if b:
        r["F_ub"] = b["F_ub"]
        r["logq_traj"] = b["logq_traj"]
        r["masked_logD"] = b["masked_logD"]
    return r


rows = []
t_meas = [0.0]


def log(d):
    if "update" in d and (d["update"] == 1 or d["update"] % every == 0):
        t0 = time.time()
        r = measure(NSf)
        r.update(update=d["update"], t=d["t"], tau=d["tau"], lam=d["lam"][0], train_viol=d["raw_viol"][0],
                 cost_per_world=d["cost_per_world"])
        rows.append(r)
        t_meas[0] += time.time() - t0
        print(json.dumps(r), flush=True)


c.train(train_s, log)
fin = measure(16, ("raw", "final"))
fin["update"] = c.updates
fin["measure_time"] = t_meas[0]
print("FINAL", json.dumps(fin), flush=True)
json.dump({"name": name, "cfg": {k: v for k, v in c.cfg.items() if k != "net" or True}, "rows": rows, "final": fin,
           "info": {k: v for k, v in c.info().items() if k in ("updates", "lam", "timing", "bound", "R", "rho")}},
          open(out, "w"), default=float)
