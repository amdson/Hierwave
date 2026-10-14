"""Diagnostic B: dimers orientation fractions, untrained (greedy, cfast lr=0) and through cfast training."""
import sys, time, json, numpy as np
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problem import Region
from castlegen.bench import energy as en
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.variant import load_variants
mode, train_s, every = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
pr = get_problem("dimers"); region = Region(pr.W_eval); cells = pr.H * pr.W_eval
V = load_variants()
NS = 8
def measure(v, tag, upd=None, t=None, kinds=("raw",)):
    rows = {k: [] for k in kinds}; lq = []; viol = []
    for i in range(NS):
        seed = EVAL_SEED0 + 100 + i
        g = v.generate(seed, region)
        for k in kinds:
            x = _with_designed(pr, g[k], seed, region); s = pr.stats(x)
            rows[k].append([s["horizontal"], s["vertical"], s["diagonal"]])
            if k == "raw":
                E = en.energy(pr, x, region); viol.append(sum(E.viol.values()) / max(sum(E.nhard.values()), 1) if isinstance(E.nhard, dict) else float(np.sum(list(E.viol.values()))))
                q = v.score(x, region); lq.append(-q / cells if q is not None else np.nan)
    out = {"tag": tag, "update": upd, "t": t, "ent_per_cell": float(np.nanmean(lq))}
    for k in kinds: out[k] = np.mean(rows[k], 0).round(4).tolist()
    return out
res = []
if mode == "untrained":
    pr_ = pr
    g = V["greedy"](pr, seed=0, workdir="/tmp/probesB"); g.train(0, lambda d: None)
    r = measure(g, "greedy", kinds=("raw", "final")); r["ent_per_cell"] = None; print(r, flush=True); res.append(r)
    for lr_ in (0.0,):
        c = V["cfast"](pr, seed=0, workdir="/tmp/probesB", lr=lr_)
        c.train(6, lambda d: None)
        print("updates", c.updates, flush=True)
        r = measure(c, "cfast_lr0", kinds=("raw", "final")); print(r, flush=True); res.append(r)
else:
    c = V[mode](pr, seed=0, workdir="/tmp/probesB")
    nxt = [0.0]; 
    res.append(None)
    def log(d):
        if "update" in d and d["t"] >= nxt[0]:
            nxt[0] = d["t"] + every
            r = measure(c, mode, d["update"], d["t"]); r["tau"] = d["tau"]; r["raw_viol"] = d["raw_viol"]
            res.append(r); print(json.dumps(r), flush=True)
    c.train(train_s, log)
    res[0] = measure(c, mode + "_end", c.updates, train_s, kinds=("raw", "final")); print(res[0], flush=True)
json.dump(res, open("/workspace/hierwave_git/reportables/2026-10-14_probes/B_%s.json" % mode, "w"))
