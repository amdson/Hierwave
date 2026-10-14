"""cbound (copied from cdim/run.py): dimers orientation/entropy/violations through a cfast-variant training run."""
import sys, json, numpy as np
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problem import Region
from castlegen.bench import energy as en
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.variant import load_variants
name, train_s = sys.argv[1], float(sys.argv[2]); every = int(sys.argv[3]); NSf = int(sys.argv[4]) if len(sys.argv) > 4 else 4
pr = get_problem("dimers"); region = Region(pr.W_eval); cells = pr.H * pr.W_eval
V = load_variants(); c = V[name](pr, seed=0, workdir="/tmp/cbound_" + name)
def measure(NS, kinds=("raw",)):
    o = {k: [] for k in kinds}; lq = []; vi = {k: [] for k in kinds}
    for i in range(NS):
        seed = EVAL_SEED0 + 100 + i; g = c.generate(seed, region)
        for k in kinds:
            x = _with_designed(pr, g[k], seed, region); s = pr.stats(x)
            o[k].append([s["horizontal"], s["vertical"], s["diagonal"]])
            E = en.energy(pr, x, region); vi[k].append(sum(E.viol.values()) / max(sum(E.nhard.values()), 1))
            if k == "raw": q = c.score(x, region); lq.append(-q / cells)
    r = {"ent": float(np.mean(lq))}
    for k in kinds: r[k] = np.mean(o[k], 0).round(4).tolist(); r[k + "_viol"] = float(np.mean(vi[k]))
    return r
rows = []
def log(d):
    if "update" in d and (d["update"] == 1 or d["update"] % every == 0):
        r = measure(NSf); r.update(update=d["update"], t=d["t"], tau=d["tau"], lam=d.get("lam_used_next", d["lam"])[0], bound=d.get("bound"), train_viol=d["raw_viol"][0])
        rows.append(r); print(json.dumps(r), flush=True)
c.train(train_s, log)
fin = measure(16, ("raw", "final")); fin["update"] = c.updates; print("FINAL", json.dumps(fin), flush=True)
json.dump({"name": name, "rows": rows, "final": fin}, open(f"reportables/2026-10-14_cbound/dimers_{name}.json", "w"))
