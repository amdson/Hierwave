"""gpen_p11: copy of reportables/2026-10-14_cr2_penalty/scripts/arm_eval.py (warm-up, train <train_s> seed <k> with
loop log, the bench's eval_seed) plus certificate slack per level on the same held-out eval worlds (raw and final):
A_tight - A and B - B_tight, as cmix certs.py (A_tight = full columns, B_tight = occupied columns of the block).
Nothing is written to results/bench.jsonl."""
import sys, json, time, pickle
import numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import eval_seed, EVAL_SEED0, _with_designed
from pathlib import Path


def cert_stats(p, st):
    t = np.asarray(st["t"])[None]
    out = {}
    for l in range(1, p.T + 1):
        c = p.chan[f"cert{l}"]
        v = np.asarray(st[f"cert{l}"])
        A, B = c.views["A"][v], c.views["B"][v]
        h = 1 << l
        b = t.reshape(t.shape[0], t.shape[1] // h, h, t.shape[2] // h, h)
        col = b.sum(2)
        At, Bt = (col == h).sum(-1)[0], (col > 0).sum(-1)[0]
        out[l] = [float(A.mean()), float(B.mean()), float((At - A).mean()), float((B - Bt).mean())]
    return out


name, train_s, k, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
V = load_variants(); prob = get("ground")
t0 = time.time()
w = V[name](prob, seed=99, workdir="/tmp/gpen_p11_warm", max_iters=3); w.train(1e9, lambda d: None)
print(f"warm-up {time.time() - t0:.1f} s", flush=True); del w
v = V[name](prob, seed=k, workdir=str(out))
f = open(out / "loop.jsonl", "w")
t0 = time.time()
v.train(train_s, lambda d: (f.write(json.dumps(d, default=float) + "\n"), f.flush()))
t_train = time.time() - t0
pickle.dump({"params": v.params, "lam": v.lam}, open(out / "params.pkl", "wb"))
m = eval_seed(prob, v, k, 16, out / ("eval_s%d.png" % k))
m["train_s"] = t_train; m["updates"] = v.updates; m["rkl_updates"] = v.rkl_updates; m["timing"] = v.timing
region = Region(prob.W_eval)
cs = {"raw": [], "final": []}
for i in range(16):
    seed = EVAL_SEED0 + 1000 * k + i
    g = v.generate(seed, region)
    for kind in cs:
        cs[kind].append(cert_stats(prob, _with_designed(prob, g[kind], seed, region)))
m["cert"] = {kind: {l: np.mean([c[l] for c in cs[kind]], 0).tolist() for l in range(1, prob.T + 1)} for kind in cs}
json.dump(m, open(out / "eval.json", "w"), default=float, indent=1)
print(json.dumps({"train_s": t_train, "updates": v.updates, "rkl": v.rkl_updates,
                  "final": m["viol"]["final"], "raw": m["viol"]["raw"], "F": m["free_energy"],
                  "line_match": m["stats"]["final"].get("line_match"), "dirt": m["stats"]["final"].get("dirt_frac"),
                  "n_valid_raw": m["n_valid_raw"], "cert[A,B,At-A,B-Bt]": m["cert"]}, default=float), flush=True)
