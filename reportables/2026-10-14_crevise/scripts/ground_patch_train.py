"""Train a variant on the 8 x 16 three-level ground patch (ground_patch.build) and compare its raw / repaired
draws with the annealed-importance reference (ground_patch.py mcmc): P(dirt) per cell, dirt fraction, soft
energy, violations per level, line match against the designed surface.
usage: python ground_patch_train.py <variant> <train_s> <n_eval> [out.json]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.variant import load_variants

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ground_patch import FILL, H, SURFACE, T, W, build  # noqa: E402

name, train_s, n_eval = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else f"reportables/2026-10-14_crevise/runs/ground_patch_{name}.json"
p = build()
region = Region(W)
V = load_variants()
var = V[name](p, seed=0, workdir="/tmp/crevise_gp_" + name)
rows = []


def log(d):
    if "update" in d and (d["update"] == 1 or d["update"] % 50 == 0):
        rows.append({k: d[k] for k in ("update", "t", "tau", "loss", "cost_per_world", "raw_viol", "lam")})
        print(json.dumps(rows[-1]), flush=True)


t0 = time.time()
var.train(train_s, log)
print("trained", var.updates, "updates in", time.time() - t0, flush=True)

ref = json.load(open("reportables/2026-10-14_crevise/runs/ground_patch_mcmc.json"))
pref = np.array(ref["p_dirt"])
res = {"name": name, "updates": var.updates, "lam": var.lam.tolist(), "train_rows": rows,
       "info": {k: v for k, v in var.info().items() if k in ("bound", "R", "rho", "timing")}}
for kind in ("raw", "final"):
    ts, S, viol, nh, lm = [], [], np.zeros(T + 1), np.zeros(T + 1), []
    for i in range(n_eval):
        seed = 1_000_000 + i
        g = var.generate(seed, region)[kind]
        x = {**p.designed_values(seed, region), **{k: np.asarray(v) for k, v in g.items()}}
        E = en.energy(p, x, region)
        S.append(E.soft)
        for l in range(T + 1):
            viol[l] += E.viol[l]
            nh[l] += E.nhard[l]
        t = np.asarray(x["t"])
        ts.append(t)
        yc = np.arange(H)[:, None]
        lm.append(float((t == (yc >= SURFACE[None, :])).mean()))
    ts = np.array(ts)
    pd = ts.mean(0)
    res[kind] = {"dirt": float(ts.mean()), "soft": float(np.mean(S)), "viol_frac": (viol / np.maximum(nh, 1)).tolist(),
                 "line_match": float(np.mean(lm)), "distinct": len({t.tobytes() for t in ts}),
                 "p_dirt": pd.tolist(), "p_dirt_fold8": (0.5 * (pd[:, :8] + pd[:, 8:])).tolist(),
                 "L1_to_mcmc": float(np.abs(pd - pref).mean()),
                 "max_abs_to_mcmc": float(np.abs(pd - pref).max())}
    print(kind, json.dumps({k: v for k, v in res[kind].items() if not k.startswith("p_dirt")}), flush=True)
print("mcmc ref: dirt", ref["dirt_weighted"], "soft", ref["soft_weighted"], "ess", ref["ess"], "valid", ref["valid"])
json.dump(res, open(out, "w"), default=float)
