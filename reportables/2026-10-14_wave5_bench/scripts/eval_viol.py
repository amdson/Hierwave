"""Raw vs final hard violations per level and per factor on the bench's held-out noise
(EVAL_SEED0 + 1000 k + i), for a ckernel params pickle (run_ck.py format).
usage: eval_viol.py <params_ground.pkl> <out.json> [n_per_k=16] [ks=0,1]"""
import json, os, sys
import numpy as np
from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.variant import load_variants
load_variants()          # same import order as the bench: same JAX compile-cache dir
from castlegen.variants.ckernel import CKernel

pkl, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ks = [int(k) for k in (sys.argv[4] if len(sys.argv) > 4 else "0,1").split(",")]
p = get("ground")
V = CKernel(p, 0, "/tmp", init=pkl)
region = Region(p.W_eval)
res = {kind: {"viol": np.zeros(p.T + 1), "n": np.zeros(p.T + 1), "fac": {}, "fac_n": {}} for kind in ("raw", "final")}
stats = {"raw": [], "final": []}
worlds = []
for k in ks:
    for i in range(n):
        seed = EVAL_SEED0 + 1000 * k + i
        g = V.generate(seed, region)
        row = {"seed": seed}
        for kind in ("raw", "final"):
            x = _with_designed(p, g[kind], seed, region)
            E = en.energy(p, x, region)
            r = res[kind]
            for l in range(p.T + 1):
                r["viol"][l] += E.viol[l]; r["n"][l] += E.nhard[l]
            for f, (_, v, m) in E.per_factor.items():
                r["fac"][f] = r["fac"].get(f, 0) + v
                r["fac_n"][f] = r["fac_n"].get(f, 0) + m
            row[kind] = {l: E.viol[l] for l in range(p.T + 1)}
            stats[kind].append(p.stats(x))
        worlds.append(row)
o = {"pkl": pkl, "n_worlds": len(worlds)}
for kind, r in res.items():
    o[kind] = {"frac": (r["viol"] / np.maximum(r["n"], 1)).tolist(), "count": r["viol"].tolist(), "n": r["n"].tolist(),
               "factors": {f: [v, r["fac_n"][f]] for f, v in r["fac"].items() if v}}
    o[kind]["line_match"] = float(np.mean([s["line_match"] for s in stats[kind]]))
    o[kind]["dirt"] = float(np.mean([s["dirt_frac"] for s in stats[kind]]))
o["worlds"] = worlds
if os.environ.get("BLOCKS", "1") == "1":
    from castlegen.bench.run import block_check
    b = block_check(p, V, 0, region)
    o["blocks"] = {l: {"rho": b["levels"][l]["direct"]["rho_pool"], "rho_sub": b["levels"][l]["subtree"]["rho_pool"],
                       "starved": b["levels"][l]["direct"]["starved_probe"]} for l in b["levels"]}
    print("blocks", json.dumps(o["blocks"]))
json.dump(o, open(out, "w"), indent=1, default=float)
for kind in ("raw", "final"):
    print(kind, "frac", np.round(o[kind]["frac"], 5), "count", o[kind]["count"], "lm", round(o[kind]["line_match"], 3))
    print("  factors", o[kind]["factors"])
