"""Comparison table from results/bench.jsonl: latest full (2-seed) run of each
variant x problem.  python table.py cpass ctemp"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPT = {"toy_support": -0.4688, "toy_modes": 0.2513}
runs = {}
for line in open(ROOT / "results" / "bench.jsonl"):
    r = json.loads(line)
    if r["variant"] in sys.argv[1:] and r["seeds"] == 2:
        runs[(r["variant"], r["problem"])] = r
for (v, p), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    if p not in ("toy_support", "toy_modes", "dimers"):
        continue
    vd = r["verdict"]
    fe = [m["free_energy"] for m in r["per_seed"]]
    kl = [f - OPT[p] for f in fe] if p in OPT else None
    st = [{k: round(x[0], 3) for k, x in m["stats"]["final"].items()} for m in r["per_seed"]]
    fails = [k for k in ("final_violations", "exact_stats", "raw_vs_final", "across_seeds", "probes",
                         "finite_support") if isinstance(vd[k], dict) and vd[k].get("ok") is False]
    print(f"| {p} | {v} | {vd['final_violations']['worst_frac']:.1e} | "
          f"{', '.join(f'{f:.3f}' for f in fe)} | {', '.join(f'{k:.3f}' for k in kl) if kl else '-'} | "
          f"{'; '.join(str(s) for s in st)} | probes {vd['probes']['flagged']} | fails: {', '.join(fails)} | {r['run']}")
