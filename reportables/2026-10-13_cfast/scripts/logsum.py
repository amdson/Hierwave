"""Summarise the latest run's training log(s) and verdict.  usage: logsum.py <substring>"""
import json, sys
from pathlib import Path

runs = sorted(d for d in Path("results/runs").iterdir() if sys.argv[1] in d.name)
d = runs[-1]
print(d.name)
for f in sorted(d.glob("train_s*.jsonl")):
    L = [json.loads(l) for l in open(f)]
    U = [l for l in L if "update" in l]
    print(f.name, L[0])
    for u in U[::max(1, len(U) // 8)] + U[-1:]:
        print(" ", u["update"], round(u["t"], 1), [float(f"{x:.2g}") for x in u["raw_viol"]],
              [round(x, 2) for x in u["lam"]], round(u["cost_per_world"], 1))
for l in open("results/bench.jsonl"):
    r = json.loads(l)
    if r["run"] == d.name:
        for m in r["per_seed"]:
            print("final viol", [float(f"{v:.2g}") for v in m["viol"]["final"].values()],
                  "raw", [float(f"{v:.2g}") for v in m["viol"]["raw"].values()], "F", m.get("free_energy"),
                  "train_s", round(m["train_s"], 1))
            print("stats raw", {k: round(v[0], 3) for k, v in m["stats"]["raw"].items()})
        print("exact", r["verdict"]["exact_stats"])
        print("verdict", {k: v for k, v in r["verdict"].items() if not isinstance(v, dict)},
              {k: v.get("ok") for k, v in r["verdict"].items() if isinstance(v, dict)})
