"""Per-seed raw / final violations, free energy and stats of bench runs.
python reportables/2026-10-13_ctrunc/summary.py <run_id> [...]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rs = {json.loads(l)["run"]: json.loads(l) for l in open(ROOT / "results" / "bench.jsonl")}
for run in sys.argv[1:]:
    r = rs[run]
    print(run, "accepted", r["verdict"]["accepted"], "F", r["verdict"]["free_energy"])
    for m, inf in zip(r["per_seed"], r["info"]):
        print("  raw  ", {k: round(v, 4) for k, v in m["viol"]["raw"].items()})
        print("  final", {k: round(v, 4) for k, v in m["viol"]["final"].items()}, "F", round(m["free_energy"] or 0, 3),
              "updates", inf.get("updates"), "lam", [round(x, 1) for x in inf.get("lam", [])])
        print("  stats", {k: round(v[0], 3) for k, v in m["stats"]["final"].items()}, "probes", m["probes"])
    v = r["verdict"]
    print("  exact", v["exact_stats"], "rf", v["raw_vs_final"]["ok"], "xs", v["across_seeds"]["ok"])
