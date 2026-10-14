"""Summarise the latest b300 bench runs of the given variants on a problem from results/bench.jsonl."""
import json, sys
prob = sys.argv[1]; names = sys.argv[2].split(","); tag = sys.argv[3] if len(sys.argv) > 3 else None
rows = [json.loads(x) for x in open("results/bench.jsonl")]
for n in names:
    rs = [r for r in rows if r["variant"] == n and r["problem"] == prob and r["train_budget_s"] == 300.0
          and (tag is None or r["tag"].startswith(tag))]
    if not rs:
        print(n, "none"); continue
    r = rs[-1]
    print(f"== {n} {prob} run {r['run']} wall {r['wall_s']:.0f}s accepted {r.get('verdict', {}).get('accepted', r.get('accepted'))}")
    for i, s in enumerate(r["per_seed"]):
        fv = max(s["viol"]["final"].values()); rv = max(s["viol"]["raw"].values())
        st = s["stats"]["raw"]
        upd = s.get("info", {}).get("updates")
        print(f"  seed {i}: updates {upd} worst final {fv:.2e} raw {rv:.2e} F {s.get('free_energy')}"
              + " " + " ".join(f"{k} {v[0]:.3f}" for k, v in st.items()))
    v = r.get("verdict", {})
    print("  verdict keys:", {k: (v[k].get("ok") if isinstance(v[k], dict) else v[k]) for k in v} if v else list(r.keys()))
