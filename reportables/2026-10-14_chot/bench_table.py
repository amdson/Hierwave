"""Gate-3 rows from results/bench.jsonl (tag chot_gate3)."""
import json
for l in open("results/bench.jsonl"):
    d = json.loads(l)
    if d.get("tag") != "chot_gate3":
        continue
    v = d["verdict"]
    for i, (s, inf) in enumerate(zip(d["per_seed"], d["info"])):
        fin = max(s["viol"]["final"].values()); raw = max(s["viol"]["raw"].values())
        st = s.get("stats", {}).get("raw", s.get("stats", {}))
        keys = [k for k in ("line_match", "dirt_frac", "horizontal", "vertical", "diagonal", "kl_block") if k in str(st)]
        stx = {k: (round(st[k][0], 3) if isinstance(st[k], (list, tuple)) else round(st[k], 3)) for k in keys if k in st}
        print(d["variant"], d["problem"], i, "upd", inf["updates"], "final %.2e raw %.2e" % (fin, raw),
              "F", round(s.get("free_energy") or float("nan"), 4),
              stx, "probes", s["probes"].get("flag") if isinstance(s.get("probes"), dict) else None)
    print("  verdict accepted:", v.get("accepted"), {k: (x.get("ok") if isinstance(x, dict) else x) for k, x in v.items()})
