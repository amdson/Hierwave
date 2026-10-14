"""Summarise bench.jsonl entries by tag: updates, worst final viol (level), F, ground stats.
usage: table.py tag [tag ...]"""
import json, sys
rows = [json.loads(l) for l in open("results/bench.jsonl") if l.strip()]
for r in rows:
    if r.get("tag") not in sys.argv[1:]:
        continue
    for k, s in enumerate(r["per_seed"]):
        info = r["info"][k] if k < len(r["info"]) else {}
        fv = s["viol"]["final"]
        wl = max(fv, key=lambda l: fv[l])
        st = s["stats"]["raw"]
        g = lambda n: round(st[n][0], 3) if n in st else None
        setup = round(info["timing"]["setup"], 1) if isinstance(info.get("timing"), dict) and "setup" in info["timing"] else None
        print(f'{r["tag"]:16s} {r["variant"]:18s} {r["problem"]:12s} s{k} upd {info.get("updates")} '
              f'worst_final {fv[wl]:.1e} (l{wl}) F {s["free_energy"]:.3f} '
              + (f'dirt {g("dirt_frac")} line {g("line_match")} fill_err {g("fill_err")} rough {g("roughness")} ' if "dirt_frac" in st else '')
              + f'probe_flag {(s["probes"] or {}).get("flag")} setup {setup}')
