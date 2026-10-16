import json
f=lambda d:",".join(f"{d[str(l)]:.1e}" if d[str(l)] else "0" for l in range(5))
print("| variant | seed | final viol L0-4 | F | line_match raw/final | rough raw/final | updates? | rho_pool L0-4 | starved_probe L0-4 | starved_own L0-4 | probe flag | accepted |\n|"+"---|"*12)
for l in open("results/bench.jsonl"):
    r=json.loads(l)
    if r["train_budget_s"]!=600 or r["variant"] not in("clongfit_stage","cfitpen_ml2_stage") or "104727" not in r["run"] and r["time"]<"20261016-104": continue
    for k,m in enumerate(r["per_seed"]):
        b=m["blocks"]["levels"]; g=lambda key:"/".join(f"{b[str(i)]['direct'][key]:.3f}" for i in range(5))
        s=m["stats"]
        print(f"| {r['variant']} | {k} | {f(m['viol']['final'])} | {m['free_energy']:.4f} | {s['raw']['line_match'][0]:.3f}/{s['final']['line_match'][0]:.3f} | {s['raw']['roughness'][0]:.2f}/{s['final']['roughness'][0]:.2f} | - | {g('rho_pool')} | {g('starved_probe')} | {g('starved_own')} | {m['probes']['flag']} | |")
    v=r["verdict"]; print(f"<!-- {r['variant']} run {r['run']} final ok {v['final_violations']['ok']} raw_vs_final {v['raw_vs_final']['per_stat']} accepted {v['accepted']} probes {v['probes']['flagged']} across {v['across_seeds']['ok']} -->")
