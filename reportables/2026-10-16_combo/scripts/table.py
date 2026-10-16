"""Print the stage2-style table from q2 json files. usage: table.py f1.json [f2.json ...]"""
import json, sys
print("| pass | R | offset | rough (circ) | mean abs c1 | line_match | dirt | viol L0..L4 | valid/128 | tiles changed |")
print("|---|---|---|---|---|---|---|---|---|---|")
for f in sys.argv[1:]:
    d = json.load(open(f)); lab = d["label"]
    def row(R, x, tc="–"):
        c = x["chain_obs"]; s = x["stats"]
        print(f"| {lab} | {R} | {c['offset']:+.2f} | {c['rough_circ']:.2f} | {c['c1abs']:.2f} | {s['line_match']:.3f} | {s['dirt_frac']:.3f} | {','.join(map(str, x['viol']))} | {x['valid']} | {tc} |")
    row("raw", d["pass_raw"])
    for a in d["arms"]:
        if a.get("init", 0) == 0:
            row(a["R"], a["final"], f"{100*a['final']['tile_changed']:.1f} %")
