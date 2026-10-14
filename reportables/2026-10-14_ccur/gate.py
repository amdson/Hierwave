"""Gate numbers from a run's ground_summary.json: fine cost per fine cell below L,
own coarse vs probe-clamped (fill-following), plus dirt fraction."""
import json, sys
for r in sys.argv[1:]:
    s = json.load(open(f"reportables/2026-10-14_ccur/runs/{r}/ground_summary.json"))["summary"]
    for m in ("untrained", "trained"):
        c = lambda k, L: sum(s[f"{m}/{k}"][str(l)]["cost_per_cell"][0] for l in range(L))
        print(f"{r:12s} {m:9s} L=4 own {c('own',4):.2f} fill {c('probe_L4',4):.2f} | L=3 own {c('own',3):.2f} fill {c('probe_L3',3):.2f}"
              f" | dirt own {s[f'{m}/own']['0']['dirt_frac']:.3f} clampL4 {s[f'{m}/probe_L4']['0']['dirt_frac']:.3f} clampL3 {s[f'{m}/probe_L3']['0']['dirt_frac']:.3f}")
