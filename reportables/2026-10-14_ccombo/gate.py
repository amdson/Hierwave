"""Gate table (cstart2's gate.py + line_match/dirt from change.json) for runs/<name>."""
import json, sys
for r in sys.argv[1:]:
    J = json.load(open(f"runs/{r}/ground_summary.json")); s = J["summary"]; ch = json.load(open(f"runs/{r}/change.json"))
    for m in ("untrained", "trained"):
        def c(k, L, f="cost_per_cell"):
            return sum(s[f"{m}/{k}"][str(l)][f][0] for l in range(L))
        print(f"{r:14s} {m:9s} upd {J['updates']} | L3 fill {c('probe_L3',3):.3f} soft {c('probe_L3',3,'soft_per_cell'):.3f}"
              f" -logq {-c('probe_L3',3,'logq_per_cell'):.3f} nv {c('probe_L3',3,'nv_per_cell'):.4f} own {c('own',3):.3f}"
              f" | L4 fill {c('probe_L4',4):.3f} soft {c('probe_L4',4,'soft_per_cell'):.3f} own {c('own',4):.3f}"
              f" | lm L3 {ch[m+'/L3/line_match_j0']:.3f} L4 {ch[m+'/L4/line_match_j0']:.3f} own {ch[m+'/own/line_match']:.3f} dirt {ch[m+'/own/dirt']:.3f}"
              f" | rho2 chg L3 {ch.get(m+'/L3/rho2/j0_vs_j2', float('nan')):.3f} rho3 chg L4 {ch.get(m+'/L4/rho3/j0_vs_j2', float('nan')):.3f}")
