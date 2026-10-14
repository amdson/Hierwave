"""Gate table: fine cost per fine cell (levels < L) in probe-lifted contexts vs own, nv per cell,
cost re-priced at lam = 20 for violations, log q per cell by level (fill context, L = 3)."""
import json, sys
for r in sys.argv[1:]:
    J = json.load(open(f"runs/{r}/ground_summary.json")); s = J["summary"]
    for m in ("untrained", "trained"):
        lam = J["lam"][m]
        def c(k, L, f="cost_per_cell"):
            return sum(s[f"{m}/{k}"][str(l)][f][0] for l in range(L))
        def c20(k, L):
            return sum(s[f"{m}/{k}"][str(l)]["cost_per_cell"][0] + (20 - lam[l]) * s[f"{m}/{k}"][str(l)]["nv_per_cell"][0] for l in range(L))
        print(f"{r:16s} {m:9s} upd {J['updates']} L3 fill {c('probe_L3',3):.3f} (lam20 {c20('probe_L3',3):.3f}, nv {c('probe_L3',3,'nv_per_cell'):.4f})"
              f" own {c('own',3):.3f} | L4 fill {c('probe_L4',4):.3f} (lam20 {c20('probe_L4',4):.3f}) own {c('own',4):.3f}"
              f" | soft L3 fill {c('probe_L3',3,'soft_per_cell'):.3f} | logq/cell L3 fill by level "
              + " ".join(f"{s[f'{m}/probe_L3'][str(l)]['logq_per_cell'][0]:.3f}" for l in range(3)))
