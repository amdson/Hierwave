"""Per-level entropy (own-draw diag passes, csat) over training."""
import json, sys
for r in sys.argv[1:]:
    D = [json.loads(l) for l in open(f"runs/{r}/train_ground.jsonl")]
    D = [d for d in D if "diag" in d]
    print(r, len(D), "diag rows")
    for d in D[::max(1, len(D) // 6)] + D[-1:]:
        g = d["diag"]
        print(f" u{d['update']:5d} H " + " ".join(f"L{l}={g[f'L{l}']['H']:.3f}" for l in range(5) if f"L{l}" in g)
              + f" dirt {g['dirt']:.3f} lm {g.get('line_match', float('nan')):.3f} lam {[round(x,1) for x in d['lam']]}")
