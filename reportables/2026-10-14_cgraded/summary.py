"""First-valid sweep per world and final stats per arm (lambda from argv, default 20)."""
import json
import sys

import numpy as np

lam = sys.argv[1] if len(sys.argv) > 1 else "20"
r = json.load(open(f"reportables/2026-10-14_cgraded/result_lam{lam}.json"))
for name, a in r["arms"].items():
    sw = np.array([t["sweep"] for t in a["trace"]])
    v = np.array([np.array(t["viol"])[:, :-1].sum(1) for t in a["trace"]])     # (checkpoints, B)
    first = [int(sw[np.argmax(v[:, b] == 0)]) if (v[:, b] == 0).any() else -1 for b in range(v.shape[1])]
    st = {k: np.array([s[k] for s in a["stats"]]) for k in a["stats"][0]}
    print(name, "first-valid sweep (25-sweep resolution): median", np.median(first), sorted(first))
    print("  final stats mean±sd:", {k: f"{x.mean():.3f}±{x.std():.3f}" for k, x in st.items()})
    print("  final soft energy mean", round(float(np.mean(np.array(a["trace"][-1]["viol"])[:, -1])), 1))
