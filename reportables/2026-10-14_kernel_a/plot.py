"""Figures: held-out log q per fine cell by level during ML; clamped L = 3 completions."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
R = "reportables/2026-10-14_kernel_a"
p = get("ground"); reg = Region(p.W_train)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for run, ls in (("ml_init", "-"), ("ml_warm", "--")):
    rows = [json.loads(l) for l in open(f"{R}/runs/{run}/train.jsonl") if "heldout" in l]
    u = [r["update"] for r in rows]
    for l in range(3):
        ax[l].plot(u, [r["heldout"][l] for r in rows], ls, label=run)
for l in range(3):
    ax[l].set_title(f"held-out log q / fine cell, level {l}"); ax[l].set_xlabel("update")
ax[0].legend(); plt.tight_layout(); plt.savefig(f"{R}/images/ml_heldout.png", dpi=110)
c2 = np.load("reportables/2026-10-14_cstart2/runs/cstart_f1L3_ln6/ground_raw.npz")
mi = np.load(f"{R}/runs/ml_init/ground_raw.npz")
pr = p.probes(3_000_000, reg)
rows = [("probe j0", pr[0]["t"]), ("probe j2", pr[2]["t"]),
        ("cstart2 LN6, clamp L=3 j0", c2["trained__probe_L3__t"][0]), ("cstart2 LN6, clamp L=3 j2", c2["trained__probe_L3__t"][2]),
        ("ML (kernel-a), clamp L=3 j0", mi["trained__probe_L3__t"][0]), ("ML (kernel-a), clamp L=3 j2", mi["trained__probe_L3__t"][2])]
fig, ax = plt.subplots(3, 2, figsize=(12, 6.5))
for i, (n, t) in enumerate(rows):
    a = ax[i // 2, i % 2]; a.imshow(p.render({"t": t}, scale=1), interpolation="nearest"); a.set_title(n, fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(f"{R}/images/clamped_completions.png", dpi=110)
