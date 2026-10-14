import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
arms = [("armA_w20", "A: ML, exp(-20V) weights"), ("armB2_rkl16", "B2: + 16 RKL steps/iter, lr 1e-3"),
        ("armB4_rkl32lr1", "B4: + 32 RKL steps/iter, lr 1e-3"), ("armB3_rkl32", "B3: + 32 RKL steps/iter, lr 3e-3")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for arm, lab in arms:
    try:
        L = [json.loads(l) for l in open(f"runs/{arm}/loop.jsonl")]
    except FileNotFoundError:
        continue
    R = [r for r in L if "lv_own_raw" in r]
    u = [r["update"] for r in R]
    ax[0].plot(u, [r["lv_own_raw"]["frac"][2] + 1e-5 for r in R], label=lab)
    ax[1].plot(u, [r["lv_own_raw"]["frac"][3] + 1e-5 for r in R], label=lab)
    ax[2].plot(u, [r["own_raw"]["line_match"] for r in R], label=lab)
for a, t in zip(ax, ["own raw violated frac, L2", "own raw violated frac, L3", "own raw line_match"]):
    a.set_title(t); a.set_xlabel("ML updates")
ax[0].set_yscale("log"); ax[1].set_yscale("log")
for a in ax[:2]:
    a.axhline(2e-4, color="k", ls=":", lw=1)
ax[2].legend(fontsize=8)
fig.suptitle("ckernel violation arms (upper bound: probe-lift contexts, change request #1), ground, 300 s, 1 seed")
fig.tight_layout(); fig.savefig("images/violation_arms.png", dpi=110)
