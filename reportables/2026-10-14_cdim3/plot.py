import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
R = {k: json.load(open(f"runs/diag_{k}.json")) for k in ("mlp", "table_feat")}
C = {"mlp": "#1f5fa8", "table_feat": "#d9822b"}
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for k, rec in R.items():
    s = [r["step"] for r in rec]
    for o, ls in (("h", "-"), ("v", "--"), ("d", ":")):
        ax[0].plot(s, [r["mix"][o] for r in rec], color=C[k], ls=ls, label=f"{k} {o}")
    ax[3].plot(s, [r["cos_logstripe"] for r in rec], color=C[k], label=k)
    for j, st in enumerate((0, 50, 100, 200)):
        r = next(r for r in rec if r["step"] == st)
        a = 0.25 + 0.25 * j
        ax[1].plot(range(16), r["p_h_draw"], color=C[k], alpha=a, marker="o", ms=3, label=f"{k} step {st}")
        ax[2].plot(range(16), r["H_draw"], color=C[k], alpha=a, marker="o", ms=3, label=f"{k} step {st}")
for y in (0.505, 0.248):
    ax[0].axhline(y, color="k", lw=0.6, ls=":")
ax[0].set_title("orientation mix (h solid, v dashed, diag dotted)"); ax[0].set_xlabel("exact step")
ax[1].set_title("P_q(horizontal) at draw i (raster colour order)"); ax[1].set_xlabel("draw i (row = i // 4)")
ax[2].set_title("E_q[policy entropy] at draw i (nats)"); ax[2].set_xlabel("draw i")
ax[3].axhline(0, color="k", lw=0.6)
ax[3].set_title("cos(-dJ/dθ, d log Q(stripes)/dθ)"); ax[3].set_xlabel("exact step")
ax[0].legend(fontsize=6, ncol=2); ax[1].legend(fontsize=6); ax[3].legend(fontsize=7)
plt.tight_layout(); plt.savefig("images/stripe_attractor.png", dpi=110)
