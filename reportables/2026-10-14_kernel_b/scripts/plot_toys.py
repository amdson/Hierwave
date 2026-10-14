"""Figure: toy stat traces (chain means over 128 chains) vs exact."""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "reportables/2026-10-14_kernel_b/runs/"
panels = [
    ("toy_modes, untrained q, L=2, from own draws", "toy_toy_modes_untrained_L2_lam20_K1_tau1_own.json", "mode_B"),
    ("toy_modes, bad q, L=2, from exact samples", "toy_toy_modes_bad_L2_lam20_K1_tau1_exact.json", "mode_B"),
    ("same, q ratio dropped (wrong)", "toy_toy_modes_bad_L2_lam20_K1_tau1_exact_dropq.json", "mode_B"),
    ("toy_support, untrained q, L=1,2, from own draws", "toy_toy_support_untrained_L1-2_lam20_K1_tau1_own.json", "flat_block_frac"),
    ("toy_support, bad q, L=2, from exact samples", "toy_toy_support_bad_L2_lam20_K1_tau1_exact.json", "flat_block_frac"),
    ("same, q ratio dropped (wrong)", "toy_toy_support_bad_L2_lam20_K1_tau1_exact_qscale0.json", "flat_block_frac"),
]
fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (title, f, k) in zip(axs.ravel(), panels):
    try:
        d = json.load(open(R + f))
    except FileNotFoundError:
        ax.set_title(title + " (missing)", fontsize=9); continue
    tr = d["trace"][k]
    seg = d["moves_per_chain"] / (len(tr) - 1)
    x = [i * seg for i in range(len(tr))]
    ax.plot(x, tr, color="#3060a0", lw=1.5, label="chain mean")
    ex = d["stats"][k][2]
    ax.axhline(ex, color="#c04030", ls="--", lw=1.2, label="exact")
    ax.set_title(f"{title}\n{k}: est {d['stats'][k][0]:.4f} ± {d['stats'][k][1]:.4f}, exact {ex:.4f}", fontsize=9)
    ax.set_xlabel("moves per chain (8 blocks)"); ax.set_ylabel(k)
    ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("reportables/2026-10-14_kernel_b/images/toy_invariance.png", dpi=110)
