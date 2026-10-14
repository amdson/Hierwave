import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parent
runs = [("exact_toy_modes", "toy_modes", 2), ("cbig256_toy_modes", "toy_modes", 2), ("greedy_toy_modes", "toy_modes", 2), ("cfast_ground", "ground", 4)]
for tag, pn, T in runs:
    z = np.load(H / f"{tag}.npz"); j = json.load(open(H / f"{tag}.json"))
    print("==", tag, "F", j.get("free_energy"), j["info"])
    for l in range(T + 1):
        lq, dw, sw, kind = z[f"lq{l}"], z[f"dw{l}"], z[f"sw{l}"], z[f"kind{l}"]
        own, pr = kind == 0, kind == 1
        d = j["levels"][str(l)]["direct"]
        lo, hi = np.percentile(dw[own], [5, 95])
        inr = pr & (dw >= lo) & (dw <= hi)
        line = d["slope"] * dw + d["intercept"]
        st_in = ((line - lq)[inr] > 5).mean() if inr.any() else float("nan")
        print(f" L{l} own w[5,95]=[{lo:.1f},{hi:.1f}] probe w median {np.median(dw[pr]):.1f} own {np.median(dw[own]):.1f}; "
              f"logq share median own {np.median(lq[own]):.2f} probe {np.median(lq[pr]):.2f}; probes in own range {inr.sum()/pr.sum():.2f}, starved among them {st_in:.2f}; "
              f"median deficit {d['median_deficit_probe']:.2f}")
        if tag == "cfast_ground":
            print("   worst", [(w["probe"], w["block"], round(w["deficit"], 1)) for w in d["worst"]])
# figure
fig, axs = plt.subplots(4, 5, figsize=(16, 11))
for ax in axs.ravel(): ax.axis("off")
for row0, (tag, T, title) in enumerate([("cbig256_toy_modes", 2, "toy_modes cbig256"), ("cfast_ground", 4, "ground cfast 60 s")]):
    z = np.load(H / f"{tag}.npz")
    for r, (wk, wn) in enumerate([("dw", "direct"), ("sw", "subtree")]):
        for l in range(T + 1):
            ax = axs[2 * row0 + r, l]; ax.axis("on")
            kind = z[f"kind{l}"]; w = z[f"{wk}{l}"]; lq = z[f"lq{l}"]
            rng = np.random.default_rng(0)
            for k, c, lab in ((0, "tab:blue", "own"), (1, "tab:red", "probe")):
                idx = np.where(kind == k)[0]; idx = rng.choice(idx, min(len(idx), 1500), replace=False)
                ax.scatter(w[idx], lq[idx], s=4, alpha=0.35, c=c, label=lab, linewidths=0)
            ax.set_title(f"{title} L{l} ({wn})", fontsize=8); ax.tick_params(labelsize=7)
            ax.set_xlabel("target log-weight share", fontsize=7); ax.set_ylabel("log q share", fontsize=7)
            if l == 0 and r == 0: ax.legend(fontsize=7, markerscale=3)
fig.tight_layout(); fig.savefig(H / "images" / "block_scatter.png", dpi=110)
