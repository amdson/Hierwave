"""Equilibrium times for allair.json from 32-chain means, calibrated on the probe arm.
Reference = probe arm's mean over its last 500 steps; band = 3 x the sd of its 50-step rolling mean
over that window (floor: 0.05 rows offset, 0.01 roughness, 0.2 % S).  An arm is equilibrated from
the first step after which its 50-step rolling mean stays inside the band for every quantity.
Also writes allair.png."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "reportables/2026-10-14_cmix"
r = json.load(open(f"{D}/allair.json"))
WIN = 10                                                            # records of 5 steps = 50 steps


def roll(x):
    return np.convolve(x, np.ones(WIN) / WIN, mode="valid")


cal = np.array(r["probe_naive"]["curve"])
late = cal[:, 0] >= cal[-1, 0] - 500
ref, band = {}, {}
for j, name, floor in ((2, "offset", 0.05), (3, "rough", 0.01), (4, "S", None)):
    ref[name] = cal[late, j].mean()
    sd = roll(cal[late, j]).std()
    band[name] = max(3 * sd, floor if floor is not None else 0.002 * abs(ref[name]))
res = {}
for arm, o in r.items():
    c = np.array(o["curve"])
    steps, secs = c[WIN - 1:, 0], c[WIN - 1:, 1]
    ok = np.ones(len(steps), bool)
    for j, name in ((2, "offset"), (3, "rough"), (4, "S")):
        ok &= np.abs(roll(c[:, j]) - ref[name]) < band[name]
    ok &= roll(c[:, 5]) == 0
    bad = np.flatnonzero(~ok)
    i = 0 if bad.size == 0 else bad[-1] + 1
    res[arm] = {"eq_step": int(steps[i]) if i < len(steps) else None, "eq_sec": float(secs[i]) if i < len(steps) else None,
                "valid_from": o["all_valid_from_step"], "line_match": round(o["line_match"], 3),
                "secs_per_step": round(o["secs_total"] / c[-1, 0], 4)}
    print(arm, res[arm])
print("reference", {k: round(v, 3) for k, v in ref.items()}, "band", {k: round(v, 3) for k, v in band.items()})
json.dump({"ref": ref, "band": band, "arms": res}, open(f"{D}/allair_eq.json", "w"), indent=1)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
COL = {"probe_naive": "#888888", "air_naive": "#2a78d6", "dirt_naive": "#eb6834", "random_naive": "#1baf7a",
       "air_ms": "#2a78d6", "dirt_ms": "#eb6834"}
for arm, o in r.items():
    c = np.array(o["curve"])
    ls = "--" if arm.endswith("_ms") else "-"
    for ax, j in zip(axs, (2, 3, 4)):
        ax.plot(c[:, 0] + 1, c[:, j], color=COL[arm], ls=ls, lw=1.8, label=arm)
for ax, (name, j) in zip(axs, (("surface offset (rows)", "offset"), ("roughness", "rough"), ("soft energy S", "S"))):
    ax.axhspan(ref[j] - band[j], ref[j] + band[j], color="#cccccc", alpha=0.5)
    ax.set_xscale("log")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("step + 1")
    ax.grid(alpha=0.25, lw=0.5)
axs[2].set_yscale("log")
axs[0].legend(frameon=False, fontsize=7)
fig.suptitle("certificate-free ground: convergence from far starts (32 chains; dashed = multiscale; grey band = equilibrium)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{D}/allair.png", dpi=110)
fig.savefig("images/cmix_allair.png", dpi=110)
