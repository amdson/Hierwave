"""Figure: when the entropy goes (training-batch stats vs time) for the diagnostic runs.
Run from the repo root."""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(pat):
    f = sorted(glob.glob(f"results/runs/*_{pat}/train_s0.jsonl"))[-1]
    return [d for d in map(json.loads, open(f)) if any(k.startswith("st_") for k in d)]


fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for i, (pat, title) in enumerate([("ctemp_dimers_diag_cpasscfg", "dimers, cpass schedule (λ0 = 10)"),
                                  ("ctemp_dimers_lamramp", "dimers, λ ramp from 1")]):
    r = load(pat)
    t = [d["t"] for d in r]
    a = ax[i]
    for k, c in (("horizontal", "C3"), ("vertical", "C0"), ("diagonal", "C2")):
        a.plot(t, [d["st_" + k] for d in r], color=c, label=k)
    b = a.twinx()
    b.semilogy(t, [d["raw_viol"][0] for d in r], "k--", lw=1, label="raw viol")
    b.set_ylim(1e-4, 1e-1)
    a.set(title=title, xlabel="training s", ylabel="orientation fraction (training draws)")
    a.legend(loc="center left", fontsize=8)
    b.legend(loc="upper right", fontsize=8)
r = load("ctemp_toy_modes_diag_cpasscfg")
t = [d["t"] for d in r]
ax[2].plot(t, [d["st_mode_other"] for d in r], label="mode_other (exact 0.256)")
ax[2].plot(t, [d["tau"] - 1 for d in r], label="τ − 1")
ax[2].axhline(0.256, color="grey", lw=0.8)
ax[2].set(title="toy_modes, cpass schedule", xlabel="training s")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig("reportables/2026-10-13_ctemp/diag_collapse.png", dpi=110)
