"""Figures: chain energy per cycle (long chains and in-loop), loop curves per arm."""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_sweep2a"
c = json.load(open(f"{D}/runs/curve/curve.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, name, ttl in ((ax[0], "probe", "probe-lift contexts (jitter 0-3)"), (ax[1], "own_untrained", "own draws of the init model")):
    x = [r["cycle"] for r in c[name]]; y = np.array([r["soft"] for r in c[name]]); se = np.array([r["soft_se"] for r in c[name]])
    a.errorbar(x, y, yerr=se, marker="o", ms=3, label="total soft / cell")
    if name == "probe":
        a.axhline(0.014, ls="--", c="gray", lw=1); a.text(20, 0.02, "probe j0 (0.014)", fontsize=8)
        a.axhline(0.128, ls=":", c="C3", lw=1); a.text(20, 0.135, "kernel-a 2-cycle states (0.128)", fontsize=8, color="C3")
    a.set_xlabel("exact block-Gibbs cycle [(2,0),(1,1)]"); a.set_ylabel("soft energy per cell"); a.set_title(ttl, fontsize=9)
plt.tight_layout(); plt.savefig(f"{D}/images/energy_per_cycle.png", dpi=110)
arms = [a for a in sys.argv[1:]]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for arm in arms:
    L = [json.loads(l) for l in open(f"{D}/runs/{arm}/loop.jsonl")]
    t = [r["t"] for r in L]
    ax[0].plot(t, [r["loss_per_cell"] for r in L], label=arm)
    ax[1].plot(t, [r["probe_soft_curve"][-1] for r in L], label=f"{arm} refined probe")
    o = [(r["t"], r["own_soft"], r["own_soft_curve"][-1]) for r in L if "own_soft" in r]
    if o:
        o = np.array(o)
        ax[2].plot(o[:, 0], o[:, 1], label=f"{arm} own draws"); ax[2].plot(o[:, 0], o[:, 2], label=f"{arm} refined own")
ax[0].set_title("ML loss per cell (-log q, levels 0-2)", fontsize=9); ax[1].set_title("soft/cell of refined probe contexts", fontsize=9)
ax[2].set_title("soft/cell, own draws before / after refinement", fontsize=9); ax[2].set_yscale("log")
for a in ax: a.set_xlabel("train s"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{D}/images/loop_curves.png", dpi=110)
