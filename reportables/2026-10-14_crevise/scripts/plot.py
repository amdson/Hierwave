"""Figures for the crevise report: dimers training curves (orientation fractions, raw violations, trajectory
entropy / entropy) for crevise and the cfast control; ground-patch P(dirt) maps against the MCMC reference."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "reportables/2026-10-14_crevise"
C = {"h": "#2a78d6", "v": "#eb6834", "d": "#1baf7a", "viol": "#eda100", "ent": "#e87ba4"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6})


def dimers():
    runs = [(n, json.load(open(f"{D}/runs/dimers_{n}.json"))) for n in
            ("crevise", "crevise_guided", "crevise_guided_fixed", "cfast") if os.path.exists(f"{D}/runs/dimers_{n}.json")]
    fig, ax = plt.subplots(3, len(runs), figsize=(3.6 * len(runs), 6.4), sharex="col", squeeze=False)
    for j, (n, r) in enumerate(runs):
        rows = r["rows"]
        u = np.array([x["update"] for x in rows])
        hvd = np.array([x["raw"] for x in rows])
        for i, (lab, col) in enumerate((("horizontal", C["h"]), ("vertical", C["v"]), ("diagonal", C["d"]))):
            ax[0, j].plot(u, hvd[:, i], color=col, lw=1.6, label=lab)
        lab = {"crevise": "crevise, unguided (biased term)", "crevise_guided": "guided (biased term)",
               "crevise_guided_fixed": "guided, fixed estimator", "cfast": "cfast control"}[n]
        ax[0, j].set(ylim=(-0.02, 1.02), title=f"dimers, {lab}\n({r['final']['update']} updates)")
        ax[0, j].legend(frameon=False, loc="upper right")
        ax[1, j].plot(u, np.maximum([x["raw_viol"] for x in rows], 1e-6), color=C["viol"], lw=1.6)
        ax[1, j].axhline(2e-4, color="#777", lw=0.8, ls="--")
        ax[1, j].set(yscale="log", ylim=(5e-6, 1e-1))
        ent = [(-x["logq_traj"] if x.get("logq_traj") is not None else x["ent"]) for x in rows]
        ax[2, j].plot(u, ent, color=C["ent"], lw=1.6)
        ax[2, j].set(xscale="symlog", xlabel="update")
        if j == 0:
            ax[0, j].set_ylabel("orientation fraction (raw)")
            ax[1, j].set_ylabel("raw violated fraction")
            ax[2, j].set_ylabel("entropy / cell (nats)")
        ax[2, j].text(0.02, 0.9, "H(trajectory) = -log Q(traj) / cell" if n.startswith("crevise") else "H = -log q(x) / cell",
                      transform=ax[2, j].transAxes, color="#555")
    fig.tight_layout()
    fig.savefig(f"{D}/images/dimers_curves.png", dpi=130)
    plt.close(fig)


def ground():
    ref = json.load(open(f"{D}/runs/ground_patch_mcmc.json"))
    runs = [(n, json.load(open(f"{D}/runs/ground_patch_{n}.json"))) for n in ("crevise", "crevise_guided", "cfast")
            if os.path.exists(f"{D}/runs/ground_patch_{n}.json")]
    panels = [("MCMC reference (weighted)", np.array(ref["p_dirt"]))]
    for n, r in runs:
        panels.append((f"{n} final", np.array(r["final"]["p_dirt"])))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.4))
    surf = np.array([3, 3, 3, 3, 2, 2, 2, 2] * 2)
    for a, (t, pd) in zip(np.atleast_1d(ax), panels):
        im = a.imshow(pd, cmap="copper_r", vmin=0, vmax=1, interpolation="nearest")
        a.plot(np.arange(17) - 0.5, np.r_[surf, surf[-1]] - 0.5, drawstyle="steps-post", color="c", lw=1)
        a.set(title=f"{t}\ndirt {pd.mean():.3f}", xticks=[], yticks=[])
    fig.colorbar(im, ax=ax, fraction=0.02, label="P(dirt)")
    fig.savefig(f"{D}/images/ground_patch_pdirt.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dimers()
    if os.path.exists(f"{D}/runs/ground_patch_mcmc.json"):
        ground()
