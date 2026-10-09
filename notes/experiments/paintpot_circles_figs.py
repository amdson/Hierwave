"""Figures for notes/paintpot_test.md from images/paintpot_circles.json.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/paintpot_circles_figs.py"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from castlegen.channels.circles import Circles

R = json.load(open("images/paintpot_circles.json"))
CP = json.load(open("images/convpot_circles.json"))
IND = json.load(open("images/induce_circles.json"))
C = Circles(6, 6)
RAW = C.reference(centred=False)
REF = C.reference()


def dc(a):
    a = np.asarray(a, float)
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


# ---------------------------------------------------------------- mid
M = np.array(R["S1"]["mask"])
v = R["S1"]["ho"]["unary+OFF8"]
gh, gv = np.array(v["mid_h"]), np.array(v["mid_v"])
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
fig.subplots_adjust(wspace=0.5)
lim = np.abs(RAW["mid_h"][1:, 1:]).max()
for a, t, title in ((ax[0], RAW["mid_h"], "reference mid_h (pp block, reference gauge)"),
                    (ax[1], gh, "materialised mid_h, held-out fit (x = held out)")):
    im = a.imshow(t[1:, 1:], cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.set_title(title, fontsize=9)
    yy, xx = np.nonzero(M[1:, 1:])
    a.scatter(xx, yy, marker="x", s=14, c="k", lw=0.7)
    a.set_xlabel("v' (east)"); a.set_ylabel("v")
fig.colorbar(im, ax=ax[:2], fraction=0.025)
d = (gh - RAW["mid_h"])[1:, 1:]
im2 = ax[2].imshow(d, cmap="RdBu_r", vmin=-1e-5, vmax=1e-5)
ax[2].set_title(f"materialised - reference (max {np.abs(d).max():.1e})", fontsize=9)
fig.colorbar(im2, ax=ax[2], fraction=0.045)
pp_ = np.zeros_like(M); pp_[1:, 1:] = True
for mask, lab, col in ((pp_ & ~M, "seen", "#2a6fb0"), (pp_ & M, "held out", "#d0532b")):
    ax[3].scatter(np.r_[RAW["mid_h"][mask], RAW["mid_v"][mask]], np.r_[gh[mask], gv[mask]], s=10, c=col, label=lab)
ax[3].plot([-lim, lim], [-lim, lim], c="0.6", lw=1)
ax[3].set_xlabel("reference (mid_h, mid_v)"); ax[3].set_ylabel("materialised"); ax[3].legend(frameon=False)
ax[3].set_title("pp entries, h and v pooled", fontsize=9)
fig.savefig("images/paintpot_circles_mid.png", dpi=110, bbox_inches="tight")

# ---------------------------------------------------------------- top
orc = {k: np.asarray(x) for k, x in IND["oracle"].items()}
tu = orc["top_u"]
olog = {k: dc(-np.log(orc[k] / np.outer(tu, tu))) for k in ("top_h", "top_v")}
cols = [("oracle -log(p/pp)", {"top_h": olog["top_h"], "top_v": olog["top_v"]}),
        ("reference (lam -> inf)", {"top_h": REF["top_h"], "top_v": REF["top_v"]}),
        ("convpot (b0), 2a", R["S2a"]["variants"]["(b0) one-hot convpot, no halo features"]),
        ("2a paint V6 OFF2", R["S2a"]["variants"]["paint slot+halo presence V6 OFF2"]),
        ("2b (b0+halo)", R["S2b"]["variants"]["(b0+halo) one-hot convpot, halo tops as features"]),
        ("2b paint OFF8", R["S2b"]["variants"]["paint slot V2 OFF8"]),
        ("2b paint OFF2", R["S2b"]["variants"]["paint slot V2 OFF2"])]
rows = ["top_h", "top_v", "diag_se", "diag_sw"]
fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.2 * len(cols), 2.2 * len(rows)))
lim = 0.4
for j, (name, d) in enumerate(cols):
    for i, r in enumerate(rows):
        a = ax[i, j]
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(["NW", "NE", "SW", "SE"], fontsize=6); a.set_yticklabels(["NW", "NE", "SW", "SE"], fontsize=6)
        if r not in d:
            a.axis("off")
            continue
        t = dc(d[r])
        a.imshow(t, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for y in range(4):
            for x in range(4):
                a.text(x, y, f"{t[y, x]:+.2f}", ha="center", va="center", fontsize=5.5)
        if i == 0:
            a.set_title(name, fontsize=8)
        if j == 0 or (j == 2 and i >= 2):
            a.set_ylabel(r, fontsize=8)
fig.suptitle("top tables, double-centred (row = this cell, column = neighbour); colour scale +-0.4", fontsize=9)
fig.tight_layout()
fig.savefig("images/paintpot_circles_top.png", dpi=120, bbox_inches="tight")
print("ok")
