"""Figures for notes/experiments/convpot_circles.py (reads images/convpot_circles.json).

    PYTHONPATH=. python notes/experiments/convpot_circles_figs.py"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from castlegen.channels.circles import Circles

R = json.load(open("images/convpot_circles.json"))
IND = json.load(open("images/induce_circles.json"))
C = Circles(6, 6)
REF = C.reference()
RAW = C.reference(centred=False)


def dc(a):
    a = np.asarray(a, float)
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def olog(orc, k):
    tu = np.asarray(orc["top_u"])
    return dc(-np.log(np.asarray(orc[k]) / np.outer(tu, tu)))


# ------------------------------------------------------------------ top
V = R["S1"]["variants"]
orc = IND["oracle"]
cols = [("reference", dict(top_h=REF["top_h"], top_v=REF["top_v"])),
        ("oracle -log(p/pp)", dict(top_h=olog(orc, "top_h"), top_v=olog(orc, "top_v"))),
        ("(a) tab h+v", V["a3"]), ("(b0) one-hot", V["b03"]), ("(b) k16", V["b3"]), ("(c) m32", V["c3"]),
        ("(b0) no halo", V["b03n"]), ("(c) m32 no halo", V["c3n"])]
rows = [("top_h (east)", "top_h"), ("top_v (south)", "top_v"), ("diag SE (1,1)", "diag_se"), ("diag SW (1,-1)", "diag_sw")]
fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.1 * len(cols), 2.2 * len(rows)))
lab = ["NW", "NE", "SW", "SE"]
for j, (cn, d) in enumerate(cols):
    for i, (rn, k) in enumerate(rows):
        a = ax[i, j]
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(lab, fontsize=6); a.set_yticklabels(lab, fontsize=6)
        if k not in d:
            a.axis("off")
            continue
        t = dc(d[k])
        a.imshow(t, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        for y in range(4):
            for x in range(4):
                a.text(x, y, f"{t[y, x]:+.2f}", ha="center", va="center", fontsize=5.5)
        if i == 0:
            a.set_title(cn, fontsize=8)
        if j == 0:
            a.set_ylabel(rn, fontsize=8)
fig.suptitle("top tables, double-centred (row = this cell's corner, column = the neighbour's); 3 x 3 windows", fontsize=9)
fig.tight_layout()
fig.savefig("images/convpot_circles_top.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ mid
S2 = R["S2"]
M = np.array(S2["mask"])
names = ["b0", "b", "c", "b-patch", "c-patch"]
fig, ax = plt.subplots(2, 1 + len(names), figsize=(3.0 * (1 + len(names)), 6.2))
hy, hx = np.nonzero(M[1:, 1:])
vm = np.abs(RAW["mid_h"]).max()
for j, n in enumerate(["reference"] + names):
    t = RAW["mid_h"] if n == "reference" else np.array(S2["variants"][n]["g_h"])
    a = ax[0, j]
    a.imshow(t[1:, 1:], cmap="RdBu_r", vmin=-vm, vmax=vm)
    a.scatter(hx, hy, s=4, c="k", marker="x", linewidths=0.5)
    a.set_title(n if n == "reference" else f"{n} (held-out fit)", fontsize=8)
    a.set_xticks([]); a.set_yticks([])
    b = ax[1, j]
    if n == "reference":
        b.axis("off")
        b.text(0.0, 0.5, "mid_h present-present block,\nreference gauge (absent = 0);\nx = held-out (v, v')\n\n"
               "below: fitted vs reference,\nseen (grey) / held out (red)", fontsize=8, va="center")
        continue
    pp = np.zeros_like(M); pp[1:, 1:] = True
    for mask, c in ((pp & ~M, "0.6"), (pp & M, "tab:red")):
        b.scatter(RAW["mid_h"][mask], t[mask], s=5, c=c)
    b.plot([-vm, vm], [-vm, vm], "k-", lw=0.5)
    sc = S2["variants"][n]["score"]
    b.set_title(f"held corr {sc['held']['corr']:.2f} slope {sc['held']['slope']:.2f}", fontsize=8)
    b.set_xlabel("reference", fontsize=7); b.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig("images/convpot_circles_mid.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------- curves
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, c in R["curves_top"].items():
    if k.endswith("_all"):
        ax[0].plot(c, label=k[:-4], lw=1)
ax[0].set_yscale("log"); ax[0].set_title("top Adam (init: the m = 0 ls solution), all windows", fontsize=9)
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss (mean sq. + ridge)"); ax[0].legend(fontsize=7)
for k, c in R["curves_mid"].items():
    ax[1].plot(c, label=k, lw=1)
ax[1].set_yscale("log"); ax[1].set_title("mid Adam (init: the m = 0 ls solution)", fontsize=9)
ax[1].set_xlabel("step"); ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("images/convpot_circles_curves.png", dpi=130)
plt.close(fig)
print("ok")
