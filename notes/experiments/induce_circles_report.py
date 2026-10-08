"""Markdown tables and figures for notes/induce_test.md from
images/induce_circles.json (written by induce_circles.py).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/induce_circles_report.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from castlegen.channels.circles import Circles

R = json.load(open("images/induce_circles.json"))
SP = json.load(open("images/circles_selfplay_eta0.5.json"))["k4_mu0.3"]
SPF = json.load(open("images/circles_selfplay_fwdscale.json"))["k4_mu0.3"]
C = Circles(6, 6)
REF = C.reference()
RAW = C.reference(centred=False)
FIT = ["mid_h", "mid_v", "mid_u", "top_h", "top_v", "top_u"]
MON = ["present", "viol", "conflict", "contact", "slot_viol", "edge_air_top"]
L = []


def dc(a):
    a = np.asarray(a, float)
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def tu(o):
    t = np.asarray(o["top_u"])
    return dc(-np.log(np.asarray(o["top_h"]) / np.outer(t, t)))


# ---- A
A = R["A"]
nobj = np.array(A["nobj"])
L.append("### A. Estimator check (200 windows, 2 x 2 mid, presence 0.5)\n")
L.append("Error = AIS minus exact `F(z_W) - F(r_W)` (nats; `F(r_W)` by AIS is exact here, no `E_rest`); "
         "'se' = the delta-method standard error the estimator reports; 'within 3 se' = fraction of windows "
         "with |error| <= 3 se.  Mean error by number of objects in the window (1 / 2 / 3 / 4).\n")
L.append("| K | M | mean err | sd | rms | mean se | within 3 se | mean err by #objects 1 / 2 / 3 / 4 | ms / window |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in A["rows"]:
    e = np.array(r["err"])
    by = " / ".join(f"{e[nobj == n].mean():+.2f}" if (nobj == n).any() else "-" for n in (1, 2, 3, 4))
    L.append(f"| {r['K']} | {r['M']} | {r['mean']:+.3f} | {r['sd']:.3f} | {r['rms']:.3f} | {r['se']:.3f} | "
             f"{r['within3']:.2f} | {by} | {r['ms']:.1f} |")
L.append(f"\nWindows with 0 objects: {(nobj == 0).sum()} (error exactly 0).  `L` = 30 vs 5 with the same "
         f"seed: max |dF| = {A['L_check']:.1e} (circles has no inf entry).\n")

# ---- B
B = R["B"]
L.append("### B. Mid tables\n")
L.append(f"N = 4000 windows, uniform offsets, presence 0.5, ridge 1e-3, reference gauge (row / column "
         f"'absent' of the pair tables 0).  Residual RMS / target RMS: exact {B['exact']['res']:.4f} / "
         f"{B['exact']['rms']:.2f}; ais (K = {B['ais']['K']}, M = {B['ais']['M']}) {B['ais']['res']:.3f} / "
         f"{B['ais']['rms']:.2f} (per-window AIS error RMS {B['ais']['err_rms']:.3f}, mean se "
         f"{B['ais']['se']:.3f}).  Wall: exact {B['exact']['t']:.1f} s, ais {B['ais']['t']:.0f} s.\n")
c0 = B["exact"]["cmp"]
hdr = ("| tables | mid_h max abs err | mid_h pp slope / corr | mid_v pp slope / corr | mean attract / repel / zero "
       f"(ref {c0['means_ref'][0]:+.2f} / {c0['means_ref'][1]:+.2f} / {c0['means_ref'][2]:+.2f}) | "
       + " | ".join(f"{n} ({c0['hand'][n][1]:+.2f})" for n in "ABCD")
       + f" | mid_u present - absent / spread (F_obj {C.object_cost():.2f} / 0) | mid_u + pres max err |")
L.append(hdr)
L.append("|---" * 11 + "|")


def mrow(name, c):
    h = c["hand"]
    return (f"| {name} | {c['mid_h']['max_err']:.2f} | {c['mid_h']['slope']:.3f} / {c['mid_h']['corr']:.3f} | "
            f"{c['mid_v']['slope']:.3f} / {c['mid_v']['corr']:.3f} | "
            f"{c['means'][0]:+.2f} / {c['means'][1]:+.2f} / {c['means'][2]:+.2f} | "
            + " | ".join(f"{h[n][0]:+.2f}" for n in "ABCD")
            + f" | {c['u_gap']:.2f} / {c['u_spread']:.2f} | {c['u_max_err']:.2f} |")


L.append(mrow("exact F", B["exact"]["cmp"]))
L.append(mrow(f"ais F (K {B['ais']['K']}, M {B['ais']['M']})", B["ais"]["cmp"]))
if "E" in R:
    L.append(mrow("ais F, iterated (E)", R["E"]["cmp_mid"]))
L.append("\nFor comparison (circles_test.md): self-play fwdscale S0 max err 1.36, pp slope / corr 1.17 / 0.91; "
         "eta0.5 S1 2.25, 0.16 / 0.84.\n")

# ---- C
Cc = R["C"]
o = R["oracle"]
olog = tu(o)
L.append("### C. Top tables\n")
L.append("N = 2000 windows of 2 x 2 top cells (uniform corners), reference corner 0 (NW), `ais` over the "
         "window's 4 x 4 mid cells.  'halo': the one-mid-cell ring around them fixed to a sample of the "
         "forward mid kernel under all-NW tops (32 halo samples, `F(r_W)` per halo by AIS at 4x K); "
         "'no halo': the ring left out (free boundary).  Contrast = mean{NE->SW, SE->NW} - "
         "mean{NE->NW, SE->SW, NW->NE, SW->SE} of the double-centred table (top_v through NE <-> SW); "
         f"oracle `-log(p/pp)` contrast {np.mean([olog[1,2], olog[3,0]]) - np.mean([olog[1,0], olog[3,2], olog[0,1], olog[2,3]]):.2f}, "
         f"reference {np.mean([REF['top_h'][1,2], REF['top_h'][3,0]]) - np.mean([REF['top_h'][1,0], REF['top_h'][3,2], REF['top_h'][0,1], REF['top_h'][2,3]]):.2f}.\n")
L.append("| mid tables used | halo | residual / target RMS | mean AIS se | contrast h / v | top_h corr / slope vs oracle | "
         "top_v corr vs oracle | corr vs reference | wall s |")
L.append("|---|---|---|---|---|---|---|---|---|")
for k, v in Cc.items():
    c, i = v["cmp"], v["info"]
    L.append(f"| {k.split('_')[0]} | {'no' if 'nohalo' in k else 'yes'} | {i['res']:.3f} / {i['rms']:.3f} | "
             f"{i['se']:.3f} | {c['contrast_h']:.2f} / {c['contrast_v']:.2f} | {c['corr_o']:.2f} / {c['slope_o']:.2f} | "
             f"{c['corr_o_v']:.2f} | {c['corr_r']:.2f} | {i['t']:.0f} |")
if "E" in R:
    c, i = R["E"]["cmp_top"], R["E"]["info"]
    L.append(f"| ais, iterated (E) | yes | {i['res']:.3f} / {i['rms']:.3f} | {i['se']:.3f} | "
             f"{c['contrast_h']:.2f} / {c['contrast_v']:.2f} | {c['corr_o']:.2f} / {c['slope_o']:.2f} | "
             f"{c['corr_o_v']:.2f} | {c['corr_r']:.2f} | {i['t']:.0f} |")
L.append("\nDouble-centred top_h (rows T = NW NE SW SE):\n")
L.append("    oracle -log(p/pp): " + "; ".join(" ".join(f"{x:+.2f}" for x in r) for r in olog))
L.append("    reference:         " + "; ".join(" ".join(f"{x:+.2f}" for x in r) for r in REF["top_h"]))
for k, v in Cc.items():
    L.append(f"    {k:18s} " + "; ".join(" ".join(f"{x:+.2f}" for x in r) for r in v["cmp"]["top_h_dc"]))
if "E" in R:
    L.append(f"    {'iterated (E)':18s} " + "; ".join(" ".join(f"{x:+.2f}" for x in r) for r in R["E"]["cmp_top"]["top_h_dc"]))
L.append("\nSelf-play for comparison (circles_test.md): eta0.5 S0 contrast 0.34 / 0.27, corr / slope 0.99 / 0.85; "
         "fwdscale S1 0.30 / 0.24, 0.88 / 0.73.\n")

# ---- D
D = R["D"]
L.append("### D. End to end (no moment matching)\n")
L.append("L1 to the oracle moments on the fitted features (64 forward runs; oracle 50 + 400 sweeps, symmetrised; "
         "same seeds and conventions as circles_test.md):\n")
L.append("| | " + " | ".join(FIT) + " |")
L.append("|---" * 7 + "|")
L.append("| untrained | " + " | ".join(f"{D['untrained']['L1'][k]:.3f}" for k in FIT) + " |")
rows = [(k, v) for k, v in D.items() if k != "untrained"]
if "E" in R:
    rows.append(("iterated (E): ais-mid + top, forward windows", R["E"]))
for k, v in rows:
    L.append(f"| {k} | " + " | ".join(f"{v['L1'][x]:.3f}" for x in FIT) + " |")
for k, v in rows:
    L.append(f"| eval noise {k} | " + " | ".join(f"{v['noise'][x]:.3f}" for x in FIT) + " |")
for nm, src, sc in (("self-play eta0.5 S0", SP, "S0"), ("self-play eta0.5 S1", SP, "S1"),
                    ("self-play eta0.5 S2", SP, "S2"), ("self-play fwdscale S0", SPF, "S0")):
    L.append(f"| {nm} | " + " | ".join(f"{src['schemes'][sc]['L1'][x]:.3f}" for x in FIT) + " |")
L.append("\nMonitors (edge_air_top held out):\n")
L.append("| | " + " | ".join(MON) + " |")
L.append("|---" * 7 + "|")
L.append("| oracle | " + " | ".join(f"{o[k]:.4f}" for k in MON) + " |")
L.append("| untrained | " + " | ".join(f"{D['untrained']['final'][k]:.4f}" for k in MON) + " |")
for k, v in rows:
    L.append(f"| {k} | " + " | ".join(f"{v['final'][x]:.4f}" for x in MON) + " |")
for nm, src, sc in (("self-play eta0.5 S0", SP, "S0"), ("self-play eta0.5 S1", SP, "S1"),
                    ("self-play eta0.5 S2", SP, "S2"), ("self-play fwdscale S0", SPF, "S0")):
    L.append(f"| {nm} | " + " | ".join(f"{src['schemes'][sc]['final'][x]:.4f}" for x in MON) + " |")

# ---- E
if "E" in R:
    E = R["E"]
    L.append("\n### E. Iteration\n")
    L.append(f"Windows: mid {E['n_fwd_mid']} cut from forward samples of the D 'ais-mid + top(ais-mid)' model "
             f"(presence {E['pres_fwd']:.3f}) + {E.get('n_mid', 4000) - E['n_fwd_mid']} uniform; top {E['n_fwd_top']} forward-cut + "
             f"{E['info']['n'] - E['n_fwd_top']} uniform; halos regenerated with the refit mid tables.  Mid residual / "
             f"target RMS {E['res_mid']:.3f} / {E['rms_mid']:.2f}.  Table movement against D's tables "
             "(double-centred pairs, centred unaries):\n")
    L.append("| table | max abs change | rms change | rms of the D table |")
    L.append("|---|---|---|---|")
    for k in FIT:
        m = E["move"][k]
        L.append(f"| {k} | {m['max']:.3f} | {m['rms']:.3f} | {m['scale']:.3f} |")
open("images/induce_circles.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---- figures
ea = A["rows"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for M, col in ((4, "#2a6fbb"), (16, "#c8553d")):
    rr = [r for r in ea if r["M"] == M]
    K = [r["K"] for r in rr]
    axs[0].errorbar(K, [r["mean"] for r in rr], yerr=[r["sd"] for r in rr], color=col, lw=2, marker="o", ms=6,
                    capsize=3, label=f"M = {M}: mean error +- sd")
    axs[0].plot(K, [r["se"] for r in rr], color=col, lw=1, ls="--", label=f"M = {M}: mean reported se")
axs[0].axhline(0, color="0.5", lw=0.8)
axs[0].set_xscale("log", base=2); axs[0].set_xlabel("K (beta steps)"); axs[0].set_ylabel("AIS - exact, F (nats)")
axs[0].set_title("A: estimator error vs budget (200 windows)", fontsize=10); axs[0].legend(fontsize=8, frameon=False)
r = [r for r in ea if r["K"] == B["ais"]["K"] and r["M"] == B["ais"]["M"]][0]
jit = (np.random.default_rng(0).random(len(nobj)) - 0.5) * 0.3
axs[1].scatter(nobj + jit, r["err"], s=10, color="#2a6fbb")
axs[1].axhline(0, color="0.5", lw=0.8)
axs[1].set_xlabel("objects in the window"); axs[1].set_ylabel("AIS - exact (nats)")
axs[1].set_title(f"A: error per window at K = {r['K']}, M = {r['M']} (the B budget)", fontsize=10)
fig.savefig("images/induce_circles_estimator.png", dpi=110); plt.close(fig)

cols = [("reference", dc(RAW["mid_h"]), REF["top_h"]),
        ("oracle -log p/pp", None, olog),
        ("exact-F fit", dc(B["exact"]["theta"]["mid_h"]), dc(Cc["exact"]["theta"]["top_h"])),
        ("ais-F fit", dc(B["ais"]["theta"]["mid_h"]), dc(Cc["ais"]["theta"]["top_h"]))]
if "E" in R:
    cols.append(("iterated (E)", dc(R["E"]["theta"]["mid_h"]), dc(R["E"]["theta"]["top_h"])))
fig, axs = plt.subplots(2, len(cols), figsize=(2.9 * len(cols), 6.0), constrained_layout=True)
vm_mid = np.abs(dc(RAW["mid_h"])).max()
for c, (title, mh, tt) in enumerate(cols):
    ax = axs[0, c]
    if mh is None:
        ax.axis("off")
    else:
        ax.imshow(mh, cmap="RdBu_r", vmin=-vm_mid, vmax=vm_mid)
        ax.set_title(f"{title} mid_h", fontsize=9)
        ax.set_xticks([0, 4, 8, 12, 16]); ax.set_yticks([0, 4, 8, 12, 16]); ax.tick_params(labelsize=7)
    ax = axs[1, c]
    vm = np.abs(tt).max() + 1e-9
    ax.imshow(tt, cmap="RdBu_r", vmin=-vm, vmax=vm)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{tt[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["NW", "NE", "SW", "SE"], fontsize=7); ax.set_yticklabels(["NW", "NE", "SW", "SE"], fontsize=7)
    ax.set_title(f"{title} top_h (+-{vm:.2f})", fontsize=9)
fig.suptitle(f"circles: double-centred tables; mid_h on one scale (+-{vm_mid:.2f}), top_h each its own; "
             "rows = left cell; mid value 0 = absent", fontsize=10)
fig.savefig("images/induce_circles_tables.png", dpi=110); plt.close(fig)
print("wrote images/induce_circles.md, images/induce_circles_estimator.png, images/induce_circles_tables.png")
