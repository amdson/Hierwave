"""Analysis of a circles self-play run (notes/circles_test.md, Results):
mid_h entries by offset pair vs the exact reference, mid_u gap, top_h vs the
oracle's pair pattern and the lam -> inf reference, the top contrast.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/circles_analyse.py images/circles_selfplay_eta0.5.json
"""
import sys, json
import numpy as np
from castlegen.channels.circles import Circles

R = json.load(open(sys.argv[1]))["k4_mu0.3"]
C = Circles(6, 6)
ref = C.reference()
raw = C.reference(centred=False)
D = C.D


def dc(a):
    a = np.asarray(a, float)
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def val(cy, cx):
    return 1 + (cy - 2) * 4 + (cx - 2)


def name(v):
    return "-" if v == 0 else f"({C.OY[v]},{C.OX[v]})"


print(f"F_obj {C.object_cost():.2f}  b {C.b:.2f}  log16 {np.log(16):.2f}")
# mid_h: present-present block (rows/cols 1..16) is where the physics is; also full table
rh = raw["mid_h"]
pp = rh[1:, 1:]
order = np.argsort(pp.ravel())
print("most attractive raw ref pairs (left, right):",
      [(name(1 + k // 16), name(1 + k % 16), round(pp.ravel()[k], 2)) for k in order[:4]])
print("most repulsive raw ref pairs:",
      [(name(1 + k // 16), name(1 + k % 16), round(pp.ravel()[k], 2)) for k in order[::-1][:4]])
print("fraction of pp entries exactly 0:", float((np.abs(pp) < 1e-9).mean()))

pairs = [(val(3, 5), val(3, 2)), (val(3, 5), val(3, 3)), (val(2, 5), val(2, 2)), (val(3, 4), val(3, 2)),
         (val(3, 2), val(3, 5))]
attr = (pp < -1e-9)
rep = (pp > 1e-9)
zero = ~attr & ~rep

oracle = R["oracle"]
th_o = oracle["top_h"]; tu = np.asarray(oracle["top_u"])
th_o = np.asarray(th_o)
exc = th_o / np.outer(tu, tu) - 1
olog = dc(-np.log(th_o / np.outer(tu, tu)))
print("\noracle top_h relative excess (rows left T, cols right T'):\n", np.round(exc, 3))
print("oracle -log(p/pp) double-centred:\n", np.round(olog, 2))
print("reference top_h (raw):\n", np.round(raw["top_h"], 2))
print("reference top_h dc:\n", np.round(ref["top_h"], 2))
ENH = [(1, 0), (3, 2), (0, 1), (2, 3)]
DEP = [(1, 2), (3, 0)]


def contrast(t):
    t = dc(t)
    return np.mean([t[a] for a in DEP]) - np.mean([t[a] for a in ENH])


def contrast_v(t):        # vertical analogue: the transpose map (NE<->SW) of the horizontal pattern
    tr = np.array([0, 2, 1, 3])
    return contrast(np.asarray(t)[np.ix_(tr, tr)])


print(f"contrast (mean depleted - mean enhanced, energy units): oracle -log ratio {contrast(olog):.3f}, "
      f"reference {contrast(ref['top_h']):.3f}")
olog_v = dc(-np.log(np.asarray(oracle['top_v']) / np.outer(tu, tu)))
print(f"  vertical: oracle {contrast_v(olog_v):.3f} reference {contrast_v(ref['top_v']):.3f}")

rows = []
s0c = None
for sc, S in R["schemes"].items():
    th = {k: np.asarray(v) for k, v in S["theta"].items()}
    mh = dc(th["mid_h"]); rr = ref["mid_h"]
    a, b = mh[1:, 1:].ravel(), dc(rh)[1:, 1:].ravel()
    mu = th["mid_u"]
    gap = mu[1:].mean() - mu[0]
    th_t = dc(th["top_h"])
    c, cv = contrast(th["top_h"]), contrast_v(th["top_v"])
    if sc == "S0":
        s0c = (c, cv)
    corr_o = np.corrcoef(th_t.ravel(), olog.ravel())[0, 1]
    corr_r = np.corrcoef(th_t.ravel(), ref["top_h"].ravel())[0, 1]
    slope_o = float(th_t.ravel() @ olog.ravel() / (olog.ravel() @ olog.ravel()))
    m_attr = (mh[1:, 1:][attr].mean(), dc(rh)[1:, 1:][attr].mean())
    m_rep = (mh[1:, 1:][rep].mean(), dc(rh)[1:, 1:][rep].mean())
    m_zero = (mh[1:, 1:][zero].mean(), dc(rh)[1:, 1:][zero].mean())
    ent = "; ".join(f"{name(p)}|{name(q)} {mh[p, q]:+.2f} ({dc(rh)[p, q]:+.2f})" for p, q in pairs)
    print(f"\n== {sc}: mid_h max err {np.abs(mh - rr).max():.2f}  pp-block max err {np.abs(a - b).max():.2f} "
          f"slope {a @ b / (b @ b):.2f} corr {np.corrcoef(a, b)[0, 1]:.2f}")
    print(f"   means over pp entries learned (ref): attract {m_attr[0]:+.2f} ({m_attr[1]:+.2f}) "
          f"repel {m_rep[0]:+.2f} ({m_rep[1]:+.2f}) zero {m_zero[0]:+.2f} ({m_zero[1]:+.2f})")
    print("   entries:", ent)
    print(f"   mid_u present-mean minus absent {gap:.2f}  spread over present offsets {mu[1:].std():.2f}  "
          f"(interior/edge offsets: centre-ish {mu[[val(3,3),val(3,4),val(4,3),val(4,4)]].mean() - mu[0]:.2f})")
    print("   top_h dc:", np.round(th_t, 2).tolist())
    print(f"   top_h vs oracle -log ratio: corr {corr_o:.2f} slope {slope_o:.2f}; vs ref corr {corr_r:.2f}")
    print(f"   contrast h {c:.3f} v {cv:.3f}  ratio to S0 {(c + cv) / (s0c[0] + s0c[1]) if s0c else float('nan'):.2f}")

# ---- legible tables figure: per-panel colour scale (stated in the title), oracle top_h column
if len(sys.argv) > 2:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scs = list(R["schemes"])
    cols = [("reference", dc(rh), ref["top_h"])] + \
           [("oracle -log p/pp", None, olog)] + \
           [(sc, dc(np.asarray(R["schemes"][sc]["theta"]["mid_h"])), dc(R["schemes"][sc]["theta"]["top_h"])) for sc in scs]
    fig, axs = plt.subplots(2, len(cols), figsize=(2.9 * len(cols), 6.0), constrained_layout=True)
    for c, (title, mh, tt) in enumerate(cols):
        ax = axs[0, c]
        if mh is None:
            ax.axis("off")
        else:
            vm = np.abs(mh).max() + 1e-9
            ax.imshow(mh, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_title(f"{title} mid_h\n(scale +-{vm:.2f})", fontsize=9)
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
    fig.suptitle("circles kappa 4: double-centred tables, each panel its own colour scale; rows = left cell; "
                 "mid value 0 = absent, 1 + 4 (cy-2) + (cx-2)")
    fig.savefig(sys.argv[2], dpi=110); plt.close(fig)
    print("wrote", sys.argv[2])
