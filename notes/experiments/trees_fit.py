"""Stage A of notes/trees_test.md: tree slots over a sampled chain terrain.
The bootstrap fit (train.fit) of the tree level's potential (tabular over
the hclass view: tree_u, tree_p1, tree_p2), targets = the collapsed
conditional over the surf window by
  hook   the transfer matrix (exact)
  ais    AISTargets over the surf sampler (pins in p_0, smooth annealed)
x K in {0, 3}; end to end: RUNS forward runs with the fitted tables
against the oracle's moments (two independent oracle chains give the eval
noise).  The learned unary and pair tables against the oracle's -log p(h)
and -PMI(h, h').

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/trees_fit.py

Env: N (24), ITERS (4), NCTX (8), NPAIR (2), RUNS (64), BURN (100), SWEEPS
(2000), SEED (0), REACH (1), B (1.0), C (1.0), KAPPA (0.1).  Outputs
images/trees_stageA.json, images/trees_tables.png, images/trees_{oracle,<setting>}.png."""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from castlegen.channels import train
from castlegen.channels.aistargets import AISTargets
from castlegen.channels.trees import LEARNED, Bench, Forward, Oracle, Trees

E = os.environ.get
N, ITERS, NCTX, NPAIR = int(E("N", 24)), int(E("ITERS", 4)), int(E("NCTX", 8)), int(E("NPAIR", 2))
RUNS, BURN, SWEEPS, SEED, REACH = int(E("RUNS", 64)), int(E("BURN", 100)), int(E("SWEEPS", 2000)), int(E("SEED", 0)), int(E("REACH", 1))
B_, C_, KAPPA = float(E("B", 1.0)), float(E("C", 1.0)), float(E("KAPPA", 0.1))
SETTINGS = [s for s in E("SETTINGS", "hook_K0,hook_K3,ais_K0").split(",") if s]
os.makedirs("images", exist_ok=True)
T = Trees(N, margin=2, b=B_, c=C_, kappa=KAPPA, reach=REACH)
OUT = dict(config=dict(N=N, ITERS=ITERS, NCTX=NCTX, NPAIR=NPAIR, RUNS=RUNS, BURN=BURN, SWEEPS=SWEEPS, SEED=SEED,
                       REACH=REACH, b=B_, c=C_, kappa=KAPPA, BT=T.BT, D=T.D, S=(30, 30), AIS=dict(K=32, M=16, L=30.0)))
NAMES = None


def save(img, name):
    Image.fromarray(img).save(f"images/{name}.png")


def eval_forward(tables, runs, seed, S=(30, 30)):
    fw = Forward(T, tables=tables, seed=seed)
    acc = []
    for _ in range(runs):
        acc.append(fw.run(*S, fresh=True))
    keys = [k for k in acc[0] if k != "h_hist"]
    mean = {k: float(np.mean([a[k] for a in acc])) for k in keys}
    se = {k: float(np.std([a[k] for a in acc]) / np.sqrt(runs)) for k in keys}
    mean["h_hist"] = np.mean([a["h_hist"] for a in acc], 0).tolist()
    return mean, se, fw


def dist(a, b, keys):
    return float(np.mean([abs(a[k] - b[k]) for k in keys]))


# ------------------------------------------------------------ oracle
t0 = time.time()
orcA = Oracle(T, seed=SEED)
mA = orcA.moments(BURN, SWEEPS)
orcB = Oracle(T, seed=SEED + 100)
mB = orcB.moments(BURN, SWEEPS)
KEYS = [k for k in mA if k != "h_hist"]
noise = dist(mA, mB, KEYS)
T.paint_tile(orcA.tree, orcA.surf, orcA.tile)
save(T.render(orcA.tile), "trees_oracle")
u_cnt, pm_cnt = orcA.height_tables(0, SWEEPS)
OUT["oracle"] = dict(A=mA, B=mB, noise=noise, seconds=time.time() - t0)
print(f"oracle: {time.time() - t0:.0f}s  present {mA['present']:.3f}/{mB['present']:.3f}  "
      f"p1_eq {mA['p1_eq']:.3f}/{mB['p1_eq']:.3f}  rough {mA['terr_rough']:.3f}/{mB['terr_rough']:.3f}  "
      f"noise {noise:.4f}", flush=True)

# ---------------------------------------------------------- baseline
t0 = time.time()
m0, se0, fw0 = eval_forward(T.theta0(), RUNS, SEED + 1)
save(T.render(fw0.tile), "trees_theta0")
OUT["theta0"] = dict(mean=m0, se=se0, L1=dist(m0, mA, KEYS), seconds=time.time() - t0)
print(f"theta0: L1 {OUT['theta0']['L1']:.4f} (noise {noise:.4f})  present {m0['present']:.3f}  "
      f"p1_eq {m0['p1_eq']:.3f}  rough {m0['terr_rough']:.3f}", flush=True)


# --------------------------------------------------------------- fits
def run_setting(name):
    kind, K = name.split("_K")
    K = int(K)
    B = Bench(T, seed=SEED + 10)
    if kind == "hook":
        tg = AISTargets("surf", B.region, B.designed_e, B.repaint, hook=B.hook)
    else:
        tg = AISTargets("surf", B.region, B.designed_e, B.repaint, K=32, M=16, L=30.0, seed=SEED + 20)
    t0 = time.time()
    theta, rep = train.fit(B.contexts, "tree", tg, B.feats, B.designed_e, iters=ITERS, n_contexts=NCTX, K=K,
                           after_change=B.move, repaint=B.repaint, n_pairs=NPAIR, rng=SEED, verbose=True)
    tfit = time.time() - t0
    tabs = B.tables_of(theta)
    m, se, fw = eval_forward(tabs, RUNS, SEED + 2)
    save(T.render(fw.tile), f"trees_{name}")
    res = dict(mean=m, se=se, L1=dist(m, mA, KEYS), fit_seconds=tfit, report={k: v for k, v in rep.items()},
               tables={n: np.asarray(v).tolist() for n, v in tabs.items()},
               ais=dict(n_ais=tg.n_ais, t_ais=tg.t_ais, n_hook=tg.n_hook))
    print(f"{name}: fit {tfit:.0f}s  KL held {rep['kl_held'][-1]:.4f}  viol fit {rep['viol_fit'][-1]:.2e}  "
          f"L1 {res['L1']:.4f} (noise {noise:.4f})  present {m['present']:.3f}  p1_eq {m['p1_eq']:.3f}  "
          f"rough {m['terr_rough']:.3f}", flush=True)
    return res, tabs


RES = {}
for name in SETTINGS:
    RES[name], tabs = run_setting(name)
    OUT[name] = RES[name]
    with open("images/trees_stageA.json", "w") as f:
        json.dump(OUT, f, indent=1)

# ------------------------------------------------------------- tables
# oracle: -log p(h) (per present tree), -PMI at distance 1, 2; fitted: tree_u over classes 1.. (height
# h = BT - 1 - r), pair tables double-centred over the present block.
h_of_class = T.BT - 1 - (np.arange(1, T.NC) - 1) % T.BT             # class -> height (NB = 1: one chunk column)
with np.errstate(divide="ignore"):
    u_orc = -np.log(u_cnt / u_cnt.sum())
    u_orc -= u_orc[np.isfinite(u_orc)].mean()
fig, ax = plt.subplots(1, 1 + 2 * len(RES), figsize=(4 * (1 + 2 * len(RES)), 3.6))
ax = np.atleast_1d(ax)
ax[0].plot(np.arange(T.BT), u_orc, "k.-", label="oracle -log p(h)")
for name, r in RES.items():
    tu = np.asarray(r["tables"]["tree_u"])[1:]
    tu = tu - tu.mean()
    ax[0].plot(h_of_class, tu, ".-", label=name)
ax[0].set_xlabel("tree height h")
ax[0].set_title("induced unary (centred)")
ax[0].legend(fontsize=7)
for i, (name, r) in enumerate(RES.items()):
    for k, nm in ((1, "tree_p1"), (2, "tree_p2")):
        a = ax[1 + 2 * i + (k - 1)]
        P = pm_cnt[k] + 0.5
        P /= P.sum()
        pmi = -np.log(P / (P.sum(1, keepdims=True) * P.sum(0, keepdims=True)))
        tp = np.asarray(r["tables"][nm])[1:, 1:]
        tp = train.double_centre(tp)
        # reorder fitted (class order = r ascending = h descending) to h ascending
        order = np.argsort(h_of_class)
        tp = tp[np.ix_(order, order)]
        pmi = train.double_centre(pmi)
        a.plot(pmi.ravel(), tp.ravel(), ".", ms=3)
        lim = [min(pmi.min(), tp.min()), max(pmi.max(), tp.max())]
        a.plot(lim, lim, "k-", lw=0.5)
        cc = np.corrcoef(pmi.ravel(), tp.ravel())[0, 1]
        a.set_title(f"{name} {nm}: fit vs -PMI (corr {cc:.3f})", fontsize=8)
        a.set_xlabel("oracle -PMI (dc)")
        a.set_ylabel("fitted (dc)")
        RES[name][f"corr_{nm}"] = float(cc)
        OUT[name][f"corr_{nm}"] = float(cc)
plt.tight_layout()
plt.savefig("images/trees_tables.png", dpi=110)
with open("images/trees_stageA.json", "w") as f:
    json.dump(OUT, f, indent=1)

print("\n| stat | oracle A | oracle B | theta0 |" + "".join(f" {n} |" for n in RES))
print("|---|---|---|---|" + "---|" * len(RES))
for k in KEYS:
    print(f"| {k} | {mA[k]:.3f} | {mB[k]:.3f} | {m0[k]:.3f} |" + "".join(f" {RES[n]['mean'][k]:.3f} |" for n in RES))
print(f"| L1 to oracle A | - | {noise:.4f} | {OUT['theta0']['L1']:.4f} |"
      + "".join(f" {RES[n]['L1']:.4f} |" for n in RES))
