"""Stage 6 of notes/circles_biome_test.md on today's Potts (kappa 8, J 0.1):
train.fit on the mid level (mid_h, mid_v, mid_u, tabular feats, designed
offset = the lam row) with
  a  ExactTargets(Oracle.mid_probs) on forward contexts, K = 0  (S1)
  b  the same with K = 3 (draws from mid_probs, block tiles redrawn by the
     oracle's backward sampling: one collapsed p* move per site)
  c  SampledTargets on oracle joint samples (Oracle.sweep between contexts)
iters 5, n_contexts 8, nty = ntx = 6.  mid_h / mid_v double-centred against
BM J d(m, m') (-0.40 / 0 / +0.40 by cyclic distance) and S1's published
-0.40 / +0.01 / +0.38.  Then (CIRCLES=1) the old circles mid level with
ExactTargets(Oracle.mid_probs), K = 0, against circles.reference().

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/bootstrap_potts.py
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from castlegen.channels.potts import Potts, Forward, Oracle, dist
from castlegen.channels.targets import ExactTargets, SampledTargets
from castlegen.channels import train

NT, ITERS, NCTX = int(os.environ.get("NT", 6)), int(os.environ.get("ITERS", 5)), int(os.environ.get("NCTX", 8))
CIRCLES = os.environ.get("CIRCLES", "1") == "1"
MID = ["mid_h", "mid_v", "mid_u"]
S1_PUB = [-0.40, 0.01, 0.38]
OUT = "images/bootstrap_potts.json"
os.makedirs("images", exist_ok=True)


def by_dist(A, q=4):
    cc = np.arange(q)
    d = dist(cc[:, None], cc[None, :])
    return [float(A[d == k].mean()) for k in range(q // 2 + 1)]


def potts_setting(name, K, sampled, seed):
    P = Potts(NT, NT, J=0.1, kappa=8.0, lam=3.0)
    orc = Oracle(P, seed=seed)
    shapes = {n: P.theta0()[n].shape for n in MID}

    def tables(th):
        t = P.theta0()
        if th is not None:
            t.update(train.unpack(th, shapes))
        return t

    if sampled:
        orc.sweep(50)

        def contexts(th, n, rng):
            for _ in range(n):
                orc.sweep(10)
                yield P.model(orc.top, orc.mid, orc.tile, tables(th))
        targets = SampledTargets()
    else:
        fw = Forward(P, P.theta0(), seed=seed + 1)
        fw.top, fw.mid, fw.tile = orc.top, orc.mid, orc.tile               # one state: forward, oracle, models

        def contexts(th, n, rng):
            for _ in range(n):
                fw.theta = tables(th)
                fw.run(30, 30, 20, fresh=True)
                yield fw.model
        targets = ExactTargets(lambda m, h, y, x: orc.mid_probs(y, x))

    def redraw(m, h, y, x):                                                 # tiles of the block given the new m
        orc._filter(y, x, int(orc.mid.grid[y, x]), True)

    print(f"== {name}: K {K}  {'sampled' if sampled else 'exact'} targets", flush=True)
    th, rep = train.fit(contexts, "mid", targets, train.tabular_feats(MID), train.designed_energies(orc.model, "mid"),
                        iters=ITERS, n_contexts=NCTX, K=K, l2=1e-4, after_change=redraw, rng=seed, verbose=True)
    tab = train.unpack(th, shapes)
    cc = np.arange(P.q)
    ref = train.double_centre(P.BM * P.J * dist(cc[:, None], cc[None, :]))
    res = dict(K=K, targets="sampled" if sampled else "exact", report=rep,
               tables={k: v.tolist() for k, v in tab.items()})
    for k in ("mid_h", "mid_v"):
        dc = train.double_centre(tab[k])
        bd = by_dist(dc)
        res[k + "_dc"] = dc.tolist()
        res[k + "_by_dist"] = bd
        res[k + "_maxerr_ref"] = float(np.abs(dc - ref).max())
        res[k + "_maxerr_S1pub"] = float(np.abs(np.array(bd) - S1_PUB).max())
    # C1: tiles with mid clamped, at the final context's state
    res["tau_tile_given_mid"] = train.autocorr_clamped(orc.model, "tile", ["mid", "top"], sweeps=400, burn=20)
    print(f"   mid_h by d {np.round(res['mid_h_by_dist'], 3)}  max err vs BM J d {res['mid_h_maxerr_ref']:.3f}  "
          f"mid_v by d {np.round(res['mid_v_by_dist'], 3)}  ({res['mid_v_maxerr_ref']:.3f})  "
          f"tau(tile | mid) {res['tau_tile_given_mid']:.2f}", flush=True)
    return res


def circles_run(seed=0):
    from castlegen.channels.circles import Circles, Forward as CF, Oracle as CO
    C = Circles(NT, NT)
    orc = CO(C, seed=seed)
    fw = CF(C, C.theta0(), seed=seed + 1)
    fw.chans = orc.chans
    fw.top, fw.mid, fw.tile, fw.slot, fw.dem = orc.chans
    shapes = {n: C.theta0()[n].shape for n in MID}

    def tables(th):
        t = C.theta0()
        if th is not None:
            t.update(train.unpack(th, shapes))
        return t

    def contexts(th, n, rng):
        for _ in range(n):
            fw.theta = tables(th)
            fw.run(30, 30, 20, fresh=True)
            yield fw.model

    print("== circles: K 0  exact targets", flush=True)
    th, rep = train.fit(contexts, "mid", ExactTargets(lambda m, h, y, x: orc.mid_probs(y, x)),
                        train.tabular_feats(MID), train.designed_energies(orc.model, "mid"),
                        iters=ITERS, n_contexts=NCTX, K=0, l2=1e-4, rng=seed, verbose=True)
    tab = train.unpack(th, shapes)
    ref = C.reference()
    val = lambda cy, cx: 1 + (cy - 2) * 4 + (cx - 2)
    ent = dict(A=(val(3, 5), val(3, 2)), B=(val(2, 5), val(2, 2)), C=(val(3, 5), val(3, 3)), D=(val(3, 4), val(3, 2)))
    res = dict(report=rep)
    for k in ("mid_h", "mid_v"):
        mh = train.double_centre(tab[k])
        a, b = mh[1:, 1:].ravel(), ref[k][1:, 1:].ravel()
        res[k] = dict(maxerr=float(np.abs(mh - ref[k]).max()), pp_maxerr=float(np.abs(a - b).max()),
                      pp_slope=float(a @ b / (b @ b)), pp_corr=float(np.corrcoef(a, b)[0, 1]),
                      entries={n: [float(mh[p]), float(ref[k][p])] for n, p in ent.items()} if k == "mid_h" else None,
                      table=mh.tolist())
        print(f"   {k}: max err {res[k]['maxerr']:.2f}  pp max err {res[k]['pp_maxerr']:.2f}  "
              f"slope {res[k]['pp_slope']:.2f}  corr {res[k]['pp_corr']:.2f}", flush=True)
    print("   A..D learned (ref):", {n: np.round(v, 2).tolist() for n, v in res["mid_h"]["entries"].items()})
    return res


out = dict(config=dict(NT=NT, ITERS=ITERS, NCTX=NCTX, kappa=8.0, J=0.1, lam=3.0, l2=1e-4, holdout=0.1,
                       ref_by_dist=[-0.4, 0.0, 0.4], S1_published=S1_PUB))
out["a_exact_K0"] = potts_setting("a", 0, False, 0)
out["b_exact_K3"] = potts_setting("b", 3, False, 0)
out["c_sampled"] = potts_setting("c", 0, True, 0)
if CIRCLES:
    out["circles_exact_K0"] = circles_run()
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)

P = Potts(1, 1, J=0.1)
cc = np.arange(4)
ref = train.double_centre(P.BM * P.J * dist(cc[:, None], cc[None, :]))
panels = [("BM J d (ref)", ref)] + [(f"{k[0]}: {k[2:]}", np.array(out[k]["mid_h_dc"]))
                                    for k in ("a_exact_K0", "b_exact_K3", "c_sampled")]
fig, ax = plt.subplots(1, 4, figsize=(11, 3))
for a, (t, A) in zip(ax, panels):
    im = a.imshow(A, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    a.set_title(t, fontsize=9)
    a.set_xticks(range(4)); a.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{A[i, j]:+.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Potts kappa 8, J 0.1: mid_h double-centred (rows = left mid colour)", fontsize=10)
fig.savefig("images/bootstrap_potts_tables.png", dpi=110, bbox_inches="tight")
print("wrote images/bootstrap_potts_tables.png")
