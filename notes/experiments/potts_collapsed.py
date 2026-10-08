"""Stage 6b of notes/circles_biome_test.md: the Potts leak with collapsed
targets (notes/circles_biome_stage6b.md).  Per setting (kappa 8 J 0.1,
kappa 1 J 0.3; nty = ntx = 6, lam = 3):

  mid    train.fit of mid_h, mid_v, mid_u: ExactTargets(Oracle.mid_probs), K 0,
         iters 5, n_contexts 8 (stage 6 / bootstrap_potts.py a), top tables 0
  top    train.fit of top_h, top_v, top_u with the fitted mid tables installed,
         contexts = fresh forward runs at the current tables, designed offset 0
         (no factor is homed on top), tabular feats, iters 5, n_contexts 8:
    frozen   ExactTargets(softmax(-site_energies(top, below=True))) = S1's
             p*(T | mid); K 0 and K 3
    ais      AISTargets(child mid, region = the cell's 2 x 2 mid block, K 32,
             M 16): log Z_t over the 4 mid cells under lam + the learned mid
             tables, the neighbouring mid cells a fixed boundary (rows kept)
    hook     the same window, log Z exact (column transfer matrix); K 0, K 3
    hook4    the window dilated by one mid cell (4 x 4, the ring is the reach of
             mid_h / mid_v), exact; boundary at distance 2
  K-steps (K 3): T drawn from the target, then Oracle.mid_move at its 4 mid
  cells (collapsed draw given T and the boundary tiles, tiles redrawn).
Eval: top contrast, 64-run forward L1 to the symmetrised oracle (50 + 400
sweeps) on the fitted features, eval noise (two independent 64-run evals),
edge_same_top, fit consistency violation, tau of the mid kernel with top clamped.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/potts_collapsed.py
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from castlegen.channels.potts import Potts, Forward, Oracle, dist
from castlegen.channels.targets import ExactTargets
from castlegen.channels.aistargets import AISTargets
from castlegen.channels import train

NT, ITERS, NCTX, RUNS = 6, int(os.environ.get("ITERS", 5)), int(os.environ.get("NCTX", 8)), int(os.environ.get("RUNS", 64))
BURN, SWEEPS, SEED = 50, int(os.environ.get("SWEEPS", 400)), 0
S = (30, 30, 20)
MID, TOP = ["mid_h", "mid_v", "mid_u"], ["top_h", "top_v", "top_u"]
FIT = MID + TOP
SETTINGS = [(8.0, 0.1), (1.0, 0.3)]
PUB = {  # notes/potts_test.md "top_h / top_v: the leak", disjoint-same h/v, one-colour-same h/v
    "k8": dict(S0=(0.73, 0.71, 0.31, 0.32), S1=(0.50, 0.53, 0.25, 0.26), S2=(0.61, 0.63, 0.28, 0.29),
               S3=(0.67, 0.70, 0.30, 0.29), edge=dict(oracle=.395, S0=.397, S1=.381, S3=.392)),
    "k1": dict(S0=(0.60, 0.58, 0.26, 0.26), S1=(0.38, 0.39, 0.20, 0.19), S2=(0.45, 0.47, 0.21, 0.24),
               S3=(0.52, 0.52, 0.22, 0.24), edge=dict(oracle=.365, S0=.365, S1=.356, S3=.361))}
S0_TOPH = {"k8": [[-.31, -.02, .38, -.05], [-.02, -.38, -.04, .38], [.41, -.02, -.30, -.06], [-.00, .40, .00, -.36]],
           "k1": [[-.27, -.02, .34, -.06], [-.03, -.31, -.03, .30], [.32, -.00, -.23, -.04], [-.00, .34, -.01, -.31]]}
MID_PUB = {"k8": dict(S0=[-.39, .02, .36], S1=[-.40, .01, .38], ref=[-.40, 0, .40]),
           "k1": dict(S0=[-.31, .02, .26], S1=[-.29, .02, .26])}
OUT = "images/potts_collapsed.json"
os.makedirs("images", exist_ok=True)


def symmetrise(P, st):
    """notes/experiments/potts_train.py's symmetrise (that script runs on import)."""
    q = P.q
    out = {k: np.zeros_like(np.asarray(st[k], float)) for k in FIT}
    n = 0
    for sgn in (1, -1):
        for r in range(q):
            sig = (sgn * np.arange(q) + r) % q
            pi = np.array([np.flatnonzero((P.PAL[:, sig[P.PAL[p]]]).all(1))[0] for p in range(P.P)])
            for k in FIT:
                a = np.asarray(st[k], float)
                m = sig if k.startswith("mid") else pi
                b = np.zeros_like(a)
                if a.ndim == 1:
                    b[m] = a
                else:
                    b[np.ix_(m, m)] = a
                out[k] += b
            n += 1
    out = {k: v / n for k, v in out.items()}
    for lv in ("mid", "top"):
        h, v = out[lv + "_h"], out[lv + "_v"]
        s = (h + h.T + v + v.T) / 4
        out[lv + "_h"], out[lv + "_v"] = s, s.copy()
    res = dict(st)
    res.update(out)
    return res


def mean_stats(sts):
    return {k: np.mean([s[k] for s in sts], axis=0) for k in sts[0]}


def l1(a, b, keys=FIT):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def by_dist(A, q=4):
    cc = np.arange(q)
    d = dist(cc[:, None], cc[None, :])
    return [float(A[d == k].mean()) for k in range(q // 2 + 1)]


def contrast(A):
    """(disjoint - same, one-colour - same) of a (P, P) table, double-centred."""
    A = train.double_centre(A)
    p = np.arange(A.shape[0])
    P = len(p)
    return float(np.mean(A[p, (p + 2) % P] - A[p, p])), float(np.mean((A[p, (p + 1) % P] + A[p, (p - 1) % P]) / 2 - A[p, p]))


def window_logz(tabs, PAL_lam, topg, midg, y0, y1, x0, x1, BT):
    """Exact log Z over the mid cells [y0, y1) x [x0, x1) (clipped) under
    lam[top, m] + mid_u + mid_h / mid_v (within and to the fixed cells
    around it, none off the grid): column transfer matrix, column state =
    the window height's colours.  AISTargets' Z convention."""
    H, W = midg.shape
    y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, H), min(x1, W)
    h, q = y1 - y0, tabs["mid_u"].shape[0]
    mh, mv, mu = tabs["mid_h"], tabs["mid_v"], tabs["mid_u"]
    NS = q ** h
    digs = np.stack([(np.arange(NS) // q ** r) % q for r in range(h)], 1)          # (NS, h)
    vert = sum(mv[digs[:, r], digs[:, r + 1]] for r in range(h - 1)) if h > 1 else np.zeros(NS)
    trans = sum(mh[digs[:, r][:, None], digs[:, r][None, :]] for r in range(h))    # (NS, NS) left -> right
    la = None
    for x in range(x0, x1):
        u = np.zeros((h, q))
        for r, y in enumerate(range(y0, y1)):
            u[r] = PAL_lam[topg[y // BT, x // BT]] + mu
            if x == x0 and x - 1 >= 0:
                u[r] += mh[midg[y, x - 1], :]
            if x == x1 - 1 and x + 1 < W:
                u[r] += mh[:, midg[y, x + 1]]
        if y0 - 1 >= 0:
            u[0] += mv[midg[y0 - 1, x], :]
        if y1 < H:
            u[h - 1] += mv[:, midg[y1, x]]
        e = vert + u[np.arange(h)[None, :], digs].sum(1)
        if la is None:
            la = -e
        else:
            a = la[:, None] - trans
            m = a.max(0)
            la = m + np.log(np.exp(a - m).sum(0)) - e
    m = la.max()
    return float(m + np.log(np.exp(la - m).sum()))


class AISRec(AISTargets):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.se = []

    def log_z(self, *a):
        lz, se = super().log_z(*a)
        self.se += list(se)
        return lz, se


def run_setting(kappa, J):
    tag = f"k{kappa:g}"
    P = Potts(NT, NT, J=J, kappa=kappa, lam=3.0)
    res = dict(kappa=kappa, J=J)
    t0 = time.time()
    raw = Oracle(P, seed=SEED + 7).moments(BURN, SWEEPS)
    target = symmetrise(P, raw)
    res["oracle"] = {k: np.asarray(v).tolist() for k, v in target.items()}
    res["t_oracle"] = time.time() - t0
    print(f"\n===== {tag} J {J}: oracle {res['t_oracle']:.1f}s  edge_same_top {target['edge_same_top']:.3f}", flush=True)

    def bench(seed):
        orc = Oracle(P, seed=seed)
        fw = Forward(P, P.theta0(), seed=seed + 1)
        fw.top, fw.mid, fw.tile = orc.top, orc.mid, orc.tile
        return orc, fw

    def contexts_of(fw, tables_of):
        def gen(th, n, rng):
            for _ in range(n):
                fw.theta = tables_of(th)
                fw.run(*S, fresh=True)
                yield fw.model
        return gen

    # ---------------------------------------------------------------- mid
    sh_mid = {n: P.theta0()[n].shape for n in MID}
    sh_top = {n: P.theta0()[n].shape for n in TOP}
    orc, fw = bench(SEED)
    mid_tab = lambda th: dict(P.theta0(), **(train.unpack(th, sh_mid) if th is not None else {}))
    th_mid, rep = train.fit(contexts_of(fw, mid_tab), "mid", ExactTargets(lambda m, h, y, x: orc.mid_probs(y, x)),
                            train.tabular_feats(MID), train.designed_energies(orc.model, "mid"), iters=ITERS,
                            n_contexts=NCTX, K=0, l2=1e-4, rng=SEED, verbose=True)
    tm = train.unpack(th_mid, sh_mid)
    res["mid"] = dict(tables={k: v.tolist() for k, v in tm.items()}, seconds=rep["seconds_total"],
                      mid_h_by_dist=by_dist(train.double_centre(tm["mid_h"])),
                      mid_v_by_dist=by_dist(train.double_centre(tm["mid_v"])))
    print("   mid_h by d", np.round(res["mid"]["mid_h_by_dist"], 3), " mid_v by d", np.round(res["mid"]["mid_v_by_dist"], 3))

    tables_of = lambda th: dict(P.theta0(), **tm, **(train.unpack(th, sh_top) if th is not None else {}))
    lamtab = P.lam * (~P.PAL).astype(np.float64)                                    # (p, m)
    BT = P.BT

    def hook_of(orc, dil):
        return lambda m, home, I, J_: window_logz(tm, lamtab, orc.top.grid, orc.mid.grid, I * BT - dil,
                                                   (I + 1) * BT + dil, J_ * BT - dil, (J_ + 1) * BT + dil, BT)

    def region_of(dil):
        return lambda m, home, I, J_: (I * BT - dil, (I + 1) * BT + dil, J_ * BT - dil, (J_ + 1) * BT + dil)

    zero_e = lambda m, home, I, J_, cand: np.zeros(len(cand))
    noop = lambda m, home, I, J_: None

    # ---------------------------------------------------------------- top
    res["top"] = {}
    variants = [("frozen", 0), ("frozen", 3), ("ais", 0), ("hook", 0), ("hook", 3), ("hook4", 0)]
    for kind, K in variants:
        orc, fw = bench(SEED)
        A = None
        if kind == "frozen":
            def fz(m, h, I, J_, orc=orc):
                e = orc.model.site_energies("top", I, J_, below=True)
                p = np.exp(-(e - e.min()))
                return p / p.sum()
            targets = ExactTargets(fz)
        elif kind == "ais":
            A = targets = AISRec("mid", region_of(0), zero_e, noop, K=32, M=16, L=30.0, seed=SEED)
        else:
            dil = 1 if kind == "hook4" else 0
            targets = AISTargets("mid", region_of(dil), zero_e, noop, hook=hook_of(orc, dil))

        def top_move(m, home, I, J_, orc=orc):
            for a in range(I * BT, (I + 1) * BT):
                for b in range(J_ * BT, (J_ + 1) * BT):
                    orc.mid_move(a, b)

        print(f"== {tag} top {kind} K {K}", flush=True)
        th, rep = train.fit(contexts_of(fw, tables_of), "top", targets, train.tabular_feats(TOP),
                            train.designed_energies(orc.model, "top"), iters=ITERS, n_contexts=NCTX, K=K, l2=1e-4,
                            after_change=top_move, rng=SEED, verbose=True)
        tab = tables_of(th)
        r = dict(kind=kind, K=K, theta=th.tolist(), tables={k: tab[k].tolist() for k in TOP},
                 report={k: (np.round(v, 6).tolist() if isinstance(v, list) else v) for k, v in rep.items()})
        ch, cv = contrast(tab["top_h"]), contrast(tab["top_v"])
        r["contrast"] = dict(disj_h=ch[0], disj_v=cv[0], one_h=ch[1], one_v=cv[1],
                             ratio_S0=(ch[0] + cv[0]) / (PUB[tag]["S0"][0] + PUB[tag]["S0"][1]))
        r["top_h_dc"] = train.double_centre(tab["top_h"]).tolist()
        evs = []
        for es in (SEED + 555, SEED + 777):
            f = Forward(P, tab, seed=es)
            evs.append(mean_stats([f.run(*S, fresh=True) for _ in range(RUNS)]))
        r["L1"] = l1(evs[0], target)
        r["L1_b"] = l1(evs[1], target)
        r["noise"] = l1(evs[0], evs[1])
        r["edge_same_top"] = [float(e["edge_same_top"]) for e in evs]
        r["pal_viol"] = [float(e["pal_viol"]) for e in evs]
        r["viol_targets"], r["viol_fit"] = rep["viol_targets"][-1], rep["viol_fit"][-1]
        fw.theta = tab
        fw.run(*S, fresh=True)
        r["tau_mid_given_top"] = train.autocorr_clamped(fw.model, "mid", ["top"], 400, 20, seed=SEED)
        if A is not None:
            se = np.array(A.se)
            r["ais"] = dict(n_ais=A.n_ais, ms_per_run=1e3 * A.t_ais / max(A.n_ais, 1), se_median=float(np.median(se)),
                            se_p90=float(np.quantile(se, .9)), se_max=float(se.max()), annealed=A._pk.anneal_names,
                            seconds=A.t_calls)
            # AIS vs the exact hook on this (final) context: log Z and target L1, every cell, every candidate
            Hx = AISTargets("mid", region_of(0), zero_e, noop, hook=hook_of(orc, 0))
            A2 = AISRec("mid", region_of(0), zero_e, noop, K=32, M=16, L=30.0, seed=SEED + 9)
            dz, zse, dpi = [], [], []
            for I in range(P.nty):
                for J_ in range(P.ntx):
                    cand = np.arange(P.P)
                    lz, se_ = A2.log_z(fw.model, "top", I, J_, cand)
                    lx, _ = Hx.log_z(fw.model, "top", I, J_, cand)
                    dz += list(lz - lx); zse += list(se_)
                    pa, px = np.exp(lz - lz.max()), np.exp(lx - lx.max())
                    dpi.append(float(np.abs(pa / pa.sum() - px / px.sum()).sum()))
            dz, zse = np.array(dz), np.array(zse)
            r["ais_vs_hook"] = dict(dlogz_mean=float(dz.mean()), dlogz_maxabs=float(np.abs(dz).max()),
                                    se_median=float(np.median(zse)), frac_within_3se=float(np.mean(np.abs(dz) <= 3 * zse + 1e-12)),
                                    target_L1_mean=float(np.mean(dpi)), target_L1_max=float(np.max(dpi)))
        print(f"   contrast disj h/v {ch[0]:.2f}/{cv[0]:.2f} one {ch[1]:.2f}/{cv[1]:.2f} ratio S0 {r['contrast']['ratio_S0']:.2f}"
              f"  edge_same_top {r['edge_same_top']}  viol targ/fit {r['viol_targets']:.3g}/{r['viol_fit']:.3g}"
              f"  tau {r['tau_mid_given_top']:.2f}", flush=True)
        print("   L1", {k: round(v, 3) for k, v in r["L1"].items()}, " noise", {k: round(v, 3) for k, v in r["noise"].items()},
              flush=True)
        if A is not None:
            print("   ais", r["ais"], "\n   ais vs hook", r["ais_vs_hook"], flush=True)
        res["top"][f"{kind}_K{K}"] = r
    return tag, res


T0 = time.time()
OUTD = dict(config=dict(NT=NT, ITERS=ITERS, NCTX=NCTX, RUNS=RUNS, BURN=BURN, SWEEPS=SWEEPS, S=S, lam=3.0, l2=1e-4,
                        AIS=dict(K=32, M=16, L=30.0)), published=PUB)
for kappa, J in SETTINGS:
    tag, r = run_setting(kappa, J)
    OUTD[tag] = r
    json.dump(OUTD, open(OUT, "w"), indent=1)
OUTD["seconds"] = time.time() - T0
json.dump(OUTD, open(OUT, "w"), indent=1)
print(f"wrote {OUT}  ({OUTD['seconds']:.0f}s)")

# ---------------------------------------------------------------- figure
kinds = ["frozen_K0", "frozen_K3", "ais_K0", "hook_K0", "hook_K3", "hook4_K0"]
fig, ax = plt.subplots(2, len(kinds) + 1, figsize=(2.3 * (len(kinds) + 1), 5))
for row, tag in enumerate(("k8", "k1")):
    panels = [("S0 (published)", train.double_centre(np.array(S0_TOPH[tag])))] + \
             [(k.replace("_", " "), np.array(OUTD[tag]["top"][k]["top_h_dc"])) for k in kinds]
    for a, (t, A) in zip(ax[row], panels):
        im = a.imshow(A, cmap="RdBu_r", vmin=-0.45, vmax=0.45)
        a.set_title(f"{tag}: {t}", fontsize=8)
        a.set_xticks(range(4)); a.set_yticks(range(4))
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{A[i, j]:+.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.6)
fig.suptitle("Potts top_h double-centred (rows = left palette): published S0 vs target types; "
             "k8 = kappa 8 J 0.1, k1 = kappa 1 J 0.3", fontsize=9)
fig.savefig("images/potts_collapsed_tables.png", dpi=110, bbox_inches="tight")
print("wrote images/potts_collapsed_tables.png")
