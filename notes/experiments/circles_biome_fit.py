"""Stages 4-5 of notes/circles_biome_test.md: the bootstrap fit (train.fit) of
the learned potentials on the biome circles model, mid level then top.

Stage 4 (obj): stamp features (CirclesBiome.stamp_features, OFF8 over dem),
designed offset = mask + pres + support (designed + support model; INF on
the inadmissible values, which fit drops), contexts = the forward chain
(old path) at the current theta's materialised tables.  Targets:
  exact   ExactTargets(Oracle.mid_probs)
  ais     AISTargets over the tile window (block + spill border), default
          split: honour reads dem (p_0), nothing annealed -> exact
  aisH    the same with the honour row forced into the annealed part (INF ->
          L = 30): the estimated-targets column (NCTX_H contexts, NPAIR_H
          probes per context: AIS probes cost 2 D window runs each)
x K in {0, 3} (K-step: draw from pi, dem repainted, changed tiles redrawn).
Tables materialised (biome_fit.materialise) against reference(),
double-centred over the finite entries; held-out pairs = (t, t') never a
(candidate, neighbour value) pair of a data record.

Stage 5 (biome): the exact-K0 mid tables installed; tabular bio_h, bio_v,
bio_u; designed offset bio_u0.  Targets:
  exact   ExactTargets(Oracle.top_probs): p*(T | its four slots), not collapsed
  ais     AISTargets over the cell's 2 x 2 obj slots (obj pairs + support
          annealed, INF -> L), after_change = paint allow + dormancy
  hook    the same window, log Z by enumeration (biome_fit.top_hook): the
          exact collapsed target under the installed mid tables
x K in {0, 3} (K-step: paint allow, dormancy, Oracle.mid_move on the four
slots).  End to end: 64 forward runs with everything installed against the
oracle moments, eval noise = two independent 64-run evals.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/circles_biome_fit.py

Env: NT (6), ITERS (5), NCTX (8), NCTX_H (4), NPAIR_H (1), RUNS (64), BURN (50),
SWEEPS (400), SEED (0).  Outputs images/cbio_stage45.json,
images/cbio_fit_tables.png, images/cbio_fit_{oracle,<setting>}.png."""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from castlegen.channels import train
from castlegen.channels.aistargets import AISTargets
from castlegen.channels.biome_fit import BIO, Bench, all_tables, materialise, mid_tables
from castlegen.channels.circles_biome import OFFNAMES, OFFS, CirclesBiome, Forward, Oracle, dcentre_finite
from castlegen.channels.targets import ExactTargets

E = os.environ.get
NT, ITERS, NCTX = int(E("NT", 6)), int(E("ITERS", 5)), int(E("NCTX", 8))
NCTX_H, NPAIR_H = int(E("NCTX_H", 4)), int(E("NPAIR_H", 1))
RUNS, BURN, SWEEPS, SEED = int(E("RUNS", 64)), int(E("BURN", 50)), int(E("SWEEPS", 400)), int(E("SEED", 0))
FIT = OFFNAMES + ["obj_u", "bio_h", "bio_v", "bio_u"]
MON = ["present_disc", "present_bar", "conflict", "contact", "dormant", "mask_viol", "edge_air_top"]
INF = np.inf
os.makedirs("images", exist_ok=True)
big = lambda a: Image.fromarray(a).resize((a.shape[1] * 4, a.shape[0] * 4), Image.NEAREST)
C =CirclesBiome(NT, NT)
REF = C.reference()
OUT = dict(config=dict(NT=NT, ITERS=ITERS, NCTX=NCTX, NCTX_H=NCTX_H, NPAIR_H=NPAIR_H, RUNS=RUNS, BURN=BURN,
                       SWEEPS=SWEEPS, mu=C.mu, l2=1e-4, holdout=0.1, AIS=dict(K=32, M=16, L=30.0), S=(30, 30, 20)))


class AISRec(AISTargets):
    """AISTargets recording the standard errors of the finite log Z's; a
    window with no finite candidate (a probe state with a conflict tile in
    the window) gives nan targets, which consistency() skips."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.se, self.n_dead = [], 0

    def log_z(self, *a):
        lz, se = super().log_z(*a)
        self.se += list(se[np.isfinite(lz)])
        return lz, se

    def at(self, model, home, y, x, cand):
        try:
            return super().at(model, home, y, x, cand)
        except AssertionError:
            self.n_dead += 1
            return np.full(len(cand), np.nan)


def ais_report(A):
    se = np.array(A.se) if A.se else np.zeros(1)
    return dict(n_ais=A.n_ais, n_hook=A.n_hook, t_ais=A.t_ais, t_calls=A.t_calls, n_calls=A.n_calls,
                ms_per_run=1e3 * A.t_ais / max(A.n_ais, 1), se_median=float(np.median(se)),
                se_p90=float(np.quantile(se, 0.9)), se_max=float(se.max()), n_dead=A.n_dead,
                annealed=A._pk.anneal_names if A._pk is not None else None)


def slim(rep):
    return {k: (np.round(v, 6).tolist() if isinstance(v, list) else v) for k, v in rep.items()}


# ------------------------------------------------------------ stage 4
def mid_metrics(theta, seen):
    T = materialise(C, theta)
    out = dict(tables={})
    hits = {}
    for k, (n, d) in enumerate(zip(OFFNAMES, OFFS)):
        L = dcentre_finite(np.where(C.CONF[d], INF, T[n]))
        R = REF[n]
        fin = np.isfinite(R)
        fin[0, :] = fin[:, 0] = False                                    # present-present entries
        held = fin & ~seen[k]
        inter = held & (C.PAIR[d] != 0)
        err = np.abs(L - R)
        a, b = L[fin], R[fin]
        out[n] = dict(maxerr=float(err[fin].max()), held_n=int(held.sum()), held_inter_n=int(inter.sum()),
                      held_maxerr=float(err[held].max()) if held.any() else float("nan"),
                      held_inter_maxerr=float(err[inter].max()) if inter.any() else float("nan"),
                      seen_n=int((fin & seen[k]).sum()), n_finite=int(fin.sum()),
                      slope=float(a @ b / (b @ b)), corr=float(np.corrcoef(a, b)[0, 1]))
        out["tables"][n] = L
    u = T["obj_u"] + C.pres_e
    u = u - u.mean()
    out["obj_u"] = dict(maxerr=float(np.abs(u - REF["obj_u"]).max()))
    for nm, (t, tp) in ENT.items():
        out.setdefault("entries", {})[nm] = [float(out["tables"]["obj_h"][t, tp]), float(REF["obj_h"][t, tp])]
    out["held_maxerr"] = max(out[n]["held_maxerr"] for n in OFFNAMES if np.isfinite(out[n]["held_maxerr"])) \
        if any(np.isfinite(out[n]["held_maxerr"]) for n in OFFNAMES) else float("nan")
    out["maxerr"] = max(out[n]["maxerr"] for n in OFFNAMES)
    return out


# two conflict-free shared-ring pairs in obj_h: the most attractive finite raw entries, distinct families
_P = np.where(C.CONF[(0, 1)], INF, C.PAIR[(0, 1)])
_dd = [(t, tp) for t in range(1, 17) for tp in range(1, 17)]
_bb = [(t, tp) for t in range(17, 33) for tp in range(17, 33)]
ENT = dict(disc_disc=min(_dd, key=lambda p: _P[p]), bar_bar=min(_bb, key=lambda p: _P[p]))
OUT["named_entries"] = {k: dict(pair=[int(v) for v in p], raw=float(_P[p])) for k, p in ENT.items()}


def run_mid(kind, K, seed):
    B = Bench(C, seed=seed)
    tables_of = lambda th: mid_tables(C, th)
    de = B.designed_e("obj")
    A = None
    if kind == "exact":
        targets = ExactTargets(lambda m, h, i, j: B.orc.mid_probs(i, j))
    else:
        A = targets = AISRec("tile", B.mid_region, de, B.mid_paint, K=32, M=16, L=30.0, seed=seed,
                             anneal_names=["honour"] if kind == "aisH" else None)
    nctx, npair = (NCTX_H, NPAIR_H) if kind == "aisH" else (NCTX, 8)
    print(f"== mid {kind} K {K}  ({nctx} contexts, {npair} probes each)", flush=True)
    th, rep = train.fit(B.contexts(tables_of, "obj", K), "obj", targets, B.mid_feats, de, iters=ITERS,
                        n_contexts=nctx, K=K, after_change=B.mid_move, repaint=B.mid_paint, rng=seed, n_pairs=npair,
                        verbose=True)
    res = dict(kind=kind, K=K, n_contexts=nctx, n_pairs=npair, report=slim(rep), theta=th.tolist())
    met = mid_metrics(th, B.seen)
    res.update({k: v for k, v in met.items() if k != "tables"})
    # C1 at the final theta: obj with biome clamped, tiles with obj clamped
    B.fw.theta = tables_of(th)
    B.fw.run(*B.S, fresh=True)
    res["dormant_frac"] = float(B.obj.fixed.mean())
    res["tau_obj_given_biome"] = train.autocorr_clamped(B.fw.model, "obj", ["biome", "allow"], 400, 20, seed=seed)
    C.paint_dem(B.obj, B.dem)
    res["tau_tile_given_obj"] = train.autocorr_clamped(B.fw.model, "tile", ["obj", "dem"], 400, 20, seed=seed)
    if A is not None:
        res["ais"] = ais_report(A)
    print(f"   held-out max err {met['held_maxerr']:.4f}  all {met['maxerr']:.4f}  obj_u {met['obj_u']['maxerr']:.4f}  "
          f"held n {[met[n]['held_n'] for n in OFFNAMES]}  slope/corr obj_h {met['obj_h']['slope']:.3f}/"
          f"{met['obj_h']['corr']:.3f}  tau obj|biome {res['tau_obj_given_biome']:.2f} tile|obj "
          f"{res['tau_tile_given_obj']:.2f}  {rep['seconds_total']:.0f}s", flush=True)
    if A is not None:
        print("   AIS", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["ais"].items()}, flush=True)
    return res, th, met["tables"]


def ais_mid_check(n=20, seed=7):
    """Default AIS (nothing annealed) and the forced honour anneal against
    mid_probs at n active sites of one forward context, all candidates."""
    B = Bench(C, seed=seed)
    B.fw.theta = mid_tables(C)
    B.fw.run(*B.S, fresh=True)
    m = B.fw.model
    de = B.designed_e("obj")
    Ad = AISRec("tile", B.mid_region, de, B.mid_paint, K=32, M=16, seed=seed)
    Ah = AISRec("tile", B.mid_region, de, B.mid_paint, K=32, M=16, seed=seed, anneal_names=["honour"])
    Hx = AISRec("tile", B.mid_region, de, B.mid_paint, hook=B.mid_hook)
    act = np.argwhere(~B.obj.fixed)
    rng = np.random.default_rng(seed)
    out = dict(default_L1=[], honour_L1=[], honour_dlogz_over_se=[])
    for i, j in act[rng.choice(len(act), min(n, len(act)), replace=False)]:
        cand = np.flatnonzero(np.isfinite(de(m, "obj", i, j, np.arange(C.D))))
        pe = B.orc.mid_probs(i, j)[cand]
        pe = pe / pe.sum()
        out["default_L1"].append(float(np.abs(Ad.at(m, "obj", i, j, cand) - pe).sum()))
        out["honour_L1"].append(float(np.abs(Ah.at(m, "obj", i, j, cand) - pe).sum()))
        lz, se = Ah.log_z(m, "obj", i, j, cand)
        lx, _ = Hx.log_z(m, "obj", i, j, cand)
        ok = se > 0
        out["honour_dlogz_over_se"] += list(((lz - lx)[ok] / se[ok]).astype(float))
    out["default"], out["honour"] = ais_report(Ad), ais_report(Ah)
    z = np.array(out["honour_dlogz_over_se"])
    out["summary"] = dict(default_L1_max=max(out["default_L1"]), honour_L1_median=float(np.median(out["honour_L1"])),
                          honour_L1_max=max(out["honour_L1"]), honour_dlogz_over_se_median=float(np.median(z)),
                          honour_frac_within_3se=float((np.abs(z) <= 3).mean()) if len(z) else float("nan"))
    print("   AIS mid check", out["summary"], flush=True)
    return out


# ------------------------------------------------------------ stage 5
def oracle_moments(seed):
    O = Oracle(C, seed=seed)
    t0 = time.time()
    mo = O.moments(BURN, SWEEPS)
    print(f"oracle moments {time.time() - t0:.0f}s", flush=True)
    return O, mo


def neglog_ratio(J):
    """double-centred -log(p(T, T') / (p(T) p(T'))) of a joint frequency table (INF where 0)."""
    J = J / J.sum()
    pp = np.outer(J.sum(1), J.sum(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(J > 0, -np.log(J / pp), INF)
    return dcentre_finite(A)


def evaluate(tables, runs, seed):
    fw = Forward(C, tables, seed=seed)
    sts = [fw.run(30, 30, 20, fresh=True) for _ in range(runs)]
    mean = {k: np.mean([s[k] for s in sts], axis=0) for k in sts[0]}
    return mean, fw


def l1(a, b, keys):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def run_top(kind, K, theta_mid, seed, mo):
    B = Bench(C, seed=seed)
    tables_of = lambda th: all_tables(C, theta_mid, th)
    tab0 = tables_of(None)
    de = B.designed_e("biome")
    feats = train.tabular_feats(BIO)
    A = None
    if kind == "exact":
        targets = ExactTargets(lambda m, h, i, j: B.orc.top_probs(i, j))
    else:
        A = targets = AISRec("obj", B.top_region, de, B.top_dormancy, K=32, M=16, L=30.0, seed=seed,
                             hook=B.top_hook(tab0) if kind == "hook" else None)
    npair = 2 if kind == "hook" else 8
    print(f"== top {kind} K {K}", flush=True)
    th, rep = train.fit(B.contexts(tables_of, "biome", K), "biome", targets, feats, de, iters=ITERS,
                        n_contexts=NCTX, K=K, after_change=B.top_move, repaint=B.top_paint, rng=seed, n_pairs=npair,
                        verbose=True)
    tab = tables_of(th)
    res = dict(kind=kind, K=K, n_pairs=npair, report=slim(rep), theta=th.tolist())
    for n in ("bio_h", "bio_v"):
        L = train.double_centre(tab[n])
        R = neglog_ratio(mo[n])
        fin = np.isfinite(R)
        res[n] = dict(dc=L.tolist(), oracle_neglog_ratio_dc=np.where(fin, R, np.nan).tolist(),
                      maxdiff_vs_oracle=float(np.abs(L - R)[fin].max()))
    bu = tab["bio_u"] - tab["bio_u"][0]
    ref_bu = -C.BT ** 2 * np.log(np.array([1.0 + C.FAMOK[1:, v].sum() for v in range(C.P)]))
    res["bio_u_none0"] = bu.tolist()
    res["bio_u_zeroth_order"] = ref_bu.tolist()
    res["bio_u_maxdiff"] = float(np.abs(bu - ref_bu).max())
    # Phi_top(none) = 0: AIS log Z of the none candidate at every cell of a fresh context at the final theta
    B.fw.theta = tab
    B.fw.run(*B.S, fresh=True)
    Ac = AISRec("obj", B.top_region, de, B.top_dormancy, K=32, M=16, L=30.0, seed=seed + 5)
    Hc = AISRec("obj", B.top_region, de, B.top_dormancy, hook=B.top_hook(tab))
    lz0, dz, zse = [], [], []
    for I in range(C.nty):
        for J in range(C.ntx):
            lz, se = Ac.log_z(B.fw.model, "biome", I, J, np.arange(C.P))
            lx, _ = Hc.log_z(B.fw.model, "biome", I, J, np.arange(C.P))
            lz0 += [float(lz[0]), float(lx[0])]
            dz += list((lz - lx)[1:]); zse += list(se[1:])
    res["phi_none_logz_max_abs"] = float(np.abs(lz0).max())
    res["ais_vs_hook"] = dict(dlogz_mean=float(np.mean(dz)), dlogz_maxabs=float(np.abs(dz).max()),
                              se_median=float(np.median(zse)),
                              frac_within_3se=float(np.mean(np.abs(dz) <= 3 * np.maximum(zse, 1e-12))))
    res["tau_obj_given_biome"] = train.autocorr_clamped(B.fw.model, "obj", ["biome", "allow"], 400, 20, seed=seed)
    if A is not None:
        res["ais"] = ais_report(A)
    ev, fw = evaluate(tab, RUNS, seed + 11)
    res["eval"] = {k: np.asarray(v).tolist() for k, v in ev.items()}
    res["L1_fit"] = l1(ev, mo, FIT)
    res["L1_mon"] = l1(ev, mo, MON)
    res["conflict"] = float(ev["conflict"])
    big(C.render(fw.biome, fw.obj, fw.tile)).save(f"images/cbio_fit_top_{kind}_K{K}.png")
    print(f"   bio_u (none = 0) {np.round(bu, 2)} vs {np.round(ref_bu, 2)}  Phi(none) |logz| {res['phi_none_logz_max_abs']}"
          f"  AIS-hook {res['ais_vs_hook']}  conflict {res['conflict']}", flush=True)
    print("   L1 fit", {k: round(v, 3) for k, v in res["L1_fit"].items()}, flush=True)
    print("   L1 mon", {k: round(v, 4) for k, v in res["L1_mon"].items()}, flush=True)
    return res


def run_mid_family_holdout(fam, seed):
    """Exact targets, K = 0, contexts with every biome fixed to admit one
    family only (biome = 1 << fam): the other family's values never occur, so
    every pair involving them is held out (the strict test of the stamp
    features' generalisation)."""
    B = Bench(C, seed=seed)
    fw = B.fw

    def gen(theta, n, rng):
        for _ in range(n):
            fw.theta = mid_tables(C, theta)
            B.biome.grid[:] = 1 << fam
            B.obj.grid[:] = 0
            B.tile.grid[:] = 0
            fw.run(0, 30, 20, fresh=False)
            B._budget = int((~B.obj.fixed).sum())
            yield fw.model
    print(f"== mid exact K 0, contexts admit {C.names[fam]} only", flush=True)
    th, rep = train.fit(gen, "obj", ExactTargets(lambda m, h, i, j: B.orc.mid_probs(i, j)), B.mid_feats,
                        B.designed_e("obj"), iters=ITERS, n_contexts=NCTX, K=0, repaint=B.mid_paint, rng=seed,
                        verbose=True)
    met = mid_metrics(th, B.seen)
    other = C.FAM != fam + 1
    other[0] = False
    out = dict(family=C.names[fam], report=slim(rep), seen_values=np.flatnonzero(B.seen.any((0, 2))).tolist())
    for n, d in zip(OFFNAMES, OFFS):
        R, L = REF[n], met["tables"][n]
        fin = np.isfinite(R)
        fin[0, :] = fin[:, 0] = False
        for nm, msk in (("unseen_family_pairs", fin & (other[:, None] | other[None, :])),
                        ("unseen_family_both", fin & other[:, None] & other[None, :])):
            e = np.abs(L - R)[msk]
            out.setdefault(n, {})[nm] = dict(n=int(msk.sum()), maxerr=float(e.max()),
                                             n_interacting=int((msk & (C.PAIR[d] != 0)).sum()))
    u = materialise(C, th)["obj_u"] + C.pres_e
    out["obj_u_maxerr"] = float(np.abs(u - u.mean() - REF["obj_u"]).max())
    out["obj_u_maxerr_unseen"] = float(np.abs(u - u.mean() - REF["obj_u"])[other].max())
    print("  ", {n: out[n] for n in OFFNAMES}, "obj_u", out["obj_u_maxerr"], flush=True)
    return out


if __name__ == "__main__" and E("HELDOUT"):
    res = {C.names[f]: run_mid_family_holdout(f, SEED + 3) for f in range(C.NF)}
    json.dump(res, open("images/cbio_stage45_heldout.json", "w"), indent=1)
elif __name__ == "__main__":
    T0 = time.time()
    # ---------------- stage 4
    mids, mtabs, theta_mid = {}, {}, None
    for kind in ("exact", "ais", "aisH"):
        for K in (0, 3):
            r, th, tabs = run_mid(kind, K, SEED)
            mids[f"{kind}_K{K}"] = r
            mtabs[f"{kind}_K{K}"] = tabs
            if kind == "exact" and K == 0:
                theta_mid = th
            json.dump(dict(OUT, mid=mids), open("images/cbio_stage45.json", "w"), indent=1)
    OUT["mid"] = mids
    OUT["gate_mid"] = dict(held_maxerr=mids["exact_K0"]["held_maxerr"], passed=bool(mids["exact_K0"]["held_maxerr"] < 0.05))
    OUT["ais_mid_check"] = ais_mid_check()
    # tables figure
    keys = list(mtabs)
    fig, ax = plt.subplots(2, len(keys) + 1, figsize=(3.0 * (len(keys) + 1), 6.2))
    for r, n in enumerate(("obj_h", "obj_v")):
        for c, (t, A) in enumerate([("reference", REF[n])] + [(k, mtabs[k][n]) for k in keys]):
            Am = np.where(np.isfinite(A), A, np.nan)
            v = np.nanmax(np.abs(Am))
            cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.6")
            im = ax[r, c].imshow(Am, cmap=cm, vmin=-v, vmax=v)
            ax[r, c].set_title(f"{n} {t}", fontsize=8)
            ax[r, c].set_xticks([0, 16, 32]); ax[r, c].set_yticks([0, 16, 32])
            fig.colorbar(im, ax=ax[r, c], shrink=0.7)
    fig.suptitle("mid tables, double-centred over finite entries (grey: support INF); own scale per panel", fontsize=10)
    fig.savefig("images/cbio_fit_tables.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # ---------------- stage 5
    O, mo = oracle_moments(SEED + 100)
    OUT["oracle"] = {k: np.asarray(v).tolist() for k, v in mo.items()}
    big(C.render(O.biome, O.obj, O.tile)).save("images/cbio_fit_oracle.png")
    tops = {}
    for kind in ("exact", "ais", "hook"):
        for K in (0, 3):
            tops[f"{kind}_K{K}"] = run_top(kind, K, theta_mid, SEED, mo)
            json.dump(dict(OUT, top=tops), open("images/cbio_stage45.json", "w"), indent=1)
    OUT["top"] = tops
    # eval noise: two independent 64-run evals of one setting's tables
    th = np.array(tops["hook_K0"]["theta"])
    e1, _ = evaluate(all_tables(C, theta_mid, th), RUNS, 901)
    e2, _ = evaluate(all_tables(C, theta_mid, th), RUNS, 902)
    OUT["eval_noise"] = dict(fit=l1(e1, e2, FIT), mon=l1(e1, e2, MON))
    # support-only + mid tables (no top bias) for scale
    e0, _ = evaluate(all_tables(C, theta_mid, None), RUNS, 903)
    OUT["mid_only"] = dict(fit=l1(e0, mo, FIT), mon=l1(e0, mo, MON))
    print("eval noise", OUT["eval_noise"], "\nmid only", OUT["mid_only"], flush=True)
    OUT["seconds"] = time.time() - T0
    json.dump(OUT, open("images/cbio_stage45.json", "w"), indent=1)
    print(f"wrote images/cbio_stage45.json  ({OUT['seconds']:.0f}s)")
