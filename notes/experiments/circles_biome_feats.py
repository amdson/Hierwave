"""Stage 4c of the biome circles test (notes/circles_biome_stage4c.md): do we
need the derived stamp features, or do learned value embeddings generalise?

Mid level (obj, D = 33), the stage 4 setup: CirclesBiome(6, 6), default
dials, contexts = the forward chain at the current fit's installed tables,
targets ExactTargets(Oracle.mid_probs), K = 0, ITERS iterations x NCTX
contexts, holdout 0.1, l2 1e-4.  Feature sets:
  tabular   train.tabular_feats(obj_h, obj_v, obj_d1, obj_d2, obj_u)  (train.fit)
  stamp     Bench.mid_feats, 68 parameters                             (train.fit)
  embed k   psi = u[c] + sum_d e(c)^T A_d e(nb_d)  (embed_fit.bootstrap / fit_embed;
            e(absent) learned, off-grid = a zero row: the tabular rows' pad -1
            convention, so both carry the world-edge discount the same way;
            tables folded to materialise's gauge by canon)
  mixed     stamp . theta + the bilinear term with k = 4
  mlp       psi = u[c] + e(c) . MLP(e(nb_1..8)) (optional: MLP > 0); contexts
            from the pairwise projection of psi (single-neighbour contexts)
Evaluated by held-out KL (own records and a common test set at the stamp fit's
tables), materialised tables against reference() (double-centred over finite
entries) on pairs never seen adjacent, and the strict never-seen-family
holdout (biomes forced to admit one family).  Top (biome, D = 4): tabular and
embed k in {2, 4} with the exact collapsed hook targets (Bench.top_hook) and
the stamp fit's mid tables installed; held-out KL only.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/circles_biome_feats.py

Env: ITERS (5), NCTX (8), SEED (0), KS (4,8,16), RESTARTS (3), STEPS (2000),
POLISH (2000), MLP (32; 0 skips), TOP (1), NTEST (4), HRESTARTS (1: random
restarts at the last iteration of a family-holdout fit), ROWS (debug filter,
e.g. embed8,mixed4).  Outputs
images/cbio_stage4c.json, images/cbio_feats.png."""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from castlegen.channels import train, embed_fit as ef
from castlegen.channels.aistargets import AISTargets
from castlegen.channels.biome_fit import BIO, Bench, all_tables, materialise, mid_tables
from castlegen.channels.circles_biome import DIR8, OFFNAMES, OFFS, CirclesBiome, dcentre_finite
from castlegen.channels.targets import ExactTargets

E = os.environ.get
ITERS, NCTX, SEED = int(E("ITERS", 5)), int(E("NCTX", 8)), int(E("SEED", 0))
KS = [int(k) for k in E("KS", "4,8,16").split(",")]
RESTARTS, STEPS, POLISH, MLP = int(E("RESTARTS", 3)), int(E("STEPS", 2000)), int(E("POLISH", 2000)), int(E("MLP", 32))
TOP, NTEST, L2, HRESTARTS = int(E("TOP", 1)), int(E("NTEST", 4)), 1e-4, int(E("HRESTARTS", 1))
INF = np.inf
C = CirclesBiome(6, 6)
REF = C.reference()
D = C.D
NAMES = OFFNAMES + ["obj_u"]
SHAPES = {n: (D, D) for n in OFFNAMES} | {"obj_u": (D,)}
DIRS8 = [(OFFS.index(d), True) if d in OFFS else (OFFS.index((-d[0], -d[1])), False) for d in DIR8]
DIR4, TOFFS = [(-1, 0), (0, -1), (0, 1), (1, 0)], [(0, 1), (1, 0)]
DIRS4 = [(TOFFS.index(d), True) if d in TOFFS else (TOFFS.index((-d[0], -d[1])), False) for d in DIR4]
os.makedirs("images", exist_ok=True)


def nb_of(g, i, j, dirs):
    return np.array([g[i + dy, j + dx] if 0 <= i + dy < g.shape[0] and 0 <= j + dx < g.shape[1] else -1
                     for dy, dx in dirs], np.int64)


# ------------------------------------------------------------ tables
def stamp_full(d, th):
    out = np.zeros((d.N, d.D))
    out[d.R, d.Cf] = (d.XS @ th)[d.R, d.Cc]
    return out


def canon(G, u):
    """Raw pairwise tables (pad -1) -> materialise's gauge (u[0] = 0, g[0, .] =
    g[., 0] = 0): an isolated interior object's unary, pairs with the absent
    rows folded out."""
    G = np.asarray(G, float)
    un = np.array(u, float)
    for o, f in DIRS8:
        un += (G[o] if f else G[o].T)[:, 0]
    out = dict(obj_u=un - un[0])
    for o, n in enumerate(OFFNAMES):
        g = G[o]
        out[n] = g - g[:, :1] - g[:1, :] + g[0, 0]
    return out


def to_theta(G, u):
    t = C.theta0()
    t.update({n: np.array(G[o]) for o, n in enumerate(OFFNAMES)}, obj_u=np.array(u))
    return t


def mlp_project(p):
    """Pairwise projection of a non-pairwise psi: u[t] = psi(t | absent
    neighbours) - psi(0 | ...), g_d[t, t'] from the context with t' alone at
    +d (forward slots only)."""
    recs = [(np.arange(D), np.zeros(D), np.ones(D) / D, np.zeros(8, np.int64))]
    for o in range(4):
        j = DIRS8.index((o, True))
        for tp in range(D):
            nb = np.zeros(8, np.int64); nb[j] = tp
            recs.append((np.arange(D), np.zeros(D), np.ones(D) / D, nb))
    P = ef.psi(p, ef.Data(recs, D), DIRS8)
    P = P - P[:, :1]
    u = P[0]
    G = np.stack([(P[1 + o * D:1 + (o + 1) * D] - u[None]).T for o in range(4)])
    return G, u


# ----------------------------------------------------------- metrics
def pair_metrics(T, seen):
    with np.errstate(all="ignore"):
        return _pair_metrics(T, seen)


def _pair_metrics(T, seen):
    out = {}
    for k, (n, d) in enumerate(zip(OFFNAMES, OFFS)):
        L = dcentre_finite(np.where(C.CONF[d], INF, T[n]))
        R = REF[n]
        fin = np.isfinite(R)
        fin[0, :] = fin[:, 0] = False
        held = fin & ~seen[k]
        err = np.abs(L - R)
        a, b, ha, hb = L[fin], R[fin], L[held], R[held]
        out[n] = dict(maxerr=float(err[fin].max()), held_n=int(held.sum()),
                      held_inter_n=int((held & (C.PAIR[d] != 0)).sum()),
                      held_maxerr=float(err[held].max()) if held.any() else float("nan"),
                      held_slope=float(ha @ hb / (hb @ hb)) if held.any() and hb @ hb > 0 else float("nan"),
                      held_corr=float(np.corrcoef(ha, hb)[0, 1]) if held.sum() > 2 and hb.std() > 0 else float("nan"),
                      slope=float(a @ b / (b @ b)), corr=float(np.corrcoef(a, b)[0, 1]))
    u = T["obj_u"] + C.pres_e
    out["obj_u_maxerr"] = float(np.abs(u - u.mean() - REF["obj_u"]).max())
    hm = [out[n]["held_maxerr"] for n in OFFNAMES if np.isfinite(out[n]["held_maxerr"])]
    out["held_maxerr"] = max(hm) if hm else float("nan")
    out["maxerr"] = max(out[n]["maxerr"] for n in OFFNAMES)
    # pooled held-out slope / corr over the four offsets (vs reference)
    a, b = [], []
    for k, (n, d) in enumerate(zip(OFFNAMES, OFFS)):
        L = dcentre_finite(np.where(C.CONF[d], INF, T[n])); R = REF[n]
        fin = np.isfinite(R); fin[0, :] = fin[:, 0] = False
        h = fin & ~seen[k]
        a.append(L[h]); b.append(R[h])
    a, b = np.concatenate(a), np.concatenate(b)
    out["held_slope"] = float(a @ b / (b @ b)) if len(b) and b @ b > 0 else float("nan")
    out["held_corr"] = float(np.corrcoef(a, b)[0, 1]) if len(b) > 2 and b.std() > 0 else float("nan")
    out["held_n"] = int(len(b))
    out["held_ref_maxabs"] = float(np.abs(b).max()) if len(b) else float("nan")
    return out


def family_metrics(T, fam):
    with np.errstate(all="ignore"):
        return _family_metrics(T, fam)


def _family_metrics(T, fam):
    other = C.FAM != fam + 1
    other[0] = False
    out = {}
    for n, d in zip(OFFNAMES, OFFS):
        L = dcentre_finite(np.where(C.CONF[d], INF, T[n]))
        R = REF[n]
        fin = np.isfinite(R)
        fin[0, :] = fin[:, 0] = False
        m = fin & (other[:, None] | other[None, :])
        out[n] = float(np.abs(L - R)[m].max())
    u = T["obj_u"] + C.pres_e
    out["pairs_maxerr"] = max(out[n] for n in OFFNAMES)
    out["obj_u_unseen_maxerr"] = float(np.abs(u - u.mean() - REF["obj_u"])[other].max())
    return out


# -------------------------------------------------------------- rows
class Marking:
    """A feats callback that marks (candidate, neighbour) pairs in Bench.seen
    while the context's budget lasts (Bench.mid_feats' bookkeeping)."""

    def __init__(self, B, f):
        self.B, self.f = B, f

    def __call__(self, model, home, i, j, cand):
        if self.B._budget > 0:
            self.B._budget -= 1
            self.B._mark(i, j, cand)
        return self.f(model, home, i, j, cand)


def make_gen(B, tables_of, fam=None):
    if fam is None:
        return B.contexts(tables_of, "obj", 0)
    fw = B.fw

    def gen(theta, n, rng):                             # stage 4's strict holdout contexts
        for _ in range(n):
            fw.theta = tables_of(theta)
            B.biome.grid[:] = 1 << fam
            B.obj.grid[:] = 0
            B.tile.grid[:] = 0
            fw.run(0, 30, 20, fresh=False)
            B._budget = int((~B.obj.fixed).sum())
            yield fw.model
    return gen


def fit_row(spec, seed, fam=None, theta_stamp=None):
    """One feature set on a fresh Bench.  Returns (canonical tables or None,
    psi_fn(Data) for the test set, raw for the projection, report, nparam,
    seen)."""
    kind, k = spec
    B = Bench(C, seed=seed)
    de = B.designed_e("obj")
    targets = ExactTargets(lambda m, h, i, j: B.orc.mid_probs(i, j))
    if kind in ("tabular", "stamp"):
        if kind == "tabular":
            tf = train.tabular_feats(NAMES)
            feats, tables_of = Marking(B, tf), lambda th: mid_tables(C) if th is None else \
                to_theta(*(lambda t: (np.stack([t[n] for n in OFFNAMES]), t["obj_u"]))(train.unpack(th, SHAPES)))
        else:
            feats, tables_of = B.mid_feats, lambda th: mid_tables(C, th)
        th, rep = train.fit(make_gen(B, tables_of, fam), "obj", targets, feats, de, iters=ITERS, n_contexts=NCTX,
                            K=0, repaint=B.mid_paint, rng=seed, n_pairs=8 if fam is None else 0, l2=L2, verbose=True)
        if kind == "tabular":
            t = train.unpack(th, SHAPES)
            G, u = np.stack([t[n] for n in OFFNAMES]), t["obj_u"]
            return canon(G, u), lambda d: ef.tables_energy(G, u, d, DIRS8), th, rep, len(th), B.seen
        return materialise(C, th), lambda d: stamp_full(d, th), th, rep, len(th), B.seen
    stamp = kind == "mixed"
    kw = dict(nf=68 if stamp else 0, mlp=MLP if kind == "mlp" else 0, theta0=theta_stamp if stamp else None,
              scale=0.1 if stamp else None, fix0=False)

    def tables_of(p):
        if p is None:
            return mid_tables(C, theta_stamp if stamp else None)
        G, u = mlp_project(p) if kind == "mlp" else ef.tables(p)
        t = to_theta(G, u)
        if stamp:
            m = materialise(C, p["theta"])
            for n in NAMES:
                t[n] = t[n] + m[n]
        return t

    def ctx(model, home, i, j, cand):
        X = B.mid_feats(model, home, i, j, cand) if stamp else None
        if not stamp and B._budget > 0:
            B._budget -= 1
            B._mark(i, j, cand)
        return nb_of(B.obj.grid, i, j, DIR8), X

    it = [0]

    def fitter(d, prev, held):
        it[0] += 1
        last = it[0] == ITERS
        R = RESTARTS if fam is None else HRESTARTS
        p, info = ef.fit_embed(d, D, k, DIRS8, 4, l2=L2, restarts=R if (last or prev is None) else 0,
                               steps=STEPS if last else STEPS // 2, polish=POLISH if last else POLISH // 2,
                               seed=seed + it[0], init_p=prev, **kw)
        kt = ef.kl(ef.psi(p, d, DIRS8), d)[0]
        kh = ef.kl(ef.psi(p, held, DIRS8), held)[0] if held is not None else float("nan")
        print(f"    starts {np.round(info['losses'], 5)}  {info['seconds']:.1f}s", flush=True)
        fitter.info = info
        return p, kt, kh

    p, rep = ef.bootstrap(make_gen(B, tables_of, fam), "obj", targets, ctx, de, fitter, ITERS, NCTX, K=0,
                          rng=seed, verbose=True)
    rep.pop("held")
    rep["restart_losses"] = fitter.info["losses"]
    nparam = ef.n_params(p, fix0=False)
    if kind == "mlp":
        return None, lambda d: ef.psi(p, d, DIRS8), p, rep, nparam, B.seen
    G, u = ef.tables(p)
    T = canon(G, u)
    if stamp:
        m = materialise(C, p["theta"])
        T = {n: T[n] + m[n] for n in NAMES}
        Tb = canon(G, u)                            # the learned part alone, materialise's gauge
        with np.errstate(all="ignore"):
            rep["bilinear_part_dc_maxabs"] = float(max(
                np.nanmax(np.abs(np.where(np.isfinite(R := dcentre_finite(np.where(C.CONF[d], INF, Tb[n]))), R, np.nan)))
                for n, d in zip(OFFNAMES, OFFS)))
        rep["bilinear_part_u_maxabs"] = float(np.abs(Tb["obj_u"] - Tb["obj_u"].mean()).max())
        rep["theta_shift_maxabs"] = float(np.abs(p["theta"] - theta_stamp).max())
    return T, lambda d: ef.psi(p, d, DIRS8), p, rep, nparam, B.seen


def test_set(theta_stamp, seed):
    """Records (cand, e, pi, nb, X_stamp) at every active site of NTEST fresh
    forward contexts at the stamp fit's tables."""
    B = Bench(C, seed=seed)
    de = B.designed_e("obj")
    recs = []
    for _ in range(NTEST):
        B.fw.theta = mid_tables(C, theta_stamp)
        B.fw.run(*B.S, fresh=True)
        m = B.fw.model
        for y, x in train._site_order(m, "obj")[0]:
            e = de(m, "obj", y, x, np.arange(D))
            cand = np.flatnonzero(np.isfinite(e))
            if len(cand) > 1:
                pi = B.orc.mid_probs(y, x)[cand]
                recs.append((cand, e[cand], pi / pi.sum(), nb_of(B.obj.grid, y, x, DIR8),
                             C.stamp_features(B.obj, y, x, cand).astype(float)))
    return ef.Data(recs, D)


# --------------------------------------------------------------- top
def run_top(kind, k, theta_stamp, seed):
    B = Bench(C, seed=seed)
    de = B.designed_e("biome")
    tab0 = all_tables(C, theta_stamp, None)
    targets = AISTargets("obj", B.top_region, de, B.top_dormancy, hook=B.top_hook(tab0))
    t0 = time.time()
    if kind == "tabular":
        th, rep = train.fit(B.contexts(lambda th: all_tables(C, theta_stamp, th), "biome", 0), "biome", targets,
                            train.tabular_feats(BIO), de, iters=ITERS, n_contexts=NCTX, K=0, repaint=B.top_paint,
                            rng=seed, n_pairs=0, l2=L2, verbose=True)
        return dict(kind=kind, nparam=len(th), kl_train=rep["kl_train"][-1], kl_held=rep["kl_held"][-1],
                    seconds=time.time() - t0)

    def tables_of(p):
        t = all_tables(C, theta_stamp, None)
        if p is not None:
            G, u = ef.tables(p)
            t.update(bio_h=G[0], bio_v=G[1], bio_u=u)
        return t

    def fitter(d, prev, held):
        p, _ = ef.fit_embed(d, C.P, k, DIRS4, 2, l2=L2, restarts=RESTARTS, steps=STEPS, polish=POLISH, seed=seed,
                            init_p=prev, fix0=False)
        return p, ef.kl(ef.psi(p, d, DIRS4), d)[0], ef.kl(ef.psi(p, held, DIRS4), held)[0] if held else float("nan")

    p, rep = ef.bootstrap(B.contexts(tables_of, "biome", 0), "biome", targets,
                          lambda m, h, I, J, c: (nb_of(B.biome.grid, I, J, DIR4), None), de, fitter, ITERS, NCTX,
                          rng=seed, verbose=True)
    return dict(kind=kind, k=k, nparam=ef.n_params(p, fix0=False), kl_train=rep["kl_train"][-1],
                kl_held=rep["kl_held"][-1], seconds=time.time() - t0)


# --------------------------------------------------------------- main
if __name__ == "__main__":
    T0 = time.time()
    OUT = dict(config=dict(ITERS=ITERS, NCTX=NCTX, SEED=SEED, KS=KS, RESTARTS=RESTARTS, STEPS=STEPS, POLISH=POLISH,
                           MLP=MLP, NTEST=NTEST, l2=L2, K=0, holdout=0.1), rows={})
    specs = [("stamp", 0), ("tabular", 0)] + [("embed", k) for k in KS] + [("mixed", 4)] + \
        ([("mlp", 8)] if MLP else [])
    if E("ROWS"):                                      # debugging: stamp always runs (test set, mixed init)
        specs = [s for s in specs if s[0] == "stamp" or f"{s[0]}{s[1]}" in E("ROWS").split(",")]
    tabs, th_stamp, TEST, th_fam = {}, None, None, {}
    for spec in specs:
        name = spec[0] + (f" k={spec[1]}" if spec[0] in ("embed", "mixed", "mlp") else "")
        print(f"== {name}", flush=True)
        t0 = time.time()
        T, psi_fn, par, rep, nparam, seen = fit_row(spec, SEED, theta_stamp=th_stamp)
        sec = time.time() - t0
        if spec[0] == "stamp":
            th_stamp = par
            TEST = test_set(th_stamp, SEED + 50)
            print(f"   test set: {TEST.N} records", flush=True)
        row = dict(nparam=int(nparam), seconds=sec, kl_train=rep["kl_train"][-1], kl_held=rep["kl_held"][-1],
                   n_records=rep["n_records"][-1], kl_test=float(ef.kl(psi_fn(TEST), TEST)[0]),
                   report={kk: v for kk, v in rep.items() if kk not in ("kl_train", "kl_held")})
        if T is not None:
            row["pairs"] = pair_metrics(T, seen)
            tabs[name] = T
        else:                                          # mlp: its pairwise projection, for reference only
            row["projection_pairs"] = pair_metrics(dict(zip(NAMES, (lambda G, u: list(G) + [u])(*mlp_project(par)))), seen)
        fams = {}
        for f in range(C.NF):
            print(f"  -- holdout: contexts admit {C.names[f]} only", flush=True)
            Tf, pf, pa, rf, _, _ = fit_row(spec, SEED + 3, fam=f, theta_stamp=th_fam.get(f))
            if spec[0] == "stamp":
                th_fam[f] = pa                         # the mixed holdout starts from the holdout's own stamp fit
            fams[C.names[f]] = dict(kl_test=float(ef.kl(pf(TEST), TEST)[0]), kl_held=rf["kl_held"][-1],
                                    **(family_metrics(Tf, f) if Tf is not None else {}))
        row["family_holdout"] = fams
        row["seconds_with_holdout"] = time.time() - t0
        pm = row.get("pairs", {})
        print(f"   {name}: params {nparam}  KL train {row['kl_train']:.2e} held {row['kl_held']:.2e} test "
              f"{row['kl_test']:.2e}  held-out pairs n {pm.get('held_n')} max err {pm.get('held_maxerr', np.nan):.4f} "
              f"slope {pm.get('held_slope', np.nan):.3f} corr {pm.get('held_corr', np.nan):.3f}  obj_u "
              f"{pm.get('obj_u_maxerr', np.nan):.4f}  family {fams}  {sec:.0f}s", flush=True)
        OUT["rows"][name] = row
        json.dump(OUT, open("images/cbio_stage4c.json", "w"), indent=1, default=float)
    if TOP:
        OUT["top"] = {}
        for kind, k in [("tabular", 0), ("embed", 2), ("embed", 4)]:
            print(f"== top {kind} {k}", flush=True)
            OUT["top"][f"{kind}" + (f" k={k}" if k else "")] = r = run_top(kind, k, th_stamp, SEED)
            print("  ", r, flush=True)
            json.dump(OUT, open("images/cbio_stage4c.json", "w"), indent=1, default=float)
    OUT["seconds"] = time.time() - T0
    json.dump(OUT, open("images/cbio_stage4c.json", "w"), indent=1, default=float)

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(15, 6.6))
    ax = fig.add_subplot(1, 3, 1)
    for name, r in OUT["rows"].items():
        n = r["nparam"]
        if "pairs" in r:
            ax.scatter(n, max(r["pairs"]["held_maxerr"], 1e-5), c="C0", marker="o")
            ax.annotate(name, (n, max(r["pairs"]["held_maxerr"], 1e-5)), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        fe = [v.get("pairs_maxerr") for v in r["family_holdout"].values() if "pairs_maxerr" in v]
        if fe:
            ax.scatter(n, max(max(fe), 1e-5), c="C3", marker="x")
    ax.scatter([], [], c="C0", marker="o", label="held-out pairs (seen values)")
    ax.scatter([], [], c="C3", marker="x", label="never-seen family pairs")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameters"); ax.set_ylabel("max abs error vs reference (double-centred)")
    ax.legend(fontsize=7); ax.set_title("mid tables: generalisation vs size", fontsize=9)
    show = [("reference", REF)] + [(k, tabs[k]) for k in ("embed k=8", "stamp") if k in tabs]
    for r, n in enumerate(("obj_h", "obj_v")):
        for c, (t, A) in enumerate(show):
            A = A[n] if t == "reference" else dcentre_finite(np.where(C.CONF[OFFS[OFFNAMES.index(n)]], INF, A[n]))
            Am = np.where(np.isfinite(A), A, np.nan)
            v = np.nanmax(np.abs(REF[n][np.isfinite(REF[n])]))
            cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.6")
            a = fig.add_subplot(2, 6, 6 * r + 3 + c + (c > 0) * 0)
            a.imshow(Am, cmap=cm, vmin=-v, vmax=v)
            a.set_title(f"{n} {t}", fontsize=8); a.set_xticks([0, 16, 32]); a.set_yticks([0, 16, 32])
    fig.suptitle("stage 4c: feature sets (tables double-centred over finite entries, reference scale; grey INF)",
                 fontsize=10)
    fig.savefig("images/cbio_feats.png", dpi=100, bbox_inches="tight")
    print(f"wrote images/cbio_stage4c.json, images/cbio_feats.png  ({OUT['seconds']:.0f}s)")
