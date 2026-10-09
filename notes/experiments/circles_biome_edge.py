"""Stage 5b of notes/circles_biome_test.md: is the world boundary the whole
story behind the failed stage 5 end-to-end gate?  Results in
notes/circles_biome_stage5b.md.

Part 1 (open world, the stage 5 potentials: exact-K0 mid theta, hook-K0 top
theta from images/cbio_stage45.json): every fitted feature and monitor over
the whole world and over the interior (top cells, slots and tiles at least
one block of their own level from the edge; "deep": slots and tiles inside
the interior top cells), oracle (BURN + SWEEPS sweeps) against the forward
(RUNS runs), eval noise = two independent RUNS-run forward evals, oracle noise
= two independent oracle chains.  Seeds as in stage 5, so the "all" row
reproduces the stage 5 hook K0 row.
Part 2 (torus, CirclesBiome(periodic=True)): the stage 4 exact-K0 mid fit and
the stage 5 hook-K0 top fit rerun on the torus, then the same end-to-end
evaluation; also the open-world thetas installed on the torus, and the final
setting at S_T = S_M = 100.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/circles_biome_edge.py

Env: NT (6), ITERS (5), NCTX (8), RUNS (64), BURN (50), SWEEPS (400), SEED (0).
Outputs images/cbio_stage5b.json, images/cbio_edge_{oracle,forward,forward_open}.png."""
import os, json, time
import numpy as np
from PIL import Image

from castlegen.channels import train
from castlegen.channels.aistargets import AISTargets
from castlegen.channels.biome_fit import BIO, Bench, all_tables, materialise, mid_tables
from castlegen.channels.circles_biome import OFFNAMES, OFFS, CirclesBiome, Forward, Oracle, dcentre_finite
from castlegen.channels.targets import ExactTargets

E = os.environ.get
NT, ITERS, NCTX = int(E("NT", 6)), int(E("ITERS", 5)), int(E("NCTX", 8))
RUNS, BURN, SWEEPS, SEED = int(E("RUNS", 64)), int(E("BURN", 50)), int(E("SWEEPS", 400)), int(E("SEED", 0))
FIT = OFFNAMES + ["obj_u", "bio_h", "bio_v", "bio_u"]
MON = ["present_disc", "present_bar", "dormant", "contact", "bio_hist", "edge_air_top", "conflict", "mask_viol"]
INF = np.inf
big = lambda a: Image.fromarray(a).resize((a.shape[1] * 4, a.shape[0] * 4), Image.NEAREST)
C0 = CirclesBiome(NT, NT)                       # open world; also the materialiser (tables do not depend on the world)
CP = CirclesBiome(NT, NT, periodic=True)
REF = C0.reference()
S45 = json.load(open("images/cbio_stage45.json"))
TH_MID0, TH_TOP0 = np.array(S45["mid"]["exact_K0"]["theta"]), np.array(S45["top"]["hook_K0"]["theta"])
OUT = dict(config=dict(NT=NT, ITERS=ITERS, NCTX=NCTX, RUNS=RUNS, BURN=BURN, SWEEPS=SWEEPS, S=(30, 30, 20)))


def variants(C):
    if C.periodic:
        return dict(all=C.stats)
    return dict(all=C.stats, int=lambda b, o, t: C.stats_region(b, o, t, 1),
                deep=lambda b, o, t: C.stats_region(b, o, t, 1, deep=True))


def edge_presence(C, og):
    """(presence on the slots of the outer ring, presence on the others) of the world (torus: its first/last rows and cols)."""
    g = C.inner(og) > 0
    e = np.zeros(g.shape, bool)
    e[0] = e[-1] = True
    e[:, 0] = e[:, -1] = True
    return np.array([g[e].mean(), g[~e].mean()])


def acc_add(acc, C, b, o, t):
    for v, f in variants(C).items():
        s = f(b, o, t)
        acc[v] = s if v not in acc else {k: acc[v][k] + s[k] for k in s}
    s = edge_presence(C, o.grid)
    acc["edge_int_presence"] = s if "edge_int_presence" not in acc else acc["edge_int_presence"] + s


def acc_mean(acc, n):
    return {v: ({k: np.asarray(x) / n for k, x in a.items()} if isinstance(a, dict) else a / n) for v, a in acc.items()}


def oracle_moments(C, seed, sweeps=None):
    sweeps = SWEEPS if sweeps is None else sweeps
    O = Oracle(C, seed=seed)
    t0 = time.time()
    O.sweep(BURN)
    acc = {}
    for _ in range(sweeps):
        O.sweep(1)
        acc_add(acc, C, O.biome, O.obj, O.tile)
    print(f"oracle seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return acc_mean(acc, sweeps), O


def evaluate(C, tab, runs, seed, S=(30, 30, 20)):
    fw = Forward(C, tab, seed=seed)
    acc = {}
    for _ in range(runs):
        fw.run(*S, fresh=True)
        acc_add(acc, C, fw.biome, fw.obj, fw.tile)
    return acc_mean(acc, runs), fw


def l1(a, b, keys=FIT + MON):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def compare(ev, mo, n1, n2, on):
    """{variant: {L1, noise, oracle_noise, pass}} for every variant."""
    out = {}
    for v in mo:
        if v == "edge_int_presence":
            continue
        L, N, ON = l1(ev[v], mo[v]), l1(n1[v], n2[v]), l1(on[v], mo[v])
        out[v] = dict(L1=L, noise=N, oracle_noise=ON,
                      fit_within_noise={k: bool(L[k] <= N[k]) for k in FIT},
                      passed=bool(all(L[k] <= N[k] for k in FIT) and ev[v]["conflict"] == 0))
    out["edge_int_presence"] = dict(forward=ev["edge_int_presence"].tolist(), oracle=mo["edge_int_presence"].tolist())
    return out


def show(title, cmp):
    print(f"== {title}", flush=True)
    for v, r in cmp.items():
        if v == "edge_int_presence":
            print("   presence edge / interior: forward", np.round(r["forward"], 3), "oracle", np.round(r["oracle"], 3))
            continue
        if not isinstance(r, dict) or "passed" not in r:
            continue
        print(f"   {v:5s} passed {r['passed']}", flush=True)
        for nm in ("L1", "noise", "oracle_noise"):
            print(f"     {nm:12s}", " ".join(f"{k}={r[nm][k]:.3f}" for k in FIT + MON), flush=True)


# ------------------------------------------------------------ fits on the torus (copied from circles_biome_fit.py)
def run_mid(C, seed):
    B = Bench(C, seed=seed)
    tables_of = lambda th: mid_tables(C0, th)
    de = B.designed_e("obj")
    targets = ExactTargets(lambda m, h, i, j: B.orc.mid_probs(i, j))
    print("== mid exact K 0 (torus)", flush=True)
    th, rep = train.fit(B.contexts(tables_of, "obj", 0), "obj", targets, B.mid_feats, de, iters=ITERS,
                        n_contexts=NCTX, K=0, after_change=B.mid_move, repaint=B.mid_paint, rng=seed, n_pairs=8,
                        verbose=True)
    T = materialise(C0, th)
    err = {}
    for n, d in zip(OFFNAMES, OFFS):
        L, R = dcentre_finite(np.where(C0.CONF[d], INF, T[n])), REF[n]
        fin = np.isfinite(R)
        fin[0, :] = fin[:, 0] = False
        err[n] = float(np.abs(L[fin] - R[fin]).max())
    u = T["obj_u"] + C0.pres_e
    err["obj_u"] = float(np.abs(u - u.mean() - REF["obj_u"]).max())
    T0 = materialise(C0, TH_MID0)
    res = dict(theta=th.tolist(), maxerr=err, seconds=rep["seconds_total"], n_records=rep["n_records"][-1],
               viol_targets=rep["viol_targets"][-1], viol_fit=rep["viol_fit"][-1],
               table_maxdiff_vs_open={n: float(np.abs(T[n] - T0[n])[~C0.CONF[OFFS[OFFNAMES.index(n)]]].max())
                                      if n in OFFNAMES else float(np.abs(T[n] - T0[n]).max()) for n in T})
    print("   maxerr", {k: round(v, 5) for k, v in err.items()}, " vs open-world tables",
          {k: round(v, 5) for k, v in res["table_maxdiff_vs_open"].items()}, flush=True)
    return res, th


def run_top(C, theta_mid, seed):
    B = Bench(C, seed=seed)
    tables_of = lambda th: all_tables(C0, theta_mid, th)
    de = B.designed_e("biome")
    targets = AISTargets("obj", B.top_region, de, B.top_dormancy, hook=B.top_hook(tables_of(None)), seed=seed)
    print("== top hook K 0 (torus)", flush=True)
    th, rep = train.fit(B.contexts(tables_of, "biome", 0), "biome", targets, train.tabular_feats(BIO), de, iters=ITERS,
                        n_contexts=NCTX, K=0, after_change=B.top_move, repaint=B.top_paint, rng=seed, n_pairs=2,
                        verbose=True)
    return dict(theta=th.tolist(), seconds=rep["seconds_total"], n_records=rep["n_records"][-1],
                kl_held=rep["kl_held"][-1], viol_targets=rep["viol_targets"][-1], viol_fit=rep["viol_fit"][-1]), th


def bio_u_none0(theta_top):
    t = all_tables(C0, TH_MID0, theta_top)["bio_u"]
    return (t - t[0]).tolist()


def js(x):
    if isinstance(x, dict):
        return {k: js(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, list, tuple)):
        return np.asarray(x).tolist()
    return x


def save():
    json.dump(js(OUT), open("images/cbio_stage5b.json", "w"), indent=1)


if __name__ == "__main__":
    T0 = time.time()
    # ---------------- part 1: open world, interior-only evaluation
    mo, O = oracle_moments(C0, SEED + 100)
    on, _ = oracle_moments(C0, SEED + 200)
    tab = all_tables(C0, TH_MID0, TH_TOP0)
    ev, fw = evaluate(C0, tab, RUNS, SEED + 11)
    n1, _ = evaluate(C0, tab, RUNS, 901)
    n2, _ = evaluate(C0, tab, RUNS, 902)
    OUT["open"] = compare(ev, mo, n1, n2, on)
    OUT["open"]["bio_hist"] = dict(oracle=mo["all"]["bio_hist"], forward=ev["all"]["bio_hist"],
                                   oracle_int=mo["int"]["bio_hist"], forward_int=ev["int"]["bio_hist"])
    OUT["open"]["moments"] = {s: {v: {k: mm[v][k] for k in MON} for v in ("all", "int", "deep")}
                              for s, mm in (("oracle", mo), ("forward", ev))}
    show("open world, stage 5 potentials", OUT["open"])
    big(C0.render(fw.biome, fw.obj, fw.tile)).save("images/cbio_edge_forward_open.png")
    save()
    # ---------------- part 2: the torus
    mp, OP = oracle_moments(CP, SEED + 100)
    onp, _ = oracle_moments(CP, SEED + 200)
    OUT["torus_oracle_vs_open_interior"] = l1(mp["all"], mo["int"])
    big(CP.render(OP.biome, OP.obj, OP.tile)).save("images/cbio_edge_oracle.png")
    rm, thm = run_mid(CP, SEED)
    OUT["torus_mid"] = rm
    save()
    rt, tht = run_top(CP, thm, SEED)
    rt["bio_u_none0"], rt["bio_u_none0_open"] = bio_u_none0(tht), bio_u_none0(TH_TOP0)
    OUT["torus_top"] = rt
    print("   bio_u (none = 0) torus", np.round(rt["bio_u_none0"], 3), "open", np.round(rt["bio_u_none0_open"], 3), flush=True)
    save()
    tabp = all_tables(C0, thm, tht)
    evp, fwp = evaluate(CP, tabp, RUNS, SEED + 11)
    n1p, _ = evaluate(CP, tabp, RUNS, 901)
    n2p, _ = evaluate(CP, tabp, RUNS, 902)
    OUT["torus"] = compare(evp, mp, n1p, n2p, onp)
    OUT["torus"]["bio_hist"] = dict(oracle=mp["all"]["bio_hist"], forward=evp["all"]["bio_hist"])
    OUT["torus"]["moments"] = {s: {k: mm["all"][k] for k in MON} for s, mm in (("oracle", mp), ("forward", evp))}
    show("torus, torus fits", OUT["torus"])
    big(CP.render(fwp.biome, fwp.obj, fwp.tile)).save("images/cbio_edge_forward.png")
    save()
    # open-world thetas on the torus
    e0, _ = evaluate(CP, tab, RUNS, SEED + 13)
    OUT["torus_open_thetas"] = dict(L1=l1(e0["all"], mp["all"]))
    # sweep budget: S_T = S_M = 100
    e100, _ = evaluate(CP, tabp, RUNS, SEED + 17, S=(100, 100, 20))
    OUT["torus_S100"] = dict(L1=l1(e100["all"], mp["all"]), L1_vs_S30=l1(e100["all"], evp["all"]))
    # a second independent forward eval of the final setting against the oracle (stability of the gate call)
    e2b, _ = evaluate(CP, tabp, RUNS, SEED + 19)
    OUT["torus_repeat"] = dict(L1=l1(e2b["all"], mp["all"]))
    for k in ("torus_open_thetas", "torus_S100", "torus_repeat"):
        print(f"== {k}", " ".join(f"{a}={b:.3f}" for a, b in OUT[k]["L1"].items()), flush=True)
    print("   S100 vs S30", " ".join(f"{a}={b:.3f}" for a, b in OUT["torus_S100"]["L1_vs_S30"].items()), flush=True)
    # residual vs bias: NREP evals per budget pooled against a long oracle chain
    NREP = int(E("NREP", 4))
    mlong, _ = oracle_moments(CP, SEED + 300, 5 * SWEEPS)
    OUT["torus_long_oracle_vs_oracle"] = l1(mlong["all"], mp["all"])
    for S in ((30, 30, 20), (100, 100, 20)):
        es = [evaluate(CP, tabp, RUNS, 1000 + 10 * r + S[0], S=S)[0]["all"] for r in range(NREP)]
        pool = {k: np.mean([e[k] for e in es], 0) for k in es[0]}
        h = NREP // 2
        a = {k: np.mean([e[k] for e in es[:h]], 0) for k in es[0]}
        b = {k: np.mean([e[k] for e in es[h:]], 0) for k in es[0]}
        OUT[f"torus_pool_S{S[0]}"] = dict(
            L1_each_mean={k: float(np.mean([l1(e, mlong["all"])[k] for e in es])) for k in FIT + MON},
            L1_pool=l1(pool, mlong["all"]), noise_pool=l1(a, b), runs=NREP * RUNS)
        r = OUT[f"torus_pool_S{S[0]}"]
        print(f"== pooled S{S[0]} ({NREP} x {RUNS} runs) vs long oracle ({5 * SWEEPS} sweeps)", flush=True)
        for nm in ("L1_each_mean", "L1_pool", "noise_pool"):
            print(f"     {nm:12s}", " ".join(f"{k}={r[nm][k]:.3f}" for k in FIT + MON), flush=True)
    OUT["seconds"] = time.time() - T0
    save()
    print(f"wrote images/cbio_stage5b.json ({OUT['seconds']:.0f}s)")
