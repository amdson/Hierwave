"""The biome circles test, stage 7: scaling probe (notes/circles_biome_test.md).

N random families (random_families: connected stamps of 6-14 cells in a 5 x 5
box, 16 in-block offsets each) for N in NS, a biome of 8 explicit masks
(none + 7 family pairs, random_pair_masks), nty = ntx = 6, default dials.
Per N: build (support + pair reference tables), oracle 20 + 100 sweeps
(conflict, families present), the Sampler forward at "+bu+ou" (theta0 with
bio_u = reference bio_u, obj_u = F(o) - F(absent); the four obj pair tables
are installed as zero tables, as in stage 3) at K = None and K = 8, and a
cost probe at active fraction 1 (biome set by hand to non-none values):
full rows vs the hard rows alone (mask + support, no soft rows), K None / 8.
Optional (FIT=1): at N = 5 the mid fit with exact targets.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/circles_biome_scale.py

Env: NS (2,5,10,20), NT (6), RUNS (16), S_T, S_M (30), S_F (20), BURN (20),
SWEEPS (100), PROBE (10), FIT (1), SEED.  Outputs images/cbio_stage7.json,
images/cbio_scaling.png, images/cbio_scale20.png.
"""
import os, time, json
import numpy as np
from PIL import Image

from castlegen.channels.core import Model
from castlegen.channels.sampler import Sampler
from castlegen.channels.circles_biome import (CirclesBiome, Forward, Oracle, random_families, random_pair_masks,
                                              OFFNAMES, OFFS, dcentre_finite)

E = os.environ.get
NS = [int(v) for v in E("NS", "2,5,10,20").split(",")]
NT, RUNS = int(E("NT", 6)), int(E("RUNS", 16))
S_T, S_M, S_F = int(E("S_T", 30)), int(E("S_M", 30)), int(E("S_F", 20))
BURN, SWEEPS, PROBE, FIT, SEED = int(E("BURN", 20)), int(E("SWEEPS", 100)), int(E("PROBE", 40)), int(E("FIT", 1)), \
    int(E("SEED", 0))
os.makedirs("images", exist_ok=True)


def png(img, path, px=3):
    Image.fromarray(np.repeat(np.repeat(img, px, 0), px, 1)).save(path)


def side(a, b, gap=6):
    return np.concatenate([a, np.full((a.shape[0], gap, 3), 255, np.uint8), b], axis=1)


def thetas(C):
    ref = C.reference(centred=False)
    th = C.theta0()
    th["bio_u"] = ref["bio_u"]
    th["obj_u"] = ref["obj_u"] - C.pres_e
    return th


def oracle(C):
    O = Oracle(C, seed=SEED + 1)
    t = time.time()
    O.sweep(BURN)
    conf, seen, pres = 0.0, np.zeros(C.NF + 1, bool), []
    for _ in range(SWEEPS):
        O.sweep(1)
        conf = max(conf, float((C.dem_of(O.obj) == 3).mean()))
        seen[np.unique(C.FAM[O.obj.grid])] = True
        pres.append(float((O.obj.grid > 0).mean()))
    adm = np.zeros(C.NF, bool)
    for m in C.MASKS:
        adm |= (int(m) >> np.arange(C.NF)) & 1 == 1
    return O, dict(seconds=time.time() - t, conflict_max=conf, fams_seen=int(seen[1:].sum()),
                   fams_admitted=int(adm.sum()), present=float(np.mean(pres)),
                   bio_hist=(np.bincount(O.biome.grid.ravel(), minlength=C.P) / O.biome.grid.size).tolist())


def forward(C, th, K):
    F = Forward(C, th, seed=SEED + 555, use_sampler=True, K=K)
    F.run(1, 1, 1)                                       # compile
    tms, act, conf, pres, walls = [], [], [], [], []
    for _ in range(RUNS):
        t = time.perf_counter()
        F.run(S_T, S_M, S_F)
        walls.append(time.perf_counter() - t)
        tms.append(dict(F.times))
        act.append(1 - F.dormant_frac)
        conf.append(float((C.dem_of(F.obj) == 3).mean()))
        pres.append(float((F.obj.grid > 0).mean()))
    nslot = C.nmy * C.nmx
    tm = {k: float(np.mean([t[k] for t in tms])) for k in tms[0]}
    tm["us_per_active_site"] = float(np.sum([t["obj"] for t in tms]) / (S_M * nslot * np.sum(act)) * 1e6)
    S = F.S["obj"]
    return F, dict(times=tm, wall=float(np.mean(walls)), dormant=float(1 - np.mean(act)), conflict=float(np.max(conf)),
                   present=float(np.mean(pres)), nrows=int(S.P.fac.shape[0]), nhard=int(S.P.nhard))


def probe(C, th):
    """S_M obj sweeps at active fraction 1 (every biome non-none, obj from
    empty after init): full rows / hard rows only, K None / 8, configurations
    interleaved within each of PROBE reps (one warm-up) so that machine load
    hits all four alike; median and min over reps."""
    rng = np.random.default_rng(SEED + 9)
    F = Forward(C, th, seed=SEED + 3, use_sampler=True)
    F.build()
    hard = Model(C.H, C.W, list(F.chans), [f for f in C.designed_factors() if f.name != "pres"] + C.support_factors())
    S = {f"{n} K={K}": Sampler(m, "obj", K=K, hb=C.BT) for n, m in (("full", F.model), ("hard", hard)) for K in (None, 8)}
    ts = {k: [] for k in S}
    for r in range(PROBE + 1):
        b = rng.integers(1, C.P, (C.nty, C.ntx))
        for k, Sk in S.items():
            F.biome.grid[:] = b
            F.obj.grid[:] = 0
            F.obj.fixed[:] = False
            C.paint_allow(F.biome, F.allow)
            Sk.init()
            t = time.perf_counter()
            Sk.sweep(S_M, seed=int(rng.integers(1 << 30)))
            if r:
                ts[k].append(time.perf_counter() - t)
            assert not (C.dem_of(F.obj) == 3).any()
    n = C.nmy * C.nmx * S_M
    return {k: dict(rows=int(S[k].P.fac.shape[0]), nhard=int(S[k].P.nhard), us=float(np.median(v)) / n * 1e6,
                    us_min=float(np.min(v)) / n * 1e6) for k, v in ts.items()}


def fit_mid(C):
    """train.fit with exact targets, stamp features, K = 0, 3 iters, 4 contexts;
    the materialised pair tables against reference() on value pairs never seen
    as (candidate, neighbour) in the data passes."""
    from castlegen.channels import train
    from castlegen.channels.targets import ExactTargets
    from castlegen.channels.biome_fit import Bench, mid_tables, materialise
    B = Bench(C, seed=SEED + 7)
    tgt = ExactTargets(lambda m, h, y, x: B.orc.mid_probs(y, x))
    t = time.time()
    theta, rep = train.fit(B.contexts(lambda th: mid_tables(C, th), "obj", K=0), "obj", tgt, B.mid_feats,
                           B.designed_e("obj"), iters=3, n_contexts=4, K=0, repaint=B.mid_paint, rng=SEED)
    tab = materialise(C, theta)
    ref = C.reference(centred=True)
    out = dict(seconds=time.time() - t, kl_held=rep["kl_held"], kl_train=rep["kl_train"], nfeat=rep["nfeat"],
               n_records=rep["n_records"])
    errs, xs, ys, nheld = [], [], [], 0
    for k, (n, d) in enumerate(zip(OFFNAMES, OFFS)):
        A = dcentre_finite(np.where(C.CONF[d], np.inf, tab[n]))
        R = ref[n]
        held = np.isfinite(R) & ~B.seen[k]
        held[0, :] = held[:, 0] = False
        nheld += int(held.sum())
        if held.any():
            errs.append(float(np.abs(A[held] - R[held]).max()))
            xs.append(R[held]); ys.append(A[held])
    x, y = np.concatenate(xs), np.concatenate(ys)
    out.update(held_pairs=nheld, max_err_held=float(max(errs)), mean_err_held=float(np.abs(x - y).mean()),
               slope=float(np.polyfit(x, y, 1)[0]), corr=float(np.corrcoef(x, y)[0, 1]),
               ref_range=float(x.max() - x.min()))
    return out


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = [r["N"] for r in R]
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, lab, c, ls in (("fw K=None", "forward, K = None", c1, "-"), ("fw K=8", "forward, K = 8", c2, "-")):
        ax[0].plot(Ns, [r[key]["times"]["us_per_active_site"] for r in R], ls, color=c, lw=2, marker="o", ms=6,
                   label=lab)
    ax[0].plot(Ns, [r["probe"]["hard K=8"]["us"] for r in R], "--", color=c3, lw=2, marker="s", ms=6,
               label="hard rows only, K = 8 (probe)")
    ax[0].set(xscale="log", yscale="log", xlabel="families N", ylabel="us per active obj site per sweep",
              title="obj sweep cost")
    ax[0].set_yticks([1, 2, 4, 8]); ax[0].set_yticklabels(["1", "2", "4", "8"])
    ax[1].plot(Ns, [r["tab_params"] for r in R], "-", color=c1, lw=2, marker="o", ms=6, label="tabular 4D² + D")
    ax[1].plot(Ns, [r["nfeat"] for r in R], "-", color=c2, lw=2, marker="o", ms=6, label="stamp features")
    ax[1].set(xscale="log", yscale="log", xlabel="families N", ylabel="learned mid parameters",
              title="parameter count")
    for a in ax:
        a.minorticks_off(); a.set_xticks(Ns); a.set_xticklabels(Ns)
        a.grid(alpha=0.25, lw=0.6); a.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    T0 = time.time()
    res = dict(config=dict(NS=NS, NT=NT, RUNS=RUNS, S_T=S_T, S_M=S_M, S_F=S_F, BURN=BURN, SWEEPS=SWEEPS, PROBE=PROBE,
                           SEED=SEED), N={})
    R = []
    for N in NS:
        rng = np.random.default_rng(SEED + 100 + N)
        fams, masks = random_families(N, rng), random_pair_masks(N, rng)
        t = time.perf_counter(); C = CirclesBiome(NT, NT, families=fams, masks=masks); t_build = time.perf_counter() - t
        t = time.perf_counter(); C._canvas(); t_canvas = time.perf_counter() - t
        t = time.perf_counter(); C.reference(); t_ref = time.perf_counter() - t
        th = thetas(C)
        og = np.zeros((C.nmy, C.nmx), np.int32)
        nfeat = int(C.stamp_features(og, 2, 2, np.arange(1, C.D)).shape[1])
        O, orc = oracle(C)
        r = dict(N=N, D=C.D, D_biome=C.P, masks=[int(m) for m in C.MASKS], cells=[len(f[1]) for f in fams],
                 build_s=t_build, support_s=t_canvas, reference_s=t_ref, nfeat=nfeat, tab_params=4 * C.D ** 2 + C.D,
                 support_entries=4 * C.D ** 2, support_bytes=int(sum(C.CONF[d].nbytes for d in OFFS)),
                 support_density=float(np.mean([C.CONF[d].mean() for d in OFFS])), oracle=orc)
        for K in (None, 8):
            F, r[f"fw K={K}"] = forward(C, th, K)
            if N == max(NS) and K == 8:
                png(side(C.render(O.biome, O.obj, O.tile), C.render(F.biome, F.obj, F.tile)), "images/cbio_scale20.png")
        r["probe"] = probe(C, th)
        if FIT and N == 5:
            r["fit"] = fit_mid(C)
            print(f"  fit N=5: {r['fit']}", flush=True)
        R.append(r)
        res["N"][str(N)] = r
        a, b = r["fw K=None"], r["fw K=8"]
        print(f"N {N:2d} D {C.D:3d} build {t_build:.3f}s (support+pair {t_canvas:.3f}s) oracle {orc['seconds']:.1f}s "
              f"conf {orc['conflict_max']} fams {orc['fams_seen']}/{orc['fams_admitted']}/{N} | "
              f"us/site {a['times']['us_per_active_site']:.2f} / {b['times']['us_per_active_site']:.2f} "
              f"wall {a['wall'] * 1e3:.1f} / {b['wall'] * 1e3:.1f} ms dormant {a['dormant']:.3f} conf {a['conflict']} "
              f"{b['conflict']} present {a['present']:.3f}/{b['present']:.3f} rows {a['nrows']} hard {a['nhard']} | "
              f"probe {' '.join(f'{k}: {v["us"]:.2f}' for k, v in r['probe'].items())}", flush=True)
    figure(R, "images/cbio_scaling.png")
    res["total_s"] = time.time() - T0
    json.dump(res, open("images/cbio_stage7.json", "w"), indent=1)
    print(f"total {res['total_s']:.0f} s")
