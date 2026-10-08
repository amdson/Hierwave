"""The biome circles test (notes/circles_biome_test.md), experiment stages.

STAGE=3, the support-only baseline: Forward on the Sampler (designed +
support factors, theta = theta0: no finite part) against the oracle on every
monitor and on the fitted features, at (p, K) in (0, None), (0, 8), (2,
None).  The same three with the zeroth-order top unary installed (bio_u =
reference()["bio_u"], uncentred: it cancels bio_u0) as "+bu", and with the
induced mid unary as well (obj_u = F(o) - F(absent), the interior closed
form: it cancels most of pres) as "+bu+ou": what is left for the finite
part's pair terms once the two unaries the designed dials were built to
cancel are in.  Then the dormancy dial
(bio_u0 + DORM nats on every non-none value, oracle none ~0.3) at p = 0, and
a cost probe: obj sweep time against the active fraction with the biome set
by hand.

    STAGE=3 NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/circles_biome.py

Env: STAGE, NT (6), RUNS (64), BURN (50), SWEEPS (400), S_T, S_M (30), S_F
(20), DORM (2.5), DBURN (100), DSWEEPS (800), PROBE (20 repeats), SEED.
Outputs images/cbio_base_{oracle,p0,p2,dormant}.png, images/cbio_stage3.json.
"""
import os, time, json
import numpy as np
from PIL import Image

from castlegen.channels.circles_biome import CirclesBiome, Forward, Oracle, OFFNAMES

E = os.environ.get
STAGE = int(E("STAGE", "3"))
NT, RUNS, BURN, SWEEPS = int(E("NT", 6)), int(E("RUNS", 64)), int(E("BURN", 50)), int(E("SWEEPS", 400))
S_T, S_M, S_F = int(E("S_T", 30)), int(E("S_M", 30)), int(E("S_F", 20))
DORM, DBURN, DSWEEPS = float(E("DORM", 2.5)), int(E("DBURN", 100)), int(E("DSWEEPS", 800))
PROBE, SEED = int(E("PROBE", 20)), int(E("SEED", 0))
FIT = OFFNAMES + ["obj_u", "bio_h", "bio_v", "bio_u"]
MON = ["present_disc", "present_bar", "pd_allowed", "pb_allowed", "conflict", "contact", "bio_hist", "dormant",
       "mask_viol", "edge_air_top"]
os.makedirs("images", exist_ok=True)


def extra(C, biome, obj):
    """present per family among the slots whose biome admits it ('in isolation')."""
    a = np.repeat(np.repeat(biome, C.BT, 0), C.BT, 1)
    f = C.FAM[obj]
    out = {}
    for k, n in enumerate(C.names):
        ok = C.FAMOK[k + 1, a]
        out[f"p{n[0]}_allowed"] = float((f[ok] == k + 1).mean()) if ok.any() else np.nan
    return out


def mean(sts):
    return {k: np.nanmean([s[k] for s in sts], axis=0) for k in sts[0]}


def l1(a, b, keys=FIT):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def oracle_moments(C, burn, sweeps, seed):
    O = Oracle(C, seed=seed)
    t = time.time()
    O.sweep(burn)
    sts = []
    for _ in range(sweeps):
        O.sweep(1)
        s = C.stats(O.biome, O.obj, O.tile)
        s.update(extra(C, O.biome.grid, O.obj.grid))
        sts.append(s)
    m = mean(sts)
    m = {**C.symmetrise({k: v for k, v in m.items() if k not in ("pd_allowed", "pb_allowed")}),
         "pd_allowed": m["pd_allowed"], "pb_allowed": m["pb_allowed"]}
    return m, O, time.time() - t


def evaluate(C, theta, n, seed, **kw):
    F = Forward(C, theta, seed=seed, use_sampler=True, **kw)
    F.run(1, 1, 1)                                       # compile
    sts, tms, act = [], [], []
    for _ in range(n):
        t = time.perf_counter()
        s = F.run(S_T, S_M, S_F)
        s["wall"] = time.perf_counter() - t
        s.update(extra(C, F.biome.grid, F.obj.grid))
        sts.append(s)
        tms.append(dict(F.times))
        act.append(1 - F.dormant_frac)
    m = mean(sts)
    tm = {k: float(np.mean([t[k] for t in tms])) for k in tms[0]}
    nslot = C.nmy * C.nmx
    tm["us_per_active_site"] = float(np.sum([t["obj"] for t in tms]) / (S_M * nslot * np.sum(act)) * 1e6)
    tm["wall"] = float(m.pop("wall"))
    return m, tm, F


def png(img, path, px=4):
    Image.fromarray(np.repeat(np.repeat(img, px, 0), px, 1)).save(path)


def side(a, b, gap=6):
    return np.concatenate([a, np.full((a.shape[0], gap, 3), 255, np.uint8), b], axis=1)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    a = np.asarray(x)
    return a.tolist() if a.ndim else float(a)


def fmt(v):
    a = np.asarray(v, float)
    return f"{float(a):.4f}" if a.ndim == 0 else "/".join(f"{x:.3f}" for x in a.ravel())


def mon_table(rows):
    print("| | " + " | ".join(MON) + " |")
    for name, m in rows:
        print(f"| {name} | " + " | ".join(fmt(m[k]) for k in MON) + " |")


VARIANTS = [("p0 K=None", dict(p_relax=0, K=None)), ("p0 K=8", dict(p_relax=0, K=8)),
            ("p2 K=None", dict(p_relax=2, K=None))]


def run_dial(C, tag, variants, burn, sweeps, renders):
    t0 = time.time()
    orc, O, t_orc = oracle_moments(C, burn, sweeps, SEED + 1)
    print(f"\n== {tag}: oracle {burn} + {sweeps} sweeps {t_orc:.1f} s; bio_hist {fmt(orc['bio_hist'])}")
    ref = C.reference(centred=False)
    th0, thb, thu = C.theta0(), C.theta0(), C.theta0()
    thb["bio_u"] = thu["bio_u"] = ref["bio_u"]
    thu["obj_u"] = ref["obj_u"] - C.pres_e                # F(o) - F(absent): the induced mid unary, interior
    R = dict(oracle=orc, oracle_s=t_orc, variants={})
    rows = [("oracle", orc)]
    fw = {}
    for th_name, th in (("theta0", th0), ("+bu", thb), ("+bu+ou", thu)):
        for vn, kw in variants:
            name = f"{th_name} {vn}"
            m1, tm1, F = evaluate(C, th, RUNS, SEED + 555, **kw)
            m2, _, _ = evaluate(C, th, RUNS, SEED + 556, **kw)
            R["variants"][name] = dict(stats=m1, L1=l1(m1, orc), eval_noise=l1(m1, m2), times=tm1,
                                       mon_noise={k: np.abs(np.asarray(m1[k]) - m2[k]).sum() for k in MON},
                                       bio_L1_uniform=float(np.abs(m1["bio_hist"] - 0.25).sum()))
            fw[name] = F
            rows.append((name, m1))
            print(f"  {name}: wall {tm1['wall'] * 1e3:.1f} ms/run, obj {tm1['obj'] * 1e3:.2f} ms, "
                  f"{tm1['us_per_active_site']:.3f} us/active site, dormant {m1['dormant']:.3f}")
    print("\nmonitors")
    mon_table(rows)
    mon_table([(f"eval noise {n}", v["mon_noise"]) for n, v in R["variants"].items()])
    print("\nL1 to oracle (fitted features)")
    print("| | " + " | ".join(FIT) + " |")
    for name, v in R["variants"].items():
        print(f"| {name} | " + " | ".join(f"{v['L1'][k]:.3f}" for k in FIT) + " |")
    for name, v in R["variants"].items():
        print(f"| eval noise {name} | " + " | ".join(f"{v['eval_noise'][k]:.3f}" for k in FIT) + " |")
    print("\ntimes (ms per forward run; obj us per active site per sweep)")
    for name, v in R["variants"].items():
        t = v["times"]
        print(f"| {name} | {t['wall'] * 1e3:.1f} | {t['biome'] * 1e3:.2f} | {t['obj_init'] * 1e3:.2f} | "
              f"{t['obj'] * 1e3:.2f} | {t['relax'] * 1e3:.2f} | {t['tile'] * 1e3:.2f} | {t['us_per_active_site']:.3f} |")
    for path, what in renders:                           # what: "oracle", a variant, or "oracle|<variant>"
        ro = C.render(O.biome, O.obj, O.tile)
        rf = lambda n: C.render(fw[n].biome, fw[n].obj, fw[n].tile)
        png(ro if what == "oracle" else side(ro, rf(what[7:])) if what.startswith("oracle|") else rf(what), path)
    print(f"({time.time() - t0:.0f} s)")
    return R


def probe(C, reps):
    """obj stage time (init + S_M sweeps) against the none fraction, biome by
    hand (non-none values uniform), K None and 8."""
    rng = np.random.default_rng(SEED + 9)
    out = {}
    nt = C.nty * C.ntx
    for K in (None, 8):
        F = Forward(C, C.theta0(), seed=SEED + 3, use_sampler=True, K=K)
        F.build()
        S = F.S["obj"]
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            ts, ti, acts = [], [], []
            for r in range(reps + 1):
                b = rng.integers(1, C.P, nt)
                b[rng.permutation(nt)[:int(round(f * nt))]] = 0
                F.biome.grid[:] = b.reshape(C.nty, C.ntx)
                F.obj.grid[:] = 0
                C.paint_allow(F.biome, F.allow)
                t = time.perf_counter(); d = S.init(); t1 = time.perf_counter()
                S.sweep(S_M, seed=int(rng.integers(1 << 30))); t2 = time.perf_counter()
                if r:                                     # first rep: warm-up
                    ti.append(t1 - t); ts.append(t2 - t1); acts.append(1 - d)
            a = float(np.mean(acts))
            out[f"K{K}_f{f}"] = dict(none=f, active=a, init_ms=float(np.mean(ti)) * 1e3,
                                     sweeps_ms=float(np.mean(ts)) * 1e3,
                                     us_per_active=float(np.mean(ts)) / max(a * C.nmy * C.nmx * S_M, 1e-9) * 1e6)
            o = out[f"K{K}_f{f}"]
            print(f"| {K} | {f:.2f} | {a:.3f} | {o['init_ms']:.3f} | {o['sweeps_ms']:.3f} | "
                  f"{o['us_per_active'] if a > 0 else float('nan'):.3f} |")
    return out


if STAGE == 3:
    T0 = time.time()
    print(f"STAGE 3  nty=ntx={NT}  RUNS {RUNS}  BURN {BURN}  SWEEPS {SWEEPS}  S_T/S_M/S_F {S_T}/{S_M}/{S_F}  "
          f"DORM {DORM} ({DBURN} + {DSWEEPS})")
    C = CirclesBiome(NT, NT)
    res = dict(config=dict(NT=NT, RUNS=RUNS, BURN=BURN, SWEEPS=SWEEPS, S_T=S_T, S_M=S_M, S_F=S_F, DORM=DORM,
                           DBURN=DBURN, DSWEEPS=DSWEEPS, PROBE=PROBE, SEED=SEED, mu=C.mu, b=C.b.tolist(),
                           bio_u0=C.bio_u0.tolist()))
    res["base"] = run_dial(C, "default dial", VARIANTS, BURN, SWEEPS,
                           [("images/cbio_base_oracle.png", "oracle"), ("images/cbio_base_p0.png", "theta0 p0 K=None"),
                            ("images/cbio_base_p2.png", "theta0 p2 K=None")])
    Cd = CirclesBiome(NT, NT)
    Cd.bio_u0 = Cd.bio_u0 + DORM * (np.arange(Cd.P) > 0)
    res["dorm"] = run_dial(Cd, f"dormancy dial (bio_u0 + {DORM} on non-none)", VARIANTS[:2], DBURN, DSWEEPS,
                           [("images/cbio_base_dormant.png", "oracle|+bu+ou p0 K=None")])
    print("\nobj cost against the none fraction (theta0, biome by hand)")
    print("| K | none | active | init ms | sweeps ms | us / active site |")
    res["probe"] = probe(C, PROBE)
    res["total_s"] = time.time() - T0
    json.dump(jsonable(res), open("images/cbio_stage3.json", "w"), indent=1)
    print(f"\ntotal {res['total_s']:.0f} s -> images/cbio_stage3.json")
