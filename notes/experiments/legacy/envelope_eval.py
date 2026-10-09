"""Evaluate the envelope-promise support pipeline against the old one and the
exact p_G reference (castlegen.legacy.refheights).  Prints one table per experiment.

    .venv/bin/python notes/experiments/legacy/envelope_eval.py [--seeds 6] [--png DIR]
"""
import argparse
import copy
import json
import time

import numpy as np

from castlegen.legacy import exemplar as ex
from castlegen.legacy import pipeline as PL
from castlegen.legacy import refheights as RH
from castlegen.legacy import tileset
from castlegen.legacy.quantities import envelope as EN

G, K, N = 8, 16, 128


def env_spec(lam=1.0, w_tex=1.0, T=0.9, clamps=None, reference=None, heights_sweeps=10, sweeps=3):
    spec = PL.load_spec("cliffs_envelope")
    spec = copy.deepcopy(spec)
    h, t = spec["stages"]
    h.update(lam=lam, sweeps=sweeps)
    q = {}
    if clamps:
        q["clamps"] = [list(k) + [v] for k, v in clamps.items()]
    if reference:
        q["reference"] = reference
    if q:
        h["envelope"] = q
    t.update(T=T, w_tex=w_tex, sweeps=heights_sweeps)
    return spec


def run_many(spec, seeds, png=None, tag=""):
    maps, rows = [], []
    for s in seeds:
        ctx = []
        ts, tiles, rep = PL.run(spec, seed=s, log=lambda *a: None, ctx_out=ctx)
        maps.append(tiles)
        last = rep[-1]
        hier = next(r for r in rep if r["stage"] == "hier")
        sat_h = hier.get("sat", {})
        rows.append(dict(unsup=last["unsupported"], sat=min(last.get("sat", {"x": 1.0}).values()),
                         tex_sat16=sat_h.get("envelope16", np.nan), tex_unsup=hier["unsupported"],
                         changed=last["changed"], sec=sum(r.get("seconds", 0) for r in rep)))
        if png and s == seeds[0]:
            ex.to_png(ts, tiles, f"{png}/{tag}_s{s}.png", 4)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["unsup_max"] = int(max(r["unsup"] for r in rows))
    return maps, agg


def summary(ts, maps, ref_stats):
    st = RH.stats_many(ts, maps, G, K, 64)
    c = RH.compare(st, ref_stats)
    return st, c


def fmt_cmp(c):
    out = []
    for k in ("solid", "H_mean", "H_std", "abs_slope", "E"):
        v = c[k]
        out.append(f"{k} {v['a']:.3f}/{v['b']:.3f} (z {v['z']:+.1f})")
    out.append(f"H_tv {c['H_hist_tv']:.2f} slope_tv {c['slope_hist_tv']:.2f} surf_tv {c['surface_tv']:.2f}")
    out.append("promise_tv " + " ".join(f"{h}:{v:.2f}" for h, v in c["promise_tv"].items()))
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--png", default=None)
    ap.add_argument("--only", default="all", help="all | unclamped | clamps")
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))
    if args.only == "clamps":
        return clamp_experiment(seeds, args.png)
    ts = tileset.load("cliffs")
    ref = RH.corpus(ts, n=N, T=0.9, ground=G, log=lambda *a: None)
    ref_stats = RH.stats_many(ts, list(ref[:len(seeds) * 2]), G, K, 64)

    print("== 1. old pipeline vs envelope (unclamped, mu = 0 tables)")
    exps = {
        "old (fulfil)": PL.load_spec("cliffs_promise"),
        "env w_tex=1": env_spec(w_tex=1.0),
        "env w_tex=0": env_spec(w_tex=0.0),
        "env lam=5": env_spec(lam=5.0),
        "env lam=20": env_spec(lam=20.0),
    }
    for name, spec in exps.items():
        maps, agg = run_many(spec, seeds, args.png, name.replace(" ", "_").replace("(", "").replace(")", ""))
        _, c = summary(ts, maps, ref_stats)
        print(f"{name:14s} {json.dumps({k: round(v, 3) for k, v in agg.items()})}")
        print(f"{'':14s} vs p_G: {fmt_cmp(c)}")




# ------------------------------------------------------------------ clamps
def clamp_sets():
    lg = EN.lang()
    F = lg.F
    full = int(lg.encode([F, F], [F, F]))
    cliff = int(lg.encode([0, 0], [F, F]))           # each interval has an empty and a full column
    return {
        "mountain (h32 block full)": {(32, 1, 1): full},
        "cliff (h16 block empty+full)": {(16, 4, 2): cliff},
        "two peaks (h16 fulls)": {(16, 2, 1): full, (16, 3, 6): full},
    }


def column_profile(ts, maps):
    return np.mean([RH.heights(ts, t, G) for t in maps], 0)


def clamp_experiment(seeds, png=None):
    ts = tileset.load("cliffs")
    lg = EN.lang()
    for name, cl in clamp_sets().items():
        t0 = time.time()
        ref = [RH.sample_conditioned(ts, cl, n=N, seed=100 + s, T=0.9, ground=G, sweeps=400) for s in seeds]
        t_ref = time.time() - t0
        ref_st = RH.stats_many(ts, ref, G, K, 64)
        ref_prof = column_profile(ts, ref)
        print(f"== clamp: {name}   (reference {t_ref / len(seeds):.1f}s/sample)")
        for lab, spec in (("env w_tex=0", env_spec(w_tex=0.0, clamps=cl)),
                          ("env w_tex=1", env_spec(w_tex=1.0, clamps=cl)),
                          ("env lam=20", env_spec(lam=20.0, clamps=cl))):
            maps, agg = run_many(spec, seeds, png, (name.split()[0] + "_" + lab).replace(" ", "_"))
            honoured = np.mean([all(EN.pyramid(ts, t, K, G, 64)[0][h][y, x] == v for (h, y, x), v in cl.items())
                                for t in maps])
            prof = column_profile(ts, maps)
            _, c = summary(ts, maps, ref_st)
            print(f"  {lab:12s} clamps honoured {honoured:.2f}  unsup {agg['unsup_max']}  sat {agg['sat']:.2f}  "
                  f"changed {agg['changed']:.3f}  mean|profile - ref| {np.abs(prof - ref_prof).mean():.2f} rows")
            print(f"  {'':12s} vs conditioned p_G: {fmt_cmp(c)}")
        if png:
            ex.to_png(ts, ref[0], f"{png}/{name.split()[0]}_reference.png", 4)


if __name__ == "__main__":
    main()
