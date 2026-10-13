"""Bench runner: train a variant on a problem and score it (notes/bench.md).

FROZEN.  python -m castlegen.bench run <variant> <problem> [--seeds 2]
[--train 150] [--eval 16]; results append to results/bench.jsonl."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from . import energy as en
from .problem import Problem, Region
from .problems import get as get_problem
from .variant import load_variants

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
EVAL_SEED0 = 1_000_000          # evaluation noise seeds start here; variants train below it
BUDGET = 2e-4                   # final violated fraction allowed at every level
PROBE_FLAG = 0.05               # flag a probe set with more than 5 % above the 99th percentile
Z = 3.0                         # agreement: |difference| <= Z * standard error + tolerance


def harness_hash():
    h = hashlib.sha256()
    for p in sorted(BENCH_DIR.rglob("*.py")):
        h.update(p.relative_to(BENCH_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_rev():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _with_designed(problem, x, seed, region):
    return {**problem.designed_values(seed, region), **{k: np.asarray(v) for k, v in x.items()}}


def _mean_se(vals):
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]                 # a stat undefined on a world (e.g. no gaps) is skipped there
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def _agree(m1, s1, m2, s2, tol):
    if any(np.isnan(v) for v in (m1, m2)):
        return None
    return bool(abs(m1 - m2) <= Z * np.hypot(s1, s2) + tol)


def eval_seed(problem: Problem, v, k: int, n_eval: int, img_path: Path | None):
    """Metrics of one trained variant on held-out noise."""
    T = problem.T
    region = Region(problem.W_eval)
    cells = problem.H * problem.W_eval
    out = {"viol": {}, "stats": {}}
    acc = {kind: {"viol": np.zeros(T + 1), "n": np.zeros(T + 1), "stats": {}} for kind in ("raw", "final")}
    S_raw, logq, d_own = [], [], []
    t0 = time.time()
    for i in range(n_eval):
        seed = EVAL_SEED0 + 1000 * k + i
        g = v.generate(seed, region)
        for kind in ("raw", "final"):
            x = _with_designed(problem, g[kind], seed, region)
            E = en.energy(problem, x, region)
            for l in range(T + 1):
                acc[kind]["viol"][l] += E.viol[l]
                acc[kind]["n"][l] += E.nhard[l]
            if problem.stats:
                for name, val in problem.stats(x).items():
                    acc[kind]["stats"].setdefault(name, []).append(float(val))
            if kind == "raw":
                q = v.score(x, region)
                S_raw.append(E.soft)
                if q is not None:
                    logq.append(float(q))
                    if sum(E.viol.values()) == 0:
                        d_own.append(-E.soft - float(q))
            if kind == "final" and i == 0 and img_path is not None and problem.render:
                import matplotlib.pyplot as plt
                img_path.parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(img_path, problem.render(x))
    out["gen_s_per_world"] = (time.time() - t0) / n_eval
    for kind in ("raw", "final"):
        out["viol"][kind] = {str(l): float(acc[kind]["viol"][l] / max(acc[kind]["n"][l], 1))
                             for l in range(T + 1)}
        out["viol"][kind + "_n"] = {str(l): int(acc[kind]["n"][l]) for l in range(T + 1)}
        out["stats"][kind] = {name: _mean_se(vals) for name, vals in acc[kind]["stats"].items()}
    out["free_energy"] = (float(np.mean(np.asarray(S_raw) + np.asarray(logq)) / cells)
                          if len(logq) == len(S_raw) and logq else None)
    out["soft_per_cell"] = float(np.mean(S_raw) / cells)
    out["n_valid_raw"] = len(d_own)

    # probes: valid states from elsewhere, scored by the variant
    out["probes"] = None
    if problem.probes and logq and len(d_own) >= 2:
        thr = float(np.quantile(d_own, 0.99))
        ds = []
        for i in range(max(2, n_eval // 4)):
            seed = EVAL_SEED0 + 1000 * k + 500 + i
            for y in problem.probes(seed, region):
                y = _with_designed(problem, y, seed, region)
                E = en.energy(problem, y, region)
                assert sum(E.viol.values()) == 0, "probe states must be valid"
                q = v.score(y, region)
                ds.append(-E.soft - float(q))
        ds = np.asarray(ds)
        out["probes"] = {"n": int(ds.size), "frac_above": float((ds > thr).mean()),
                         "median_gap_per_cell": float(np.median(ds - thr) / cells),
                         "flag": bool((ds > thr).mean() > PROBE_FLAG)}

    # finite support: two overlapping open windows must agree on the overlap
    W = problem.W_eval
    half = (W // 2) >> T << T
    seed = EVAL_SEED0 + 1000 * k + 999
    ga = v.generate(seed, Region(W, periodic=False, x0=0))
    gb = v.generate(seed, Region(W, periodic=False, x0=half))
    mism, tot = 0, 0
    for kind in ("raw", "final"):
        for c in problem.sampled():
            s = half >> c.level
            a = np.asarray(ga[kind][c.name])[:, s:]
            b = np.asarray(gb[kind][c.name])[:, :a.shape[1]]
            mism += int((a != b).sum())
            tot += a.size
    out["support_mismatch"] = mism / max(tot, 1)
    return out


def run(variant: str, problem_name: str, seeds=2, train_s=150.0, n_eval=16, tag="", quiet=False):
    variants = load_variants()
    problem = get_problem(problem_name)
    cls = variants[variant]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{variant}_{problem_name}" + (f"_{tag}" if tag else "")
    rdir = ROOT / "results" / "runs" / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    per_seed, infos = [], []
    for k in range(seeds):
        logf = open(rdir / f"train_s{k}.jsonl", "w")

        def log(d, _f=logf):
            _f.write(json.dumps(d, default=float) + "\n")

        v = cls(problem, seed=k, workdir=str(rdir))
        t0 = time.time()
        v.train(train_s, log)
        t_train = time.time() - t0
        logf.close()
        img = ROOT / "images" / "bench" / f"{variant}_{problem_name}_s{k}.png"
        m = eval_seed(problem, v, k, n_eval, img)
        m["train_s"] = t_train
        per_seed.append(m)
        infos.append(v.info())
        if not quiet:
            print(f"[seed {k}] train {t_train:.0f}s  final viol {m['viol']['final']}  "
                  f"F {m['free_energy']}  support {m['support_mismatch']:.3g}", flush=True)

    verdict = judge(problem, per_seed, train_s)
    rec = {"run": run_id, "time": stamp, "variant": variant, "problem": problem_name,
           "problem_version": problem.version, "harness": harness_hash(), "git": git_rev(),
           "machine": f"{platform.node()} {platform.machine()}", "train_budget_s": train_s,
           "seeds": seeds, "n_eval": n_eval, "wall_s": time.time() - t_all, "tag": tag,
           "per_seed": per_seed, "info": infos, "verdict": verdict}
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "bench.jsonl", "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    if not quiet:
        print(json.dumps(verdict, indent=1, default=float))
    return rec


def judge(problem: Problem, per_seed, train_s):
    """The constitution's acceptance, check by check."""
    T = problem.T
    v = {}
    worst = max(max(m["viol"]["final"].values()) for m in per_seed)
    v["final_violations"] = {"worst_frac": worst, "ok": bool(worst < BUDGET)}
    v["raw_violations_worst"] = max(max(m["viol"]["raw"].values()) for m in per_seed)
    # exact stats
    ex = {}
    for name, val in problem.exact.items():
        res = []
        for m in per_seed:
            mean, se = m["stats"]["final"].get(name, (float("nan"), 0.0))
            res.append(_agree(mean, se, val, 0.0, problem.tol.get(name, 0.0)))
        ex[name] = all(r is True for r in res)
    v["exact_stats"] = {"per_stat": ex, "ok": all(ex.values()) if ex else None}
    # raw vs final, and across seeds
    rf, xs = {}, {}
    names = per_seed[0]["stats"]["final"].keys()
    for name in names:
        tol = problem.tol.get(name, 0.0)
        rf[name] = all(_agree(*m["stats"]["raw"][name], *m["stats"]["final"][name], tol) is not False
                       for m in per_seed)
        if len(per_seed) > 1:
            xs[name] = all(_agree(*per_seed[0]["stats"]["final"][name], *m["stats"]["final"][name], tol)
                           is not False for m in per_seed[1:])
    v["raw_vs_final"] = {"per_stat": rf, "ok": all(rf.values()) if rf else None}
    v["across_seeds"] = {"per_stat": xs, "ok": all(xs.values()) if xs else None}
    pr = [m["probes"] for m in per_seed]
    v["probes"] = {"flagged": [p["flag"] if p else None for p in pr],
                   "ok": all(p is not None and not p["flag"] for p in pr) if problem.probes else None}
    v["finite_support"] = {"worst_mismatch": max(m["support_mismatch"] for m in per_seed),
                           "ok": all(m["support_mismatch"] == 0 for m in per_seed)}
    fe = [m["free_energy"] for m in per_seed]
    v["free_energy"] = None if any(f is None for f in fe) else float(np.mean(fe))
    v["within_budget"] = all(m["train_s"] <= 1.1 * train_s + 5 for m in per_seed)
    checks = [v["final_violations"]["ok"], v["exact_stats"]["ok"], v["raw_vs_final"]["ok"],
              v["across_seeds"]["ok"], v["probes"]["ok"], v["finite_support"]["ok"], v["within_budget"]]
    v["accepted"] = all(c is not False for c in checks) and v["free_energy"] is not None
    return v
