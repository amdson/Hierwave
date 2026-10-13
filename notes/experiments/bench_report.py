"""Figures for a reportables folder from bench runs.

python notes/experiments/bench_report.py <out_dir> <variant> [<variant> ...]

For the latest run of each variant on each problem: training curves (raw
violated fraction per level, penalties, cost per world) and a grid of the
bench images (one final world per seed, every variant side by side)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def latest(variants):
    runs = {}
    for line in open(ROOT / "results" / "bench.jsonl"):
        r = json.loads(line)
        if r["variant"] in variants:
            runs[(r["variant"], r["problem"])] = r
    return runs


def curves(run, out):
    logs = sorted((ROOT / "results" / "runs" / run["run"]).glob("train_s*.jsonl"))
    rows = [[json.loads(l) for l in open(p)] for p in logs]
    rows = [[d for d in r if "update" in d] for r in rows]
    if not any(rows):
        return None
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for s, r in enumerate(rows):
        if not r:
            continue
        t = np.array([d["t"] for d in r])
        v = np.array([d["raw_viol"] for d in r])
        lam = np.array([d["lam"] for d in r])
        for l in range(v.shape[1]):
            ls = "-" if s == 0 else "--"
            ax[0].plot(t, np.maximum(v[:, l], 1e-5), ls, color=f"C{l}", label=f"level {l}" if s == 0 else None)
            ax[1].plot(t, lam[:, l], ls, color=f"C{l}")
        ax[2].plot(t, [d["cost_per_world"] for d in r], "-" if s == 0 else "--", color="k")
    ax[0].set_yscale("log")
    ax[0].axhline(2e-4, color="grey", lw=0.8)
    ax[0].set(title="raw violated fraction (training draws)", xlabel="s")
    ax[0].legend(fontsize=7)
    ax[1].set(title="penalty λ per level", xlabel="s")
    ax[2].set(title="cost per world (S + λH + τ log q)", xlabel="s")
    fig.suptitle(f"{run['variant']} on {run['problem']} (solid seed 0, dashed seed 1)", fontsize=10)
    fig.tight_layout()
    p = out / f"curves_{run['variant']}_{run['problem']}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def grid(variants, problems, out):
    fig, ax = plt.subplots(len(problems), len(variants), figsize=(4.2 * len(variants), 2.0 * len(problems)),
                           squeeze=False)
    for i, prob in enumerate(problems):
        for j, var in enumerate(variants):
            a = ax[i, j]
            a.axis("off")
            f = ROOT / "images" / "bench" / f"{var}_{prob}_s0.png"
            if f.exists():
                a.imshow(plt.imread(f), interpolation="nearest")
            a.set_title(f"{var} · {prob}", fontsize=8)
    fig.tight_layout()
    p = out / "samples.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p.name


def main():
    out = Path(sys.argv[1])
    variants = sys.argv[2:]
    out.mkdir(parents=True, exist_ok=True)
    runs = latest(variants)
    problems = sorted({p for _, p in runs})
    made = [grid(variants, problems, out)]
    for (var, prob), r in sorted(runs.items()):
        c = curves(r, out)
        if c:
            made.append(c)
    print("\n".join(made))


main()
