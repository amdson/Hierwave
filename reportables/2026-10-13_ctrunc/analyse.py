"""Throughput and value-bias summary of ctrunc bench runs.

python reportables/2026-10-13_ctrunc/analyse.py <run_id> [<run_id> ...]

For each run and seed: updates, updates per second after the first update,
and for each level l the value estimate sum V_{l-1} (made before the update
that used the draw) against the realised cost of every level below l on the
worlds that reached level 0, over the second half of training.  Also writes
value_<run>.png (pred vs realised, per level)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

for run in sys.argv[1:]:
    rd = ROOT / "results" / "runs" / run
    rec = [json.loads(l) for l in open(ROOT / "results" / "bench.jsonl") if json.loads(l)["run"] == run][0]
    from castlegen.bench.problems import get
    p = get(rec["problem"])
    cells = p.H * p.W_train
    figs = []
    for s, f in enumerate(sorted(rd.glob("train_s*.jsonl"))):
        ups = [d for d in (json.loads(l) for l in open(f)) if "update" in d]
        if not ups:
            continue
        rate = (len(ups) - 1) / max(ups[-1]["t"] - ups[0]["t"], 1e-9)
        print(f"{run} seed {s}: {len(ups)} updates, first at {ups[0]['t']:.0f} s, {rate:.2f} updates/s")
        half = ups[len(ups) // 2:]
        levels = sorted({k for d in half for k in d.get("vcheck", {})}, key=int)
        for l in levels:
            pr = np.concatenate([d["vcheck"][l][0] for d in half]) / cells
            re = np.concatenate([d["vcheck"][l][1] for d in half]) / cells
            c = np.corrcoef(pr, re)[0, 1] if re.std() > 0 and pr.std() > 0 else float("nan")
            print(f"  below level {l}: pred {pr.mean():+.3f} real {re.mean():+.3f} bias {(pr - re).mean():+.3f} "
                  f"(se {(pr - re).std() / np.sqrt(len(re)):.3f}) rmse {np.sqrt(((pr - re) ** 2).mean()):.3f} "
                  f"sd(real) {re.std():.3f} corr {c:.2f}  [per fine cell, n={len(re)}]")
            figs.append((s, l, pr, re))
    if figs:
        ls = sorted({f[1] for f in figs}, key=int)
        fig, ax = plt.subplots(1, len(ls), figsize=(3.2 * len(ls), 3.2), squeeze=False)
        for s, l, pr, re in figs:
            a = ax[0, ls.index(l)]
            a.plot(re, pr, ".", ms=3, alpha=0.5, color=f"C{s}", label=f"seed {s}")
            lo, hi = min(re.min(), pr.min()), max(re.max(), pr.max())
            a.plot([lo, hi], [lo, hi], "k-", lw=0.6)
            a.set(title=f"below level {l}", xlabel="realised / fine cell", ylabel="V / fine cell")
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f"{run}: value estimate vs realised downstream cost (2nd half of training)", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"value_{rec['variant']}_{rec['problem']}.png", dpi=110)
        plt.close(fig)
