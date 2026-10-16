"""Render held-out worlds: chain reference vs best models (raw pass draw and staged final).
usage: python samples.py <out.png> <n_worlds>"""
import pickle, sys, os
import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.energy import energy

out, n = sys.argv[1], int(sys.argv[2])
p = get("ground"); reg = Region(p.W_eval); names = [c.name for c in p.chans]
V = load_variants()

def gen(vname, path):
    ck = pickle.load(open(path, "rb")); tr = ck["trained"] if "trained" in ck else ck
    v = V[vname](p, 0, "/tmp/samp_" + vname)
    v.params = jax.tree_util.tree_map(jnp.asarray, tr["params"]); v.lam = np.asarray(tr["lam"], float)
    rows = {"raw": [], "final": []}
    for i in range(n):
        g = v.generate(EVAL_SEED0 + i, reg)
        for k in rows: rows[k].append(_with_designed(p, g[k], EVAL_SEED0 + i, reg))
    return rows

def label(x):
    e = energy(p, x, reg); nv = int(sum(e.viol.values())); s = p.stats(x)
    return f"viol {nv}  line_match {s['line_match']:.2f}  rough {s['roughness']:.2f}"

panels = []
z = np.load("reportables/2026-10-14_cgraded/final_binary_lam20.npz")
panels.append(("chain reference (long MCMC, valid)", [{k: z[k][b] for k in names} for b in range(n)]))
models = [("clongfit_stage", "reportables/2026-10-16_longfit2/runs/big300/params_s0.pkl", "clongfit 300 s")]
mp = "results/runs/20261016-061934_cfitpen_ml2_stage_ground_warm/params_combo.pkl"
if os.path.exists(mp): models.append(("cfitpen_ml2_stage", mp, "cfitpen_ml2 300 s"))
for vname, path, lab in models:
    r = gen(vname, path)
    panels.append((f"{lab}: staged final (R 30)", r["final"]))
    panels.append((f"{lab}: raw pass draw", r["raw"]))

fig, axes = plt.subplots(len(panels), n, figsize=(3.2 * n, 1.9 * len(panels)))
for i, (title, xs) in enumerate(panels):
    for j, x in enumerate(xs):
        ax = axes[i, j]; ax.imshow(p.render(x), interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label(x), fontsize=7)
        if j == 0: ax.set_ylabel(title, fontsize=7, rotation=0, ha="right", va="center")
plt.tight_layout(); plt.savefig(out, dpi=140); print("saved", out)
