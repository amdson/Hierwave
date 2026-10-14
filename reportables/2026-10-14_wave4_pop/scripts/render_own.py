"""Own tau=1 draws (tiles) of the start params and each run's trained params.
usage (repo root): render_own.py out.png run1,run2,...   (runs under the report's runs/)"""
import sys, pickle
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.variants.ckernel_own import CKernelOwn
from castlegen.variants._cfast_core import train_pass
D = "reportables/2026-10-14_wave4_pop/runs"
out, runs = sys.argv[1], sys.argv[2].split(",")
p = get("ground"); reg = Region(p.W_train)
V = CKernelOwn(p, 0, "/tmp"); lay = V.layout(reg)
lam_gen = jnp.asarray(V.lam, jnp.float32)
gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lam_gen, record=False)[0], in_axes=(None, 0, 0)))
seeds = np.arange(5_000_000, 5_000_004)
ds = [p.designed_values(int(s), reg) for s in seeds]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
sets = [("start (sweep2a arm B)", V.params)] + [(r, jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{D}/{r}/params_ground.pkl", "rb"))["trained"]["params"])) for r in runs]
fig, ax = plt.subplots(len(sets), 4, figsize=(16, 2.3 * len(sets)))
for i, (name, pr) in enumerate(sets):
    st = gen(pr, jnp.asarray(seeds, jnp.int32), dj)
    for j in range(4):
        x = {"t": np.asarray(st["t"][j]), "fill": np.asarray(dj["fill"][j]), "rho4": np.zeros(1)}
        s = p.stats({**x, "rho4": np.asarray(st["rho4"][j])})
        ax[i, j].imshow(p.render(x)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        from castlegen.bench.problems.ground import _line_from_fill
        g = _line_from_fill(x["fill"], 128)
        ax[i, j].plot(np.arange(128) * 2 + 1, g * 2, "y-", lw=0.8)
        ax[i, j].set_title(f"{name}: dirt {s['dirt_frac']:.2f} lm {s['line_match']:.2f}", fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=100)
print("saved", out)
# population examples (final kernel-refined chain states) of each run
fig, ax = plt.subplots(len(runs), 4, figsize=(16, 2.3 * len(runs)), squeeze=False)
for i, r in enumerate(runs):
    P = pickle.load(open(f"{D}/{r}/population.pkl", "rb"))
    for j in range(4):
        x = {"t": P["t"][j], "fill": P["fill"][j], "rho4": P["rho4"][j]}
        s = p.stats(x)
        ax[i, j].imshow(p.render(x)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        g = _line_from_fill(x["fill"], 128); ax[i, j].plot(np.arange(128) * 2 + 1, g * 2, "y-", lw=0.8)
        ax[i, j].set_title(f"{r} population: dirt {s['dirt_frac']:.2f} lm {s['line_match']:.2f}", fontsize=8)
plt.tight_layout(); plt.savefig(out.replace(".png", "_population.png"), dpi=100)
print("saved population image", flush=True)
