"""Per-world actor-gradient SNR per level, at init, cfast-equivalent (no
branches, blev=0) vs creplay (branches), same worlds and parameters, critic
zeroed (returns alone carry the correction).  Per level: |mean|^2,
E|g - mean|^2 and SNR = |mean|^2 / E|g - mean|^2 (per world, B = 1).
usage: snr.py problem N 'R=(16,16),Kr=4'"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.creplay import CReplay

prob = problems.get(sys.argv[1])
N = int(sys.argv[2])
spec = eval(f"dict({sys.argv[3]})")
region = Region(prob.W_train)
v0 = CReplay(prob, 0, "/tmp", B=1, blev=0, R=())
v1 = CReplay(prob, 0, "/tmp", B=1, **spec)
lay = v0.layout(region)
f0, f1 = v0.train_fns(lay), v1.train_fns(lay)
params = v0.params
cparams = jax.tree_util.tree_map(jnp.zeros_like, v0.cparams)
gs = {k: jnp.float32(1.0) for k in params}
lam = jnp.asarray(v0.lam, jnp.float32)
tau = jnp.float32(1.0)
levels = sorted({int(k[1:k.index("_")]) for k in params})


def fl(g):
    return {l: np.concatenate([np.ravel(np.asarray(x)) for k in sorted(g) if k.startswith(f"l{l}_")
                               for x in jax.tree_util.tree_leaves(g[k])]) for l in levels}


g0, g1 = {l: [] for l in levels}, {l: [] for l in levels}
t = time.time()
for i in range(N):
    seeds = jnp.asarray([3_000 + i], jnp.int32)
    des = v0._batch(np.asarray(seeds), region)
    a = fl(f0["agrad"](params, cparams, gs, seeds, des, tau, lam, jax.random.PRNGKey(i))[0])
    b = fl(f1["agrad"](params, cparams, gs, seeds, des, tau, lam, jax.random.PRNGKey(i))[0])
    for l in levels:
        g0[l].append(a[l]); g1[l].append(b[l])
    if i == 0:
        print(f"first world (compile) {time.time() - t:.1f}s", flush=True)
print(f"{N} worlds in {time.time() - t:.1f}s")
out = {"spec": sys.argv[3], "N": N, "branch": {str(l): {"R": x["R"], "Nc": x["Nc"]} for l, x in f1["btab"].items()}}
for l in levels:
    A, B = np.stack(g0[l]), np.stack(g1[l])
    ma, mb = A.mean(0), B.mean(0)
    va, vb = ((A - ma) ** 2).sum(1).mean(), ((B - mb) ** 2).sum(1).mean()
    d = A - B
    md = d.mean(0)
    se2 = ((d - md) ** 2).sum(1).mean() / N
    # SNR with the pooled mean (both estimate the same gradient)
    m = (ma + mb) / 2
    out[l] = {"var_cfast": float(va), "var_creplay": float(vb), "var_ratio": float(va / max(vb, 1e-30)),
              "mean_sq": float((m ** 2).sum()), "snr_cfast": float((m ** 2).sum() / va),
              "snr_creplay": float((m ** 2).sum() / vb), "diff_sq_over_se2": float((md ** 2).sum() / max(se2, 1e-30))}
    print(l, json.dumps(out[l]), flush=True)
json.dump(out, open(f"reportables/2026-10-13_creplay/snr_{sys.argv[1]}.json", "w"), indent=1)
