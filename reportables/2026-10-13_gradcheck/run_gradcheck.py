"""Spec test 5 for cpass: estimator mean vs exact grad J on the enumerable
tiny problem.  PYTHONPATH=. python reportables/2026-10-13_gradcheck/run_gradcheck.py [M] [B]"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.variants import _gradcheck as gc
from castlegen.variants.cpass import CPass

M = int(sys.argv[1]) if len(sys.argv) > 1 else 400
B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = "reportables/2026-10-13_gradcheck/results.json"
TAU, LAM = 1.5, np.array([2.0, 3.0])

t0 = time.time()
p = gc.tiny_problem()
var = CPass(p, 0, "/tmp/gcw", K=4)
var.params = gc.randomize(var.params, jax.random.PRNGKey(1), scale=1.0)
var.cparams = gc.zero_critic(var.cparams)
ex = gc.Exact(var, LAM, cache="/tmp/gradcheck_tiny_exact.npz")
J, gJ, info = ex.J_and_grad(var.params, TAU)
keys = sorted(gJ)
gflat = gc.flat(gJ, keys)
sl = gc.block_slices(gJ, keys)
print(f"exact: {ex.n} reachable outcomes, J = {J:.4f}, info {info}, {time.time() - t0:.0f}s", flush=True)

# finite-difference sanity check of the exact gradient along its own direction
eps = 1e-2
u = jax.tree_util.tree_map(lambda x: x / np.linalg.norm(gflat), gJ)
pp = jax.tree_util.tree_map(lambda a, b: a + eps * b, var.params, u)
pm = jax.tree_util.tree_map(lambda a, b: a - eps * b, var.params, u)
fd = (ex.J_and_grad(pp, TAU)[0] - ex.J_and_grad(pm, TAU)[0]) / (2 * eps)
print(f"finite difference along grad: {fd:.5f} vs |grad| {np.linalg.norm(gflat):.5f}", flush=True)

results = {"exact": {"n": ex.n, "J": J, **info, "fd": fd, "norm": float(np.linalg.norm(gflat)),
                     "block_norms": {k: float(np.linalg.norm(gflat[s])) for k, s in sl.items()}}}


def run(name, part="all", seed0=1_000_000):
    t = time.time()
    est, cost, enum = [], [], []
    for m in range(M):
        g, bi = gc.estimator_grad(var, np.arange(seed0 + m * B, seed0 + (m + 1) * B), TAU, LAM, part=part)
        est.append(gc.flat(g, keys))
        cost.append(bi["cost"])
        enum.append(bi["enum"].mean())
    est = np.array(est)
    cost = np.concatenate(cost)
    res = {"enum_frac": float(np.mean(enum)), "mean_cost": float(np.mean(cost)),
           "cost_z": float((cost.mean() - J) / (cost.std() / np.sqrt(len(cost)))), "secs": time.time() - t}
    blocks = [k for k in keys if part == "all" or k in var.layout(gc.Region(8)).topkeys]
    for k in blocks:
        res[k] = gc.ztest(est[:, sl[k]], gflat[sl[k]], B)
    if part == "all":
        res["all"] = gc.ztest(est, gflat, B)
    results[name] = res
    print(name, json.dumps({k: (v if not isinstance(v, dict) else
                               {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
                            for k, v in res.items()}), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)


def setK(k):
    """K is baked into the built losses: rebuild."""
    var.cfg["K"] = k
    lay = var.layout(gc.Region(8))
    var._build_train(lay)
    if hasattr(lay, "jit_agrad"):
        del lay.jit_agrad


print(f"mean cost per world from exact: {J:.4f}")
# 1. zero critic, K = 4 (top D = 3 <= K: every branch enumerated)
run("zero_critic_K4")
run("zero_critic_K4_branch_only", part="branch")
run("zero_critic_K4_rest_only", part="rest")
# 2. K = 2: branches with |A| = 3 sampled with the leave-one-out baseline
setK(2)
run("zero_critic_K2", seed0=2_000_000)
run("zero_critic_K2_branch_only", part="branch", seed0=2_000_000)
setK(4)
# 3. a deliberately wrong fixed critic
cz = var.cparams
var.cparams = gc.randomize(var.cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
run("wrong_critic_K4", seed0=3_000_000)
setK(2)
run("wrong_critic_K2", seed0=3_000_000)
setK(4)
# 4. a trained critic (actor fixed)
var.cparams = cz
t = time.time()
losses = gc.fit_critic(var, TAU, LAM, steps=1500, B=16, seed0=900_000)
results["critic_fit"] = {"first": float(np.mean(losses[:20])), "last": float(np.mean(losses[-100:])),
                         "secs": time.time() - t}
print("critic fit", results["critic_fit"], flush=True)
run("trained_critic_K4", seed0=4_000_000)
setK(2)
run("trained_critic_K2", seed0=4_000_000)
print(f"total {time.time() - t0:.0f}s")
