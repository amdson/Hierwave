"""Per-world gradient SNR of the cpass estimator vs world size, no exact
gradient needed: signal |E g|^2 estimated as |mean|^2 - tr(Cov)/M (unbiased),
noise tr(Cov of one world's gradient).  Zero critic, untrained actor (the
init the bench starts from), tau 1.5, lambda as cpass starts (10).
PYTHONPATH=. python ... <M> <B>"""
import json
import sys
import time

import jax
import numpy as np

from castlegen.bench.problems import get as get_problem
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cpass import CPass

M = int(sys.argv[1]) if len(sys.argv) > 1 else 200
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TAU = 1.5
out = []


def snr(var, name, lam, seed0, random_actor):
    if random_actor:
        var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    t = time.time()
    keys = sorted(var.params)
    est = np.array([gc.flat(gc.estimator_grad(var, np.arange(seed0 + m * B, seed0 + (m + 1) * B), TAU, lam)[0], keys)
                    for m in range(M)])
    sl = gc.block_slices(var.params, keys)
    lay = var.layout(gc.Region(var.problem.W_train))
    row = {"problem": name, "W": var.problem.W_train, "secs": round(time.time() - t, 1)}
    for k, s in sl.items():
        e = est[:, s]
        m = e.mean(0)
        trc = e.var(0, ddof=1).sum()                  # tr Cov of a batch mean = tr Cov(world) / B
        sig = m @ m - trc / M
        l = int(k[1:].split("_")[0])
        nd = lay.nval[l]
        row[k] = {"snr_world": float(sig / (B * trc)), "draws": nd, "snr_per_draw": float(sig / (B * trc) / nd)}
    out.append(row)
    print(json.dumps(row), flush=True)


for W in (8, 16, 32):
    p = gc.tiny_problem()
    p.W_train = W
    snr(CPass(p, 0, "/tmp/gcw"), "tiny", np.array([2.0, 3.0]), 5_000_000, True)
for name in ("toy_support", "dimers"):
    p = get_problem(name)
    v = CPass(p, 0, "/tmp/gcw")
    snr(v, name, np.full(p.T + 1, 10.0), 6_000_000, False)
json.dump(out, open("reportables/2026-10-13_gradcheck/snr_scaling.json", "w"), indent=1)
