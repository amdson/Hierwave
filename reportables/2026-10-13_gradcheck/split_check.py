"""Follow-up on the per-parameter chi^2 > 1 cases of run_gradcheck.py: fresh,
disjoint seeds; two independent halves.  If the estimator were biased, the
deviations (mean - exact) of the two halves would be positively correlated;
heavy-tailed noise inflates chi^2 without correlating them."""
import json
import sys

import jax
import numpy as np

from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cpass import CPass

TAU, LAM = 1.5, np.array([2.0, 3.0])
M, B = int(sys.argv[1]) if len(sys.argv) > 1 else 3000, 16
p = gc.tiny_problem()
var = CPass(p, 0, "/tmp/gcw", K=4)
var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
cz = gc.zero_critic(var.cparams)
cw = gc.randomize(var.cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
ex = gc.Exact(var, LAM, cache="/tmp/gradcheck_tiny_exact.npz")
J, g, info = ex.J_and_grad(var.params, TAU)
keys = sorted(g)
gf, sl = gc.flat(g, keys), gc.block_slices(g, keys)


def setK(k):
    var.cfg["K"] = k
    lay = var.layout(Region(8))
    var._build_train(lay)
    if hasattr(lay, "jit_agrad"):
        del lay.jit_agrad


out = {}
for name, K, cp, seed0 in (("zero_critic_K2", 2, cz, 100_000_000), ("wrong_critic_K2", 2, cw, 200_000_000),
                           ("zero_critic_K4", 4, cz, 300_000_000)):
    setK(K)
    var.cparams = cp
    est = np.array([gc.flat(gc.estimator_grad(var, np.arange(seed0 + m * B, seed0 + (m + 1) * B), TAU, LAM)[0],
                            keys) for m in range(M)])
    res = {}
    for k, s in sl.items():
        e, x = est[:, s], gf[s]
        h1, h2 = e[: M // 2], e[M // 2:]
        d1, d2 = h1.mean(0) - x, h2.mean(0) - x
        sd = e.std(0, ddof=1)
        ok = sd > 0
        w = 1 / sd[ok]
        corr = float(np.corrcoef(d1[ok] * w, d2[ok] * w)[0, 1])
        z = gc.ztest(e, x, B)
        # excess kurtosis of the per-batch gradients: heavy tails make the sd estimate unreliable
        c = e[:, ok] - e[:, ok].mean(0)
        kurt = float(np.median((c ** 4).mean(0) / (c ** 2).mean(0) ** 2 - 3))
        res[k] = {"proj_z": z["proj_z"], "proj_ratio": z["proj_ratio"], "rand_z": z["rand_z"], "chi2": z["chi2"],
                  "maxz": z["maxz"], "rel": z["rel"], "rel_noise": z["rel_noise"], "snr_world": z["snr_world"],
                  "half_corr": corr, "half_corr_se": 1 / np.sqrt(ok.sum()), "median_excess_kurtosis": kurt}
    out[name] = res
    print(name, json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in res.items()}), flush=True)
json.dump(out, open("reportables/2026-10-13_gradcheck/split_results.json", "w"), indent=1)
