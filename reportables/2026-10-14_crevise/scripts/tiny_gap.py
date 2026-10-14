"""Bound gap on the enumerable tiny problems (CPU): path KL minus the exact reverse KL of x_0 (log Z cancels), at
tau = 1 and lambda = 20 (the bench's cap), for random actor parameters and after... (random parameters only:
the gap of a trained sampler is reported from the dimers / ground runs as F_ub - F where F is available)."""
import json, sys
import jax, numpy as np
from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants._crevise_gradcheck import ExactPath, enumerate_trajectories, tiny_flat_problem
from castlegen.variants.crevise import CRevise
out = []
cases = [("flat", tiny_flat_problem, 1, 0.5, [1, 2, 3], [20.0]), ("two_level_coarse_rev", gc.tiny_problem, [0, 1], 0.5, [2], [20.0, 20.0])]
for name, mk, R, rho, seeds, lam in cases:
    for pscale in (0.0, 1.0):
        for ms in seeds:
            p = mk()
            var = CRevise(p, 0, "/tmp/crv_gap", R=R, rho=rho)
            if pscale > 0:
                var.params = gc.randomize(var.params, jax.random.PRNGKey(1), scale=pscale)
            lay = var.layout(Region(p.W_train))
            forced, final, masks = enumerate_trajectories(var.sched(lay), rho, ms)
            ex = ExactPath(var, 1.0, lam, ms, forced, final)
            r = ex.reverse_kl_parts(var.params, tau=1.0)
            r.update(case=name, mask_seed=ms, n_masked=int(sum(int(m.sum()) for m in masks.values())), params="random" if pscale else "init (b = 0)")
            print(json.dumps(r), flush=True)
            out.append(r)
json.dump(out, open("reportables/2026-10-14_crevise/runs/tiny_gap.json", "w"))
