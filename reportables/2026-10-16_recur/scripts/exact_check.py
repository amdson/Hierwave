"""Guard (1): bias of -log Z_K vs exact subtree free energy on the tiny two-level problem, K = 1, 4, 16, and the
coarse-gradient check.  usage: python exact_check.py <out.json>   (CPU)"""
import json, sys, time
import numpy as np
import jax
from castlegen.variants import _crecur_exact as X
from castlegen.variants import _gradcheck as gc

out, t0 = {}, time.time()
N_C = 3000
for name, sc in (("b0_prior_fine", 0.0), ("b1_random_fine", 1.0), ("b2_random_fine_x3", 3.0)):
    p, var, lay, lam = X.setup(fine_scale=sc)
    ex = X.exact_tables(lay, var.params, lam)
    C, F = X.mc_fine_sums(lay, var.params, lam, N_C, 16)
    idx = np.array([ex["cs"].index(tuple(c.ravel())) for c in C])
    top = np.argsort(-np.bincount(idx, minlength=len(ex["cs"])))[:5]
    ex = X.exact_tables(lay, var.params, lam, all_t=top) if name == "b0_prior_fine" else ex
    rows, per_c = X.bias_table(ex, C, F)
    out[name] = {"rows": rows, "per_c": per_c, "n_coarse_states": len(ex["cs"]), "n_states": int(len(ex["ci"]))}
    if name == "b0_prior_fine":
        rng = np.random.default_rng(0)
        out["a_exact_fine"] = X.exact_sampler_bias(ex, 500, 16, rng)
    print(name, json.dumps(rows), round(time.time() - t0), flush=True)
    if name == "b1_random_fine":
        gJ, gM, info = X.exact_coarse_grads(lay, var.params, lam, ex)
        res = {"info": info}
        for K in (1, 16):
            est = X.estimator_grads(lay, var.params, lam, K, 250, 16, 300_000 + 10_000 * K)
            for tgt, g in (("joint", gJ), ("marginal", gM)):
                z = gc.ztest(est, gc.flat(g), 16)
                res[f"K{K}_vs_{tgt}"] = {k: float(z[k]) for k in ("proj_z", "rand_z", "hotelling_p", "proj_ratio", "rel_noise", "cos", "rel")}
            res[f"K{K}_cos_joint_marg"] = float(gc.flat(gJ) @ gc.flat(gM) / np.linalg.norm(gc.flat(gJ)) / np.linalg.norm(gc.flat(gM)))
            print("grad", K, json.dumps(res), round(time.time() - t0), flush=True)
        out["gradcheck_b1"] = res
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("DONE", round(time.time() - t0))
