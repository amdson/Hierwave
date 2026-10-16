"""Gradient check with the prior fine model (b0): crecur coarse gradient at K = 1, 16 vs exact joint / marginal."""
import json, sys
import numpy as np
from castlegen.variants import _crecur_exact as X
from castlegen.variants import _gradcheck as gc
p, var, lay, lam = X.setup(fine_scale=0.0)
ex = X.exact_tables(lay, var.params, lam)
gJ, gM, info = X.exact_coarse_grads(lay, var.params, lam, ex)
res = {"info": info, "cos_joint_marg": float(gc.flat(gJ) @ gc.flat(gM) / np.linalg.norm(gc.flat(gJ)) / np.linalg.norm(gc.flat(gM)))}
for K in (1, 4, 16):
    est = X.estimator_grads(lay, var.params, lam, K, 250, 16, 600_000 + 10_000 * K)
    for tgt, g in (("joint", gJ), ("marginal", gM)):
        z = gc.ztest(est, gc.flat(g), 16)
        res[f"K{K}_vs_{tgt}"] = {k: float(z[k]) for k in ("proj_z", "hotelling_p", "proj_ratio", "cos", "rel")}
    print(K, json.dumps({k: v for k, v in res.items() if k.startswith(f"K{K}")}), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("DONE")
