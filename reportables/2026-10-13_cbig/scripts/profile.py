"""s/update and GPU memory vs B for a cfast-family variant.
usage: python profile.py <variant> <problem> B1,B2,... [reps]"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.variant import load_variants

vname, pname, Bs = sys.argv[1], sys.argv[2], [int(b) for b in sys.argv[3].split(",")]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
V = load_variants()[vname]
prob = get(pname)
for B in Bs:
    v = V(prob, seed=0, workdir="/tmp", B=B)
    region = Region(v.cfg["W_train"] or prob.W_train)
    lay = v.layout(region)
    v.fn = v.train_fns(lay)
    v.gs = {k: jnp.float32(1.0) for k in v.params}
    tau, lam = jnp.float32(1.5), jnp.asarray(v.lam, jnp.float32)
    times, tb_all = [], []
    for r in range(reps + 1):
        seeds = np.arange(1000 + r * B, 1000 + (r + 1) * B)
        t0 = time.time()
        des = v._batch(seeds, region)
        jax.block_until_ready(des)
        tb = time.time() - t0
        t1 = time.time()
        o = v.fn["fused"](v.params, v.cparams, v.opt, v.copt, v.gs, jnp.asarray(seeds, jnp.int32), des, tau, lam,
                          1e-3, 1e-3, 10.0, jax.random.PRNGKey(r))
        jax.block_until_ready(o)
        tf = time.time() - t1
        if r == 0:
            tc = tf
        else:
            times.append(tf)
            tb_all.append(tb)
    ms = jax.devices()[0].memory_stats() or {}
    rec = {"variant": vname, "problem": pname, "B": B, "compile_s": tc, "fused_s": float(np.median(times)),
           "batch_s": float(np.median(tb_all)), "s_per_world": float(np.median(times) + np.median(tb_all)) / B,
           "peak_GB": ms.get("peak_bytes_in_use", 0) / 1e9}
    print(json.dumps(rec), flush=True)
