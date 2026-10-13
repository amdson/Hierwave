"""Lower vs compile time of cbig's fused update (persistent cache off).
usage: python compile_time.py <problem> <B>   (set XLA_FLAGS to compare)"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_compilation_cache", False)
from castlegen.bench.problem import Region  # noqa: E402
from castlegen.bench.problems import get  # noqa: E402
from castlegen.variants.cbig import CBig  # noqa: E402

p = get(sys.argv[1])
B = int(sys.argv[2])
v = CBig(p, seed=0, workdir="/tmp", B=B)
region = Region(p.W_train)
lay = v.layout(region)
t0 = time.time()
fn = v.train_fns(lay)
t1 = time.time()
seeds = np.arange(B)
des = v._batch(seeds, region)
gs = {k: jnp.float32(1.0) for k in v.params}
args = (v.params, v.cparams, v.opt, v.copt, gs, jnp.asarray(seeds, jnp.int32), des, jnp.float32(1.5),
        jnp.asarray(v.lam, jnp.float32), 1e-3, 1e-3, 10.0, jax.random.PRNGKey(0))
lo = fn["fused"].lower(*args)
t2 = time.time()
c = lo.compile()
t3 = time.time()
print(json.dumps({"problem": sys.argv[1], "B": B, "build_s": t1 - t0, "trace_lower_s": t2 - t1,
                  "compile_s": t3 - t2}), flush=True)
