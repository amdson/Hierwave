"""Seconds per cfine update (copy of creplay scripts/timing.py) (fused, GPU) for a few configurations.
usage: timing.py problem 'B=8,R=(None,64),Kr=4' ..."""
import sys, time, json
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
load_variants()     # the bench's import order: same compile-cache dir as the bench (round 3 cache-miss cause)
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.cfine import CFine as CReplay

prob = problems.get(sys.argv[1])
for spec in sys.argv[2:]:
    cfg = eval(f"dict({spec})")
    v = CReplay(prob, 0, "/tmp", **cfg)
    region = Region(prob.W_train)
    lay = v.layout(region)
    t0 = time.time()
    v.fn = v.train_fns(lay)
    gs = {k: jnp.float32(1.0) for k in v.params}
    B = v.cfg["B"]
    lam = jnp.asarray(v.lam, jnp.float32)

    def go(u):
        seeds = jnp.arange(B, dtype=jnp.int32) + 1000 + B * u
        des = v._batch(np.asarray(seeds), region)
        out = v.fn["fused"](v.params, v.cparams, v.opt, v.copt, gs, seeds, des, jnp.float32(1.0), lam, 1e-3, 1e-3,
                            10.0, jax.random.PRNGKey(u))
        jax.block_until_ready(out[5])
        return out
    out = go(0)
    tc = time.time() - t0
    t1 = time.time()
    n = 5
    for u in range(1, n + 1):
        out = go(u)
    dt = (time.time() - t1) / n
    d = jax.device_get(out[9])
    print(json.dumps({"spec": spec, "compile_s": round(tc, 1), "s_per_update": round(dt, 3), "replays": B * v.cfg["Kr"] * sum(b["R"] for b in v.fn["btab"].values()),
                      "branch": {str(l): {"R": b["R"], "Nc": b["Nc"]} for l, b in v.fn["btab"].items()},
                      "diag": {l: {k: round(float(x), 3) for k, x in dd.items()} for l, dd in d.items()}}),
          flush=True)
