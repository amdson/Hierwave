"""Interleaved timings of full replay passes (GPU contention varies): N, override on/off, per-level sum."""
import sys, time, json
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
load_variants()                       # the bench's import order (and compile-cache dir)
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.creplay import CReplay
from castlegen.variants._cfast_core import train_pass

prob = problems.get(sys.argv[1])
v = CReplay(prob, 0, "/tmp")
region = Region(prob.W_train)
lay = v.layout(region)
fn = v.train_fns(lay)
B = 8
lam = jnp.asarray(v.lam, jnp.float32); tau = jnp.float32(1.0)
seeds = jnp.arange(B, dtype=jnp.int32) + 1000
des = v._batch(np.asarray(seeds), region)
rc = jax.jit(fn["replay_costs"])
nov = jax.jit(lambda p, s, d: jax.vmap(lambda s1, d1: train_pass(lay, p, s1, d1, tau, lam, record=False)[1])(s, d))
Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [32, 64, 128, 256, 512]
args = {}
for N in Ns:
    b = jnp.arange(N) % B
    z = jnp.zeros((N,), jnp.int32)
    args[N] = (seeds[b], {k: x[b] for k, x in des.items()}, z + 3, z, z)
    jax.block_until_ready(rc(v.params, *args[N], ) if False else rc(v.params, args[N][0], args[N][1], tau, lam, *args[N][2:]))
    jax.block_until_ready(nov(v.params, args[N][0], args[N][1]))
res = {N: {"ov": [], "noov": []} for N in Ns}
for rnd in range(4):
    for N in Ns:
        for nm, f in (("ov", lambda: rc(v.params, args[N][0], args[N][1], tau, lam, *args[N][2:])),
                      ("noov", lambda: nov(v.params, args[N][0], args[N][1]))):
            t0 = time.time()
            for _ in range(3):
                o = f()
            jax.block_until_ready(o)
            res[N][nm].append((time.time() - t0) / 3)
for N in Ns:
    print(N, {k: [round(x, 4) for x in sorted(v_)] for k, v_ in res[N].items()},
          "ms/replay(min ov)", round(1000 * min(res[N]["ov"]) / N, 3), flush=True)
