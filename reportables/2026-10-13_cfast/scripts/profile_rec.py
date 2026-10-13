"""Record-pass cost with and without critic features, and plain generation."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import ground, columns
from castlegen.bench.problem import Region
from castlegen.variants.cpass import CPass
from castlegen.variants import _cpass_core as core

prob = {"ground": ground.build_ground, "columns": columns.build_columns}[sys.argv[1]]()
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
v = CPass(prob, 0, "/tmp")
region = Region(prob.W_train)
lay = v.layout(region)
lam = jnp.asarray(v.lam, jnp.float32)
extra = jnp.concatenate([jnp.array([2.0]), lam]).astype(jnp.float32)
seeds = jnp.arange(B, dtype=jnp.int32)
desl = [v.designed(int(s), region) for s in range(B)]
des = {k: jnp.stack([d[k] for d in desl]) for k in desl[0]}
fns = {
    "gen": jax.jit(jax.vmap(lambda p, s, d: core.coloured_pass(lay, p, s, d, 2.0, lam)[1], (None, 0, 0))),
    "rec_nocrit": jax.jit(jax.vmap(lambda p, s, d: core.coloured_pass(lay, p, s, d, 2.0, lam, record=True)[3],
                                   (None, 0, 0))),
    "rec_crit": jax.jit(jax.vmap(lambda p, s, d: core.coloured_pass(lay, p, s, d, 2.0, lam, record=True,
                                                                    critic_extra=extra)[3], (None, 0, 0))),
}
for k, f in fns.items():
    t = time.time(); jax.block_until_ready(f(v.params, seeds, des)); c = time.time() - t
    ts = []
    for _ in range(3):
        t = time.time(); jax.block_until_ready(f(v.params, seeds + 7, des)); ts.append(time.time() - t)
    print(f"{k}: compile+run {c:.2f}  run {min(ts):.3f}", flush=True)
