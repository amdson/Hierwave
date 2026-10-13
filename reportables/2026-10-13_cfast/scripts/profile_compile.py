"""Trace / lower / compile time of the record pass."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import ground, columns
from castlegen.bench.problem import Region
from castlegen.variants.cpass import CPass
from castlegen.variants import _cpass_core as core

prob = {"ground": ground.build_ground, "columns": columns.build_columns}[sys.argv[1]]()
B = 8
v = CPass(prob, 0, "/tmp")
region = Region(prob.W_train)
lay = v.layout(region)
lam = jnp.asarray(v.lam, jnp.float32)
seeds = jnp.arange(B, dtype=jnp.int32)
desl = [v.designed(int(s), region) for s in range(B)]
des = {k: jnp.stack([d[k] for d in desl]) for k in desl[0]}
f = jax.jit(jax.vmap(lambda p, s, d: core.coloured_pass(lay, p, s, d, 2.0, lam, record=True)[3], (None, 0, 0)))
t = time.time(); tr = f.trace(v.params, seeds, des); t1 = time.time()
lo = tr.lower(); t2 = time.time()
co = lo.compile(); t3 = time.time()
print(f"trace {t1-t:.2f} lower {t2-t1:.2f} compile {t3-t2:.2f}")
