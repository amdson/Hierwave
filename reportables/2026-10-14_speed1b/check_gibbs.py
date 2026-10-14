"""Fast JaxBlockGibbs vs reference path: same key -> same sweeps; timings."""
import time, jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.amortised import JaxBlockGibbs, make_energy
p = get("ground"); reg = Region(p.W_train)
for L, d in [(4, 0), (3, 0)]:
    J = JaxBlockGibbs(p, reg, L, d, lam=20.0, chunk=4096); print(L, d, "fast", J.fast)
    fl = J.K.flat; E = make_energy(fl, J.K.insts)
    v = jnp.asarray(np.stack([fl.flatten({**p.probes(int(s), reg, n=k + 1)[k], **p.designed_values(int(s), reg)}) for s, k in [(11, 0), (12, 3)] * 16]), jnp.int32)
    va, vb = v, v
    for i in range(3):
        key = jax.random.PRNGKey(i)
        va, ca = J.sweep(va, key); vb, cb = J.sweep_ref(vb, key)
        print(" sweep", i, "frac differing entries", float((va != vb).mean()), "changed", int(ca.sum()), int(cb.sum()), "soft", float(E(va)[0].mean()), float(E(vb)[0].mean()))
        vb = va
    for name, f in (("fast", J.sweep), ("ref", J.sweep_ref)):
        f(v, key)[0].block_until_ready(); t = time.time()
        for _ in range(5): r = f(v, key)[0]
        r.block_until_ready(); print("  ", name, round((time.time() - t) / 5 * 1000, 1), "ms/sweep")
