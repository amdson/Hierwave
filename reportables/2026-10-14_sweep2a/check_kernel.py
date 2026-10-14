"""Check JaxBlockGibbs energies against ExactBlockGibbs (numpy) and time sweeps."""
import time, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.exact_block import total_energy
from castlegen.kernels.amortised import JaxBlockGibbs, make_energy, internal_pruned
p = get("ground"); reg = Region(p.W_train)
xs = [{**p.probes(int(s), reg, n=k + 1)[k], **p.designed_values(int(s), reg)} for s, k in [(11, 0), (12, 3)]]
for (L, d, cm) in [(2, 0, "full"), (1, 1, "full"), (2, 1, "pruned")]:
    t = time.time(); J = JaxBlockGibbs(p, reg, L, d, lam=None, comps=cm, chunk=None if cm == "full" else 2048)
    print(L, d, "comps", J.comps.shape, "build", round(time.time() - t, 2), flush=True)
    K = J.K
    v = np.stack([K.flat.flatten(x) for x in xs])
    for k in range(K.ncol):
        BV, nbk, gs = J.classes[k]
        comps = jnp.asarray(J.comps[:300])
        sj, nj = J._class_energy(jnp.asarray(v, jnp.int32), gs, nbk, comps)
        sn, nn = K.energies(v[1], k, comps=np.asarray(J.comps[:300], np.int64))
        print("  class", k, "max |dsoft|", float(np.abs(np.asarray(sj[1]) - sn).max()), "nv eq", bool((np.asarray(nj[1]) == nn).all()))
    E = make_energy(K.flat, K.insts)
    s, nv, sf = E(jnp.asarray(v, jnp.int32))
    print("  energy jax", np.asarray(s), np.asarray(nv), "numpy", [total_energy(K.flat, K.insts, vv) for vv in v])
    vb = jnp.asarray(np.repeat(v, 16, 0), jnp.int32)
    t = time.time(); vb2, ch = J.sweep(vb, jax.random.PRNGKey(0)); vb2.block_until_ready(); t1 = time.time() - t
    t = time.time(); vb2, ch = J.sweep(vb2, jax.random.PRNGKey(1)); vb2.block_until_ready()
    print("  sweep B=32 compile+run", round(t1, 2), "run", round(time.time() - t, 3), "changed", np.asarray(ch)[:4],
          "E after", np.asarray(E(vb2)[0])[:4] / 8192, "nv", np.asarray(E(vb2)[1])[:4])
