"""Host-side cost of own draws: designed_values (numpy), stacking/transfer, device pass. usage: host_prof.py problem W n"""
import sys, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "reportables/2026-10-14_speed1b")
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass
import common
pn, W, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = get(pn); region = Region(W); lay = core.Layout(p, W, True, 0)
params = common.init_params(p, lay, 0, scale=1.0); lam = jnp.full(p.T + 1, 20.0)
gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lam, record=False)[0], in_axes=(None, 0, 0)))
def run(base):
    t0 = time.time(); ss = np.arange(base, base + n)
    ds = [p.designed_values(int(s), region) for s in ss]; t1 = time.time()
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}; js = jnp.asarray(ss, jnp.int32); jax.block_until_ready(dj); t2 = time.time()
    st = gen(params, js, dj); jax.block_until_ready(st); t3 = time.time()
    return t1 - t0, t2 - t1, t3 - t2
run(0); r = np.array([run(1000 * (i + 1)) for i in range(15)])
print(pn, W, n, "designed_values ms %.1f  stack+transfer ms %.1f  device pass ms %.1f" % tuple(1e3 * np.median(r, 0)))
