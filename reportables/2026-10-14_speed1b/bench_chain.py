"""Old (full-image) vs new chains in one process: min-of-N wall time of 4 moves on 32 worlds."""
import importlib.util, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels import block_mh as bm, coarse_flip as cf
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass
from castlegen.variants._cln_core import window_of
def load(name, path):
    sp = importlib.util.spec_from_file_location("castlegen.kernels." + name, path); m = importlib.util.module_from_spec(sp)
    m.__package__ = "castlegen.kernels"; sys.modules[sp.name] = m; sp.loader.exec_module(m); return m
bo = load("block_mh_old", "reportables/2026-10-14_speed1b/block_mh_old.py"); co = load("coarse_flip_old", "reportables/2026-10-14_speed1b/coarse_flip_old.py")
p = get("ground"); reg = Region(p.W_train)
ck = pickle.load(open("reportables/2026-10-14_sweep2a/runs/armB_s0/params_ground.pkl", "rb"))
lay = core.Layout(p, reg.W, True, 0, win=window_of((-2, 3)))
params = jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"]); lam_gen = jnp.asarray(ck["trained"]["lam"], jnp.float32)
NS = 32
ds = [p.designed_values(i, reg) for i in range(NS)]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
st = jax.jit(jax.vmap(lambda s, d: train_pass(lay, params, s, d, jnp.float32(1.0), lam_gen, record=False)[0]))(jnp.arange(NS, dtype=jnp.int32), dj)
keys = jax.random.split(jax.random.PRNGKey(0), NS)
def tm(f):
    jax.block_until_ready(f()); ts = []
    for _ in range(6):
        t = time.time(); jax.block_until_ready(f()); ts.append(time.time() - t)
    return min(ts) * 1000
res = {}
for name, fn in [("mtm_L2_K8_old", lambda: bo.make_chain(lay, [2], 1.0, 20.0, 8)), ("mtm_L2_K8_new", lambda: bm.make_chain(lay, [2], 1.0, 20.0, 8)),
                 ("flip_L34_K4_old", lambda: co.make_flip_chain(lay, [3, 4], 1.0, 2.0, "q", 20.0, 4)), ("flip_L34_K4_new", lambda: cf.make_flip_chain(lay, [3, 4], 1.0, 2.0, "q", 20.0, 4))]:
    ch = fn(); t0 = time.time(); jax.block_until_ready(ch(params, keys, st, 4)); comp = time.time() - t0
    res[name] = tm(lambda: ch(params, keys, st, 4)); print(name, "ms per call (4 moves x 32 worlds)", round(res[name], 1), "first call (compile) s", round(comp, 1), flush=True)
import json; json.dump(res, open("reportables/2026-10-14_speed1b/runs/bench_chain.json", "w"), indent=1)
