"""Time own draws (vmapped train_pass record=False; the ckernel/cfast generator).
usage: bench_gen.py <problem> <W> <B,B,..> [old|new] [ablate]"""
import sys, time, os, json, importlib.util
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_speed1b")
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.variants import _cfast_core as fc
import common
pn, W, Bs, which = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
if which == "old":
    here = os.path.dirname(os.path.abspath(__file__))
    sp = importlib.util.spec_from_file_location("castlegen.variants._cfast_core_old", f"{here}/_cfast_core_old.py")
    m = importlib.util.module_from_spec(sp); m.__package__ = "castlegen.variants"; sys.modules[sp.name] = m
    sp.loader.exec_module(m); tp = m.train_pass
else:
    tp = fc.train_pass
p = get(pn); region = Region(W)
lay = core.Layout(p, region.W, region.periodic, region.x0)
params = common.init_params(p, lay, 0, scale=1.0)
lam = jnp.full(p.T + 1, 20.0)
f = jax.jit(jax.vmap(lambda s, d: tp(lay, params, s, d, 1.0, lam, record=False)[0]))
for B in Bs:
    ss = np.arange(100, 100 + B)
    ds = [p.designed_values(int(s), region) for s in ss]
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
    js = jnp.asarray(ss, jnp.int32)
    t0 = time.time(); o = f(js, dj); jax.block_until_ready(o); tc = time.time() - t0
    ts = []
    for _ in range(8):
        t0 = time.time(); o = f(js, dj); jax.block_until_ready(o); ts.append(time.time() - t0)
    t = float(np.median(ts))
    print(json.dumps({"problem": pn, "W": W, "B": B, "which": which, "compile_s": round(tc, 1), "s": round(t, 4), "worlds_per_s": round(B / t, 1)}), flush=True)
    if len(sys.argv) > 5:
        np.savez(sys.argv[5] + f"_{B}.npz", **{k: np.asarray(v) for k, v in o.items()})
