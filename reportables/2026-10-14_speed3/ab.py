"""Interleaved old/new timing of the vmapped generator (GPU is shared: alternate, report min and median of N).
usage: ab.py problem W B,B [N]   (old = _cfast_core_old.py in this folder)"""
import sys, time, os, json, importlib.util
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_speed1b")
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.variants import _cfast_core as fc
import common
here = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("castlegen.variants._cfast_core_old", f"{here}/_cfast_core_old.py")
m = importlib.util.module_from_spec(sp); m.__package__ = "castlegen.variants"; sys.modules[sp.name] = m; sp.loader.exec_module(m)
pn, W, Bs = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 20
p = get(pn); region = Region(W); lay = core.Layout(p, W, True, 0)
params = common.init_params(p, lay, 0, scale=1.0); lam = jnp.full(p.T + 1, 20.0)
fs = {n: jax.jit(jax.vmap(lambda s, d, tp=tp: tp(lay, params, s, d, 1.0, lam, record=False)[0])) for n, tp in (("old", m.train_pass), ("new", fc.train_pass))}
for B in Bs:
    ss = np.arange(100, 100 + B); ds = [p.designed_values(int(s), region) for s in ss]
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}; js = jnp.asarray(ss, jnp.int32)
    outs = {n: f(js, dj) for n, f in fs.items()}; jax.block_until_ready(outs)
    diff = sum(int((np.asarray(outs["old"][k]) != np.asarray(outs["new"][k])).sum()) for k in outs["old"])
    ts = {"old": [], "new": []}
    for _ in range(N):
        for n, f in fs.items():
            t0 = time.time(); o = f(js, dj); jax.block_until_ready(o); ts[n].append(time.time() - t0)
    r = {"problem": pn, "W": W, "B": B, "differing_cells": diff}
    for n in ts:
        r[n + "_min_ms"] = round(1e3 * min(ts[n]), 1); r[n + "_med_ms"] = round(1e3 * float(np.median(ts[n])), 1)
        r[n + "_worlds_per_s_min"] = round(B / min(ts[n]), 0)
    r["speedup_min"] = round(min(ts["old"]) / min(ts["new"]), 2); r["speedup_med"] = round(float(np.median(ts["old"]) / np.median(ts["new"])), 2)
    print(json.dumps(r), flush=True)
