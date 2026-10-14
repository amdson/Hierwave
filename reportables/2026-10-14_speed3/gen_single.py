"""Single-world generate() path (cpass.CPass.generate: coloured_pass + repair + crop, host round trip), old vs new coloured_pass.
usage: gen_single.py problem W"""
import sys, time, os, importlib.util, numpy as np, jax, jax.numpy as jnp
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
here = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("castlegen.variants._old_cp", f"{here}/_cpass_core_old.py")
old = importlib.util.module_from_spec(sp); old.__package__ = "castlegen.variants"; sys.modules[sp.name] = old; sp.loader.exec_module(old)
sys.path.insert(0, "reportables/2026-10-14_speed1b"); import common
pn, W = sys.argv[1], int(sys.argv[2]); p = get(pn); region = Region(W); lay = core.Layout(p, W, True, 0)
params = common.init_params(p, lay, 0, scale=1.0); lam = jnp.full(p.T + 1, 20.0)
fs = {n: (jax.jit(lambda pr, s, d, m=m: m.coloured_pass(lay, pr, s, d, 1.0, lam)[0]), jax.jit(lambda st, s: core.repair(lay, st, s))) for n, m in (("old", old), ("new", core))}
des = lambda s: {k: jnp.asarray(v, jnp.int32) for k, v in p.designed_values(s, region).items()}
res = {}
for n, (g, r) in fs.items():
    d = des(1); np.asarray(r(g(params, 1, d), 1)["t"] if "t" in p.chan else 0)
    ts = []
    for s in range(2, 42):
        t0 = time.time(); d = des(s); raw = g(params, s, d); fin = r(raw, s)
        out = {c.name: np.asarray(fin[c.name]) for c in p.sampled()}; ts.append(time.time() - t0)
    res[n] = (min(ts) * 1e3, float(np.median(ts)) * 1e3)
print(pn, W, "generate() ms/world (min, median):", {k: tuple(round(x, 1) for x in v) for k, v in res.items()})
