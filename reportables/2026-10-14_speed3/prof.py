"""GPU kernel profile of the vmapped generator via jax.profiler (perfetto json). usage: prof.py problem W B old|new outdir"""
import sys, os, json, gzip, glob, importlib.util, collections
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_speed1b")
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.variants import _cfast_core as fc
import common
pn, W, B, which, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
tp = fc.train_pass
if which == "old":
    here = os.path.dirname(os.path.abspath(__file__))
    sp = importlib.util.spec_from_file_location("castlegen.variants._cfast_core_old", f"{here}/_cfast_core_old.py")
    m = importlib.util.module_from_spec(sp); m.__package__ = "castlegen.variants"; sys.modules[sp.name] = m; sp.loader.exec_module(m); tp = m.train_pass
p = get(pn); region = Region(W); lay = core.Layout(p, W, True, 0)
params = common.init_params(p, lay, 0, scale=1.0); lam = jnp.full(p.T + 1, 20.0)
f = jax.jit(jax.vmap(lambda s, d: tp(lay, params, s, d, 1.0, lam, record=False)[0]))
ss = np.arange(100, 100 + B); ds = [p.designed_values(int(s), region) for s in ss]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}; js = jnp.asarray(ss, jnp.int32)
jax.block_until_ready(f(js, dj))
with jax.profiler.trace(out, create_perfetto_trace=True):
    for _ in range(3): jax.block_until_ready(f(js, dj))
fn = glob.glob(out + "/**/*.trace.json.gz", recursive=True)[0]
tr = json.load(gzip.open(fn))
ev = tr["traceEvents"]; names = {}
for e in ev:
    if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"): names[(e["pid"], e.get("tid") if e["name"] == "thread_name" else None)] = e["args"]["name"]
agg = collections.defaultdict(lambda: [0, 0.0]); tot = 0.0; nk = 0
for e in ev:
    if e.get("ph") != "X": continue
    pn_ = names.get((e["pid"], None), "")
    if "GPU" not in pn_ and "gpu" not in pn_: continue
    tn = names.get((e["pid"], e["tid"]), "")
    if "Stream" not in tn and "stream" not in tn: continue
    a = agg[e["name"].split(".")[0]]; a[0] += 1; a[1] += e["dur"]; tot += e["dur"]; nk += 1
print("kernels/call", nk / 3, "gpu busy ms/call", tot / 3e3)
for k, (n, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:25]: print(f"{k:50s} n/call {n/3:8.0f} ms/call {d/3e3:7.2f}")
