"""Interleaved timing of several versions of the vmapped generator (shared GPU: alternate, report min / p10 / median).
usage: abn.py problem W B,B N file1.py file2.py ...   (files are copies of _cfast_core.py; the first is the reference)"""
import sys, time, os, json, importlib.util
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_speed1b")
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
import common
pn, W, Bs, N, files = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5:]
mods = {}
for f in files:
    nm = os.path.basename(f)[:-3]
    sp = importlib.util.spec_from_file_location("castlegen.variants._tmp_" + nm, f)
    m = importlib.util.module_from_spec(sp); m.__package__ = "castlegen.variants"; sys.modules[sp.name] = m; sp.loader.exec_module(m); mods[nm] = m
p = get(pn); region = Region(W); lay = core.Layout(p, W, True, 0)
params = common.init_params(p, lay, 0, scale=1.0); lam = jnp.full(p.T + 1, 20.0)
fs = {n: jax.jit(jax.vmap(lambda s, d, tp=m.train_pass: tp(lay, params, s, d, 1.0, lam, record=False)[0])) for n, m in mods.items()}
names = list(fs)
for B in Bs:
    ss = np.arange(100, 100 + B); ds = [p.designed_values(int(s), region) for s in ss]
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}; js = jnp.asarray(ss, jnp.int32)
    outs = {n: f(js, dj) for n, f in fs.items()}; jax.block_until_ready(outs)
    ts = {n: [] for n in names}
    for _ in range(N):
        for n in names:
            t0 = time.time(); o = fs[n](js, dj); jax.block_until_ready(o); ts[n].append(time.time() - t0)
    for n in names:
        diff = sum(int((np.asarray(outs[names[0]][k]) != np.asarray(outs[n][k])).sum()) for k in outs[n])
        a = np.array(ts[n]) * 1e3
        print(json.dumps({"problem": pn, "W": W, "B": B, "version": n, "diff_vs_first": diff, "min_ms": round(a.min(), 1), "p10_ms": round(float(np.percentile(a, 10)), 1), "med_ms": round(float(np.median(a)), 1), "worlds_per_s_at_min": round(B / a.min() * 1e3)}), flush=True)
