"""dthin (adapted from dgate_stage.py): thin alternating-orientation blocks. usage: dthin_stage.py out.json nworlds KHxKW
(orig doc) dgate part 2: fixed block-sweep stages on dimers vs exact gate 0.351/0.324/0.324 +-0.02.
usage: dgate_stage.py <out.json> <nworlds> <config> ; config in {k3, mix, k4}
Starts: collapsed cfast60 draws; random valid tiling (probes mode 2, seed per world)."""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants
from castlegen.variants._dthin import ThinKernel
from castlegen.kernels.amortised import make_energy
from castlegen.variants._cfast_core import train_pass
out, NW, cfg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
TGT = np.array([0.351, 0.324, 0.324]); EVERY = 30
KMAX = 240
CK = set(range(EVERY, KMAX + 1, EVERY))
p = get_problem("dimers"); region = Region(p.W_train)
c = load_variants()["cfast"](p, seed=0, workdir="/tmp/dg_stage"); lay = c.layout(region)
ck = pickle.load(open("reportables/2026-10-14_dimers_kernel/runs/cfast60.pkl", "rb"))
params = jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"]); lg = jnp.asarray(ck["trained"]["lam"], jnp.float32)
gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lg, record=False)[0], in_axes=(None, 0, 0)))
ss = np.arange(40_000_000, 40_000_000 + NW); ds = [p.designed_values(int(s), region) for s in ss]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
CH = 4  # GPU 0 has only ~4 GB free (another process holds 12 GB): chunk worlds
parts = [gen(params, jnp.asarray(ss[i:i+CH], jnp.int32), {k: v[i:i+CH] for k, v in dj.items()}) for i in range(0, NW, CH)]
st = {k: jnp.concatenate([q[k] for q in parts]) for k in parts[0]}
st0 = {ch.name: jnp.asarray(st[ch.name] if ch.name in st else dj[ch.name], jnp.int32) for ch in p.chans}
def to_flat(st):
    n = next(iter(st.values())).shape[0]
    return jnp.concatenate([jnp.asarray(st[ch.name]).reshape(n, -1) for ch in p.chans], 1).astype(jnp.int32)
pr = [p.probes(int(s), region, 4)[2] for s in range(100, 100 + NW)]
st1 = {ch.name: jnp.asarray(np.stack([q[ch.name] for q in pr]), jnp.int32) for ch in p.chans}
starts = {"collapsed": to_flat(st0), "tiling": to_flat(st1)}
kh, kw = map(int, cfg.split("x")); KS = [ThinKernel(p, region, kh, kw, lam=20.0, seed=0)]; print("ncomps", KS[0].nc, flush=True)
fl = KS[0].flat
E = make_energy(fl, KS[0].Js[0][0].K.insts, fine_max=0)
def pch(v): return np.asarray(v[:, fl.off["p"]:fl.off["p"] + p.H * region.W]) // 2
def stats(v):
    pp = pch(v); return [round(float((pp == i).mean()), 4) for i in range(3)], float(np.asarray(E(v)[1]).mean()) / (p.H * region.W)
res = {}
for sn, v0 in starts.items():
    for i in range(2):
        for K in KS: jax.block_until_ready(K.sweep(v0[:CH], jax.random.PRNGKey(5 + i))[0])
    v = v0; key = jax.random.PRNGKey(1); t = 0.0; rows = []
    fr, viol = stats(v); print(cfg, sn, "start", fr, viol, flush=True); rows.append(dict(k=0, hvd=fr, viol=viol))
    for s in range(1, KMAX + 1):
        t0 = time.time()
        for K in KS:
            key, sub = jax.random.split(key); v = jnp.concatenate([K.sweep(v[i:i+CH], sub)[0] for i in range(0, NW, CH)])
        jax.block_until_ready(v); t += time.time() - t0
        if s in CK:
            fr, viol = stats(v); ok = bool(np.all(np.abs(np.array(fr) - TGT) <= 0.02))
            rows.append(dict(k=s, hvd=fr, viol=viol, ms_per_world=1000 * t / NW, in_gate=ok))
            print(cfg, sn, s, fr, f"viol {viol:.2e} ms/w {1000*t/NW:.1f} gate {ok}", flush=True)
    res[sn] = rows; json.dump(res, open(out, "w"))
