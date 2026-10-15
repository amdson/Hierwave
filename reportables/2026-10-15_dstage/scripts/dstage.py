"""Pure kernel mixing (no learning): p*-invariant fine-block Gibbs chains on dimers (bench region 32 x 64) from several starts.
usage: chain_relax.py <out.json> <cycles> ; starts: cfast60 own draws (two orientations), untrained-init own draws, probe tilings
(squares = h/v only, rhombi = h/d only).  Kernels: k=2 only, k=3 only, k=2+3.  Logs orientation fractions every `every` cycles."""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants
from castlegen.kernels.fine_block import FineBlockKernel
from castlegen.variants._cfast_core import train_pass
out = sys.argv[1]; NW = int(sys.argv[2]); CK = [10, 30, 60, 120]
p = get_problem("dimers"); region = Region(p.W_train)
c = load_variants()["cfast"](p, seed=0, workdir="/tmp/dk_relax"); lay = c.layout(region)
def draws(params, lam, n=NW, s0=40_000_000):
    lg = jnp.asarray(lam, jnp.float32)
    gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lg, record=False)[0], in_axes=(None, 0, 0)))
    ss = np.arange(s0, s0 + n); ds = [p.designed_values(int(s), region) for s in ss]
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
    st = gen(params, jnp.asarray(ss, jnp.int32), dj)
    return {ch.name: jnp.asarray(st[ch.name] if ch.name in st else dj[ch.name], jnp.int32) for ch in p.chans}
ck = pickle.load(open("reportables/2026-10-14_dimers_kernel/runs/cfast60.pkl", "rb"))
from castlegen.bench.problems import dimers as dm
st0 = draws(jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"]), ck["trained"]["lam"])
FB = {(k, l): FineBlockKernel(p, region, k, lam=l, seed=0) for k in (2, 3) for l in (None, 20.0)}
fl = FB[(2, None)].flat
def to_flat(st):
    n = next(iter(st.values())).shape[0]
    return jnp.concatenate([jnp.asarray(st[ch.name]).reshape(n, -1) for ch in p.chans], 1).astype(jnp.int32)
def pch(v): return np.asarray(v[:, fl.off["p"]:fl.off["p"] + p.H * region.W]) // 2
from castlegen.kernels.amortised import make_energy
E = make_energy(fl, FB[(2, None)].Ks[0].insts, fine_max=0)
def stats(v, v0):
    pp = pch(v); fr = [round(float((pp == i).mean()), 4) for i in range(3)]
    viol = float(np.asarray(E(v)[1]).mean()) / (p.H * region.W)
    ch = float((pch(v) != pch(v0)).mean())
    return fr, viol, ch
res = {}
v0 = to_flat(st0); print("start", stats(v0, v0), flush=True)
for (k, l), K in FB.items():
    v = v0
    for i in range(3): jax.block_until_ready(K.sweep(v, jax.random.PRNGKey(5 + i))[0])  # warm
    key = jax.random.PRNGKey(1); t = 0.0; rows = []
    for s in range(1, max(CK) + 1):
        key, sub = jax.random.split(key); t0 = time.time()
        v, _ = K.sweep(v, sub); jax.block_until_ready(v); t += time.time() - t0
        if s in CK:
            fr, viol, ch = stats(v, v0)
            rows.append(dict(k=s, hvd=fr, viol=viol, changed=ch, ms_per_world=1000 * t / NW))
            print(f"block{k} lam{l} sweeps {s} hvd {fr} viol {viol:.2e} changed {ch:.3f} ms/world {1000*t/NW:.1f}", flush=True)
    res[f"k{k}_lam{l}"] = rows; json.dump(res, open(out, "w"))
