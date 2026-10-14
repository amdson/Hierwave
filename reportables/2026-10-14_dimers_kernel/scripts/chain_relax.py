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
out, cycles = sys.argv[1], int(sys.argv[2]); every = 10
p = get_problem("dimers"); region = Region(p.W_train)
c = load_variants()["cfast"](p, seed=0, workdir="/tmp/dk_relax"); lay = c.layout(region)
def draws(params, lam, n=32, s0=40_000_000):
    lg = jnp.asarray(lam, jnp.float32)
    gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lg, record=False)[0], in_axes=(None, 0, 0)))
    ss = np.arange(s0, s0 + n); ds = [p.designed_values(int(s), region) for s in ss]
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
    st = gen(params, jnp.asarray(ss, jnp.int32), dj)
    return {ch.name: jnp.asarray(st[ch.name] if ch.name in st else dj[ch.name], jnp.int32) for ch in p.chans}
ck = pickle.load(open("reportables/2026-10-14_dimers_kernel/runs/cfast60.pkl", "rb"))
starts = {"cfast60_draws": draws(jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"]), ck["trained"]["lam"]),
          "untrained_draws": draws(jax.tree_util.tree_map(jnp.asarray, ck["untrained"]["params"]), ck["untrained"]["lam"])}
for name, mode in (("probe_squares", 0), ("probe_rhombi", 1)):
    xs = []
    for i in range(32):
        from castlegen.bench.problems import dimers as dm
        pp = dm._tiling(100 + i, i, region, mode)
        xs.append({"p": pp, **p.designed_values(i, region)})
    starts[name] = {ch.name: jnp.asarray(np.stack([x[ch.name] for x in xs]), jnp.int32) for ch in p.chans}
FB = {k: FineBlockKernel(p, region, k, lam=None, seed=0) for k in (2, 3)}
fl = FB[2].flat
def to_flat(st):
    n = next(iter(st.values())).shape[0]
    return jnp.concatenate([jnp.asarray(st[ch.name]).reshape(n, -1) for ch in p.chans], 1).astype(jnp.int32)
def op(v):
    pp = np.asarray(v[:, fl.off["p"]:fl.off["p"] + p.H * region.W]) // 2
    return [round(float((pp == i).mean()), 4) for i in range(3)]
from castlegen.kernels.amortised import make_energy
E = make_energy(fl, FB[2].Ks[0].insts, fine_max=0)
res = {}
for sname, st in starts.items():
    for kname, kl in (("k2", (2,)), ("k3", (3,)), ("k2+k3", (2, 3))):
        v = to_flat(st); key = jax.random.PRNGKey(1); rows = [(0, op(v))]; t = time.time()
        for cyc in range(1, cycles + 1):
            for k in kl:
                key, sub = jax.random.split(key); v, _ = FB[k].sweep(v, sub)
            if cyc % every == 0:
                rows.append((cyc, op(v)))
        nv = float(np.asarray(E(v)[1]).mean()) / (p.H * region.W)
        res[f"{sname}/{kname}"] = {"rows": rows, "viol_per_cell_end": nv, "sec": time.time() - t}
        print(sname, kname, rows[0][1], "->", rows[-1][1], "viol/cell", round(nv, 5), "sec", round(time.time() - t, 1), flush=True)
        json.dump(res, open(out, "w"))
