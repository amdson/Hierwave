"""Worm kernel alone on dimers populations: h/v/d vs sweeps (sweep = H*W excursion attempts per world).
usage: pop_sweeps.py <out.json> <nworlds> <lmax>.  Starts: collapsed cfast60 draws (valid ones), probe tilings, all-horizontal."""
import json, pickle, sys, time
import numpy as np
import jax, jax.numpy as jnp
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants
from castlegen.variants._cfast_core import train_pass
from castlegen.variants import _dimer_loop_kernel as dk
out, NW, LMAX = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = get_problem("dimers"); region = Region(p.W_train); H, W = p.H, region.W
c = load_variants()["cfast"](p, seed=0, workdir="/tmp/dsec"); lay = c.layout(region)
ck = pickle.load(open("reportables/2026-10-14_dimers_kernel/runs/cfast60.pkl", "rb"))
params = jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"]); lg = jnp.asarray(ck["trained"]["lam"], jnp.float32)
gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lg, record=False)[0], in_axes=(None, 0, 0)))
ss = np.arange(40_000_000, 40_000_000 + 2 * NW); ds = [p.designed_values(int(s), region) for s in ss]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
col = np.asarray(gen(params, jnp.asarray(ss, jnp.int32), dj)["p"], np.int64)
vm = dk.valid_mask(col); print("collapsed draws valid", vm.mean(), "hvd(valid)", dk.hvd(col[vm]), flush=True)
col = col[vm][:NW]
til = np.stack([p.probes(int(s), region, 4)[2]["p"] for s in range(100, 100 + NW)]).astype(np.int64)
hor = np.zeros((NW, H, W), np.int64); hor[:, :, 1::2] = 1
starts = {"collapsed": col, "tiling": til, "all_h": hor}
CK = [1, 3, 10, 30, 100, 300]
res = {}
for sn, P in starts.items():
    P = P.copy(); rows = [dict(k=0, hvd=dk.hvd(P), n=len(P))]; print(sn, rows[0], flush=True)
    done = 0; t0 = time.time()
    for k in CK:
        P, info = dk.loop_refresh(P, (k - done) * H * W, lmax=LMAX, seed=1000 * k + 7); done = k
        assert dk.valid_mask(P).all()
        r = dict(k=k, hvd=dk.hvd(P), acc=info["acc"], changed=info["changed"], t=time.time() - t0)
        # spread over worlds
        per = np.array([dk.hvd(P[b:b + 1]) for b in range(len(P))]); r["std_world"] = per.std(0).round(3).tolist()
        rows.append(r); print(sn, r, flush=True)
    res[sn] = rows; json.dump(dict(lmax=LMAX, res=res), open(out, "w"))
print("DONE", flush=True)
