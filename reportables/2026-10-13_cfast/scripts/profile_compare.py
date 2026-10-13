"""Seconds per update, cpass vs cfast, interleaved in one process (the machine
is shared; interleaving exposes both to the same load).  Also the setup time
(trace + compile of what training needs before its first update; the
persistent compile caches are warm for both).  usage: profile_compare.py problem [n]"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.cpass import CPass
from castlegen.variants.cfast import CFast

prob = problems.get(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
region = Region(prob.W_train)
blk = jax.block_until_ready

# ---- cpass
cp = CPass(prob, 0, "/tmp")
lay = cp.layout(region)
cp.nhard = np.ones(prob.T + 1)


def cpass_update(it):
    B = cp.cfg["B"]
    tau, lam = jnp.float32(2.0), jnp.asarray(cp.lam, jnp.float32)
    seeds = np.arange(cp.train_seed, cp.train_seed + B); cp.train_seed += B
    desl = [cp.designed(int(s), region) for s in seeds]
    des = {k: jnp.stack([d[k] for d in desl]) for k in desl[0]}
    extra = jnp.concatenate([jnp.array([2.0]), lam]).astype(jnp.float32)
    st, tot, recs = lay.jit_rec(cp.params, jnp.asarray(seeds, jnp.int32), des, tau, lam, extra)
    G, after = cp._returns(lay, tot)
    if cp.gscale is None:
        cp.gscale = float(max(np.std(np.asarray(after)), 1.0))
    br = cp._branches(lay, recs, after, jnp.asarray(seeds, jnp.int32), des, tau, lam)
    out = blk(lay.jit_step(cp.params, cp.cparams, cp.opt, cp.copt, recs, G, tau, lam, br,
                           jnp.float32(cp.gscale), 1e-3, 1e-3, 10.0))
    cp.params, cp.cparams, cp.opt, cp.copt = out[:4]


t = time.time()
cp._build_train(lay)
cpass_update(0)
setup_cp = time.time() - t                         # (cpass's baseline generate/repair compile not included)

# ---- cfast
cf = CFast(prob, 0, "/tmp")
lay2 = cf.layout(region)
t = time.time()
cf.fn = cf.train_fns(lay2)
cf._measure_baseline(lay2, lambda d: None)


def cfast_update(it):
    B = cf.cfg["B"]
    tau, lam = jnp.float32(2.0), jnp.asarray(cf.lam, jnp.float32)
    seeds = np.arange(cf.train_seed, cf.train_seed + B); cf.train_seed += B
    des = cf._batch(seeds, region)
    out = blk(cf.fn["fused"](cf.params, cf.cparams, cf.opt, cf.copt, cf.gs, jnp.asarray(seeds, jnp.int32), des,
                             tau, lam, 3e-3, 1e-3, 10.0, jax.random.PRNGKey(it)))
    cf.params, cf.cparams, cf.opt, cf.copt = out[1:5]


cfast_update(0)
setup_cf = time.time() - t                         # includes cfast's baseline (fused at lr 0 + batched repair)
res = {"cpass": [], "cfast": []}
for it in range(1, n + 1):
    t = time.time(); cpass_update(it); res["cpass"].append(time.time() - t)
    t = time.time(); cfast_update(it); res["cfast"].append(time.time() - t)
out = {"problem": sys.argv[1], "setup_s": {"cpass": setup_cp, "cfast": setup_cf},
       "s_per_update_median": {k: float(np.median(v)) for k, v in res.items()},
       "s_per_update": res}
print(json.dumps({k: v for k, v in out.items() if k != "s_per_update"}, indent=1))
json.dump(out, open(f"reportables/2026-10-13_cfast/speed_{sys.argv[1]}.json", "w"), indent=1)
