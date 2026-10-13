"""Time the parts of one cpass update (record pass, returns, replays, step)."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import ground, columns
from castlegen.bench.problem import Region
from castlegen.variants.cpass import CPass

name = sys.argv[1]
prob = {"ground": ground.build_ground, "columns": columns.build_columns}[name]()
v = CPass(prob, 0, "/tmp")
region = Region(prob.W_train)
lay = v.layout(region)
t = time.time(); v._build_train(lay); print("build", time.time() - t)
v.nhard = np.ones(prob.T + 1)
B = v.cfg["B"]


def blk(x):
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)
    return x


for it in range(4):
    T0 = time.time()
    tau, lam = jnp.float32(2.0), jnp.asarray(v.lam, jnp.float32)
    seeds = np.arange(v.train_seed, v.train_seed + B); v.train_seed += B
    desl = [v.designed(int(s), region) for s in seeds]
    des = {k: jnp.stack([d[k] for d in desl]) for k in desl[0]}
    extra = jnp.concatenate([jnp.array([2.0]), lam]).astype(jnp.float32)
    t1 = time.time()
    st, tot, recs = blk(lay.jit_rec(v.params, jnp.asarray(seeds, jnp.int32), des, tau, lam, extra))
    t2 = time.time()
    G, after = blk(v._returns(lay, tot))
    v.gscale = v.gscale or float(max(np.std(np.asarray(after)), 1.0))
    t3 = time.time()
    br = blk(v._branches(lay, recs, after, jnp.asarray(seeds, jnp.int32), des, tau, lam))
    t4 = time.time()
    out = blk(lay.jit_step(v.params, v.cparams, v.opt, v.copt, recs, G, tau, lam, br,
                           jnp.float32(v.gscale), 1e-3, 1e-3, 10.0))
    v.params, v.cparams, v.opt, v.copt = out[:4]
    t5 = time.time()
    print(f"it{it} des {t1-T0:.2f} rec {t2-t1:.2f} ret {t3-t2:.2f} branch {t4-t3:.2f} "
          f"step {t5-t4:.2f} total {t5-T0:.2f}", flush=True)
