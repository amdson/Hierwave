"""Breakdown of one creplay update on ground (GPU): jitted parts timed separately.
usage: profile.py problem [cfg overrides as python dict literal]"""
import sys, time, json
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.creplay import CReplay
from castlegen.variants._cfast_core import train_pass

prob = problems.get(sys.argv[1])
over = eval(sys.argv[2]) if len(sys.argv) > 2 else {}
v = CReplay(prob, 0, "/tmp", **over)
region = Region(prob.W_train)
lay = v.layout(region)
fn = v.train_fns(lay)
B = v.cfg["B"]
gs = {k: jnp.float32(1.0) for k in v.params}
lam = jnp.asarray(v.lam, jnp.float32)
tau = jnp.float32(1.0)
seeds = jnp.arange(B, dtype=jnp.int32) + 1000
des = v._batch(np.asarray(seeds), region)
res = {}

def tm(name, f, *a, n=5):
    t0 = time.time(); jax.block_until_ready(f(*a)); tc = time.time() - t0
    t0 = time.time()
    for _ in range(n):
        out = f(*a)
    jax.block_until_ready(out)
    res[name] = {"s": round((time.time() - t0) / n, 4), "compile": round(tc, 1)}
    print(name, res[name], flush=True)
    return out

tm("run_pass", jax.jit(fn["run_pass"]), v.params, seeds, des, tau, lam)
bat = tm("batch", fn["batch"], v.params, v.cparams, gs, seeds, des, tau, lam, jax.random.PRNGKey(0)) if False else None
tm("batch", fn["batch"], v.params, seeds, des, tau, lam, jax.random.PRNGKey(0))
tm("fused", fn["fused"], v.params, v.cparams, v.opt, v.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
   jax.random.PRNGKey(0))

# raw replays: N full passes (record=False), worlds repeated
rc = jax.jit(fn["replay_costs"])
for N in [int(x) for x in over.get("_N", [64, 256, 1024])]:
    b = jnp.arange(N) % B
    st = jnp.full((N,), 3, jnp.int32); site = jnp.zeros((N,), jnp.int32); val = jnp.zeros((N,), jnp.int32)
    tm(f"replay_costs_N{N}", rc, v.params, seeds[b], {k: x[b] for k, x in des.items()}, tau, lam, st, site, val)
    cone = fn["cone"]
    tm(f"cone_integral_N{N}", jax.jit(lambda c: cone.integral(c)),
       rc(v.params, seeds[b], {k: x[b] for k, x in des.items()}, tau, lam, st, site, val))

# per-level cost of a replay: passes restricted to one level (other levels' state stays unset)
levels = list(lay.levels)
N = 256
b = jnp.arange(N) % B
for l in levels:
    lay.levels = [l]
    f = jax.jit(lambda p, s, d: jax.vmap(lambda s1, d1: train_pass(lay, p, s1, d1, tau, lam, record=False)[1])(s, d))
    tm(f"pass_level{l}_N{N}", f, v.params, seeds[b], {k: x[b] for k, x in des.items()})
lay.levels = levels
json.dump(res, open(f"/tmp/creplay2_profile_{sys.argv[1]}.json", "w"), indent=1)
