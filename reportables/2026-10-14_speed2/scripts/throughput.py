"""Throughput of colour_gibbs on ground (W = 64): level-0 / level-3 sweeps over 64 worlds, annealed level-3 subtree blocks."""
import json, time, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.kernels import colour_gibbs as cg

p = get("ground"); lay = core.Layout(p, 64, True, 0)
B = 64
reg = Region(64)
rng = np.random.default_rng(0)
des = p.designed_values(0, reg)
st = {c.name: jnp.asarray(np.broadcast_to(des[c.name], (B,) + lay.hw[c.level]) if c.name in des
                          else rng.integers(0, c.D, (B,) + lay.hw[c.level]), jnp.int32) for c in p.chans}
res = {"B": B, "W": 64, "device": str(jax.devices()[0])}

def timeit(f, n=5):
    out = f(); jax.block_until_ready(out)
    t0 = time.time()
    for _ in range(n):
        out = f(); jax.block_until_ready(out)
    return (time.time() - t0) / n

key = jax.random.PRNGKey(0)
for l in (0, 3):
    ncells = sum(int(np.prod(lay.hw[l])) for c in lay.chans[l]) * B       # cells (channel x position x world)
    t0 = time.time(); cg.sweep_level(lay, st, l, 1.0, 20.0, key); jax.block_until_ready(st)
    comp = time.time() - t0
    dt = timeit(lambda: cg.sweep_level(lay, st, l, 1.0, 20.0, key))
    res[f"level{l}"] = dict(channels=[c.name for c in lay.chans[l]], cell_updates=ncells, sec=dt,
                            cells_per_s=ncells / dt, compile_s=comp)
    print(res[f"level{l}"], flush=True)

# annealed level-3 subtree blocks, one block per world, tau 8 -> 1, 32 steps
cells = np.stack([rng.integers(0, lay.hw[3][0], B), rng.integers(0, lay.hw[3][1], B)], 1)
mask = cg.subtree_mask(lay, 3, cells)
nblk = int(sum(np.asarray(m[0]).sum() for m in mask.values()))
def run():
    return cg.anneal_block(lay, st, mask, key, 8.0, 32, 20.0)
t0 = time.time(); jax.block_until_ready(run()); comp = time.time() - t0
dt = timeit(run, 3)
_, logw, info = run()
res["anneal_L3"] = dict(blocks=B, tau_hi=8, steps=32, sec=dt, compile_s=comp, sec_per_block=dt / B,
                        cells_in_block=nblk, logw_mean=float(logw.mean()), logw_std=float(logw.std()),
                        nv_end_mean=float(info["nv_end"].mean()))
print(res["anneal_L3"], flush=True)
json.dump(res, open("reportables/2026-10-14_speed2/runs/throughput.json", "w"), indent=1)
