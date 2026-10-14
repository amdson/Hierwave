"""Violations left inside annealed level-3 subtree blocks on ground (lam 20 and lam 60), by schedule length; timing."""
import json, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.kernels import colour_gibbs as cg
p = get("ground"); lay = core.Layout(p, 64, True, 0); B = 64
rng = np.random.default_rng(0); des = p.designed_values(0, Region(64))
st = {c.name: jnp.asarray(np.broadcast_to(des[c.name], (B,) + lay.hw[c.level]) if c.name in des
                          else rng.integers(0, c.D, (B,) + lay.hw[c.level]), jnp.int32) for c in p.chans}
cells = np.stack([rng.integers(0, lay.hw[3][0], B), rng.integers(0, lay.hw[3][1], B)], 1)
mask = cg.subtree_mask(lay, 3, cells); out = {}
for lam in (20.0, 60.0):
    for n in (32, 128):
        f = lambda: cg.anneal_block(lay, st, mask, jax.random.PRNGKey(1), 8.0, n, lam)
        jax.block_until_ready(f()); t0 = time.time(); s2, lw, info = f(); jax.block_until_ready(lw); dt = time.time() - t0
        nv = np.asarray(info["nv_end"])
        out[f"lam{lam:g}_steps{n}"] = dict(sec=dt, blocks_per_s=B / dt, nv_mean=float(nv.mean()), frac_nv0=float((nv == 0).mean()),
                                           soft_end_mean=float(np.asarray(info["soft_end"]).mean()),
                                           E0_mean=float(np.asarray(info["E0"]).mean()), E_end_mean=float(np.asarray(info["E_end"]).mean()))
        print(lam, n, out[f"lam{lam:g}_steps{n}"], flush=True)
json.dump(out, open("reportables/2026-10-14_speed2/runs/anneal_quality.json", "w"), indent=1)
