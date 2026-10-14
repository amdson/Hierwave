"""Per-level replay cost on ground: windowed vs full-world replays of the same
selected draws and actions (B = 8 worlds, R draws, K = 4 actions)."""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
load_variants()
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.cfine import CFine

prob = problems.get("ground")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 16
v = CFine(prob, 0, "/tmp/cfine_tl", blevs=(2, 1, 0), R=(R, R, R), wlev=2, chunk=8192)
region = Region(prob.W_train)
lay = v.layout(region)
fn = v.train_fns(lay)
B = 8
seeds = jnp.arange(B, dtype=jnp.int32) + 1000
des = v._batch(np.asarray(seeds), region)
tau, lam = jnp.float32(1.0), jnp.asarray(v.lam, jnp.float32)
st, recs, viol, tot = jax.jit(fn["run_pass"])(v.params, seeds, des, tau, lam)
final = {c.name: st[c.name] for c in prob.chans if c.designed is None}
out = {}
for l in (2, 1, 0):
    sel = fn["select"](recs, l, jax.random.PRNGKey(l))
    for mode in ("window", "full"):
        f = jax.jit(lambda s, fi: fn["replay"](v.params, seeds, des, tau, lam, l, s, fi))
        fi = final if mode == "window" else None
        t0 = time.time(); g = jax.block_until_ready(f(sel, fi)); tc = time.time() - t0
        ts = []
        for _ in range(4):
            t0 = time.time(); g = jax.block_until_ready(f(sel, fi)); ts.append(time.time() - t0)
        out[f"l{l}_{mode}"] = {"replays": B * R * 4, "compile_s": round(tc, 1), "s_min": round(min(ts), 4),
                               "ms_per_replay": round(1000 * min(ts) / (B * R * 4), 3)}
        print(json.dumps({f"l{l}_{mode}": out[f"l{l}_{mode}"]}), flush=True)
json.dump(out, open("reportables/2026-10-14_cfine/timing_levels.json", "w"), indent=1)
