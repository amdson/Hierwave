"""Part 2: validity / ESS / change of annealed ground blocks by level, steps, start, lambda schedule, joint moves."""
import json, pickle, time, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass
from castlegen.variants._cln_core import window_of
from castlegen.kernels import colour_gibbs as cg, anneal as an

p = get("ground"); reg = Region(64); B = 64
lay = core.Layout(p, 64, True, 0)
ck = pickle.load(open("reportables/2026-10-13_clamp/params_ground.pkl", "rb"))
layq = core.Layout(p, 64, True, 0, resid=bool(ck["cfg"].get("resid", False)), win=window_of(ck["cfg"].get("win")))
params = jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"])
seeds = np.arange(5000, 5000 + B // 4)
des = [p.designed_values(int(s), reg) for s in seeds]
probes = [p.probes(int(s), reg) for s in seeds]
def stack(ds): return {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
dall = [{**des[i], **{}} for i in range(B // 4) for _ in range(4)]
probe_st = stack([{**probes[i][j], **des[i]} for i in range(B // 4) for j in range(4)])
sd = np.repeat(seeds, 4)
f = jax.jit(jax.vmap(lambda s, d: train_pass(layq, params, s, d, 1.0, jnp.full(p.T + 1, 20.0), record=False)[0]))
q_st = {k: jnp.asarray(v, jnp.int32) for k, v in f(jnp.asarray(sd, jnp.int32), stack(dall)).items()}
rng = np.random.default_rng(0)
rand_st = {c.name: (probe_st[c.name] if c.designed is not None else
                    jnp.asarray(rng.integers(0, c.D, (B,) + lay.hw[c.level]), jnp.int32)) for c in p.chans}
starts = {"random": rand_st, "probe": probe_st, "qdraw": q_st}
joint = None
if "--joint" in sys.argv:
    from castlegen.kernels.exact_block import ExactBlockGibbs
    t0 = time.time(); joint = [ExactBlockGibbs(p, reg, level=1, lam=20.0)]; print("joint build", time.time() - t0, flush=True)

def blockstats(st0, st1, mask, valid, logw, info, dt):
    ch = [np.asarray((st0[k] != st1[k]) & mask[k]).sum((1, 2)) for k in mask]
    n = [np.asarray(mask[k]).sum((1, 2)) for k in mask]
    changed = float((sum(ch) / sum(n)).mean())
    d0 = np.asarray((st0["t"] * mask["t"]).sum((1, 2)) / mask["t"].sum((1, 2))); d1 = np.asarray((st1["t"] * mask["t"]).sum((1, 2)) / mask["t"].sum((1, 2)))
    lw = np.asarray(logw)
    return dict(start_valid=float((np.asarray(info["N0"]) == 0).mean()), validity=float(np.asarray(valid).mean()),
                ess64=an.ess(lw), ess_valid_only=an.ess(np.where(np.asarray(valid), lw, -np.inf)) if False else None,
                changed=changed, dirt0=float(d0.mean()), dirt1=float(d1.mean()),
                pure0=float(((d0 == 0) | (d0 == 1)).mean()), partial_from_pure=float((((d0 == 0) | (d0 == 1)) & (d1 > 0) & (d1 < 1)).mean()),
                nv_end=float(np.asarray(info["N_end"]).mean()), sec=dt)

rows = []
configs = []
if "--joint" in sys.argv:
    for start in ("random", "probe", "qdraw"):
        for lk in ("const", "geo"):
            configs += [(3, 32, start, lk, "hard", True)]
else:
    for L in (3, 4):
        for n in (32, 128):
            for start in ("random", "probe", "qdraw"):
                for lk in ("const", "geo"):
                    configs.append((L, n, start, lk, "hard", False))
for (L, n, start, lk, fin, jt) in configs:
    if jt and joint is None: continue
    if not jt and joint is not None and "--joint-only" in sys.argv: continue
    r2 = np.random.default_rng(1)
    cells = np.stack([r2.integers(0, lay.hw[L][0], B), r2.integers(0, lay.hw[L][1], B)], 1)
    mask = cg.subtree_mask(lay, L, cells); st = starts[start]
    tau_hi = 8.0
    run = lambda: an.anneal_block_valid(lay, st, mask, jax.random.PRNGKey(2), tau_hi, n, lk, 20.0, final=fin, n_hard=2,
                                        joint=joint if jt else None)
    if not jt: jax.block_until_ready(run()[1])
    t0 = time.time(); s1, lw, valid, info = run(); jax.block_until_ready(lw); dt = time.time() - t0
    row = dict(level=L, steps=n, start=start, lam=lk, final=fin, joint=jt, **blockstats(st, s1, mask, valid, lw, info, dt))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open("reportables/2026-10-14_speed2/runs/valid_blocks2%s.json" % ("_joint" if joint else ""), "w"), indent=1)
