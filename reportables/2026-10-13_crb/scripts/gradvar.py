"""Gradient variance per level at frozen parameters: cfast (realised cone
returns) vs crb (Rao-Blackwellised cone returns), paired (same worlds, same
pass, same loss subsample), plus the fused-update time of each, so variance per
second can be compared.

Parameters: init (pre = 0) or after `pre` crb updates at tau = 1 (lam fixed at
lam0); critic either zeroed ("off") or as trained ("on").

usage: gradvar.py problem N_worlds pre [B]"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.crb import CRB

prob = problems.get(sys.argv[1])
N, pre = int(sys.argv[2]), int(sys.argv[3])
B = int(sys.argv[4]) if len(sys.argv) > 4 else 8
region = Region(prob.W_train)
vr = CRB(prob, 0, "/tmp", B=B)
vc = CRB(prob, 0, "/tmp", B=B, rb=False)
lay = vr.layout(region)
fr, fc = vr.train_fns(lay), vc.train_fns(lay)
tau, lam = jnp.float32(1.0), jnp.asarray(vr.lam, jnp.float32)
vr.fn = fr
vr._measure_baseline(lay, lambda d: None)       # sets gs (critic output scale) from the RB returns
gs = vr.gs
t0 = time.time()
for u in range(pre):
    seeds = np.arange(10_000 + B * u, 10_000 + B * (u + 1))
    des = vr._batch(seeds, region)
    o = fr["fused"](vr.params, vr.cparams, vr.opt, vr.copt, gs, jnp.asarray(seeds, jnp.int32), des, tau, lam,
                    3e-3, 1e-3, 10.0, jax.random.PRNGKey(u))
    vr.params, vr.cparams, vr.opt, vr.copt = o[1:5]
print(f"pre {pre} updates {time.time() - t0:.1f}s", flush=True)
params = vr.params

# time per fused update (same params; outputs discarded)
tim = {}
for name, f in (("cfast", fc), ("crb", fr)):
    ts = []
    for r in range(21):
        seeds = np.arange(90_000 + B * r, 90_000 + B * (r + 1))
        des = vr._batch(seeds, region)
        jax.block_until_ready(des)
        t1 = time.time()
        o = f["fused"](params, vr.cparams, vr.opt, vr.copt, gs, jnp.asarray(seeds, jnp.int32), des, tau, lam,
                       3e-3, 1e-3, 10.0, jax.random.PRNGKey(r))
        jax.block_until_ready(o)
        if r:
            ts.append(time.time() - t1)
    tim[name] = float(np.median(ts))
print("s/update", tim, flush=True)

levels = sorted({int(k[1:k.index("_")]) for k in params})


def by_level(g):
    """(B, P_l) per level."""
    return {l: np.concatenate([np.asarray(x).reshape(x.shape[0], -1) for k in sorted(g) if k.startswith(f"l{l}_")
                               for x in jax.tree_util.tree_leaves(g[k])], 1) for l in levels}


out = {"problem": sys.argv[1], "N": N, "pre": pre, "B": B, "s_per_update": tim}

# return statistics per level: mean and spread of G (cfast vs crb), same worlds and subsample
gst = {}
for name, f in (("cfast", fc), ("crb", fr)):
    acc = {}
    for i in range(8):
        seeds = np.arange(2_000 + B * i, 2_000 + B * (i + 1))
        des = vr._batch(seeds, region)
        G = f["batch"](params, jnp.asarray(seeds, jnp.int32), des, tau, lam, jax.random.PRNGKey(77 + i))[2]
        for k, v in G.items():
            l = int(k[1:k.index("_")])
            acc.setdefault(l, []).append(np.asarray(v).ravel())
    gst[name] = {l: {"mean": float(np.concatenate(v).mean()), "std": float(np.concatenate(v).std())}
                 for l, v in acc.items()}
print("G stats", json.dumps(gst), flush=True)
out["G_stats"] = gst
for crit in ("off", "on"):
    cparams = vr.cparams if crit == "on" else jax.tree_util.tree_map(jnp.zeros_like, vr.cparams)
    gC, gR = {l: [] for l in levels}, {l: [] for l in levels}
    t1 = time.time()
    for i in range(N // B):
        seeds = np.arange(2_000 + B * i, 2_000 + B * (i + 1))
        des = vr._batch(seeds, region)
        a, b = fr["agrad_pair"](params, cparams, gs, jnp.asarray(seeds, jnp.int32), des, tau, lam,
                                jax.random.PRNGKey(77 + i))
        a, b = by_level(a), by_level(b)
        for l in levels:
            gC[l].append(a[l])
            gR[l].append(b[l])
    print(f"critic {crit}: {N} worlds {time.time() - t1:.1f}s", flush=True)
    res = {}
    for l in levels:
        C, R = np.concatenate(gC[l]).astype(np.float64), np.concatenate(gR[l]).astype(np.float64)
        n = len(C)
        mc, mr = C.mean(0), R.mean(0)
        vC, vR = ((C - mc) ** 2).sum(1).mean(), ((R - mr) ** 2).sum(1).mean()      # per-world total variance
        d = C - R
        md = d.mean(0)
        se2 = ((d - md) ** 2).sum(1).mean() / n
        sig2 = max(float((mr ** 2).sum()) - vR / n, 0.0)                         # |grad|^2, debiased
        res[l] = {"var_world_cfast": vC, "var_world_crb": vR, "var_ratio": vC / max(vR, 1e-30),
                  "per_sec_ratio": vC * tim["cfast"] / max(vR * tim["crb"], 1e-30),
                  "snr_batch_cfast": sig2 / max(vC / B, 1e-30), "snr_batch_crb": sig2 / max(vR / B, 1e-30),
                  "mean_norm_cfast": float(np.linalg.norm(mc)), "mean_norm_crb": float(np.linalg.norm(mr)),
                  "paired_diff_sq_over_se2": float((md ** 2).sum() / max(se2, 1e-30))}
        r = res[l]
        print(f"[{crit}] level {l}: var cfast {vC:.4g} crb {vR:.4g} ratio {r['var_ratio']:.2f} "
              f"per-s {r['per_sec_ratio']:.2f}  SNR/batch {r['snr_batch_cfast']:.3g} -> {r['snr_batch_crb']:.3g}  "
              f"|d|^2/se^2 {r['paired_diff_sq_over_se2']:.2f}", flush=True)
    out["critic_" + crit] = res
json.dump(out, open(f"reportables/2026-10-13_crb/gradvar_{sys.argv[1]}_pre{pre}.json", "w"), indent=1)
