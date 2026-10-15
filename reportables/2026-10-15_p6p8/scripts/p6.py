"""P6: plug-in H(channel | all else) from the exact Gibbs conditional, averaged over chain states.
ground: valid states (cgraded final_binary_lam20, 16 worlds W=128) + hard sweeps; columns: exact iid + hard sweeps.
columns stripped chain: validity and mixing."""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_cgraded"); sys.path.insert(0, "reportables/2026-10-14_cmix")
from castlegen.bench.problems import ground as G, columns as C
from castlegen.bench.problem import Problem
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout, NCOL
W = 128
OUT = {}

def ent(lay, st, p):
    """{chan: nats per level-l cell} of the conditional entropy; plus soft-only (nv ignored) variant."""
    res = {}
    for c in p.sampled():
        l = c.level
        hs, hh = [], []
        for k in range(NCOL):
            soft, nv, (Y, X, valid) = cg.conditional(lay, st, l, c.name, k)
            soft = np.asarray(soft, np.float64); nv = np.asarray(nv); v = np.asarray(valid)
            lg = np.where(nv > 0, -np.inf, -soft)
            m = np.isfinite(lg).any(-1)
            lg = lg - np.where(m, lg.max(-1), 0)[..., None]
            pr = np.exp(lg); pr /= np.maximum(pr.sum(-1, keepdims=True), 1e-300)
            H = -(pr * np.log(np.maximum(pr, 1e-300))).sum(-1)
            hs.append(H[:, v].mean())
            # support size
            hh.append(np.isfinite(lg).sum(-1)[:, v].mean())
        res[c.name] = (float(np.mean(hs)), float(np.mean(hh)))
    return res

def avg(rs):
    return {k: [float(np.mean([r[k][i] for r in rs])) for i in range(2)] for k in rs[0]}

OUT["ground"] = json.loads(open("reportables/2026-10-15_p6p8/runs/ground_ent.json").read())
# ---------------- columns
pc = C.build_columns()
parts = C.chain_parts(pc.factors[0].table, pc.factors[1].table)
rng = np.random.default_rng(0)
B = 16
ks = C.sample_chains(rng, B * W, parts)                # (H, B*W)
ks = ks.reshape(C.H, B, W).transpose(1, 0, 2)
xs = [C.lift(ks[b]) for b in range(B)]
cn = [c.name for c in pc.chans]
stc = {k: jnp.asarray(np.stack([x[k] for x in xs]).astype(np.int32)) for k in cn}
layc = Layout(pc, W, True)
_, N = cg.state_energy(layc, stc); print("columns exact samples viol", int(np.asarray(N).sum()), flush=True)
OUT["columns"] = ent(layc, stc, pc); print(json.dumps(OUT["columns"]), flush=True)
# stripped
ps = Problem(name="columns_strip", T=pc.T, H=pc.H, chans=[c for c in pc.chans if c.name == "k"],
             factors=[f for f in pc.factors if f.name in ("column", "mu")], W_train=W, W_eval=W, stats=pc.stats)
lays = Layout(ps, W, True)
sts = {"k": stc["k"]}
OUT["columns_strip"] = ent(lays, sts, ps); print(json.dumps(OUT["columns_strip"]), flush=True)

def chain(L, st, lam, n=300, hard=True):
    k0 = np.asarray(st["k"]); dens, chg, nv = [], [], []
    key = jax.random.PRNGKey(1); prev = k0
    for s in range(n):
        key, sub = jax.random.split(key)
        st = cg.sweep_all(L, st, 1.0, None if hard else lam, sub)
        kk = np.asarray(st["k"]); chg.append(float((kk != prev).mean())); prev = kk
        dens.append((kk > 0).mean((1, 2)))
        _, N = cg.state_energy(L, st); nv.append(int(np.asarray(N).sum()))
    dens = np.array(dens)                      # (n, B)
    d = dens - dens.mean(0)
    ac = lambda lag: float((d[:-lag] * d[lag:]).mean() / max((d * d).mean(), 1e-12))
    return {"frac_cells_changed_per_sweep": float(np.mean(chg)), "frac_changed_vs_start": float((np.asarray(st["k"]) != k0).mean()),
            "final_viol": nv[-1], "max_viol": max(nv), "density_acf_lag1_10_50": [ac(1), ac(10), ac(50)],
            "density_mean_std_over_chains": [float(dens[-1].mean()), float(dens[-1].std())]}
OUT["col_chain_full_hard"] = chain(layc, stc, None); print(OUT["col_chain_full_hard"], flush=True)
OUT["col_chain_strip_hard"] = chain(lays, sts, None); print(OUT["col_chain_strip_hard"], flush=True)
OUT["col_chain_strip_pen20"] = chain(lays, sts, 20.0, hard=False); print(OUT["col_chain_strip_pen20"], flush=True)
# stripped from empty start (can it find valid states?) 
st0 = {"k": jnp.zeros_like(sts["k"])}
OUT["col_chain_strip_hard_fromempty"] = chain(lays, st0, None); print(OUT["col_chain_strip_hard_fromempty"], flush=True)
json.dump(OUT, open("reportables/2026-10-15_p6p8/runs/p6.json", "w"), indent=1)
