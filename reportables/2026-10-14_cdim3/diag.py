"""cdim3 task 2: the stripe attractor.  Exact training (cdim2 settings: lam 20, tau 2 -> 1 over 20 % of
1500 steps, Adam, clip 10) of the tanh MLP and table_feat for the first 200 steps; every 10 steps:
orientation mix, per-draw (raster colour order) P_q(horizontal) and q-weighted policy entropy, logit
margin, and the cosine between the descent direction -dJ/dtheta and the stripe directions
d E_q[h-fraction]/dtheta and d log Q(all-horizontal)/dtheta.  Also, at init, E_q[violations | draw i's
orientation].  Writes runs/diag_<kind>.json."""
import json, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.variants import _exactopt as eo
from castlegen.variants import _cpass_core as core

kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mlp", "table_feat"]
STEPS, EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 200, 10
problem, region = eo.get_problem("dimers4")
lam = np.full(1, 20.0)
base = eo.make_variant("cpass", problem, region, 0)
enum = eo.Enum(problem, region, base.params, lam, cache="/tmp/exactopt_enum_dimers4.pkl")
k = enum.keys[0]
ctx, act = enum.ctx[k], enum.act[k]                       # (N, 16): draw i = colour i (raster)
orient = act // 2
st = eo.dimer_stats(enum.X)
hfrac = jnp.asarray(st["horizontal"], jnp.float32)
stripe = np.asarray(st["horizontal"] == 1.0)
viol = enum.H.sum(1)
print("all-horizontal outcomes:", stripe.sum(), "valid among them:", (viol[stripe] == 0).sum())
mask = jnp.asarray(enum.Umask[k])

for kind in kinds:
    logq = enum._logq(kind)

    def eh(params, tau):
        return jnp.sum(jnp.exp(logq(params, tau)) * hfrac)

    def lstripe(params, tau):
        lq = logq(params, tau)
        return jax.scipy.special.logsumexp(jnp.where(jnp.asarray(stripe), lq, -jnp.inf))
    geh, gls = jax.jit(jax.grad(eh)), jax.jit(jax.grad(lstripe))
    _, jg = enum.fns(kind)
    flat = lambda t: np.concatenate([np.asarray(x, np.float64).ravel() for x in jax.tree_util.tree_leaves(t)])
    rec = []

    def hook(i, params, tau):
        lq, g = jg(params, jnp.float32(tau))
        q = np.exp(np.asarray(lq, np.float64))
        b = enum.b_unique(kind, params, k)
        lg = jnp.where(mask, -jnp.asarray(enum.Usoft[k]) - b, -jnp.inf)
        pi = jax.nn.softmax(lg, -1)
        Hc = np.asarray(-jnp.sum(jnp.where(pi > 0, pi * jnp.log(jnp.maximum(pi, 1e-30)), 0.0), -1), np.float64)
        srt = np.sort(np.asarray(jnp.where(mask, lg, -1e9)), -1)
        marg = srt[:, -1] - srt[:, -2]                       # top-1 minus top-2 logit (masked)
        nallow = np.asarray(mask.sum(-1))
        marg = np.where(nallow > 1, marg, np.nan)
        r = {"step": i, "tau": tau,
             "mix": {o: float(q @ (orient == j).mean(1)) for j, o in enumerate(("h", "v", "d"))},
             "p_h_draw": [float(q @ (orient[:, d] == 0)) for d in range(16)],
             "H_draw": [float(q @ Hc[ctx[:, d]]) for d in range(16)],
             "margin_draw": [float(np.nansum(q * marg[ctx[:, d]]) / max(np.sum(q * ~np.isnan(marg[ctx[:, d]])), 1e-30))
                             for d in range(16)],
             "q_stripes": float(q[stripe].sum()), "entropy_cell": float(-(q * np.log(np.maximum(q, 1e-300))).sum() / 16)}
        gJ = -flat(g)
        for name, gg in (("cos_eh", geh(params, jnp.float32(tau))), ("cos_logstripe", gls(params, jnp.float32(tau)))):
            v = flat(gg)
            r[name] = float(gJ @ v / (np.linalg.norm(gJ) * np.linalg.norm(v) + 1e-30))
        r["gnorm"] = float(np.linalg.norm(gJ))
        if i == 0:
            # why horizontal: expected violations under q given draw d's orientation
            r["viol_given"] = {o: [float((q * viol)[orient[:, d] == j].sum() / max(q[orient[:, d] == j].sum(), 1e-30))
                                   for d in range(16)] for j, o in enumerate(("h", "v", "d"))}
            r["pviol_given"] = {o: [float((q * (viol > 0))[orient[:, d] == j].sum() / max(q[orient[:, d] == j].sum(), 1e-30))
                                    for d in range(16)] for j, o in enumerate(("h", "v", "d"))}
        rec.append(r)
        print(f"  [{kind}] step {i}: mix {r['mix']} q_stripes {r['q_stripes']:.3f} H/cell {r['entropy_cell']:.3f} "
              f"cos(-gJ, dEh) {r['cos_eh']:+.3f} cos(-gJ, dlogQs) {r['cos_logstripe']:+.3f} "
              f"Hdraw[0,1,4,15] {[round(r['H_draw'][d], 3) for d in (0, 1, 4, 15)]}", flush=True)
    params = enum.init_params(kind, base.params)
    lr = 3e-2 if kind.startswith("table") else 3e-3
    # cdim2's schedule: tau anneals over 20 % of 1500 steps; we stop at STEPS
    eo.train_exact(enum, kind, params, STEPS, lr, lam, EVERY, eo.dimer_stats, tau0=2.0,
                   anneal=0.2 * 1500 / STEPS, hook=hook)
    json.dump(rec, open(f"reportables/2026-10-14_cdim3/runs/diag_{kind}.json", "w"))
