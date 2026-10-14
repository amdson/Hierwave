"""ML fit of cfast's actor to EXACT target samples on the enumerable toys, vs exact reverse-KL.
usage: ml_fit.py <support|modes> <tanh|ln> <ml_sample|ml_exact|rev_exact> <seconds> [lr]"""
import sys, json, time
import numpy as np, jax, jax.numpy as jnp
from castlegen.kernels import logq as lq
from castlegen.variants import _exactopt as eo
from castlegen.variants.cpass import adam_init, adam_step, clip_by_level

prob, net, mode, secs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-3
B = 256
problem, region = eo.get_problem(prob)
var = eo.make_variant("cfast", problem, region, 0, net=net)
lay = var.layout(region)
params = var.params
lam = np.full(problem.T + 1, 10.0)
enum = eo.Enum(problem, region, params, lam, cache=f"/tmp/exactopt_enum_{prob}.pkl")
stat_fn = eo.stat_fn_for(prob)
valid = enum.H.sum(1) == 0
pstar = np.where(valid, np.exp(-enum.S), 0.0); pstar /= pstar.sum()
Fstar = float(-np.log(np.exp(-enum.S[valid]).sum()) / 16)
logp = np.log(np.maximum(pstar, 1e-300))
stats = stat_fn(enum.X)
logq_enum = enum.fns("mlp")[0]

def evaluate(params):
    l = np.asarray(logq_enum(params, jnp.float32(1.0)), np.float64); q = np.exp(l)
    out = {"fwd_kl": float(np.sum(pstar[valid] * (logp[valid] - l[valid]))),
           "rev_kl_lam10": float(np.sum(q * (l + enum.E + enum.logZ))),
           "p_viol": float(q @ ~valid), "F_per_cell": float(np.sum(q * (enum.S + enum.H @ lam + l)) / 16),
           "ent_per_cell": float(-(q * l).sum() / 16), "sum_q": float(q.sum())}
    out.update({k: float(q @ v) for k, v in stats.items()})
    return out

ref = {"Fstar": Fstar, "ent_pstar_per_cell": float(-(pstar[valid] * logp[valid]).sum() / 16),
       **{k: float(pstar @ v) for k, v in stats.items()}, "n_valid": int(valid.sum()), "N": enum.N}
print(json.dumps(ref), flush=True)
rng = np.random.default_rng(0)
X = {k: np.asarray(v) for k, v in enum.X.items()}
levels = list(range(problem.T + 1))
if mode == "ml_sample":
    step = lq.make_ml_update(lay); opt = lq.ml_opt_init(params)
elif mode == "ml_exact":
    wt = jnp.asarray(pstar, jnp.float32); lj = enum.fns("mlp")[0]
    @jax.jit
    def step_e(params, opt):
        g = jax.grad(lambda p: -jnp.sum(wt * jnp.where(wt > 0, lj(p, 1.0), 0.0)))(params)
        g = clip_by_level(g, levels, 10.0)
        return adam_step(params, g, opt, lr)
    opt = lq.ml_opt_init(params)
else:  # rev_exact: exact gradient of reverse KL (train_exact's update), tau 2->1 over first 20 %
    _, jg = enum.fns("mlp"); opt = lq.ml_opt_init(params)
    @jax.jit
    def step_r(params, opt, tau):
        _, g = jg(params, tau); g = clip_by_level(g, levels, 10.0)
        return adam_step(params, g, opt, lr)
curve = []; t0 = time.time(); i = 0; nxt = 0
while True:
    el = time.time() - t0
    if el >= nxt or el >= secs:
        m = {"step": i, "t": el, **evaluate(params)}; curve.append(m)
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()), flush=True)
        nxt += 20
    if el >= secs: break
    if mode == "ml_sample":
        idx = rng.choice(enum.N, B, p=pstar)
        params, opt, info = lq.ml_update(params, opt, {k: v[idx] for k, v in X.items()}, np.ones(B), step, lr)
        assert int(info["n_infeasible"]) == 0
    elif mode == "ml_exact":
        params, opt = step_e(params, opt)
    else:
        tau = 1.0 + max(0.0, 1 - el / (0.2 * secs))
        params, opt = step_r(params, opt, jnp.float32(tau))
    i += 1
json.dump({"ref": ref, "mode": mode, "net": net, "prob": prob, "lr": lr, "B": B, "curve": curve},
          open(f"runs/{prob}_{net}_{mode}.json", "w"))
