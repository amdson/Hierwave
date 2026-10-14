"""Gate-2 training (from cfine's train_gate.py): a variant on ground for a
budget with csat's diagnostics every 25 updates (per-level entropy of q over
non-forced draws, dirt fraction) plus line_match of the same 8 diagnostic
worlds (generation-style pass: own draws at every level, current tau);
saves untrained/trained params for the clamp gate.
usage: train_gate.py <problem> <budget_s> <outdir> <variant> [key=value ...]"""
import ast, json, os, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants
from castlegen.variants.csat import CSat
from castlegen.variants._cfast_core import train_pass

prob, budget, out, vname = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
over = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[5:])}
os.makedirs(out, exist_ok=True)
Base = load_variants()[vname]
_LM = {}


class V(CSat, Base):
    name = vname + "_sat"

    def diag(self, tau=None, lam=None):
        res = super().diag(tau, lam)
        if self.problem.name != "ground":
            return res
        from castlegen.bench.problems.ground import stats
        region = Region(self.cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        if "f" not in _LM:
            _LM["f"] = jax.jit(jax.vmap(lambda pr, s, d, t, lm: train_pass(lay, pr, s, d, t, lm, record=False)[0]["t"],
                                        in_axes=(None, 0, 0, None, None)))
        seeds = np.arange(900_000, 900_008)
        des = self._batch(seeds, region)
        tau = self.tau if tau is None else tau
        lam = self.lam if lam is None else lam
        t = np.asarray(_LM["f"](self.params, jnp.asarray(seeds, jnp.int32), des, jnp.float32(tau),
                                jnp.asarray(lam, jnp.float32)))
        fills = np.asarray(des["fill"])
        lm = [stats({"t": t[i], "fill": fills[i], "rho4": np.zeros_like(fills[i])})["line_match"] for i in range(len(t))]
        res["line_match"] = float(np.mean(lm))
        return res


p = get_problem(prob)
tw = time.time()
w = V(p, seed=99, workdir=out, **over)
class _Stop(Exception):
    pass


def _wl(d):
    if d.get("update", 0) >= 2:
        raise _Stop


try:                                   # warm-up: at least two real updates (compiles the clamped update)
    w.train(1e9, _wl)
except _Stop:
    pass
w.diag(tau=2.0)
print("warm-up compile", time.time() - tw, flush=True)
v = V(p, seed=0, workdir=out, **over)
p0 = jax.device_get(v.params); lam0 = v.lam.copy()
logf = open(f"{out}/train_{prob}.jsonl", "w")
t0 = time.time()
v.train(budget, lambda d: (logf.write(json.dumps(d, default=float) + "\n"), logf.flush()))
logf.close()
pickle.dump({"untrained": {"params": p0, "lam": lam0.tolist()},
             "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist(),
                         "cparams": jax.device_get(v.cparams), "gs": jax.device_get(v.gs)},
             "updates": v.updates, "cfg": v.cfg, "train_s": time.time() - t0,
             "diag_time": getattr(v, "_dtime", 0.0), "devices": str(jax.devices())},
            open(f"{out}/params_{prob}.pkl", "wb"))
print("done", prob, vname, v.updates, "updates", v.lam.tolist(), flush=True)

# ---- end-of-training per-level actor-gradient SNR, all worlds clamped at L = 3 (the training distribution)
from castlegen.variants import _gradcheck as gc
lay = v.layout(region := Region(v.cfg["W_train"] or p.W_train)); fn = v.train_fns(lay)
B, M = v.cfg["B"], 48
tau, lam = jnp.float32(1.0), jnp.asarray(v.lam, jnp.float32)
keys = sorted(v.params); sl = gc.block_slices(v.params, keys)
snr = {}
p0j = jax.tree_util.tree_map(jnp.asarray, p0)
for crit, pa, cp in (("trained", v.params, v.cparams), ("trained_zero_critic", v.params, gc.zero_critic(v.cparams)),
                     ("init", p0j, gc.zero_critic(v.cparams))):
    est = []
    for m in range(M):
        seeds = np.arange(8_000_000 + m * B, 8_000_000 + (m + 1) * B)
        des = v._batch(seeds, region)
        Ls_w = np.full(B, 3, np.int64)
        ctx, Lj = v.contexts(seeds, region, Ls_w)
        st, rs, G, wgt, viol, tot = fn["batch"](pa, jnp.asarray(seeds, jnp.int32), des, tau, lam, ctx, Lj,
                                                 jax.random.PRNGKey(m))
        g = fn["actor_grad"](pa, cp, rs, G, tau, lam, v.gs)
        est.append(gc.flat(g, keys))
    est = np.array(est)
    res = {}
    for l in range(p.T + 1):
        cols = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in keys if k.startswith(f"l{l}_")])
        e = est[:, cols]; mu = e.mean(0); trc = e.var(0, ddof=1).sum(); sig = mu @ mu - trc / M
        res[l] = {"snr_batch": float(sig / max(trc, 1e-30)), "signal": float(sig), "noise": float(trc),
                  "mean_norm": float(np.sqrt(max(mu @ mu, 0)))}
    snr[crit] = res
    print("SNR", crit, json.dumps(res), flush=True)
json.dump({"B": B, "M": M, "snr": snr}, open(f"{out}/snr.json", "w"), indent=1)

