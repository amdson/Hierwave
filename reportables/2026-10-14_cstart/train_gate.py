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
             "trained": {"params": jax.device_get(v.params), "lam": v.lam.tolist()},
             "updates": v.updates, "cfg": v.cfg, "train_s": time.time() - t0,
             "diag_time": getattr(v, "_dtime", 0.0), "devices": str(jax.devices())},
            open(f"{out}/params_{prob}.pkl", "wb"))
print("done", prob, vname, v.updates, "updates", v.lam.tolist(), flush=True)
