"""Train cfast on ground (default cfg, B=8, 150 s budget schedule), snapshot params/lam/tau at
update 0 (init), 50 and 200, then stop.  Diagnostics only."""
import pickle, sys, time
import jax
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants

out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
V = load_variants()["cfast"]; p = get_problem("ground")
w = V(p, seed=99, workdir="/tmp/ctop_w"); w.train(1.0, lambda d: None)        # warm compile
v = V(p, seed=seed, workdir="/tmp/ctop_v")
snaps = {"init": {"params": jax.device_get(v.params), "lam": v.lam.tolist(), "tau": 2.0, "update": 0}}
class Stop(Exception): pass
def log(d):
    u = d.get("update")
    if u in (50, 200):
        snaps[f"u{u}"] = {"params": jax.device_get(v.params), "lam": v.lam.tolist(), "tau": float(v.tau), "update": u}
        print("snap", u, "tau", v.tau, "lam", v.lam, flush=True)
    if u == 200: raise Stop
t0 = time.time()
try: v.train(150.0, log)
except Stop: pass
print("train s", time.time() - t0, flush=True)
pickle.dump(snaps, open(out, "wb"))
