"""Long ML-only run.  usage: longfit.py <variant> <budget_s> <outdir> [smoke]
Logs every iteration (filtered) to <outdir>/train.jsonl; light eval row every ~100 updates is the iteration's
own8 stats; heavy eval (eval_seed n=8 + reduced block_check) every EVERY_S; params every 15 min (keep last 3)."""
import json, os, pickle, sys, time
import jax, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench import run as R
variant, budget, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
smoke = len(sys.argv) > 4
os.makedirs(out, exist_ok=True)
EVERY_S = 600 if not smoke else 30
SAVE_S = 900 if not smoke else 45
p = get("ground")
V = load_variants()
v = V[variant](p, 0, out)
lf = open(f"{out}/train.jsonl", "a"); ef = open(f"{out}/eval.jsonl", "a")
state = dict(t0=time.time(), last_eval=0.0, last_save=0.0, last_upd=0, saved=[])
KEEP = ("iter", "update", "t", "t_train", "loss_per_cell", "own8", "lv_own", "lv_pop", "own_raw", "time_refine", "time_ml", "setup_done", "pop")

def save(tag):
    f = f"{out}/params_{tag}.pkl"
    pickle.dump({"params": jax.device_get(v.params), "lam": np.asarray(v.lam), "updates": v.updates}, open(f, "wb"))
    state["saved"].append(f)
    while len(state["saved"]) > 3:
        os.remove(state["saved"].pop(0))
    return f

def heavy(tag, full=False):
    t = time.time()
    if not full:
        R.N_POOL_OWN, R.N_POOL_PROBE_SEEDS = 24, 6
    m = {"tag": tag, "updates": v.updates}
    reg = R.Region(p.W_eval)
    # eval_seed minus probes/support (cheap): reuse it fully only for the final
    r = R.eval_seed(p, v, 0, 16 if full else 8, None)
    m["viol_raw"] = r["viol"]["raw"]; m["stats_raw"] = r["stats"]["raw"]; m["viol_final"] = r["viol"]["final"]
    m["blocks"] = r["blocks"]; m["support_mismatch"] = r["support_mismatch"]
    m["eval_s"] = time.time() - t
    ef.write(json.dumps(m, default=float) + "\n"); ef.flush()
    print("EVAL", tag, v.updates, m["stats_raw"].get("line_match"), m["viol_raw"], f"{m['eval_s']:.0f}s", flush=True)

def log(rec):
    row = {k: rec[k] for k in KEEP if k in rec}
    row["wall"] = time.time() - state["t0"]
    lf.write(json.dumps(row, default=float) + "\n"); lf.flush()
    if "iter" not in rec:
        return
    now = time.time() - state["t0"]
    if now - state["last_save"] > SAVE_S:
        state["last_save"] = now; save(f"{int(now/60):03d}min")
    if now - state["last_eval"] > EVERY_S:
        state["last_eval"] = now
        try:
            heavy(f"{int(now/60):03d}min")
        except Exception as e:
            print("EVAL FAILED", repr(e), flush=True)

nv = os.popen("nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv").read()
open(f"{out}/nvidia_before.txt", "w").write(nv)
v.train(budget, log)
print("trained", v.timing, "updates", v.updates, flush=True)
f = f"{out}/params.pkl"
pickle.dump({"params": jax.device_get(v.params), "lam": np.asarray(v.lam), "updates": v.updates}, open(f, "wb"))
heavy("final", full=True)
open(f"{out}/nvidia_after.txt", "w").write(os.popen("nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv").read())
print("DONE", flush=True)
