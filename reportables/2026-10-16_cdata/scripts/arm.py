"""cdata arm: continue training ccdata from the saved cfitpen_ml2 params for <train_s> seconds, log every iteration,
reduced block checks mid-run, then the bench's eval_seed (full block check, violations, stats), per-level offset, params.pkl.
usage: arm.py <label> '<json cfg>' <train_s> <outdir>"""
import sys, json, time, pickle
import numpy as np, jax, jax.numpy as jnp
from pathlib import Path
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench import run as R
from castlegen.bench.problems import ground as G

label, cfg, train_s, out = sys.argv[1], json.loads(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
INIT = "reportables/2026-10-16_fitpen/runs/arm_cfitpen_ml2/params.pkl"
cfg = {"init_pkl": INIT, **cfg}
V = load_variants(); prob = get("ground"); reg = Region(128)
t0 = time.time()
w = V["ccdata"](prob, seed=99, workdir="/tmp/cdata_warm", max_iters=3, **cfg); w.train(1e9, lambda d: None)
print(f"warm-up {time.time() - t0:.1f} s", flush=True); del w
v = V["ccdata"](prob, seed=0, workdir=str(out), **cfg)
f = open(out / "loop.jsonl", "w")
full = (R.N_POOL_OWN, R.N_POOL_PROBE_SEEDS)
checks, due = [], [train_s / 3, 2 * train_s / 3]
KEEP = ("iter", "t", "t_train", "update", "loss_per_cell", "time_refine", "time_ml", "time_rkl", "rkl_raw_viol", "lam")

def log(d):
    r = {k: d[k] for k in KEEP if k in d}
    if "own_raw" in d:
        o = d["own_raw"]; r["own_raw"] = {k: o.get(k) for k in ("line_match", "dirt", "offset")}
    f.write(json.dumps(r, default=float) + "\n"); f.flush()
    if due and d.get("t", 0) >= due[0]:
        due.pop(0); tc = time.time()
        R.N_POOL_OWN, R.N_POOL_PROBE_SEEDS = 16, 8            # reduced mid-run check (cost); final eval is the full one
        try:
            bc = R.block_check(prob, v, 0, reg)
        finally:
            R.N_POOL_OWN, R.N_POOL_PROBE_SEEDS = full
        checks.append({"t": d["t"], "update": d.get("update"), "check_s": time.time() - tc, "blocks": bc})
        print("mid check", d["t"], json.dumps(bc, default=float)[:400], flush=True)

t1 = time.time()
v.train(train_s, log)
t_train = time.time() - t1
pickle.dump({"params": v.params, "lam": v.lam}, open(out / "params.pkl", "wb"))
m = R.eval_seed(prob, v, 0, 16, out / "eval_s0.png")
m["train_s"] = t_train; m["updates"] = v.updates; m["rkl_updates"] = v.rkl_updates; m["timing"] = v.timing
m["mid_checks"] = checks; m["cfg"] = {k: cfg[k] for k in cfg}
# per-level implied surface offset of the raw pass draws (offset.py measure), 16 eval worlds
H, W = prob.H, 128
seeds = [R.EVAL_SEED0 + i for i in range(16)]
gs = [G._line_from_fill(np.asarray(v.designed(s, reg)["fill"], np.int64), W) for s in seeds]
xs = [v.generate(s, reg)["raw"] for s in seeds]
off = []
for l in range(prob.T + 1):
    h = 1 << l
    off.append(float(np.mean([(H - np.asarray(x["t" if l == 0 else f"rho{l}"], np.float64).sum(0) / h) - g.reshape(-1, h).mean(1)
                              for x, g in zip(xs, gs)])))
m["offset_raw_L0_L4"] = off
json.dump(m, open(out / "eval.json", "w"), default=float, indent=1)
print("ARM_DONE", label, json.dumps({"updates": v.updates, "timing": v.timing, "offset": off,
      "raw": m["viol"]["raw"], "lm": m["stats"]["final"].get("line_match")}, default=float), flush=True)
