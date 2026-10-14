"""wave5-pop: where does ckpop setup time go?  Constructs the variant through the bench's import
path (load_variants), wraps setup stages with timers, counts backend compiles and persistent-cache
hits via jax.monitoring, runs train(budget) for seeds given, all in ONE process (like run.py)."""
import sys, json, time, collections
import jax
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
V = load_variants()
print("cache dir:", jax.config.jax_compilation_cache_dir,
      "min_compile_s:", jax.config.jax_persistent_cache_min_compile_time_secs, flush=True)
EV = collections.Counter(); DUR = collections.defaultdict(float)
jax.monitoring.register_event_listener(lambda e, **kw: EV.update([e]))
def _d(e, s, **kw):
    DUR[e] += s; EV.update(["n:" + e])
jax.monitoring.register_event_duration_secs_listener(_d)
name = sys.argv[1]; budget = float(sys.argv[2]); seeds = [int(s) for s in sys.argv[3].split(",")]
out = []
for k in seeds:
    v = V[name](get("ground"), seed=k, workdir="/tmp/wave5-pop_prof")
    T = collections.defaultdict(float); N = collections.Counter()
    def wrap(obj, attr, label=None):
        f = getattr(obj, attr)
        def g(*a, **kw):
            t = time.time(); r = f(*a, **kw)
            jax.block_until_ready(r) if r is not None else None
            T[label or attr] += time.time() - t; N[label or attr] += 1
            return r
        setattr(obj, attr, g)
    for a in ("_build", "_own_ctx", "refine", "_pop_stats", "_rkl_setup", "_rkl_steps", "_level_viol", "_ground_stats"):
        wrap(v, a)
    import castlegen.kernels.colour_gibbs as cg
    if not hasattr(cg, "_orig_sweep_all"):
        cg._orig_sweep_all = cg.sweep_all
    def sa(*a, **kw):
        t = time.time(); r = cg._orig_sweep_all(*a, **kw); jax.block_until_ready(r)
        T["cg.sweep_all"] += time.time() - t; N["cg.sweep_all"] += 1; return r
    cg.sweep_all = sa
    EV.clear(); DUR.clear()
    logs = []
    t0 = time.time()
    v.train(budget, logs.append)
    setup = [r for r in logs if r.get("setup_done")]
    rec = {"seed": k, "setup": setup[0]["t"] if setup else None, "total": time.time() - t0, "updates": v.updates,
           "rkl_updates": getattr(v, "rkl_updates", 0), "iters": v.timing.get("iters"),
           "stage_s": {a: round(x, 2) for a, x in T.items()}, "stage_n": dict(N),
           "events": {e: c for e, c in EV.items() if "cache" in e or "compile" in e},
           "dur": {e: round(x, 2) for e, x in DUR.items()}}
    # per-iteration wall from log
    ts = [r["t"] for r in logs if "iter" in r]
    rec["iter_t"] = [round(x, 1) for x in ts[:8]]
    print(json.dumps(rec), flush=True); out.append(rec)
