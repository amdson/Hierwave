"""wave5-pop wrapper (the bench harness itself is NOT edited).

  python bench_warm.py smoke <variant> <train_s> <seeds>   # mimics run.py's per-seed loop (construct,
        train) without evaluation or writing results/bench.jsonl; prints setup per seed
  python bench_warm.py bench <variant> <train_s> <seeds>   # warm, then castlegen.bench.run.run(...)

Warm-up: one instance of the variant (seed 99, max_iters=3) is constructed and trained in THIS
process before the bench's clock starts, through the bench's import path (load_variants).  Its
jitted callables are memoised per process (CKernelOwn._build), so the bench's seed instances
reuse the traced functions: the same as running each seed in an already-warmed process.  The
warm-up instance shares no state with the bench instances (own params, rng, population)."""
import sys, json, time
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get

mode, name, train_s, nseeds = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
warm = "--cold" not in sys.argv
V = load_variants()
prob = get("ground")
if warm:
    t0 = time.time()
    w = V[name](prob, seed=99, workdir="/tmp/wave5-pop_warm", max_iters=3)
    w.train(1e9, lambda d: None)
    print(f"warm-up: {time.time() - t0:.1f} s ({w.timing})", flush=True)
    del w
if mode == "smoke":
    for k in range(nseeds):
        logs = []
        v = V[name](prob, seed=k, workdir="/tmp/wave5-pop_smoke")
        t0 = time.time()
        v.train(train_s, logs.append)
        st = [r["t"] for r in logs if r.get("setup_done")][0]
        print(json.dumps({"seed": k, "setup_s": round(st, 1), "train_s": round(time.time() - t0, 1),
                          "updates": v.updates, "rkl_updates": v.rkl_updates, "iters": v.timing["iters"],
                          "iter_t": [round(r["t"], 1) for r in logs if "iter" in r][:6]}), flush=True)
else:
    from castlegen.bench.run import run
    from castlegen.bench.board import board
    rec = run(name, "ground", nseeds, train_s, 16, "fitpen")
    board()
    for k, m in enumerate(rec["per_seed"]):
        print("seed", k, "train_s", m["train_s"], flush=True)
