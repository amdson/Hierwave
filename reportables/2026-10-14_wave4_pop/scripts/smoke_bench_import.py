"""Warm the compile cache for a bench variant through the bench's import path (load_variants), 30 s train."""
import sys, json, time
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
name = sys.argv[1]
V = load_variants()[name](get("ground"), seed=0, workdir="/tmp/wave4-pop_smoke")
t0 = time.time()
V.train(float(sys.argv[2]) if len(sys.argv) > 2 else 30.0, lambda d: None)
print("smoke done", name, round(time.time() - t0, 1), "updates", V.updates, V.timing, flush=True)
