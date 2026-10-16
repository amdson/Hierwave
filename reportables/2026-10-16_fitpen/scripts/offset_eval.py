"""Per-level implied surface offset (rows) of the pass's raw draws on the bench eval worlds; measure copied from
reportables/2026-10-16_offset/scripts/offset.py (no clamped runs).  usage: offset_eval.py <params.pkl>... (n=16 worlds)"""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as G
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0
p = get("ground"); reg = Region(128); H = p.H; W = 128; n = 16
V = load_variants()
v = V["ckpop_hard_pen"](p, 0, "/tmp/fitpen_off")
seeds = [EVAL_SEED0 + i for i in range(n)]
gs = [G._line_from_fill(np.asarray(v.designed(s, reg)["fill"], np.int64), W) for s in seeds]
for path in sys.argv[1:]:
    ck = pickle.load(open(path, "rb"))
    v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)
    xs = [v.generate(s, reg)["raw"] for s in seeds]
    off = []
    for l in range(p.T + 1):
        h = 1 << l
        off.append(float(np.mean([(H - np.asarray(x["t" if l == 0 else f"rho{l}"], np.float64).sum(0) / h) - g.reshape(-1, h).mean(1)
                                  for x, g in zip(xs, gs)])))
    print(json.dumps({"params": path, "offset_raw_L0..L4": [round(o, 3) for o in off]}), flush=True)
