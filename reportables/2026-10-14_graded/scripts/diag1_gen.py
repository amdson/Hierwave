"""Diagnostic 1 (ideas #21): graded violation mass next to binary counts on held-out eval worlds of a
saved model, raw and final (repaired), no retraining.  usage: diag1_gen.py <label> <variant> <params.pkl>
<out.json> [n_worlds=16].  Pickle layouts: {"params","lam"} or {"trained": {"params","lam"}}.
The variant class is constructed only for its layout / network shape (no training, no kernels)."""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench import energy as en
from castlegen import graded as gr

label, vname, path, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 16
V = load_variants(); p = get("ground"); reg = Region(p.W_eval)
ck = pickle.load(open(path, "rb"))
tr = ck["trained"] if "trained" in ck else ck
t0 = time.time()
v = V[vname](p, 0, "/tmp/graded_" + label)
assert set(tr["params"]) == set(v.params), "params do not match the layout"
v.params = jax.tree_util.tree_map(jnp.asarray, tr["params"])
v.lam = np.asarray(tr["lam"], float)
print(f"{label}: built {vname} in {time.time()-t0:.1f} s, lam {np.round(v.lam, 3).tolist()}", flush=True)
worlds = []
for i in range(n):
    seed = EVAL_SEED0 + i
    t1 = time.time()
    g = v.generate(seed, reg)
    rec = {"seed": seed}
    for kind in ("raw", "final"):
        x = _with_designed(p, g[kind], seed, reg)
        E = en.energy(p, x, reg); G = gr.graded(p, x, reg)
        assert G.viol == E.viol and G.nhard == E.nhard
        rec[kind] = {"viol": [E.viol[l] for l in range(p.T + 1)], "nhard": [E.nhard[l] for l in range(p.T + 1)],
                     "mass": [G.mass[l] for l in range(p.T + 1)],
                     "per_factor": {k: [vv, m] for k, (vv, m, _) in G.per_factor.items() if vv},
                     "hist": {str(l): G.hist(l) for l in range(p.T + 1)},
                     "stats": {k: float(s) for k, s in p.stats(x).items()}, "soft": E.soft}
    worlds.append(rec)
    print(f"  world {i}: gen {time.time()-t1:.1f} s  raw viol {rec['raw']['viol']} mass {rec['raw']['mass']} | "
          f"final viol {rec['final']['viol']} mass {rec['final']['mass']}", flush=True)
json.dump({"label": label, "variant": vname, "params": path, "lam": v.lam.tolist(), "n": n,
           "nhard": worlds[0]["raw"]["nhard"], "worlds": worlds}, open(out, "w"), indent=1)
print(f"{label}: done in {time.time()-t0:.1f} s", flush=True)
