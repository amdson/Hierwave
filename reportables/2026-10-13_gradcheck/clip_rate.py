"""How often does cpass's per-level gradient clip (norm 10) bind in real training?
A binding clip rescales each batch by a batch-dependent factor, which biases
the expected update away from grad J.  PYTHONPATH=. python ... <problem> <secs>"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problems import get as get_problem  # noqa
from castlegen.variants import cpass

norms = []


def logged_clip(g, levels, c):
    ns = []
    for l in levels:
        ks = [k for k in g if k.startswith(f"l{l}_")]
        ns.append(jnp.sqrt(sum(jnp.sum(x * x) for k in ks for x in jax.tree_util.tree_leaves(g[k]))))
    jax.debug.callback(lambda *a: norms.append([float(x) for x in a]), *ns)
    return _orig(g, levels, c)


_orig = cpass.clip_by_level
cpass.clip_by_level = logged_clip

name, secs = sys.argv[1], float(sys.argv[2])
prob = get_problem(name)
v = cpass.CPass(prob, 0, "/tmp/gc_clip")
v.train(secs, lambda d: None)
cpass.clip_by_level = _orig
n = np.array(norms)        # (updates * 2: actor then critic, levels)
act, cri = n[0::2], n[1::2]
out = {"problem": name, "updates": len(act), "levels": list(range(prob.T + 1)),
       "actor_norm_median": np.median(act, 0).tolist(), "actor_norm_p90": np.percentile(act, 90, 0).tolist(),
       "actor_clip_frac": (act > 10).mean(0).tolist(), "critic_clip_frac": (cri > 10).mean(0).tolist()}
print(json.dumps(out))
json.dump(out, open(f"reportables/2026-10-13_gradcheck/clip_{name}.json", "w"), indent=1)
