"""function-preservation check + one tiny speed look: l3cap arms vs control on random contexts (generation with same noise)."""
import sys, jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
p = get("ground"); V = load_variants(); reg = Region(p.W_eval)
ref = None
for n in ("l3cap_ctl", "l3cap_w2", "l3cap_c2", "l3cap_wc2"):
    v = V[n](p, 0, "/tmp/l3chk_" + n)
    shapes = {k: v.params[k]["W1"].shape for k in ("l3_cert3", "l2_cert2") if k in v.params} or {k: x["W1"].shape for k, x in list(v.params.items())[:3]}
    g = v.generate(5, reg)["raw"]
    x = np.concatenate([np.asarray(g[k]).ravel() for k in sorted(g)])
    if ref is None: ref = x
    print(n, shapes, "same draw as ctl:", bool((x == ref).all()), flush=True)
