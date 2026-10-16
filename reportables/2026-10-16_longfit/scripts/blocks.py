"""block_check (full size: 128 own + 32 probe seeds) on saved params.  usage: blocks.py <label>=<params.pkl> ..."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import block_check
p = get("ground"); V = load_variants(); reg = Region(p.W_eval)
for a in sys.argv[1:]:
    lab, path = a.split("=")
    ck = pickle.load(open(path, "rb")); ck = ck.get("trained", ck)
    v = V["cfitpen_ml2"](p, 0, "/tmp/lf_blocks")
    v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)
    b = block_check(p, v, 0, reg)["levels"]
    print(json.dumps({"label": lab, **{k: [round(b[str(l)]["direct"][k], 4) for l in range(5)] for k in ("rho_pool", "rho_own", "starved_probe", "starved_own")}}), flush=True)
