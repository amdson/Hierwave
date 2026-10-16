"""v2 block check on ground_cf (pass only). usage: blockcf.py <params.pkl> <seed_k> <out.json>"""
import json, pickle, sys, jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants, VARIANTS
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import block_check
path, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
p = get("ground"); V = load_variants(); ck = pickle.load(open(path, "rb"))
v = V["ccertfree"](p, k, "/tmp/certfree_blk")
v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)
base = VARIANTS["cfitpen_ml2"]
class Shim:
    def generate(self, seed, region):
        des, raw = v._pass(seed, region); x = {c.name: np.asarray(raw[c.name]) for c in v.problem.sampled()}
        return {"raw": x, "final": x}
    def score(self, x, region): return base.score(v, x, region)
    def score_sites(self, x, region): return base.score_sites(v, x, region)
res = block_check(v.problem, Shim(), k, Region(128))
json.dump(res, open(out, "w"), default=float)
for l, r in res["levels"].items():
    d = r["direct"]; print(k, l, "rho_pool %.2f own %.2f probe %.2f starved_probe %.1f%%" % (d["rho_pool"], d["rho_own"], d["rho_probe"], 100*d["starved_probe"]))
print("DONE")
