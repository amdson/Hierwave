"""cdim3 task 1: distil table_feat's converged per-context policies into neural actors by supervised
KL (same features, masks applied), weighted by target visits ('target') or uniform over all reachable
contexts ('all'); then the exact KL of the distilled actor.  Writes runs/distill_<kind>_<how>.json and
/tmp/cdim3_distilled_<kind>_<how>.pkl."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.variants import _exactopt as eo

kinds = sys.argv[1].split(",")
hows = sys.argv[2].split(",") if len(sys.argv) > 2 else ["target"]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-3
problem, region = eo.get_problem("dimers4")
lam = np.full(1, 20.0)
base = eo.make_variant("cpass", problem, region, 0)
enum = eo.Enum(problem, region, base.params, lam, cache="/tmp/exactopt_enum_dimers4.pkl")
src = jax.tree_util.tree_map(jnp.asarray, pickle.load(open("/tmp/cdim3_table_feat.pkl", "rb")))
print("source table_feat:", enum.evaluate("table_feat", src, eo.dimer_stats), flush=True)
for kind in kinds:
    for how in hows:
        p0 = enum.init_params(kind, base.params)
        p, curve = eo.distill(enum, "table_feat", src, kind, p0, steps=steps, lr=lr, how=how, every=500)
        tag = f"{kind}_{how}"
        pickle.dump(jax.device_get(p), open(f"/tmp/cdim3_distilled_{tag}.pkl", "wb"))
        json.dump({"kind": kind, "how": how, "steps": steps, "lr": lr, "curve": curve},
                  open(f"reportables/2026-10-14_cdim3/runs/distill_{tag}.json", "w"))
        m = curve[-1]
        print(f"== distilled {tag}: loss {m['distill_loss']:.4f} kl {m['kl']:.4f} H/cell {m['entropy'] / 16:.3f} "
              f"p_viol {m['p_viol']:.4f} h/v/d {m['horizontal']:.3f}/{m['vertical']:.3f}/{m['diagonal']:.3f}", flush=True)
