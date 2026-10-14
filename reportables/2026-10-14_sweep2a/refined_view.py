"""Where do probe contexts go under the exact chain?  4 probe lifts (jitters 0-3),
n cycles of [(2,0),(1,1)] (lam = 20): render t before/after, line_match, dirt
fraction, and soft energy per cell broken down by factor."""
import json, sys
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench.problems.ground import stats
from castlegen.kernels.amortised import JaxBlockGibbs, refine
from castlegen.kernels.exact_block import _lookup

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D = "reportables/2026-10-14_sweep2a"
p = get("ground"); reg = Region(p.W_train); cells = p.H * reg.W
J20 = JaxBlockGibbs(p, reg, 2, 0, lam=20.0); J11 = JaxBlockGibbs(p, reg, 1, 1, lam=20.0)
fl, insts = J11.K.flat, J11.K.insts
seeds = list(range(3_000_000, 3_000_016))
xs = [{**p.probes(s, reg, n=4)[j % 4], **p.designed_values(s, reg)} for j, s in enumerate(seeds)]
v0 = np.stack([fl.flatten(x) for x in xs])
v, _ = refine([J20, J11], jnp.asarray(v0, jnp.int32), jax.random.PRNGKey(5), NC)
v = np.asarray(v)


def by_factor(vv):
    out = {}
    for d in insts:
        vals = [np.where(g >= 0, view[vv[np.maximum(g, 0)]], c) for g, c, view in d["members"]]
        e = _lookup(d, vals, d["pos"] if d.get("by_pos") else None)
        f = p.factors[d["f"]]
        nm = f"{d['f']}:{f.kind}:{f.a[0]}-{f.b[0] if f.b else ''}"
        out[nm] = out.get(nm, 0.0) + float(np.where(np.isinf(e), 0, e).sum()) / cells
    return out


res = {"line_match_probe": [], "line_match_ref": [], "dirt_probe": [], "dirt_ref": []}
bf0, bf1 = {}, {}
for b in range(len(seeds)):
    x0, x1 = fl.unflatten(v0[b]), fl.unflatten(v[b])
    for tag, x in (("probe", x0), ("ref", x1)):
        res[f"line_match_{tag}"].append(stats({"t": x["t"], "fill": x["fill"], "rho4": x["rho4"]})["line_match"])
        res[f"dirt_{tag}"].append(float(x["t"].mean()))
    for k, val in by_factor(v0[b]).items(): bf0[k] = bf0.get(k, 0) + val / len(seeds)
    for k, val in by_factor(v[b]).items(): bf1[k] = bf1.get(k, 0) + val / len(seeds)
summ = {k: float(np.mean(x)) for k, x in res.items()}
summ["by_jitter_line_match_ref"] = [float(np.mean(res["line_match_ref"][j::4])) for j in range(4)]
summ["by_jitter_line_match_probe"] = [float(np.mean(res["line_match_probe"][j::4])) for j in range(4)]
summ["soft_by_factor"] = {k: (round(bf0[k], 4), round(bf1[k], 4)) for k in bf0 if abs(bf0[k]) + abs(bf1[k]) > 1e-4}
summ["n_cycles"] = NC
print(json.dumps(summ, indent=1))
json.dump(summ, open(f"{D}/runs/refined_view.json", "w"), indent=1)
fig, ax = plt.subplots(4, 2, figsize=(12, 8))
for i in range(4):
    for j, (tag, vv) in enumerate((("probe (jitter %d)" % i, v0[i]), (f"after {NC} exact cycles", v[i]))):
        ax[i, j].imshow(p.render({"t": fl.unflatten(vv)["t"]}, scale=1), interpolation="nearest")
        ax[i, j].set_title(tag, fontsize=9); ax[i, j].axis("off")
plt.tight_layout(); plt.savefig(f"{D}/images/refined_probe_states.png", dpi=110)
