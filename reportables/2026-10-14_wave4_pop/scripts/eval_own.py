"""64 own tau=1 draws per params file: dirt, line_match, top-row cert B4, violations per cell
(energy.py), valid fraction, by-row dirt.  usage (repo root): eval_own.py out.json name=path.pkl ..."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import energy as en
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.variants.ckernel_own import CKernelOwn
from castlegen.variants._cfast_core import train_pass
p = get("ground"); reg = Region(p.W_train)
V = CKernelOwn(p, 0, "/tmp"); lay = V.layout(reg)
gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), jnp.asarray(V.lam, jnp.float32), record=False)[0], in_axes=(None, 0, 0)))
seeds = np.arange(6_000_000, 6_000_064)
ds = [p.designed_values(int(s), reg) for s in seeds]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
Bv = p.chan["cert4"].views["B"]
out = {}
for a in sys.argv[2:]:
    name, path = a.split("=", 1)
    pr = V.params if path == "start" else jax.tree_util.tree_map(jnp.asarray, pickle.load(open(path, "rb"))["trained"]["params"])
    st = {k: np.asarray(v) for k, v in gen(pr, jnp.asarray(seeds, jnp.int32), dj).items()}
    st = {**{k: np.asarray(v) for k, v in dj.items()}, **st}
    S = [p.stats({"t": st["t"][b], "fill": st["fill"][b], "rho4": st["rho4"][b]}) for b in range(64)]
    Es = [en.energy(p, {k: v[b] for k, v in st.items()}, reg) for b in range(64)]
    nv = [sum(E.viol.values()) for E in Es]
    lvf = [float(sum(E.viol[l] for E in Es) / max(sum(E.nhard[l] for E in Es), 1)) for l in range(p.T + 1)]
    out[name] = {"dirt": float(np.mean([s["dirt_frac"] for s in S])), "line_match": float(np.mean([s["line_match"] for s in S])),
                 "line_match_sd": float(np.std([s["line_match"] for s in S])),
                 "B4_top_row": float(Bv[st["cert4"]][:, 0].mean() / 16), "viol_per_cell": float(np.mean(nv) / (64 * 128)),
                 "valid_frac": float(np.mean(np.asarray(nv) == 0)), "viol_frac_per_level": lvf, "dirt_by_L4_row": st["t"].reshape(64, 4, 16, 128).mean((0, 2, 3)).round(3).tolist(),
                 "fill_by_L4_row": (st["fill"].mean((0, 2)) / 16).round(3).tolist(), "fill_err": float(np.mean([s["fill_err"] for s in S]))}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
