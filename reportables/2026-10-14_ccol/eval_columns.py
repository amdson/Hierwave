"""Clamp diagnostic on ground: levels >= L clamped to (a) a probe's lifted
coarse channels (fill-following) or (b) left to the model's own draw, same
noise seeds.  Per level below L: cost / soft / violations / log q per fine
cell, forced and no-option fractions; raw energy violations per level."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.variants import _cpass_core as core
from castlegen.variants._creplay_clamp import clamped_pass
from clamp_tools import FIELDS, clamped_pass_stats

D = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
p = get_problem("columns"); T = p.T
reg = Region(p.W_train)
lay = core.Layout(p, reg.W, True, 0, resid=bool(int(sys.argv[3])) if len(sys.argv) > 3 else False)
ck = pickle.load(open(f"{D}/params_columns.pkl", "rb"))
cells = p.H * reg.W
seeds = np.arange(3_000_000, 3_000_000 + NS)
des = [p.designed_values(int(s), reg) for s in seeds]
LS = (4, 3, 2, 1)
probes = [p.probes(int(s), reg) for s in seeds]          # 4 per seed: jitter 0..3 rows
NP = len(probes[0])
sampled = [c.name for c in p.sampled()]


def stack(ds):
    return {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}



def run(params, lam, sd, ds, cl):
    key = tuple(sorted(cl))
    out_s, out_a = [], []
    CH = 32
    for i in range(0, len(sd), CH):
        global_lam = jnp.asarray(lam, jnp.float32)
        f = jax.jit(jax.vmap(lambda pr, lm, s, d, c: clamped_pass_stats(lay, pr, s, d, 1.0, lm, c),
                             in_axes=(None, None, 0, 0, 0))) if key not in fns2 else fns2[key]
        fns2[key] = f
        st, acc = f(params, global_lam, jnp.asarray(sd[i:i + CH], jnp.int32),
                    {k: v[i:i + CH] for k, v in ds.items()}, {k: v[i:i + CH] for k, v in cl.items()})
        out_s.append(jax.device_get(st)); out_a.append(np.asarray(acc))
    st = {k: np.concatenate([o[k] for o in out_s]) for k in out_s[0]}
    return st, np.concatenate(out_a)
fns2 = {}


def energy_viol(st, sd, dsl):
    v = np.zeros((len(sd), T + 1)); n = np.zeros(T + 1)
    for b in range(len(sd)):
        x = {**dsl[b], **{c: st[c][b] for c in sampled}}
        E = en.energy(p, x, reg)
        for l in range(T + 1):
            v[b, l] = E.viol[l]; n[l] = E.nhard[l]
    return v, n


def coarse_clamp(L, prs):
    return {c.name: np.stack([pr[c.name] for pr in prs]) for c in p.sampled() if c.level >= L}


res = {}
images = {}
for model in ("untrained", "trained"):
    params = jax.tree_util.tree_map(jnp.asarray, ck[model]["params"])
    lam = ck[model]["lam"]
    # (b) own coarse draw: one unclamped run per seed (the same run serves every L)
    st_own, acc_own = run(params, lam, seeds, stack(des), {})
    ev_own, nh = energy_viol(st_own, seeds, des)
    res[(model, "own")] = {"acc": acc_own, "ev": ev_own, "st": st_own}
    # sanity: the stats pass reproduces clamped_pass's costs (seed 0, L=4 probe clamp)
    for L in LS:
        sd = np.repeat(seeds, NP); dsl = [d for d in des for _ in range(NP)]
        prs = [pr for ps in probes for pr in ps]
        cl = {k: jnp.asarray(v, jnp.int32) for k, v in coarse_clamp(L, prs).items()}
        st, acc = run(params, lam, sd, stack(dsl), cl)
        ev, _ = energy_viol(st, sd, dsl)
        res[(model, f"probe_L{L}")] = {"acc": acc, "ev": ev, "st": st, "jit": np.tile(np.arange(NP), NS)}
    # check against the reference clamped_pass on one world
    d0 = {k: jnp.asarray(v, jnp.int32) for k, v in des[0].items()}
    cl0 = {k: jnp.asarray(v[0], jnp.int32) for k, v in coarse_clamp(L, probes[0]).items()}
    _, costs, _ = jax.jit(lambda pr: clamped_pass(lay, pr, int(seeds[0]), d0, 1.0, jnp.asarray(lam, jnp.float32), cl0))(params)
    ref = {l: sum(float(jnp.sum(costs[core.key_of(l, c.name)])) for c in lay.chans[l]) for l in range(L)}
    mine = res[(model, f"probe_L{L}")]["acc"][0][:, FIELDS.index("cost")]
    print(model, "check vs clamped_pass", {l: (ref[l], float(mine[l])) for l in range(L)}, flush=True)

# ------------------------------------------------------------------ summary
summ = {}
for (model, cond), r in res.items():
    acc, ev = r["acc"], r["ev"]
    s = {}
    for l in range(T + 1):
        n = acc[:, l, FIELDS.index("n")]
        row = {}
        for f in ("cost", "soft", "nv", "logq"):
            x = acc[:, l, FIELDS.index(f)] / cells
            row[f + "_per_cell"] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))))
        for f in ("forced", "noopt", "noopt_now"):
            x = acc[:, l, FIELDS.index(f)] / np.maximum(n, 1)
            row[f + "_frac"] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))))
        x = ev[:, l] / max(nh[l], 1)
        row["viol_frac"] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))))
        kk = r["st"]["k"]
        row["dirt_frac"] = float((kk > 0).mean())
        row["empty_xcol"] = float((~(kk > 0).any(1)).mean())
        s[l] = row
    if "jit" in r:                                   # per probe jitter: cost of levels < 4 per cell
        s["by_jitter"] = {int(j): float((acc[r["jit"] == j][:, :T, FIELDS.index("cost")].sum(1) / cells).mean())
                          for j in range(NP)}
    summ[f"{model}/{cond}"] = s
json.dump({"summary": summ, "n_seeds": NS, "n_probes_per_seed": NP, "updates": ck["updates"],
           "lam": {m: ck[m]["lam"] for m in ("untrained", "trained")}},
          open(f"{D}/columns_summary.json", "w"), indent=1)
np.savez_compressed(f"{D}/columns_raw.npz", **{f"{m}__{c}__acc": r["acc"] for (m, c), r in res.items()},
                    **{f"{m}__{c}__ev": r["ev"] for (m, c), r in res.items()},
                    **{f"{m}__{c}__t": r["st"]["k"][:8] for (m, c), r in res.items()})
print(json.dumps({k: {l: {f: v[f] for f in ("cost_per_cell", "viol_frac", "forced_frac", "noopt_frac")}
                      for l, v in s.items() if l != "by_jitter"} for k, s in summ.items()}, indent=0)[:6000])
