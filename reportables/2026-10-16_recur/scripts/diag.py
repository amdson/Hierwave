"""Step-4 diagnosis: which coarse cells the subtree bound favours.  For each params file: 32 own worlds (W 128), K = 16
fine completions per world (split_pass, l0 = 3).  Per L3 / L4 cell: footprint cost per replica (fine draws L0-L2 under
the cell, soft + lam nv + log q, tau = 1), R_1 (replica 0), R_16, log-weight std, ESS; the cell's dirt fraction (rho / h^2);
the coarse draw's entropy (sum_a q log q) and -log q of the drawn value.  Grouped by dirt class (air = 0, dirt = 1, mixed).
Also the same for L3 cells under probe-lift L4 (L4 clamped).  usage: diag.py <out.json> <label=params.pkl> ..."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.variants._crecur_core import split_pass, foot_sums
from castlegen.variants._cpass_core import key_of
p = get("ground"); reg = Region(p.W_train); V = load_variants(); T = p.T
v = V["crecur"](p, 0, "/tmp/recur_diag", rec_n=0); lay = v.layout(reg)
NW, K = 32, 16
seeds = np.arange(4_000_000, 4_000_000 + NW)
des = {k: jnp.asarray(np.stack([p.designed_values(int(s), reg)[k] for s in seeds]), jnp.int32) for k in ("fill",)}
probes = [p.probes(int(s), reg, n=4)[int(s) % 4] for s in seeds]
cl = {c.name: jnp.asarray(np.stack([pr[c.name] for pr in probes]), jnp.int32) for c in lay.chans[T]}
fkeys = [key_of(l, c.name) for l in range(3) for c in lay.chans[l]]
out = {}
for arg in sys.argv[2:]:
    label, path = arg.split("=")
    ck = pickle.load(open(path, "rb")); ck = ck.get("trained", ck)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); lam = jnp.asarray(ck["lam"], jnp.float32)
    res = {}
    for cond in ("own", "probeL4"):
        flag = cond == "probeL4"

        @jax.jit
        def one(s, d, c):
            fs = s * 1000 + jnp.arange(K)
            st, recs = split_pass(lay, params, s, fs[0], d, 1.0, lam, 3, clamp=c, cflag=flag, Lc=T)
            _, costs = jax.vmap(lambda f: split_pass(lay, params, s, f, d, 1.0, lam, 3, record=False, clamp=c, cflag=flag, Lc=T))(fs)
            fine = {k: costs[k] for k in fkeys}
            o = {}
            for L in (3, 4):
                ft = foot_sums(lay, fine, L)                                   # (K, 16, n)
                r = recs[key_of(L, f"rho{L}")]
                ent = -jnp.sum(jnp.where(r["mask"], jnp.exp(r["logq"]) * jnp.where(r["mask"], r["logq"], 0.0), 0.0), -1)
                nlq = -jnp.take_along_axis(r["logq"], r["a"][..., None], -1)[..., 0]
                o[L] = {"ft": ft, "valid": r["valid"], "a": r["a"], "ent": ent, "nlq": nlq}
            return o
        rows = {3: [], 4: []}
        for b in range(NW):
            o = jax.device_get(one(jnp.int32(seeds[b]), {k: x[b] for k, x in des.items()}, {k: x[b] for k, x in cl.items()}))
            for L in (3, 4):
                m = o[L]["valid"]
                ft = o[L]["ft"][:, m]                                           # (K, ncell)
                lw = -ft
                R16 = -(np.logaddexp.reduce(lw, 0) - np.log(K))
                ess = np.exp(2 * np.logaddexp.reduce(lw, 0) - np.logaddexp.reduce(2 * lw, 0)) / K
                dirt = o[L]["a"][m] / float((1 << L) ** 2)
                rows[L].append(np.stack([dirt, ft[0], R16, ft.std(0), ess, o[L]["ent"][m], o[L]["nlq"][m]], 1))
        for L in (3, 4):
            a = np.concatenate(rows[L])
            if flag and L == 4:
                continue
            cls = {"air": a[:, 0] == 0, "dirt": a[:, 0] == 1, "mixed": (a[:, 0] > 0) & (a[:, 0] < 1)}
            res[f"{cond}_L{L}"] = {n: {"frac": float(m.mean()), "R1": float(a[m, 1].mean()), "R16": float(a[m, 2].mean()),
                                       "R1_minus_R16": float((a[m, 1] - a[m, 2]).mean()), "lw_std": float(a[m, 3].mean()),
                                       "ess": float(a[m, 4].mean()), "coarse_entropy": float(a[m, 5].mean()),
                                       "coarse_nlogq": float(a[m, 6].mean())} for n, m in cls.items() if m.any()}
    out[label] = res
    print(label, json.dumps(res), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DONE")
