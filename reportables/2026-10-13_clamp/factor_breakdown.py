"""Per-factor soft energy (per fine cell) of trained own draws vs probe-clamped
draws (L = 4, 3), 8 seeds; plus how the own top level relates to the fill."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems.ground import cert_domain
from castlegen.variants import _cpass_core as core
from clamp_tools import clamped_pass_stats
p = get_problem("ground"); T = p.T; reg = Region(p.W_train); cells = p.H * reg.W
lay = core.Layout(p, reg.W, True, 0)
ck = pickle.load(open("params_ground.pkl", "rb"))
NS = 8; seeds = np.arange(3_000_000, 3_000_000 + NS)
des = [p.designed_values(int(s), reg) for s in seeds]
prs = [p.probes(int(s), reg)[0] for s in seeds]
f = jax.jit(jax.vmap(lambda pr, lm, s, d, c: clamped_pass_stats(lay, pr, s, d, 1.0, lm, c), in_axes=(None, None, 0, 0, 0)))
dj = {k: jnp.asarray(np.stack([d[k] for d in des]), jnp.int32) for k in des[0]}
out = {}
for model in ("untrained", "trained"):
    params = jax.tree_util.tree_map(jnp.asarray, ck[model]["params"]); lam = jnp.asarray(ck[model]["lam"], jnp.float32)
    for cond, L in (("own", 99), ("probe_L4", 4), ("probe_L3", 3)):
        cl = {c.name: jnp.asarray(np.stack([pr[c.name] for pr in prs]), jnp.int32) for c in p.sampled() if c.level >= L}
        st, _ = f(params, lam, jnp.asarray(seeds, jnp.int32), dj, cl)
        st = jax.device_get(st)
        pf = {}
        for b in range(NS):
            x = {**des[b], **{c.name: np.asarray(st[c.name][b]) for c in p.sampled()}}
            E = en.energy(p, x, reg)
            for k, (s_, v_, n_) in E.per_factor.items():
                pf[k] = pf.get(k, 0.0) + s_ / cells / NS
        r = {"soft_by_factor": {k: round(v, 4) for k, v in pf.items() if v > 1e-3}}
        if cond == "own":
            A4, B4 = cert_domain(16)
            c4 = np.asarray(st["cert4"]); r4 = np.asarray(st["rho4"]); fill = np.stack([d["fill"] for d in des])
            r["rho4_minus_256f_mean_abs"] = float(np.abs(r4 - 16 * fill).mean())
            r["rho4_mean"] = float(r4.mean()); r["256f_mean"] = float((16 * fill).mean())
            r["cert4_A_mean"] = float(A4[c4].mean()); r["cert4_B_mean"] = float(B4[c4].mean())
            for l in (3, 2, 1):
                rl = np.asarray(st[f"rho{l}"]); r[f"rho{l}_frac_mean"] = float(rl.mean() / 4 ** l)
            r["dirt_frac"] = float(np.asarray(st["t"]).mean())
        out[f"{model}/{cond}"] = r
        print(model, cond, json.dumps(r), flush=True)
json.dump(out, open("factor_breakdown.json", "w"), indent=1)
