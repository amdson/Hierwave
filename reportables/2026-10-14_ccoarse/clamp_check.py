"""Clamp gate with prior contexts (ground, clamp's saved untrained/trained cfast
params): fine cost per fine cell below L with levels >= L clamped to
(a) probe lifts, (b) the restricted prior (all channels), (c) the prior on its
informative channels only (rho; cert drawn by the model), (d) nothing (own).
Also fine stats (dirt, line_match) and cert/rho consistency at L."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-13_clamp")
from clamp_tools import FIELDS, clamped_pass_stats
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench.problems.ground import cert_domain
from castlegen.variants import _cpass_core as core
from castlegen.variants import _coarse_prior as cpr

p = get("ground"); T = p.T; reg = Region(p.W_train)
lay = core.Layout(p, reg.W, True, 0)
ck = pickle.load(open("reportables/2026-10-13_clamp/params_ground.pkl", "rb"))
NS = 64
seeds = np.arange(3_000_000, 3_000_000 + NS)
des = [p.designed_values(int(s), reg) for s in seeds]
ds = {k: jnp.asarray(np.stack([d[k] for d in des]), jnp.int32) for k in des[0]}
cells = p.H * reg.W
fns = {}


def run(params, lam, cl):
    key = tuple(sorted(cl))
    if key not in fns:
        fns[key] = jax.jit(jax.vmap(lambda pr, lm, s, d, c: clamped_pass_stats(lay, pr, s, d, 1.0, lm, c),
                                    in_axes=(None, None, 0, 0, 0)))
    st, acc = fns[key](params, lam, jnp.asarray(seeds, jnp.int32), ds,
                       {k: jnp.asarray(v, jnp.int32) for k, v in cl.items()})
    return {k: np.asarray(v) for k, v in jax.device_get(st).items()}, np.asarray(acc)


res = {}
for L in (4, 3):
    prb = [p.probes(int(s), reg)[i % 4] for i, s in enumerate(seeds)]
    probe = {c.name: np.stack([q[c.name] for q in prb]) for c in p.sampled() if c.level >= L}
    prior = cpr.coarse_clamp_batch(p, seeds, L, sweeps=8)
    info = cpr.informative_channels(p, L)
    ctxs = {"probe lift": probe, "prior (all)": prior,
            "prior (informative: " + ",".join(info) + ")": {k: prior[k] for k in info}, "own": {}}
    for model in ("untrained", "trained"):
        params = jax.tree_util.tree_map(jnp.asarray, ck[model]["params"])
        lam = jnp.asarray(ck[model]["lam"], jnp.float32)
        for nm, cl in ctxs.items():
            st, acc = run(params, lam, cl)
            fine = acc[:, :L].sum(1)                                     # (B, fields)
            cost = float(fine[:, FIELDS.index("cost")].mean() / cells)
            nvf = float(fine[:, FIELDS.index("nv")].mean() / cells)
            stt = [p.stats({**des[b], **{k: v[b] for k, v in st.items()}}) for b in range(NS)]
            h = 1 << L
            A, B = cert_domain(h)
            r, c = st[f"rho{L}"], st[f"cert{L}"]
            cons = float(((r >= A[c] * h) & (r <= B[c] * h)).mean())
            row = {"fine cost/cell": cost, "fine viol/cell": nvf,
                   "dirt_frac": float(np.mean([s["dirt_frac"] for s in stt])),
                   "line_match": float(np.mean([s["line_match"] for s in stt])),
                   "rho-cert consistent at L": cons}
            res[f"L{L} | {model} | {nm}"] = row
            print(L, model, nm, {k: round(v, 4) for k, v in row.items()}, flush=True)
json.dump(res, open("reportables/2026-10-14_ccoarse/clamp_check.json", "w"), indent=1)
