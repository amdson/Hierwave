"""Per-level actor-gradient SNR of one batch (B = 8) at the end of training, all worlds clamped at L = 3
to probe lifts (the training distribution).  SNR_B = (|mean|^2 - trCov/M) / trCov over M batches;
the 1/M term makes the estimate unbiased, its floor is about -1/M.
usage: snr.py <rundir> <variant> <M>"""
import json, pickle, sys, tempfile
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.variant import load_variants
from castlegen.variants import _gradcheck as gc
D, vname, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = get_problem("ground"); ck = pickle.load(open(f"{D}/params_ground.pkl", "rb"))
v = load_variants()[vname](p, 0, tempfile.mkdtemp())
region = Region(p.W_train); lay = v.layout(region); fn = v.train_fns(lay)
tr = ck["trained"]
lam = jnp.asarray(tr["lam"], jnp.float32); tau = jnp.float32(1.0)
gs = tr.get("gs") or {k: jnp.float32(1.0) for k in v.params}
keys = sorted(v.params); sl = gc.block_slices(v.params, keys); B = v.cfg["B"]
J = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
arms = [("trained_zero_critic", J(tr["params"]), gc.zero_critic(v.cparams)),
        ("init", J(ck["untrained"]["params"]), gc.zero_critic(v.cparams))]
if "cparams" in tr:
    arms.insert(0, ("trained", J(tr["params"]), J(tr["cparams"])))
out = {"M": M, "B": B, "lam": tr["lam"]}
for name, pa, cp in arms:
    est = []
    for m in range(M):
        seeds = np.arange(8_000_000 + m * B, 8_000_000 + (m + 1) * B)
        ctx, Lj = v.contexts(seeds, region, np.full(B, 3, np.int64))
        _, rs, G, _, _, _ = fn["batch"](pa, jnp.asarray(seeds, jnp.int32), v._batch(seeds, region), tau, lam,
                                        ctx, Lj, jax.random.PRNGKey(m))
        est.append(gc.flat(fn["actor_grad"](pa, cp, rs, G, tau, lam, gs), keys))
    est = np.array(est); res = {}
    for l in range(3):
        cols = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in keys if k.startswith(f"l{l}_")])
        e = est[:, cols]; mu = e.mean(0); trc = e.var(0, ddof=1).sum(); sig = mu @ mu - trc / M
        # bootstrap SE of the signal estimate over batches
        bs = []
        rng = np.random.default_rng(0)
        for _ in range(100):
            ii = rng.integers(M, size=M); eb = e[ii]; mb = eb.mean(0); tb = eb.var(0, ddof=1).sum()
            bs.append((mb @ mb - tb / M) / tb)
        res[l] = {"snr_batch": float(sig / trc), "snr_se": float(np.std(bs)), "noise": float(trc)}
    out[name] = res
    print(name, json.dumps(res), flush=True)
json.dump(out, open(f"{D}/snr_M{M}.json", "w"), indent=1)
