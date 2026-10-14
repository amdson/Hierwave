"""Per-level acceptance of q-proposal subtree resampling on ground.
States: (a) own draws of the trained q (raw pass, tau = 1), (b) probe lifts.
One move per (state, uniformly random level-L cell), no chaining.
usage: ground_accept.py params.pkl tag [N]"""
import sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_kernel_b/scripts")
from common import *
from castlegen.kernels import block_mh as bm

pk, tag = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
p = get_problem("ground"); reg = Region(p.W_train); lay = core.Layout(p, reg.W, True, 0)
params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(pk, "rb"))[sys.argv[4] if len(sys.argv) > 4 else "trained"]["params"])
NW = 64
seeds = np.arange(4_000_000, 4_000_000 + NW)
own = own_draws(p, lay, params, seeds, reg)
prb = [p.probes(int(s), reg)[int(s) % 4] for s in seeds]
probe = {k: jnp.asarray(np.stack([x[k] for x in prb]), jnp.int32) for k in prb[0]}
print("own dirt frac", float(np.asarray(own["t"]).mean()), "probe", float(np.asarray(probe["t"]).mean()), flush=True)
fns = {}
def get_fn(L, K):
    if (L, K) not in fns:
        def one(params, key, state, Y, X, tau):
            return bm.mh_move(lay, params, key, state, jnp.int32(L), Y, X, tau, 20.0, K, L)
        fns[(L, K)] = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0, None)))
    return fns[(L, K)]
out = []
rng = np.random.default_rng(0)
for L in (1, 2, 3, 4):
    for K in (1, 8):
        for tau in (1.0, 2.0):
            for sname, S in (("own", own), ("probe", probe)):
                t0 = time.time()
                f = get_fn(L, K)
                chunk = max(256 // (K + 1), 16)
                h, w = lay.hw[L]
                recs, flips, nfull, dcells, nchg = [], 0, 0, [], 0
                for i in range(0, N, chunk):
                    n = min(chunk, N - i)
                    wi = rng.integers(0, NW, n)
                    Y, X = rng.integers(0, h, n), rng.integers(0, w, n)
                    st = {k: v[wi] for k, v in S.items()}
                    keys = jax.random.split(jax.random.PRNGKey(1000 * L + 10 * K + i + int(tau * 7)), n)
                    new, rec = f(params, keys, st, jnp.asarray(Y, jnp.int32), jnp.asarray(X, jnp.int32), jnp.float32(tau))
                    rec = jax.device_get(rec)
                    s_ = 1 << L
                    ys_ = (Y[:, None] * s_ + np.arange(s_)[None, :])
                    xs_ = (X[:, None] * s_ + np.arange(s_)[None, :])
                    t0_ = np.asarray(st["t"])[np.arange(n)[:, None, None], ys_[:, :, None], xs_[:, None, :]]
                    t1_ = np.asarray(new["t"])[np.arange(n)[:, None, None], ys_[:, :, None], xs_[:, None, :]]
                    isfull = t0_.reshape(n, -1).all(1)
                    part1 = ~t1_.reshape(n, -1).all(1)
                    nfull += int(isfull.sum())
                    flips += int((isfull & rec["acc"] & part1).sum())
                    dcells.append(np.abs(t1_.reshape(n, -1).sum(1) - t0_.reshape(n, -1).sum(1))[rec["acc"]])
                    chg = np.zeros(n, bool)
                    for c in p.sampled():
                        a0, a1 = np.asarray(st[c.name]), np.asarray(new[c.name])
                        chg |= (a0 != a1).reshape(n, -1).any(1)
                    nchg += int(chg.sum())
                    recs.append(rec)
                rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
                acc = float(rec["acc"].mean())
                d = {"tag": tag, "L": L, "K": K, "tau": tau, "start": sname, "N": N, "acc": acc, "moved": nchg / N,
                     "acc_se": float(np.sqrt(acc * (1 - acc) / N)),
                     "mean_min1_r": float(np.exp(rec["log_r"]).mean()),
                     "median_log_r": float(np.median(rec["log_r"])),
                     "block_size": float(rec["size"].mean()),
                     "frac_new_viol": float((rec["nv_new"] > 0).mean()), "frac_old_viol": float((rec["nv_old"] > 0).mean()),
                     "median_dE": float(np.median(rec["dE"])), "median_dlogq": float(np.median(rec["dlogq"])),
                     "old_full": nfull, "full_to_partial_acc": flips, "mean_abs_ddirt_acc": float(np.concatenate(dcells).mean()) if np.concatenate(dcells).size else 0.0, "t_s": time.time() - t0}
                print(json.dumps(d), flush=True)
                out.append(d)
                json.dump(out, open(f"reportables/2026-10-14_kernel_b/runs/ground_{tag}.json", "w"))
