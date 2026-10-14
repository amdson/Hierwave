"""Invariance test on an enumerable toy: B chains on a periodic 8 x 16 region
(8 iid 4 x 4 blocks), block-MH moves with q as proposal; per-block exact stats
vs problem.exact (z-scores from chain-to-chain variance).
usage: toy_chain.py problem q(untrained|bad|trained) Ls(e.g. 2 or 1,2) lam(20|inf)
       K tau start(own|exact) nseg segmoves [drop_q]"""
import sys, json, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_kernel_b/scripts")
from common import *
from castlegen.bench.problems import toys
from castlegen.kernels import block_mh as bm

name, qk, Ls, lam, K, tau, start, nseg, segm = sys.argv[1:10]
drop_q = False if len(sys.argv) <= 10 else (True if sys.argv[10] == "drop_q" else float(sys.argv[10].split("=")[1]))
Ls = [int(v) for v in Ls.split(",")]; lam = float(lam); K = int(K); tau = float(tau)
nseg, segm = int(nseg), int(segm)
p = get_problem(name); reg = Region(16); lay = core.Layout(p, 16, True, 0)
if qk == "trained":
    import pickle
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(
        "reportables/2026-10-13_clamp/params_toy_modes.pkl", "rb"))["trained"]["params"])
else:
    params = init_params(p, lay, 0, scale={"bad": 3.0, "mild": 1.0}.get(qk, 0.0))
B = 128
if start == "own":
    st = own_draws(p, lay, params, np.arange(1000, 1000 + B), reg)
else:
    E = toys.modes_block_energy if name == "toy_modes" else toys.support_block_energy
    pr = toys._probs(E(toys.all_blocks()))
    rng = np.random.default_rng(1)
    xs = [toys.lift(toys.sample_t(rng, pr, p.H, 16)) for _ in range(B)]
    st = {k: jnp.asarray(np.stack([x[k] for x in xs]), jnp.int32) for k in xs[0]}
stat_fn = toys.modes_block_stats if name == "toy_modes" else toys.support_block_stats
keys_ = list(p.exact)
chain = bm.make_chain(lay, Ls, tau, lam, K, drop_q)
key = jax.random.split(jax.random.PRNGKey(7), B)
traj, accs = [], {l: [] for l in Ls}
t0 = time.time()
def stats_of(st):
    t = np.asarray(st["t"])
    out = np.zeros((B, len(keys_)))
    for b in range(B):
        s = stat_fn(toys.blocks_of(t[b]))
        out[b] = [s[k].mean() for k in keys_]
    return out
traj.append(stats_of(st))
for s in range(nseg):
    st, rec = chain(params, key, st, segm)
    key = jax.vmap(lambda k: jax.random.fold_in(k, s + 1))(key)
    rec = jax.device_get(rec)
    for l in Ls:
        m = rec["L"] == l
        accs[l].append((rec["acc"][m].sum(), m.sum()))
    traj.append(stats_of(st))
traj = np.stack(traj)                                   # (nseg + 1, B, nstat)
burn = (nseg + 1) // 3
cm = traj[burn:].mean(0)                                # per-chain time means
est, se = cm.mean(0), cm.std(0, ddof=1) / np.sqrt(B)
ex = np.array([p.exact[k] for k in keys_])
z = (est - ex) / np.maximum(se, 1e-12)
acc = {l: float(sum(a for a, _ in v) / max(sum(n for _, n in v), 1)) for l, v in accs.items()}
res = {"problem": name, "q": qk, "Ls": Ls, "lam": lam, "K": K, "tau": tau, "start": start,
       "drop_q": drop_q, "moves_per_chain": nseg * segm, "chains": B, "acc": acc,
       "stats": {k: [float(est[i]), float(se[i]), float(ex[i]), float(z[i])] for i, k in enumerate(keys_)},
       "max_abs_z": float(np.abs(z).max()), "t_s": time.time() - t0,
       "initial": {k: float(traj[0, :, i].mean()) for i, k in enumerate(keys_)},
       "trace": {k: traj[:, :, i].mean(1).tolist() for i, k in enumerate(keys_)}}
print(json.dumps({k: v for k, v in res.items() if k != "trace"}))
tag = f"{name}_{qk}_L{'-'.join(map(str, Ls))}_lam{lam:g}_K{K}_tau{tau:g}_{start}" + ("" if drop_q is False else f"_qscale{0 if drop_q is True else drop_q:g}")
json.dump(res, open(f"reportables/2026-10-14_kernel_b/runs/toy_{tag}.json", "w"))
