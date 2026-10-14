"""Invariance of the coarse-flip kernel (and coarse single-cell Gibbs) on the toys:
B chains on a periodic 8 x 16 region (8 iid 4 x 4 blocks), exact stats vs problem.exact,
z from chain-to-chain variance (kernel-b's protocol).
usage: toy_flip.py problem q(untrained|bad|mild) kernel lam K tau tau_c start(own|exact) nseg segm
kernel: flip_uniform | flip_q | flip_uniform_dropq | flip_gibbs (flip_uniform L2 + exact Gibbs (2,0) every segment)"""
import sys, json, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_kernel_b/scripts")
from common import *
from castlegen.bench.problems import toys
from castlegen.kernels import coarse_flip as cf

name, qk, kern, lam, K, tau, tau_c, start, nseg, segm = sys.argv[1:11]
lam, K, tau, tau_c, nseg, segm = float(lam), int(K), float(tau), float(tau_c), int(nseg), int(segm)
p = get_problem(name); reg = Region(16); lay = core.Layout(p, 16, True, 0)
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
mode = "q" if kern == "flip_q" else "uniform"
chain = cf.make_flip_chain(lay, [2], tau, tau_c, mode, lam, K)
if kern.endswith("dropq"):      # deliberately wrong: drop the proposal density from the ratio (power check)
    import castlegen.kernels.coarse_flip as cfm
    orig = cfm.flip_pass
    def fp(*a, **k):
        r = orig(*a, **k)
        return r[0], 0.0 * r[1], r[2], r[3], r[4]
    cfm.flip_pass = fp
    chain = cfm.make_flip_chain(lay, [2], tau, tau_c, mode, lam, K)
G = None
if kern == "flip_gibbs":
    from castlegen.kernels.amortised import JaxBlockGibbs
    G = JaxBlockGibbs(p, reg, 2, 0, lam=None if np.isinf(lam) else lam)
    fl = G.K.flat
    def to_flat(s):
        return jnp.concatenate([s[c.name].reshape(B, -1) for c in p.chans], 1).astype(jnp.int32)
    def from_flat(v):
        return {c.name: v[:, fl.off[c.name]:fl.off[c.name] + int(np.prod(fl.shape[c.name]))].reshape((B,) + fl.shape[c.name]) for c in p.chans}
stat_fn = toys.modes_block_stats if name == "toy_modes" else toys.support_block_stats
keys_ = list(p.exact)
key = jax.random.split(jax.random.PRNGKey(7), B)
gkey = jax.random.PRNGKey(11)
traj, accs, gch = [], [], []
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
    accs.append((rec["acc"].sum(), rec["acc"].size))
    if G is not None:
        gkey, sub = jax.random.split(gkey)
        v, n = G.sweep(to_flat(st), sub)
        gch.append(float(np.asarray(n).mean()) / G.K.nb)
        st = from_flat(v)
    traj.append(stats_of(st))
traj = np.stack(traj)
burn = (nseg + 1) // 3
cm = traj[burn:].mean(0)
est, se = cm.mean(0), cm.std(0, ddof=1) / np.sqrt(B)
ex = np.array([p.exact[k] for k in keys_])
z = (est - ex) / np.maximum(se, 1e-12)
res = {"problem": name, "q": qk, "kernel": kern, "lam": lam, "K": K, "tau": tau, "tau_c": tau_c, "start": start,
       "moves_per_chain": nseg * segm, "chains": B, "acc": float(sum(a for a, _ in accs) / sum(n for _, n in accs)),
       "gibbs_changed_frac": float(np.mean(gch)) if gch else None,
       "stats": {k: [float(est[i]), float(se[i]), float(ex[i]), float(z[i])] for i, k in enumerate(keys_)},
       "max_abs_z": float(np.abs(z).max()), "t_s": time.time() - t0,
       "initial": {k: float(traj[0, :, i].mean()) for i, k in enumerate(keys_)}}
print(json.dumps(res))
tag = f"{name}_{qk}_{kern}_lam{lam:g}_K{K}_tau{tau:g}_tc{tau_c:g}_{start}"
json.dump(res, open(f"reportables/2026-10-14_sweep2b/runs/toy_{tag}.json", "w"))
