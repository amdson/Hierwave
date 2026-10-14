"""Joint check: clamp ALL level-4 draws to (a) a fill-following probe's lift (jitter 0..3), (b) all dirt (t = 1 everywhere, lifted),
and (c) leave the model's own; cost of the levels below, split by term and level, per fine cell.  Same noise for all.
usage: ctop_joint.py snaps.pkl out.pkl NW"""
import pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems import ground
from castlegen.bench.variant import load_variants
from clamp_tools import FIELDS, clamped_pass_stats
snapf, outf, NW = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = get_problem("ground"); T = p.T; reg = Region(p.W_train)
lay = load_variants()["cfast"](p, seed=99, workdir="/tmp/ctop_j").layout(reg)
seeds = np.arange(4_000_000, 4_000_000 + NW)
dl = [p.designed_values(int(s), reg) for s in seeds]
des = {k: jnp.asarray(np.stack([d[k] for d in dl]), jnp.int32) for k in dl[0]}
pr = [p.probes(int(s), reg) for s in seeds]
sam = [c.name for c in p.sampled() if c.level >= T]
def clamp_of(get): return {c: jnp.asarray(np.stack([get(b)[c] for b in range(NW)]), jnp.int32) for c in sam}
dirt_state = ground.lift(np.ones((p.H, reg.W), np.int64))
cl = {"probe0": clamp_of(lambda b: pr[b][0]), "probe1": clamp_of(lambda b: pr[b][1]), "probe3": clamp_of(lambda b: pr[b][3]),
      "dirt": clamp_of(lambda b: dirt_state), "own": {}}
f = {}
res = {}
for name, sn in pickle.load(open(snapf, "rb")).items():
    params = jax.tree_util.tree_map(jnp.asarray, sn["params"]); lam = jnp.asarray(sn["lam"], jnp.float32)
    for tau in sorted({sn["tau"], 1.0, 2.0}):
        for cn, c in cl.items():
            key = (cn,)
            if key not in f:
                f[key] = jax.jit(jax.vmap(lambda pa, lm, ta, s, d, cc: clamped_pass_stats(lay, pa, s, d, ta, lm, cc)[1],
                                           in_axes=(None, None, None, 0, 0, 0)))
            acc = np.asarray(f[key](params, lam, jnp.float32(tau), jnp.asarray(seeds, jnp.int32), des, c))   # (NW, T+1, nf)
            res[(name, round(tau, 3), cn)] = {"acc": acc, "lam": sn["lam"], "is_native": tau == sn["tau"]}
        print(name, tau, flush=True)
pickle.dump(res, open(outf, "wb"))
