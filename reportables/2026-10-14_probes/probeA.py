"""Diagnostic A: why are toy_modes probes flagged?  Train cbig256 (or cfast) on toy_modes, then score probes."""
import sys, time, json, numpy as np
import jax
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problem import Region
from castlegen.bench import energy as en
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.problems import toys
from castlegen.bench.variant import load_variants
name, train_s, NBIG = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
pr = get_problem("toy_modes"); region = Region(pr.W_eval); cells = pr.H * pr.W_eval
v = load_variants()[name](pr, seed=0, workdir="/tmp/probesA")
t0 = time.time(); v.train(train_s, lambda d: None); print("trained", time.time() - t0, "updates", v.updates, flush=True)
Eb = toys.modes_block_energy(toys.all_blocks()); p = toys._probs(Eb)
with np.errstate(divide="ignore"): logp_blk = np.log(p)            # per-block exact log p over 2^16 states
def exact_logp(x):
    t = np.asarray(x["t"]); b = toys.blocks_of(t)
    idx = (b.reshape(len(b), 16) * (1 << np.arange(16))).sum(1)
    return float(logp_blk[idx].sum()), logp_blk[idx]
def stats_modes(x):
    st = toys.modes_block_stats(toys.blocks_of(np.asarray(x["t"])))
    return {k: float(st[k].mean()) for k in toys.MODES_KEYS}
# own draws
own = []   # (seed, S, logq, logp, valid, d)
mstat = []
for i in range(NBIG):
    seed = EVAL_SEED0 + 7000 + i
    g = v.generate(seed, region); x = _with_designed(pr, g["raw"], seed, region)
    E = en.energy(pr, x, region); q = v.score(x, region); valid = sum(E.viol.values()) == 0
    lp = exact_logp(x)[0] if valid else -np.inf
    own.append((E.soft, q, lp, valid)); mstat.append(stats_modes(x))
own = np.array(own, float)
# probes
prb = []; blk = []
for i in range(NBIG // 4):
    seed = EVAL_SEED0 + 9000 + i
    for j, y in enumerate(pr.probes(seed, region)):
        y = _with_designed(pr, y, seed, region); E = en.energy(pr, y, region)
        assert sum(E.viol.values()) == 0
        q = v.score(y, region); lp, lb = exact_logp(y)
        prb.append((seed, j, E.soft, q, lp)); blk.append(lb)
prb = np.array(prb, float)
np.savez("/tmp/probesA_%s.npz" % name, own=own, prb=prb, blk=np.array(blk))
print("done", flush=True)
