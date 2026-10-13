"""Split the exact reverse KL on the block toys into a coarse part (the level
1-2 counts) and a fine part (t given the counts).

python kl_split.py <toy_support|toy_modes> <train_s> '<json cfg>'

Trains ctemp in process, draws 64 worlds at tau = 1 with records, and for
every 4 x 4 block reads log q of its coarse draws (rho2, rho1) and of its 16
fine draws.  With r = the block's rho1 pattern,
  KL = E[log q(r) - log p(r)] + E[S + log q(t | r) + log Z(r)],
p(r) and Z(r) exact by enumerating the 2^16 block states.  Blocks are
treated as independent (true of p; q only approximately, its windows cross
block edges)."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from castlegen.bench.problem import Region  # noqa: E402
from castlegen.bench.problems import get  # noqa: E402
from castlegen.bench.problems import toys  # noqa: E402
from castlegen.variants import _cpass_core as core  # noqa: E402
from castlegen.variants.ctemp import CTemp  # noqa: E402

name, budget = sys.argv[1], float(sys.argv[2])
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
prob = get(name)
blkE = toys.support_block_energy if name == "toy_support" else toys.modes_block_energy
allb = toys.all_blocks()
E = blkE(allb)
ok = np.isfinite(E)
r1 = np.stack([allb[:, 0:2, 0:2].sum((1, 2)), allb[:, 0:2, 2:4].sum((1, 2)),
               allb[:, 2:4, 0:2].sum((1, 2)), allb[:, 2:4, 2:4].sum((1, 2))], 1)
rkey = r1 @ np.array([125, 25, 5, 1])
logZ = {}
for k in np.unique(rkey[ok]):
    e = E[ok & (rkey == k)]
    logZ[int(k)] = float(np.log(np.sum(np.exp(-(e - e.min())))) - e.min())
lZ = np.log(sum(np.exp(v) for v in logZ.values()))
logp_r = {k: v - lZ for k, v in logZ.items()}

v = CTemp(prob, 0, "/tmp", **cfg)
v.train(budget, lambda d: None)
lay = v.layout(Region(prob.W_train))
nb_y, nb_x = prob.H // 4, prob.W_train // 4
coarse = np.zeros((0, nb_y, nb_x))
fine = np.zeros((0, nb_y, nb_x))
keys, Ss = [], []
for rep in range(8):
    B = 8
    seeds = np.arange(900_000 + B * rep, 900_000 + B * rep + B)
    des = {}
    lam = jnp.asarray(v.lam, jnp.float32)
    extra = jnp.concatenate([v.ones(), lam]).astype(jnp.float32)
    st, tot, recs = lay.jit_rec(v.params, jnp.asarray(seeds, jnp.int32), des, v.ones(), lam, extra)
    lqc = np.zeros((B, nb_y, nb_x))
    lqf = np.zeros((B, nb_y, nb_x))
    for l in range(prob.T + 1):
        for c in lay.chans[l]:
            r = recs[core.key_of(l, c.name)]
            lq = np.take_along_axis(np.asarray(r["logq"]), np.asarray(r["a"])[..., None], -1)[..., 0]
            for k in range(core.NCOL):
                Y, X, val = (np.asarray(z) for z in lay.sites(l, k))
                by, bx = Y >> (2 - l), X >> (2 - l)
                for b in range(B):
                    tgt = lqf if l == 0 else lqc
                    np.add.at(tgt[b], (by[val], bx[val]), lq[b, k][val])
    t = np.asarray(st["t"])
    for b in range(B):
        blocks = toys.blocks_of(t[b]).reshape(nb_y, nb_x, 4, 4)
        idx = (blocks.reshape(nb_y, nb_x, 16) * (1 << np.arange(16))).sum(-1)
        keys.append(rkey[idx])
        Ss.append(E[idx])
    coarse = np.concatenate([coarse, lqc])
    fine = np.concatenate([fine, lqf])
keys, Ss = np.stack(keys), np.stack(Ss)
valid = np.isfinite(Ss)
lp = np.vectorize(lambda k: logp_r.get(int(k), -np.inf))(keys)
lz = np.vectorize(lambda k: logZ.get(int(k), np.nan))(keys)
kl_c = np.mean((coarse - lp)[valid])
kl_f = np.mean((Ss + fine + lz)[valid])
print(f"{name}: valid blocks {valid.mean():.4f}; KL per block coarse {kl_c:.3f} fine {kl_f:.3f} "
      f"total {kl_c + kl_f:.3f} (per cell {(kl_c + kl_f) / 16:.4f})")
# fine KL by context: most frequent rho1 patterns under q, with p(r) and q(r)
ks, cnt = np.unique(keys[valid], return_counts=True)
order = np.argsort(-cnt)[:12]
print("rho1 pattern (TL,TR,BL,BR) | q freq | p(r) | fine KL given r")
for i in order:
    k = ks[i]
    m = valid & (keys == k)
    pat = [(k // 125) % 5, (k // 25) % 5, (k // 5) % 5, k % 5]
    print(pat, f"{cnt[i] / valid.sum():.3f}", f"{np.exp(logp_r[int(k)]):.3f}", f"{np.mean((Ss + fine + lz)[m]):.3f}")
# mass of p on patterns q rarely visits
qfreq = {int(k): c / valid.sum() for k, c in zip(ks, cnt)}
pr = np.array([np.exp(v) for v in logp_r.values()])
qr = np.array([qfreq.get(k, 0.0) for k in logp_r])
print(f"p mass on patterns with q freq < p/2: {pr[qr < pr / 2].sum():.3f}; #patterns {len(pr)}")
