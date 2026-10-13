"""Does the pass sample the q that forced scoring assigns?  Mean cost vs J and
the coarse-level marginal vs its exact value, over many worlds."""
import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cpass import CPass

TAU, LAM = 1.5, np.array([2.0, 3.0])
p = gc.tiny_problem()
var = CPass(p, 0, "/tmp/gcw", K=4)
var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
var.cparams = gc.zero_critic(var.cparams)
ex = gc.Exact(var, LAM, cache="/tmp/gradcheck_tiny_exact.npz")
lq = np.asarray(ex.logq(var.params, TAU), np.float64)
q = np.exp(lq)
J = q @ (ex.E + TAU * lq)
lay = var.layout(Region(8))
var._build_train(lay)
tau_, lam_ = jnp.float32(TAU), jnp.asarray(LAM, jnp.float32)
extra = jnp.concatenate([jnp.array([TAU]), lam_]).astype(jnp.float32)
cidx = (ex.X["c"].reshape(len(q), -1) * np.array([27, 9, 3, 1])).sum(1)
pc = np.bincount(cidx, q, 81)
cost, cnt, lqs = [], np.zeros(81), []
N, Bn = 0, 512
for i in range(400):
    seeds = jnp.asarray(np.arange(i * Bn, (i + 1) * Bn) + 7_000_000, jnp.int32)
    st, tot, recs = lay.jit_rec(var.params, seeds, {}, tau_, lam_, extra)
    cost.append(np.asarray(tot).sum(-1))
    c = np.asarray(st["c"]).reshape(Bn, -1)
    cnt += np.bincount((c * np.array([27, 9, 3, 1])).sum(1), minlength=81)
    N += Bn
cost = np.concatenate(cost)
print(f"N {N}: mean cost {cost.mean():.4f} vs J {J:.4f}, z {(cost.mean() - J) / (cost.std() / np.sqrt(N)):.2f}")
ok = pc > 0
e = N * pc
chi = ((cnt - e) ** 2 / np.maximum(e, 1e-12))[ok].sum()
print(f"coarse marginal: chi2 {chi:.1f} on {ok.sum() - 1} dof; mass off support {cnt[~ok].sum()}")
top = np.argsort(-np.abs(cnt - e) / np.sqrt(np.maximum(e, 1)))[:5]
for t in top:
    print(t, cnt[t], round(e[t], 1), round((cnt[t] - e[t]) / np.sqrt(max(e[t], 1)), 2))
