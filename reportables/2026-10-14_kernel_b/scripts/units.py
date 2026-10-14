"""Unit checks of block_mh on a toy: (1) block energy delta == energy.py delta,
(2) log q of a draw == teacher-forced log q, (3) empirical draw frequencies
match exp(log q) (L = 1 cell, 20000 draws)."""
import sys, json
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_kernel_b/scripts")
from common import *
from castlegen.bench import energy as en
from castlegen.kernels import block_mh as bm

name = sys.argv[1] if len(sys.argv) > 1 else "toy_modes"
p = get_problem(name); reg = Region(16)
lay = core.Layout(p, 16, True, 0)
params = init_params(p, lay, 0, scale=3.0)
st = own_draws(p, lay, params, np.arange(16), reg)
LAM = 20.0
def Etot(x):
    E = en.energy(p, {k: np.asarray(v) for k, v in x.items()}, reg)
    return E.soft + LAM * sum(E.viol.values())
fp = jax.jit(lambda key, s, L, Y, X: bm.block_pass(lay, params, key, s, bm.block_masks(lay, L, Y, X), 1.0, LAM, p.T))
ff = jax.jit(lambda key, s, f, L, Y, X: bm.block_pass(lay, params, key, s, bm.block_masks(lay, L, Y, X), 1.0, LAM, p.T, forced=f))
rng = np.random.default_rng(0); errs, lqerr = [], []
for i in range(40):
    b = i % 16; L = int(rng.integers(0, p.T + 1)); h, w = lay.hw[L]
    Y, X = int(rng.integers(h)), int(rng.integers(w))
    x = {k: v[b] for k, v in st.items()}
    y, lqy, esy, nvy = fp(jax.random.PRNGKey(i), x, L, Y, X)
    _, lqx, esx, nvx = ff(jax.random.PRNGKey(0), x, x, L, Y, X)
    _, lqy2, esy2, nvy2 = ff(jax.random.PRNGKey(0), x, y, L, Y, X)
    d_pass = float(esy + LAM * nvy - esx - LAM * nvx)
    d_ref = Etot(y) - Etot(x)
    errs.append(abs(d_pass - d_ref)); lqerr.append(abs(float(lqy - lqy2)) + abs(float(esy - esy2)))
print(json.dumps({"problem": name, "max_dE_err": max(errs), "max_logq_err": max(lqerr)}))
# (3) empirical frequencies of one L=1 block
x = {k: v[0] for k, v in st.items()}
L, Y, X = 1, 1, 2
N = 20000
ys = jax.jit(jax.vmap(lambda k: fp(k, x, L, Y, X)))(jax.random.split(jax.random.PRNGKey(5), N))
inb = {k: np.asarray(v) for k, v in bm.block_masks(lay, L, Y, X).items()}
cfg = np.stack([np.concatenate([np.asarray(ys[0][k])[n][inb[k]] for k in sorted(inb)]) for n in range(N)])
keys, cnt = np.unique(cfg, axis=0, return_counts=True)
lq = np.asarray(ys[1]); _, first = np.unique(cfg, axis=0, return_index=True)
pq = np.exp(lq[first])
tv = 0.5 * np.abs(cnt / N - pq).sum() + 0.5 * (1 - pq.sum())
chi = float(((cnt - N * pq) ** 2 / (N * pq)).sum())
print(json.dumps({"n_configs": int(len(keys)), "sum_q_seen": float(pq.sum()), "TV": float(tv),
                  "chi2": chi, "dof": int(len(keys) - 1)}))
