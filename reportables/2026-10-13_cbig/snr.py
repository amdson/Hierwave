"""Per-batch actor-gradient SNR vs B for cbig at init (zero critic, tau 1.5, lam 10).
SNR_B(level) = (|mean|^2 - trCov/M) / trCov  (signal |E g|^2 over noise tr Cov of a batch gradient)."""
import sys, json, time, tempfile
import numpy as np, jax, jax.numpy as jnp
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
import castlegen.variants.cbig as cb
from castlegen.variants import _gradcheck as gc
prob, B, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = get(prob); v = cb.CBig(p, 0, tempfile.mkdtemp(), B=B)
region = Region(p.W_train); lay = v.layout(region); fn = v.train_fns(lay)
cp = gc.zero_critic(v.cparams)
gs = {k: jnp.float32(1.0) for k in v.params}
lam = jnp.full(p.T + 1, 10.0, jnp.float32); tau = jnp.float32(1.5)
keys = sorted(v.params); sl = gc.block_slices(v.params, keys)
est = []; t = time.time()
for m in range(M):
    seeds = np.arange(7_000_000 + m * B, 7_000_000 + (m + 1) * B)
    g = fn["grad"](v.params, cp, gs, jnp.asarray(seeds, jnp.int32), v._batch(seeds, region), tau, lam, jax.random.PRNGKey(m))
    est.append(gc.flat(g, keys))
    if m == 0: print("first batch", round(time.time() - t, 1), flush=True)
est = np.array(est); out = {"problem": prob, "B": B, "M": M, "secs": round(time.time() - t, 1)}
for k, s in sl.items():
    e = est[:, s]; mu = e.mean(0); trc = e.var(0, ddof=1).sum(); sig = mu @ mu - trc / M
    out[k] = {"snr": float(sig / trc), "noise": float(trc), "signal": float(sig)}
print("SNR", json.dumps(out), flush=True)
