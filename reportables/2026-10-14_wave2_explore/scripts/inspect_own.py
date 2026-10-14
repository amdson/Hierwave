"""What do own draws look like at the coarse levels?  rho_L vs the tile count of the
block, cert (A, B), fill-implied rho4.  usage: inspect_own.py [params.pkl]"""
import sys, json, pickle
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.problems.ground import cert_domain
from castlegen.variants.ckernel_own import CKernelOwn
p = get("ground"); reg = Region(p.W_train)
cfg = {"init": sys.argv[1]} if len(sys.argv) > 1 else {}
V = CKernelOwn(p, 0, "/tmp", **cfg)
lay = V.layout(reg)
from castlegen.variants._cfast_core import train_pass
lam_gen = jnp.asarray(V.lam, jnp.float32)
V.gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lam_gen, record=False)[0], in_axes=(None, 0, 0)))
V.ctx_seed = 31_000_000
st = {k: np.asarray(v) for k, v in V._own_ctx(16, reg).items()}
t = st["t"]; B = t.shape[0]
out = {"dirt": float(t.mean()), "stats": V._ground_stats(st)}
for L in (1, 2, 3, 4):
    h = 1 << L
    cnt = t.reshape(B, 64 // h, h, 128 // h, h).sum((2, 4))
    r = st[f"rho{L}"]
    A, Bv = cert_domain(h)
    a, b = A[st[f"cert{L}"]], Bv[st[f"cert{L}"]]
    out[L] = {"rho_mean": float(r.mean() / h**2), "count_mean": float(cnt.mean() / h**2),
              "abs_rho_minus_count": float(np.abs(r - cnt).mean() / h**2), "rho_full": float((r == h * h).mean()),
              "count_full": float((cnt == h * h).mean()), "A_mean": float(a.mean() / h), "B_mean": float(b.mean() / h),
              "A_full": float((a == h).mean()), "B_full": float((b == h).mean())}
f = st["fill"]
out["fill_mean"] = float(f.mean() / 16)
out["rho4_by_row"] = (st["rho4"].mean((0, 2)) / 256).round(3).tolist()
out["count4_by_row"] = (t.reshape(B, 4, 16, 8, 16).sum((2, 4)).mean((0, 2)) / 256).round(3).tolist()
out["fill_by_row"] = (f.mean((0, 2)) / 16).round(3).tolist()
out["A4_by_row"] = (cert_domain(16)[0][st["cert4"]].mean((0, 2)) / 16).round(3).tolist()
out["B4_by_row"] = (cert_domain(16)[1][st["cert4"]].mean((0, 2)) / 16).round(3).tolist()
print(json.dumps(out, indent=1))
