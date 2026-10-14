"""Per-kernel timing of one ckernel iteration (arm: default CK cfg, ground).  usage: profile_ck.py <n_iters> <outjson>"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants.ckernel import CKernel
from castlegen.kernels.logq import ml_opt_init
N, out = int(sys.argv[1]), sys.argv[2]
p = get("ground"); V = CKernel(p, 0, "/tmp/speed1b_ck")
c = V.cfg; region = Region(p.W_train); V._build(region); V.ctx_seed = 30_000_000
import os
if os.environ.get("OLD") == "1":           # the pre-change coarse kernels (full-image passes)
    import importlib.util
    def load(name, path):
        sp = importlib.util.spec_from_file_location("castlegen.kernels." + name, path); m = importlib.util.module_from_spec(sp)
        m.__package__ = "castlegen.kernels"; sys.modules[sp.name] = m; sp.loader.exec_module(m); return m
    bo = load("block_mh_old", "reportables/2026-10-14_speed1b/block_mh_old.py"); co = load("coarse_flip_old", "reportables/2026-10-14_speed1b/coarse_flip_old.py")
    lay = V.layout(region)
    V.mtm = bo.make_chain(lay, [2], 1.0, float(c["kernel_lam"]), c["K_l2"])
    V.flip = co.make_flip_chain(lay, list(c["flip_Ls"]), 1.0, c["tau_c"], c["flip_mode"], float(c["kernel_lam"]), c["K_flip"])
opt = ml_opt_init(V.params); on = jnp.ones(p.T + 1, jnp.float32)
n_own = c["nctx"] // 2; n_prb = c["nctx"] - n_own
def sync(x): jax.block_until_ready(x); return time.time()
def ctx():
    parts = [V._probe_ctx(n_prb, region), V._own_ctx(n_own, region)]
    return {k: jnp.concatenate([q[k] for q in parts]) for k in parts[0]}
key = jax.random.PRNGKey(5)
T = {k: 0.0 for k in ["ctx", "flip", "gibbs_coarse", "mtm_L2", "gibbs_fine", "ml", "total"]}
acc = {}
for it in range(N + 1):
    t0 = time.time(); st = ctx(); t1 = sync(st); key, sub = jax.random.split(key)
    kf, k4, k2, kg = jax.random.split(sub, 4); n = c["nctx"]
    st, rec = V.flip(V.params, jax.random.split(kf, n), st, c["m_flip"]); t2 = sync(st)
    recf = jax.device_get(rec)
    v = V._to_flat(st)
    for L, G in V.Gc.items():
        k4, s2 = jax.random.split(k4); v, ch = G.sweep(v, s2)
    t3 = sync(v); st = V._from_flat(v)
    st, rec2 = V.mtm(V.params, jax.random.split(k2, n), st, c["m_l2"]); t4 = sync(st)
    recm = jax.device_get(rec2)
    v = V._to_flat(st)
    for _ in range(c["n_fine"]):
        for J in (V.J20, V.J11):
            kg, s3 = jax.random.split(kg); v, ch = J.sweep(v, s3)
    t5 = sync(v); st = V._from_flat(v)
    for u in range(c["upi"]):
        idx = V.rng.choice(n, c["ml_B"], replace=False)
        V.params, opt, info = V.mlstep(V.params, opt, {k: x[idx] for k, x in st.items()}, jnp.ones(c["ml_B"]), jnp.float32(c["ml_lr"]), on)
    float(info["loss"]); t6 = time.time()
    if it == 0: print("warm", t6 - t0, flush=True); continue
    for k, d in zip(T, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0]): T[k] += d
    for L in (3, 4): m = recf["L"] == L; acc.setdefault(f"flip_L{L}", []).append((int(m.sum()), float(recf["acc"][m].mean()) if m.any() else 0, float(np.mean(recf["log_r"][m])) if m.any() else 0))
    acc.setdefault("mtm_L2", []).append((int(recm["acc"].size), float(recm["acc"].mean()), float(recm["log_r"].mean())))
res = {"per_iter_s": {k: v / N for k, v in T.items()}, "moves_per_iter": {"flip": c["m_flip"] * c["nctx"], "mtm_L2": c["m_l2"] * c["nctx"]},
       "acc": {k: [float(np.mean([a[1] for a in v])), float(np.mean([a[0] for a in v]))] for k, v in acc.items()}}
res["moves_per_s"] = {"flip": res["moves_per_iter"]["flip"] / res["per_iter_s"]["flip"], "mtm_L2": res["moves_per_iter"]["mtm_L2"] / res["per_iter_s"]["mtm_L2"]}
res["updates_per_s"] = c["upi"] / res["per_iter_s"]["total"]
print(json.dumps(res, indent=1)); json.dump(res, open(out, "w"), indent=1)
