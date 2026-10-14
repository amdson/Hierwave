"""Fused RKL scan == sequential _rkl_steps, fused ML multi-update == sequential mlstep (same seeds, keys, minibatches).
usage: check_fast.py [n_rkl=4]"""
import sys, time, json
import numpy as np, jax, jax.numpy as jnp
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.variant import load_variants
from castlegen.kernels.logq import ml_opt_init
V_ = load_variants()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
p = get("ground")
V = V_["ckernel_rkl4w3f"](p, 0, "/tmp")
region = Region(p.W_train)
V._build(region)
V._rkl_setup(lambda d: None)
V.rkl_updates = 0
snap = dict(params=V.params, cparams=V.cparams, opt=V.opt, copt=V.copt, lam=V.lam.copy(), vbar=np.array(V.vbar, float).copy(),
            ts=V.train_seed)
# sequential (old path)
t = time.time(); hf_old = V._rkl_steps(n); jax.block_until_ready(V.params); t_old = time.time() - t
p_old, lam_old = V.params, V.lam.copy()
# fast path from the same snapshot
V.params, V.cparams, V.opt, V.copt, V.lam, V.vbar = snap["params"], snap["cparams"], snap["opt"], snap["copt"], snap["lam"], snap["vbar"]
seeds = np.arange(snap["ts"], snap["ts"] + n * V.cfg["B"]).reshape(n, V.cfg["B"])
ds = [p.designed_values(int(s), region) for s in seeds.ravel()]
des = {k: np.stack([d[k] for d in ds]).astype(np.int32).reshape((n, V.cfg["B"]) + ds[0][k].shape) for k in ds[0]}
hf_new = V._rkl_steps_fast(n, seeds, des)
t = time.time(); V._rkl_steps_fast(n, seeds, des, warm=True); t_new = time.time() - t
dp = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(p_old), jax.tree_util.tree_leaves(V.params)))
mp = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(p_old), jax.tree_util.tree_leaves(snap["params"])))
out = {"rkl_n": n, "max_param_diff_old_vs_fast": dp, "max_param_change_by_steps": mp,
       "lam_old": lam_old.tolist(), "lam_fast": V.lam.tolist(), "hf_old": np.asarray(hf_old).tolist(), "hf_fast": hf_new.tolist(),
       "t_old_s(incl compile? no: second call below)": t_old, "t_fast_warm_call_s": t_new}
# time the old path again (compiled)
V.params = snap["params"]; t = time.time(); V._rkl_steps(n); jax.block_until_ready(V.params); out["t_old_compiled_s"] = time.time() - t
# ML multi vs sequential
key = jax.random.PRNGKey(0)
V.ctx_seed = 123; st = V._own_ctx(32, region)
st2, _, v = V.refine(st, key)
opt = ml_opt_init(snap["params"]); on = jnp.ones(p.T + 1, jnp.float32)
rng = np.random.default_rng(1); U, Bm = 8, 8
idx = np.stack([rng.choice(32, Bm, replace=False) for _ in range(U)]); w = rng.uniform(0.5, 1.5, (U, Bm)).astype(np.float32)
pa, oa = snap["params"], opt
losses = []
for u in range(U):
    pa, oa, info = V.mlstep(pa, oa, {k: x[idx[u]] for k, x in st2.items()}, jnp.asarray(w[u]), jnp.float32(1e-3), on)
    losses.append(float(info["loss"]))
mm = V._ml_multi_build()
pb, ob, infos = mm(snap["params"], opt, st2, jnp.asarray(idx, jnp.int32), jnp.asarray(w), jnp.float32(1e-3), on)
out["ml_max_param_diff"] = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(pa), jax.tree_util.tree_leaves(pb)))
out["ml_max_param_change"] = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(pa), jax.tree_util.tree_leaves(snap["params"])))
out["ml_loss_seq"] = losses; out["ml_loss_multi"] = np.asarray(infos["loss"]).tolist()
print(json.dumps(out, indent=1))
