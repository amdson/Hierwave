import pickle, time, sys, jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.amortised import make_ml_update_drawwise
from castlegen.kernels.logq import ml_opt_init, logq_of_state
from castlegen.variants import _cpass_core as core
from castlegen.variants._cln_core import window_of
p = get("ground"); reg = Region(p.W_train)
ck = pickle.load(open("reportables/2026-10-14_cstart2/runs/cstart_f1L3_ln6/params_ground.pkl", "rb"))
lay = core.Layout(p, reg.W, True, 0, win=window_of((-2, 3)))
params = jax.tree_util.tree_map(jnp.asarray, ck["untrained"]["params"]); opt = ml_opt_init(params)
d = np.load("reportables/2026-10-14_kernel_a/runs/data_a.npz") if False else None
from castlegen.variants._cfast_core import train_pass
lam = jnp.asarray(ck["trained"]["lam"], jnp.float32)
ss = jnp.arange(16, dtype=jnp.int32)
des = [p.designed_values(int(s), reg) for s in range(16)]
dj = {k: jnp.asarray(np.stack([x[k] for x in des]), jnp.int32) for k in des[0]}
st = jax.jit(jax.vmap(lambda s, dd: train_pass(lay, params, s, dd, jnp.float32(1.0), lam, record=False)[0]))(ss, dj)
on = jnp.asarray([1, 1, 1, 0, 0], jnp.float32)
def tm(f, n=5):
    r = f(); jax.block_until_ready(r); ts = []
    for _ in range(12):
        t = time.time(); r = f(); jax.block_until_ready(r); ts.append(time.time() - t)
    return min(ts) * 1000
for B in (8, 16):
    s = {k: v[:B] for k, v in st.items()}
    for name, sl in (("all", None), ("fine", (0, 1, 2))):
        step = make_ml_update_drawwise(lay, score_levels=sl)
        fwd = jax.jit(lambda pr, s: jax.vmap(lambda x: logq_of_state(lay, pr, x, score_levels=sl)["total_finite"])(s))
        print(f"B={B} {name}: step {tm(lambda: step(params, opt, s, jnp.ones(B), jnp.float32(1e-3), on)):.1f} ms  fwd {tm(lambda: fwd(params, s)):.1f} ms", flush=True)
# equivalence: new pass vs reference pass (values and gradients)
from castlegen.kernels import logq as L
s = {k: v[:8] for k, v in st.items()}
for sl in (None, (0, 1, 2)):
    f_new = lambda pr: jax.vmap(lambda x: L.logq_of_state(lay, pr, x, score_levels=sl)["total_finite"])(s).sum()
    def f_ref(pr):
        def one(x):
            lq, ok = L._forced_pass_ref(lay, pr, {k: jnp.asarray(v) for k, v in x.items()}, 1.0, False, sl)
            return sum(v.sum() for v in lq.values())
        return jax.vmap(one)(s).sum()
    (a, ga), (b, gb) = jax.value_and_grad(f_new)(params), jax.value_and_grad(f_ref)(params)
    md = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb)))
    mg = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(gb))
    print("score_levels", sl, "value", float(a), float(b), "max grad diff", md, "max |grad|", mg)
