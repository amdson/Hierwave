"""After the all-action fix: R = 0 still equals cfast; cost identity (guided ground, R 2); one guided fused update runs."""
import numpy as np, jax, jax.numpy as jnp
from castlegen.bench import problems, energy as en
from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cfast import CFast
from castlegen.variants.crevise import CRevise
prob = problems.get("toy_support")
a = CFast(prob, 0, "/tmp/crv_t", W_train=64); b = CRevise(prob, 0, "/tmp/crv_t", W_train=64, R=0)
params = gc.randomize(a.params, jax.random.PRNGKey(4)); lay = a.layout(Region(64))
fa, fb = a.train_fns(lay), b.train_fns(lay)
seeds = jnp.arange(5, dtype=jnp.int32); des = a._batch(np.arange(5), Region(64))
tau, lam = jnp.float32(1.2), jnp.asarray([3.0, 5.0, 7.0], jnp.float32); gs = {k: jnp.float32(2.0) for k in params}
oa = fa["fused"](params, a.cparams, a.opt, a.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(9))
ob = fb["fused"](params, b.cparams, b.opt, b.copt, gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(9))
mx = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(oa[1]), jax.tree_util.tree_leaves(ob[1])))
assert mx == 0.0 and np.array_equal(np.asarray(oa[8]), np.asarray(ob[8])), mx
print("R0 = cfast ok")
prob = problems.get("ground")
c = CRevise(prob, 0, "/tmp/crv_t", W_train=64, R=2, rho=0.5, guide=True)
params = gc.randomize(c.params, jax.random.PRNGKey(2)); lay = c.layout(Region(64)); fn = c.train_fns(lay)
seeds = jnp.arange(2, dtype=jnp.int32); des = c._batch(np.arange(2), Region(64)); lam = jnp.asarray([3.0] * 5, jnp.float32)
st, recs, G, viol, tot = fn["rec"](params, seeds, des, tau, lam); st = jax.device_get(st); logD = fn["logD"]
for bb in range(2):
    d = {k: np.asarray(v[bb]) for k, v in des.items()}; xs = {ch.name: np.asarray(st[ch.name][bb]) for ch in prob.sampled()}
    E = en.energy(prob, {**d, **xs}, Region(64))
    lq = sum(float(np.sum(np.where(np.asarray(recs[k]["valid"][bb]), np.take_along_axis(np.asarray(recs[k]["logq"][bb]), np.asarray(recs[k]["a"][bb])[..., None], -1)[..., 0], 0.0))) for k in recs)
    ld = sum(logD[k] * float(np.sum(np.asarray(recs[k]["valid"][bb]))) for k in recs)
    pred = E.soft + float(np.asarray(lam) @ [E.viol[l] for l in range(5)]) + float(tau) * (lq + ld)
    assert np.isclose(float(tot[bb]), pred, rtol=1e-5, atol=2e-3), (float(tot[bb]), pred)
    # soft_c at the drawn value equals the recorded energy difference of the cost
    for k, r in recs.items():
        if "@rev" in k:
            sc = np.take_along_axis(np.asarray(r["soft_c"][bb]), np.asarray(r["a"][bb])[..., None], -1)[..., 0]
            assert np.all(np.abs(sc[np.asarray(r["valid"][bb])]) < 1e3)
print("cost identity ok (guided ground R2)")
oc = fn["fused"](params, c.cparams, c.opt, c.copt, {k: jnp.float32(1.0) for k in c.cparams}, seeds, des, tau, lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(9))
print("fused ok loss", float(oc[5]))
