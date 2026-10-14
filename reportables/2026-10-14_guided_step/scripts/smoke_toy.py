import time, numpy as np, jax, jax.numpy as jnp
from castlegen.bench import problems, energy as en
from castlegen.bench.problem import Region
from castlegen.variants import _cpass_core as core
from castlegen.variants._guided_step import HardMaps, violation_field, guided_revise, select_cells
from castlegen.variants._jitter_repair import jitter_repair
prob = problems.get("toy_support"); W = 32
lay = core.Layout(prob, W, True)
key = jax.random.PRNGKey(0); params = {}
for l in range(prob.T + 1):
    for c in lay.chans[l]:
        key, k1 = jax.random.split(key)
        params[core.key_of(l, c.name)] = core.mlp_init(k1, lay.feat_dim(l, core.ACTOR_WIN), c.D, 16)
des = {k: jnp.asarray(v, jnp.int32) for k, v in prob.designed_values(0, Region(W)).items()}
lam = jnp.zeros(prob.T + 1)
hm = HardMaps(lay)
gen = jax.jit(lambda s: core.coloured_pass(lay, params, s, des, 1.0, lam)[0])
rng = np.random.default_rng(0)
# random states: check per_level == energy
for i in range(5):
    st = {c.name: jnp.asarray(rng.integers(0, c.D, lay.hw[c.level]), jnp.int32) for c in prob.sampled()}
    st = {**des, **st}
    E = en.energy(prob, {k: np.asarray(v) for k, v in st.items()}, Region(W))
    pl = np.asarray(hm.per_level(st))
    assert all(pl[l] == E.viol[l] for l in range(prob.T + 1)), (pl, E.viol)
    f = violation_field(hm, st)
    assert (sum(int(np.asarray(v).sum()) for v in f.values()) >= pl.sum())
print("maps == energy ok")
st = gen(3)
print("raw viol", np.asarray(hm.per_level(st)))
t0 = time.time()
rev = jax.jit(lambda st, s: guided_revise(lay, params, st, s, 4))
st2, info = rev(st, 3); print("guided R=4 compile+run", time.time() - t0, "viol before each step", np.asarray(info["viol_before"]), "final", np.asarray(hm.per_level(st2)), "changed", np.asarray(info["changed"]), "logq", np.asarray(info["log_q"]))
st3, _ = rev(st, 3); assert all(np.array_equal(np.asarray(st2[k]), np.asarray(st3[k])) for k in st2); print("deterministic ok")
jr = jax.jit(lambda st, s: jitter_repair(lay, st, s, 2))
t0 = time.time(); st4 = jr(st, 3); print("jitter R=2", time.time() - t0, np.asarray(hm.per_level(st4)), "repair-only", np.asarray(hm.per_level(jax.jit(lambda st, s: core.repair(lay, st, s))(st, 3))))
# invariant on random states
for i in range(5):
    s0 = gen(10 + i)
    a = int(hm.total(s0)); b = int(hm.total(jr(s0, 10 + i)))
    assert b <= a, (a, b)
    print("jitter never increases:", a, "->", b)
