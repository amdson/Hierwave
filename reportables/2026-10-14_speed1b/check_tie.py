import sys; sys.argv = ["x"]
exec(open("reportables/2026-10-14_speed1b/check_equiv.py").read().split("rng = np.random.default_rng(0)")[0])
rng = np.random.default_rng(1)
L, mode, tc = 4, "q", 2.0; N = 256
w = rng.integers(NS, size=N); Y = rng.integers(lay.hw[L][0], size=N); X = rng.integers(lay.hw[L][1], size=N)
s = {k: v[w] for k, v in st.items()}; keys = jax.random.split(jax.random.PRNGKey(99), N)
f = lambda mod: jax.jit(jax.vmap(lambda key, state, y, x: mod.flip_move(lay, params, key, state, jnp.int32(L), y, x, 1.0, tc, mode, 20.0, 4, 4)))
ra = jax.device_get(f(cf)(keys, s, jnp.asarray(Y), jnp.asarray(X))); rb = jax.device_get(f(co)(keys, s, jnp.asarray(Y), jnp.asarray(X)))
same = np.all([(np.asarray(ra[0][k]) == np.asarray(rb[0][k])).reshape(N, -1).all(1) for k in s], 0)
print("N", N, "identical", same.mean(), "differing moves (Y,X,acc_new,acc_old,dlogq):")
for i in np.where(~same)[0]: print(Y[i], X[i], bool(ra[1]["acc"][i]), bool(rb[1]["acc"][i]), float(ra[1]["dlogq"][i]), float(rb[1]["dlogq"][i]), float(ra[1]["log_r"][i]), float(rb[1]["log_r"][i]))
print("all log_r finite agree:", float(np.abs(np.nan_to_num(ra[1]["log_r"], neginf=-1e9) - np.nan_to_num(rb[1]["log_r"], neginf=-1e9)).max()))
