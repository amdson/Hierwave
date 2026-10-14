"""New (compact) vs old (full-image) block_mh / coarse_flip: same key -> same candidates, log ratios, decisions.
Also per-move timing of old vs new chains (make_chain / make_flip_chain)."""
import importlib.util, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels import block_mh as bm, coarse_flip as cf
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass
from castlegen.variants._cln_core import window_of
def load(name, path):
    sp = importlib.util.spec_from_file_location("castlegen.kernels." + name, path); m = importlib.util.module_from_spec(sp)
    m.__package__ = "castlegen.kernels"; sys.modules[sp.name] = m; sp.loader.exec_module(m); return m
bo = load("block_mh_old", "reportables/2026-10-14_speed1b/block_mh_old.py"); co = load("coarse_flip_old", "reportables/2026-10-14_speed1b/coarse_flip_old.py")
p = get("ground"); reg = Region(p.W_train)
ck = pickle.load(open("reportables/2026-10-14_sweep2a/runs/armB_s0/params_ground.pkl", "rb"))
lay = core.Layout(p, reg.W, True, 0, win=window_of((-2, 3)))
params = jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"]); lam_gen = jnp.asarray(ck["trained"]["lam"], jnp.float32)
NS = 8
ds = [p.designed_values(i, reg) for i in range(NS)]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
st = jax.jit(jax.vmap(lambda s, d: train_pass(lay, params, s, d, jnp.float32(1.0), lam_gen, record=False)[0]))(jnp.arange(NS, dtype=jnp.int32), dj)
rng = np.random.default_rng(0)
tot = dict(n=0, agree=0, maxdlr=0.0, maxdE=0.0, maxdq=0.0, state_eq=0)
for name, mk_new, mk_old, Ls, kw in [
    ("mtm_L2_K8", lambda L, Lm: bm.make_accept(lay, L, 1.0, 20.0, 8), lambda L, Lm: bo.make_accept(lay, L, 1.0, 20.0, 8), [2], {}),
    ("mh_L1_K1", lambda L, Lm: bm.make_accept(lay, L, 1.0, 20.0, 1), lambda L, Lm: bo.make_accept(lay, L, 1.0, 20.0, 1), [1], {}),
    ("mtm_L3_K4", lambda L, Lm: bm.make_accept(lay, L, 1.0, 20.0, 4), lambda L, Lm: bo.make_accept(lay, L, 1.0, 20.0, 4), [3], {})]:
    for L in Ls:
        N = 64
        w = rng.integers(NS, size=N); Y = rng.integers(lay.hw[L][0], size=N); X = rng.integers(lay.hw[L][1], size=N)
        s = {k: v[w] for k, v in st.items()}; keys = jax.random.split(jax.random.PRNGKey(L), N)
        a = mk_new(L, L)(params, keys, s, jnp.asarray(Y), jnp.asarray(X)); b = mk_old(L, L)(params, keys, s, jnp.asarray(Y), jnp.asarray(X))
        ra, rb = jax.device_get(a[1]), jax.device_get(b[1])
        d = lambda k: float(np.abs(np.nan_to_num(ra[k], neginf=-1e9) - np.nan_to_num(rb[k], neginf=-1e9)).max())
        eq = float(np.mean(np.all([(np.asarray(a[0][k]) == np.asarray(b[0][k])).reshape(N, -1).all(1) for k in s], 0)))   # fraction of moves with identical new states
        print(name, "decisions agree", float((ra["acc"] == rb["acc"]).mean()), "acc rate", float(ra["acc"].mean()), "max|dlog_r|", d("log_r"), "max|dE|", d("dE"), "max|dlogq|", d("dlogq"), "moves with identical new state", eq, flush=True)
# flip moves, modes uniform and q
for mode, tc in (("uniform", 1.0), ("q", 2.0)):
    for L in (3, 4):
        N = 64
        w = rng.integers(NS, size=N); Y = rng.integers(lay.hw[L][0], size=N); X = rng.integers(lay.hw[L][1], size=N)
        s = {k: v[w] for k, v in st.items()}; keys = jax.random.split(jax.random.PRNGKey(10 + L), N)
        f = lambda mod: jax.jit(jax.vmap(lambda key, state, y, x: mod.flip_move(lay, params, key, state, jnp.int32(L), y, x, 1.0, tc, mode, 20.0, 4, 4)))
        ra = jax.device_get(f(cf)(keys, s, jnp.asarray(Y), jnp.asarray(X))); rb = jax.device_get(f(co)(keys, s, jnp.asarray(Y), jnp.asarray(X)))
        d = lambda k: float(np.abs(np.nan_to_num(ra[1][k], neginf=-1e9) - np.nan_to_num(rb[1][k], neginf=-1e9)).max())
        eq = float(np.mean(np.all([(np.asarray(ra[0][k]) == np.asarray(rb[0][k])).reshape(N, -1).all(1) for k in s], 0)))
        print("flip", mode, "L", L, "decisions agree", float((ra[1]["acc"] == rb[1]["acc"]).mean()), "acc rate", float(ra[1]["acc"].mean()), "max|dlog_r|", d("log_r"), "max|dE|", d("dE"), "max|dlogq|", d("dlogq"), "moves with identical new state", eq, flush=True)
