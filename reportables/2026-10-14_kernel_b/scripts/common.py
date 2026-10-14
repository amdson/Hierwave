"""Shared helpers for kernel_b scripts."""
import pickle
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass


def init_params(p, lay, seed=0, scale=0.0):
    """cfast's actor init; scale > 0 replaces the zero output layer with
    N(0, scale^2 / width) weights (a deliberately bad q)."""
    key = jax.random.PRNGKey(seed)
    out = {}
    for l in range(p.T + 1):
        for c in lay.chans[l]:
            key, k2, k3 = jax.random.split(key, 3)
            pr = core.mlp_init(k2, lay.feat_dim(l, lay.actor_win), c.D, 64)
            if scale > 0:
                pr["W3"] = jax.random.normal(k3, pr["W3"].shape) * scale / 8.0
                pr["b3"] = jax.random.normal(jax.random.fold_in(k3, 1), (c.D,)) * scale
            out[core.key_of(l, c.name)] = pr
    return out


def own_draws(p, lay, params, seeds, region):
    des = [p.designed_values(int(s), region) for s in seeds]
    lam = jnp.full(p.T + 1, 20.0)
    f = jax.jit(jax.vmap(lambda s, d: train_pass(lay, params, s, d, 1.0, lam, record=False)[0]))
    d = {k: jnp.asarray(np.stack([x[k] for x in des]), jnp.int32) for k in des[0]} if des[0] else {}
    if not d:
        st = f(jnp.asarray(seeds, jnp.int32), {k: jnp.zeros((len(seeds),), jnp.int32) for k in []})
    else:
        st = f(jnp.asarray(seeds, jnp.int32), d)
    st = {k: jnp.asarray(v, jnp.int32) for k, v in st.items()}
    return st
