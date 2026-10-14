"""score_levels (per-level teacher forcing) leaves gradients of the scored levels unchanged;
make_multi_update equals sequential updates."""
import jax
import jax.numpy as jnp
import numpy as np

from castlegen.kernels import logq as lq
from castlegen.kernels.amortised import make_ml_update_drawwise, make_multi_update
from castlegen.variants import _exactopt as eo
from castlegen.variants import _gradcheck as gc
from castlegen.variants._cfast_core import train_pass


def _setup():
    problem, region = eo.get_problem("support")
    var = eo.make_variant("cfast", problem, region, 0, net="ln")
    lay = var.layout(region)
    params = gc.randomize(var.params, jax.random.PRNGKey(3))
    lam = jnp.full(lay.T + 1, 10.0)
    st = jax.vmap(lambda s: train_pass(lay, params, s, {}, jnp.float32(1.0), lam, record=False)[0])(jnp.arange(6, dtype=jnp.int32))
    return lay, params, st


def test_score_levels_gradients():
    lay, params, st = _setup()
    keep = tuple(range(lay.T))                     # drop the top level
    def loss(sl):
        def f(pr):
            r = jax.vmap(lambda s: lq.logq_of_state(lay, pr, s, score_levels=sl))(st)
            return -r["total_finite"].sum()
        return jax.grad(f)(params)
    g_full, g_part = loss(None), loss(keep)
    for k in params:
        lev = int(k[1:k.index("_")])
        if lev in keep:
            for a, b in zip(jax.tree_util.tree_leaves(g_full[k]), jax.tree_util.tree_leaves(g_part[k])):
                assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-4, rtol=1e-4)


def test_multi_update_equals_sequential():
    lay, params, st = _setup()
    step = make_ml_update_drawwise(lay)
    multi = make_multi_update(step)
    on = jnp.ones(lay.T + 1)
    opt = lq.ml_opt_init(params)
    idx = np.array([[0, 1, 2], [3, 4, 5], [1, 3, 5]], np.int32)
    p1, o1 = params, opt
    for ix in idx:
        p1, o1, _ = step(p1, o1, {k: v[ix] for k, v in st.items()}, jnp.ones(3), jnp.float32(1e-3), on)
    p2, o2, infos = multi(params, opt, st, jnp.asarray(idx), jnp.ones(3), jnp.float32(1e-3), on)
    for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2)):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    assert infos["loss"].shape == (3,)
