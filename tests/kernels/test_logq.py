"""logq_of_state (teacher forcing) equals the log q recorded when the pass generated the state."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.kernels import logq as lq
from castlegen.variants import _cpass_core as core
from castlegen.variants import _exactopt as eo
from castlegen.variants import _gradcheck as gc
from castlegen.variants._cfast_core import train_pass


@pytest.fixture(scope="module", params=["tanh", "ln"])
def setup(request):
    problem, region = eo.get_problem("support")
    var = eo.make_variant("cfast", problem, region, 0, net=request.param)
    lay = var.layout(region)
    params = gc.randomize(var.params, jax.random.PRNGKey(3))
    return var, lay, params


def _generate(lay, params, seeds, tau):
    lam = jnp.full(lay.T + 1, 10.0)
    def one(seed):
        st, recs = train_pass(lay, params, seed, {}, tau, lam, record=True)
        tot = 0.0
        for r in recs.values():
            idx = jnp.arange(r["a"].shape[-1])
            la = jnp.take_along_axis(r["logq"], r["a"][..., None], -1)[..., 0]
            tot = tot + jnp.sum(jnp.where(r["valid"], la, 0.0))
        return st, tot
    return jax.vmap(one)(jnp.asarray(seeds, jnp.int32))


@pytest.mark.parametrize("tau", [1.0, 1.7])
def test_matches_generation(setup, tau):
    var, lay, params = setup
    st, ref = _generate(lay, params, np.arange(40), tau)
    r = lq.logq_batch(lay, params, st, tau=tau)
    assert bool(r["feasible"].all())
    assert np.abs(np.asarray(r["total"]) - np.asarray(ref)).max() < 1e-5
    # per-level breakdown adds up; per-draw too
    assert np.abs(np.asarray(r["per_level"]).sum(1) - np.asarray(r["total"])).max() < 1e-4
    pd = sum(np.asarray(v).sum((1, 2)) for v in r["per_draw"].values())
    assert np.abs(pd - np.asarray(r["total"])).max() < 1e-4


def test_infeasible_and_maskfree(setup):
    var, lay, params = setup
    st, _ = _generate(lay, params, np.arange(4), 1.0)
    s0 = {k: np.array(v[0]) for k, v in st.items()}
    # find a mask-excluded value: scan values of the level-0 channel until infeasible
    found = False
    for v in range(2):
        for y in range(4):
            for x in range(4):
                s = {k: a.copy() for k, a in s0.items()}
                s["t"][y, x] = v
                # keep coarse counts consistent? not needed: counts make a violation => infeasible
                r = lq.logq_of_state(lay, params, s)
                if not bool(r["feasible"]):
                    assert int(r["n_infeasible"]) >= 1 and np.isneginf(float(r["total"]))
                    rm = lq.logq_of_state(lay, params, s, maskfree=True)
                    assert np.isfinite(float(rm["total"]))
                    found = True
    assert found


def test_grad_and_level_mask(setup):
    var, lay, params = setup
    st, _ = _generate(lay, params, np.arange(16), 1.0)
    step = lq.make_ml_update(lay)
    opt = lq.ml_opt_init(params)
    w = np.ones(16)
    on = np.array([1.0, 0.0, 0.0])
    p2, o2, info = lq.ml_update(params, opt, st, w, step, 1e-3, on)
    assert np.isfinite(float(info["loss"]))
    for k in params:
        same = all(bool(jnp.all(a == b)) for a, b in zip(jax.tree_util.tree_leaves(p2[k]), jax.tree_util.tree_leaves(params[k])))
        assert same == (not k.startswith("l0_")), k
    p3, _, info2 = lq.ml_update(p2, o2, st, w, step, 1e-3, on)
    assert float(info2["loss"]) <= float(info["loss"]) + 1e-3
