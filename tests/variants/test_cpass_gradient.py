"""Spec test 5 for cpass: on an enumerable two-level problem the mean of the
actor estimator's gradient equals the exact grad J, J = E_q[S + sum_l lam_l H_l
+ tau log q], computed by enumerating every reachable draw outcome.  Run with
random nonzero actor parameters, tau != 1, lambda > 0 and violations likely;
zero critic and a wrong fixed critic; branches enumerated (K = 4 >= |A|) and
sampled with the leave-one-out baseline (K = 2 < |A| = 3)."""
import jax
import numpy as np
import pytest

from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cpass import CPass

TAU, LAM = 1.5, np.array([2.0, 3.0])


@pytest.fixture(scope="module")
def setup():
    p = gc.tiny_problem()
    var = CPass(p, 0, "/tmp/gc_test", K=4)
    var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
    var.cparams = gc.zero_critic(var.cparams)
    ex = gc.Exact(var, LAM, reference=False, ref_check=200)
    J, g, info = ex.J_and_grad(var.params, TAU)
    return var, ex, J, g, info


def set_K(var, K):
    var.cfg["K"] = K
    lay = var.layout(Region(8))
    var._build_train(lay)
    if hasattr(lay, "jit_agrad"):
        del lay.jit_agrad


def test_enumeration(setup):
    var, ex, J, g, info = setup
    assert abs(info["sum_q"] - 1) < 1e-4
    assert info["p_viol"] > 0.1                      # violations are common: lambda matters
    # tree expansion of the masks finds exactly the outcomes forced scoring gives q > 0
    lq = np.asarray(ex.logq(var.params, TAU))
    assert np.isfinite(lq).all()


def test_exact_gradient_finite_difference(setup):
    var, ex, J, g, info = setup
    gf = gc.flat(g)
    u = jax.tree_util.tree_map(lambda x: x / np.linalg.norm(gf), g)
    eps = 1e-2
    Jp = ex.J_and_grad(jax.tree_util.tree_map(lambda a, b: a + eps * b, var.params, u), TAU)[0]
    Jm = ex.J_and_grad(jax.tree_util.tree_map(lambda a, b: a - eps * b, var.params, u), TAU)[0]
    assert abs((Jp - Jm) / (2 * eps) - np.linalg.norm(gf)) < 0.02 * np.linalg.norm(gf)


def _estimate(var, M, B, seed0, part="all"):
    est, cost = [], []
    for m in range(M):
        g, bi = gc.estimator_grad(var, np.arange(seed0 + m * B, seed0 + (m + 1) * B), TAU, LAM, part=part)
        est.append(gc.flat(g))
        cost.append(bi["cost"])
    return np.array(est), np.concatenate(cost)


@pytest.mark.parametrize("K,critic,part", [(4, "zero", "all"), (2, "zero", "all"), (2, "wrong", "all"),
                                           (2, "zero", "branch")])
def test_estimator_unbiased(setup, K, critic, part):
    var, ex, J, g, info = setup
    set_K(var, K)
    cz = var.cparams
    if critic == "wrong":
        var.cparams = gc.randomize(var.cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    try:
        est, cost = _estimate(var, 300, 16, 10_000 + 1000 * K)
    finally:
        var.cparams = cz
    # the draw costs sum to S + lam H + tau log q: their mean is J
    assert abs(cost.mean() - J) < 4 * cost.std() / np.sqrt(len(cost))
    gf = gc.flat(g)
    sl = gc.block_slices(g)
    blocks = sl if part == "all" else {k: s for k, s in sl.items() if k.startswith("l1_")}
    for k, s in blocks.items():
        z = gc.ztest(est[:, s], gf[s], 16)
        # per-parameter chi^2 is not used: an MLP's gradient coordinates are strongly
        # correlated (low rank), so mean z^2 scatters widely around 1 even when unbiased
        assert abs(z["proj_z"]) < 4 and abs(z["rand_z"]) < 4, (k, z)
        assert z["hotelling_p"] > 1e-4, (k, z)        # heavy tails: normal-theory p is approximate
        assert abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.01, (k, z)


def test_negative_control_detects_truncated_returns(setup):
    """The test has power: returns truncated to the next 4 steps (a biased
    estimator) are rejected."""
    var, ex, J, g, info = setup
    set_K(var, 4)
    est = np.array([gc.flat(gc.estimator_grad(var, np.arange(50_000 + 16 * m, 50_000 + 16 * (m + 1)), TAU, LAM,
                                              truncate=4)[0]) for m in range(300)])
    zs = {k: gc.ztest(est[:, s], gc.flat(g)[s], 16) for k, s in gc.block_slices(g).items()}
    assert any(z["hotelling_p"] < 1e-6 or abs(z["proj_z"]) > 6 for z in zs.values()), zs
