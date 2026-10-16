"""crecur (amendment 2026-10-16 (b)): guard (1) on the enumerable two-level problem.
1. split_pass: K fine completions of one coarse seed share the coarse state; R_1's mean bias = exact KL(q_fine||p*).
2. Bias of R_K falls with K (prior fine model) and is 0 with exact fine conditionals.
3. The crecur coarse gradient at K = 1 is unbiased for the joint reverse-KL gradient of the coarse network."""
import jax
import numpy as np
import pytest

from castlegen.variants import _crecur_exact as X
from castlegen.variants import _gradcheck as gc


@pytest.fixture(scope="module")
def prior():
    p, var, lay, lam = X.setup(fine_scale=0.0)
    ex = X.exact_tables(lay, var.params, lam)
    return lay, var, lam, ex


def test_bias_falls_with_K(prior):
    lay, var, lam, ex = prior
    C, F = X.mc_fine_sums(lay, var.params, lam, 1500, 16)
    rows, _ = X.bias_table(ex, C, F)
    assert abs(rows[1]["bias_mean"] - rows["exact_KL_mean"]) < 4 * rows[1]["bias_se"] + 0.02, rows
    assert rows[1]["bias_mean"] > rows[4]["bias_mean"] > rows[16]["bias_mean"] > 0, rows


def test_exact_fine_zero_bias(prior):
    lay, var, lam, ex = prior
    r = X.exact_sampler_bias(ex, 100, 16, np.random.default_rng(0))
    assert all(abs(m) < 1e-6 and mx < 1e-6 for m, mx in r.values()), r


def test_K1_gradient_is_joint(prior):
    lay, var, lam, ex = prior
    gJ, gM, _ = X.exact_coarse_grads(lay, var.params, lam, ex)
    est = X.estimator_grads(lay, var.params, lam, 1, 200, 16, 700_000)
    z = gc.ztest(est, gc.flat(gJ), 16)
    assert abs(z["proj_z"]) < 4 and abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.02, z


def test_foot_K1_gradient_is_joint(prior):
    """mode foot (hybrid): at K = 1 the return is the cone return, so the gradient is the unbiased joint one."""
    lay, var, lam, ex = prior
    gJ, gM, _ = X.exact_coarse_grads(lay, var.params, lam, ex)
    est = X.estimator_grads(lay, var.params, lam, 1, 200, 16, 710_000, mode="foot")
    z = gc.ztest(est, gc.flat(gJ), 16)
    assert abs(z["proj_z"]) < 4 and abs(z["proj_ratio"] - 1) < 5 * z["rel_noise"] + 0.02, z


def test_foot_sums_partition(prior):
    """Footprint sums over all level-1 sites = the world's fine cost (the footprints partition the fine draws)."""
    import jax.numpy as jnp
    from castlegen.variants._crecur_core import foot_sums, split_pass
    lay, var, lam, ex = prior
    st, costs = jax.vmap(lambda s, f: split_pass(lay, var.params, s, f, {}, 1.0, jnp.asarray(lam), 1, record=False))(
        jnp.arange(8, dtype=jnp.int32), jnp.arange(100, 108, dtype=jnp.int32))
    fine = {k: v for k, v in costs.items() if k.startswith("l0_")}
    ft = foot_sums(lay, fine, 1)
    tot = sum(np.asarray(v).sum((1, 2)) for v in fine.values())
    assert np.allclose(np.asarray(ft).sum((1, 2)), tot, atol=1e-3)
