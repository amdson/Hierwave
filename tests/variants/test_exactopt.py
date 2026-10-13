"""exactopt (ideas.md #10): the context-table evaluation of log q matches the
forced pass, q sums to 1, the exact gradient matches finite differences, and
the full-history table can reach the target (KL -> 0) on the tiny problem."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.variants import _cpass_core as core
from castlegen.variants import _exactopt as eo
from castlegen.variants import _gradcheck as gc

TAU = 1.5


@pytest.fixture(scope="module")
def setup():
    problem, region = gc.tiny_problem(D1=2), eo.Region(8)
    var = eo.make_variant("cpass", problem, region, 0)
    lam = np.full(problem.T + 1, 3.0)
    enum = eo.Enum(problem, region, var.params, lam, verbose=False)
    params = gc.randomize(var.params, jax.random.PRNGKey(3))
    return problem, var, enum, params


def test_logq_matches_forced_pass(setup):
    problem, var, enum, params = setup
    lay = enum.lay
    lam = jnp.asarray(enum.lam, jnp.float32)
    f = jax.jit(jax.vmap(lambda x: core.coloured_pass(lay, params, 0, {}, TAU, lam, forced=x)[2]))
    ref = np.asarray(f({k: jnp.asarray(v) for k, v in enum.X.items()}))
    lq = np.asarray(enum.fns("mlp")[0](params, jnp.float32(TAU)))
    assert np.abs(lq - ref).max() < 1e-4


@pytest.mark.parametrize("kind", ["mlp", "table_feat", "table_hist"])
def test_normalised(setup, kind):
    problem, var, enum, params = setup
    p = params if kind == "mlp" else jax.tree_util.tree_map(
        lambda x: jax.random.normal(jax.random.PRNGKey(x.size), x.shape), enum.init_params(kind))
    lq = np.asarray(enum.fns(kind)[0](p, jnp.float32(TAU)), np.float64)
    assert abs(np.exp(lq).sum() - 1) < 1e-4


def test_gradient_finite_difference(setup):
    problem, var, enum, params = setup
    logq, jg = enum.fns("mlp")
    J = lambda p: enum.J(logq(p, jnp.float32(TAU)), TAU)[0]
    _, g = jg(params, jnp.float32(TAU))
    n = np.linalg.norm(gc.flat(g))
    u = jax.tree_util.tree_map(lambda x: x / n, g)
    eps = 1e-2
    fd = (J(jax.tree_util.tree_map(lambda a, b: a + eps * b, params, u))
          - J(jax.tree_util.tree_map(lambda a, b: a - eps * b, params, u))) / (2 * eps)
    assert abs(fd - n) < 0.03 * n, (fd, n)


def test_full_history_table_reaches_target(setup):
    """With one free logit vector per full history the order can express any
    distribution on the reachable set: exact training drives KL to ~0."""
    problem, var, enum, params = setup
    p = enum.init_params("table_hist")
    _, curve = eo.train_exact(enum, "table_hist", p, 2000, 1e-1, enum.lam, 2000, None, tau0=1.0, log=lambda s: None)
    assert curve[0]["kl"] > 0.1 and curve[-1]["kl"] < 0.01, (curve[0]["kl"], curve[-1]["kl"])
