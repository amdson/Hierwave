"""explore (wave2-explore): tracked coarse flips record the flipped cell's old/new
value consistently; annealed multiple-try candidates (x0 ~ q, AIS with hard final)
give unbiased partition functions and self-normalised conditional means on the
enumerable toys (exact references from test_colour_gibbs.toy_exact)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.kernels import explore as ex
from tests.kernels.test_block_mh import _setup, _exact_states
from tests.kernels.test_colour_gibbs import exact_samples, toy_exact


def test_tracked_flip_records():
    p, lay, params = _setup("toy_support")
    st = _exact_states(p, 8)
    f = ex.make_flip_chain_tracked(lay, [2], 1.0, 1.0, "uniform", 20.0, 2)
    new, rec = f(params, jax.random.split(jax.random.PRNGKey(3), 8), st, 1)
    rec = jax.device_get(rec)
    acc = rec["acc"][:, 0]
    for b in range(8):
        same = all(np.array_equal(np.asarray(new[k][b]), np.asarray(st[k][b])) for k in st)
        if not acc[b]:
            assert same and rec["v_old"][b, 0] == rec["v_new"][b, 0]
    assert ((rec["v_old"] != rec["v_new"]) <= rec["acc"]).all()
    # multi-move chains run and keep values in the domain
    new, rec = f(params, jax.random.split(jax.random.PRNGKey(4), 8), st, 5)
    assert (np.asarray(rec["v_new"]) <= lay.chans[2][0].D - 1).all()


@pytest.mark.parametrize("name", ["toy_support", "toy_modes"])
def test_annealed_candidates_unbiased(name):
    p, lay, params = _setup(name)
    exh = toy_exact(name, 60.0, 1.0)                       # ~ hard target (exp(-60) per violation)
    B, K = 1000, 8
    st = exact_samples(exh, np.random.default_rng(0), B)   # outside: exact; block (0, 0) is replaced
    A = ex.make_annealed_candidates(lay, 2, K, tau_hi=4.0, n_steps=128, lam=5.0, lam_kind="const")
    cand, logw, valid, info = A(params, jax.random.PRNGKey(0), st, jnp.zeros((B, 2), jnp.int32))
    lw = np.asarray(logw, np.float64).ravel()
    fin = np.isfinite(lw)
    assert fin.mean() > 0.1, fin.mean()
    m = lw[fin].max()
    lme = lambda a: np.log(np.mean(np.where(np.isfinite(a), np.exp(a - m), 0.0))) + m
    rng = np.random.default_rng(1)
    se = np.std([lme(lw[i]) for i in rng.integers(0, lw.size, (200, lw.size))])
    assert abs(lme(lw) - exh["logZ"]) < max(4 * se, 0.03), (lme(lw), exh["logZ"], se)
    # self-normalised conditional mean of rho2 at the block (pooled over contexts: the block is
    # independent of the outside on the toys)
    r2 = np.asarray(cand["rho2"])[:, 0, 0].astype(float)
    w = np.where(fin, np.exp(lw - m), 0.0)
    est = (w * r2).sum() / w.sum()
    ess = w.sum() ** 2 / (w ** 2).sum()
    sd = np.sqrt(((w * (r2 - est) ** 2).sum() / w.sum()) / ess)
    assert abs(est - exh["exact"]["rho2_mean"]) < max(4 * sd, 0.05), (est, exh["exact"]["rho2_mean"], sd, ess)


@pytest.mark.parametrize("name", ["toy_support", "toy_modes"])
def test_tempered_transition_invariant(name):
    """Exact (penalised, lam = 2) samples stay exact under repeated tempered
    transitions on random level-2 subtrees; acceptance is not trivial."""
    from tests.kernels.test_colour_gibbs import _lay, _zscores
    lam = 2.0
    p, lay = _lay(name)
    exl = toy_exact(name, lam)
    rng = np.random.default_rng(5)
    B = 256
    st = exact_samples(exl, rng, B)
    f = ex.make_tempered_transition(lay, 2, tau_hi=2.0, n_steps=4, lam=lam)
    h, w = lay.hw[2]
    key = jax.random.PRNGKey(0)
    accs = []
    for i in range(30):
        key, sub = jax.random.split(key)
        cells = jnp.asarray(np.stack([rng.integers(h, size=B), rng.integers(w, size=B)], 1), jnp.int32)
        st, la, acc, _ = f(sub, st, cells)
        accs.append(float(np.asarray(acc).mean()))
    z = _zscores(exl, st)
    assert 0.05 < np.mean(accs) < 1.0, np.mean(accs)
    assert np.abs(z).max() < 4.5, dict(zip(exl["keys"], z))


@pytest.mark.parametrize("name", ["toy_support", "toy_modes"])
def test_tempered_transition_lambda_ladder_invariant(name):
    """Same with the violation price lowered along the ladder (lam 2 -> 0.5)."""
    from tests.kernels.test_colour_gibbs import _lay, _zscores
    lam = 2.0
    p, lay = _lay(name)
    exl = toy_exact(name, lam)
    rng = np.random.default_rng(6)
    B = 256
    st = exact_samples(exl, rng, B)
    f = ex.make_tempered_transition(lay, 2, tau_hi=2.0, n_steps=4, lam=lam, lam_hi=0.5)
    h, w = lay.hw[2]
    key = jax.random.PRNGKey(1)
    accs = []
    for i in range(30):
        key, sub = jax.random.split(key)
        cells = jnp.asarray(np.stack([rng.integers(h, size=B), rng.integers(w, size=B)], 1), jnp.int32)
        st, la, acc, _ = f(sub, st, cells)
        accs.append(float(np.asarray(acc).mean()))
    z = _zscores(exl, st)
    assert 0.02 < np.mean(accs) < 1.0, np.mean(accs)
    assert np.abs(z).max() < 4.5, dict(zip(exl["keys"], z))
