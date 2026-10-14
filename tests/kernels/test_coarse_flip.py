"""coarse_flip: the flip proposal's log density equals its teacher-forced value,
the block energy delta matches energy.py, mode "q" with tau_c = tau is exactly
block_mh's move, and the exact single-cell coarse Gibbs conditional at L = 3, 4
on ground matches full-energy differences.  (Toy invariance chains:
reportables/2026-10-14_sweep2b/scripts/toy_flip.py.)"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.kernels import block_mh as bm
from castlegen.kernels import coarse_flip as cf
from castlegen.kernels.exact_block import total_energy
from tests.kernels.test_block_mh import _setup, _exact_states

LAM = 20.0


@pytest.mark.parametrize("mode,tau_c", [("uniform", 1.0), ("q", 2.0)])
def test_flip_density_and_energy(mode, tau_c):
    p, lay, params = _setup("toy_modes")
    reg = Region(16)
    st = _exact_states(p, 4)
    inb = lambda Y, X: bm.block_masks(lay, 2, Y, X)
    fp = jax.jit(lambda k, s, Y, X: cf.flip_pass(lay, params, k, s, inb(Y, X), 2, 1.0, tau_c, mode, p.T))
    ff = jax.jit(lambda k, s, f, Y, X: cf.flip_pass(lay, params, k, s, inb(Y, X), 2, 1.0, tau_c, mode, p.T, forced=f))

    def etot(x):
        E = en.energy(p, {k: np.asarray(v) for k, v in x.items()}, reg)
        return E.soft + LAM * sum(E.viol.values())
    rng = np.random.default_rng(0)
    for i in range(6):
        h, w = lay.hw[2]
        Y, X = int(rng.integers(h)), int(rng.integers(w))
        x = {k: v[i % 4] for k, v in st.items()}
        y, lqy, esy, nvy, lfy = fp(jax.random.PRNGKey(i), x, Y, X)
        _, lqx, esx, nvx, _ = ff(jax.random.PRNGKey(0), x, x, Y, X)
        _, lqy2, _, _, lfy2 = ff(jax.random.PRNGKey(0), x, y, Y, X)
        assert abs(float(esy + LAM * nvy - esx - LAM * nvx) - (etot(y) - etot(x))) < 1e-3
        assert abs(float(lqy - lqy2)) < 1e-4 and abs(float(lfy - lfy2)) < 1e-5
        assert np.isfinite(float(lqx))


def test_q_mode_equals_block_mh():
    p, lay, params = _setup("toy_support")
    st = _exact_states(p, 3)
    for i in range(3):
        x = {k: v[i] for k, v in st.items()}
        k = jax.random.PRNGKey(10 + i)
        _, r1 = bm.mh_move(lay, params, k, x, jnp.int32(2), 1, 2, 1.3, LAM, 4)
        _, r2 = cf.flip_move(lay, params, k, x, jnp.int32(2), 1, 2, 1.3, 1.3, "q", LAM, 4)
        assert abs(float(r1["log_r"]) - float(r2["log_r"])) < 1e-4
        assert bool(r1["acc"]) == bool(r2["acc"])


@pytest.mark.parametrize("L", [3, 4])
def test_coarse_gibbs_conditional_ground(L):
    p = get_problem("ground")
    reg = Region(p.W_train)
    G = cf.coarse_gibbs(p, reg, L, lam=LAM, backend="numpy")
    fl = G.flat
    x = {**p.probes(5, reg)[2], **p.designed_values(5, reg)}
    v = fl.flatten(x)
    k = 0
    blocks, _ = G.cls[k]
    soft, nv = G.energies(v, k)
    rng = np.random.default_rng(L)
    b = 0
    for _ in range(4):
        c1, c2 = rng.integers(G.comps.shape[0], size=2)
        v1, v2 = v.copy(), v.copy()
        v1[G.BV[blocks[b]]] = G.comps[c1]
        v2[G.BV[blocks[b]]] = G.comps[c2]
        s1, n1 = total_energy(fl, G.insts, v1)
        s2, n2 = total_energy(fl, G.insts, v2)
        d_full = (s1 + LAM * n1) - (s2 + LAM * n2)
        d_blk = (soft[b, c1] + LAM * nv[b, c1]) - (soft[b, c2] + LAM * nv[b, c2])
        assert abs(d_full - d_blk) < 1e-3 * max(1.0, abs(d_full)), (d_full, d_blk)
