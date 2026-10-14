"""Compact (block-sites-only, active-colours-only) passes equal the full-image reference passes:
same draws for the same key, same log q / soft / violation sums, and the batched candidate+reverse
mh_move / flip_move equal moves built from the reference passes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.kernels import block_mh as bm
from castlegen.kernels import coarse_flip as cf
from castlegen.variants._cfast_core import train_pass

from test_block_mh import _setup


@pytest.mark.parametrize("name,L", [("toy_support", 2), ("toy_modes", 3), ("toy_modes", 1)])
def test_compact_equals_reference(name, L):
    p, lay, params = _setup(name)
    if L > lay.T:
        pytest.skip("level absent")
    lam = jnp.full(lay.T + 1, 10.0)
    states = jax.vmap(lambda s: train_pass(lay, params, s, {}, jnp.float32(1.0), lam, record=False)[0])(jnp.arange(4, dtype=jnp.int32))
    hL, wL = lay.hw[L]
    for i, (Y, X) in enumerate([(0, 0), (hL - 1, wL - 1), (hL // 2, wL // 2), (0, wL - 1)]):
        st = {k: v[i % 4] for k, v in states.items()}
        Y, X = jnp.int32(Y), jnp.int32(X)
        inb = bm.block_masks(lay, L, Y, X)
        key = jax.random.PRNGKey(i)
        for Lmax in sorted({L, lay.T}):
            a = bm.compact_pass(lay, params, key, st, jnp.int32(L), Y, X, 1.0, Lmax)
            b = bm.block_pass_ref(lay, params, key, st, inb, 1.0, 20.0, Lmax)
            assert all(bool((a[0][k] == b[0][k]).all()) for k in st)
            assert abs(float(a[1] - b[1])) < 1e-2 and abs(float(a[2] - b[2])) < 1e-2 and int(a[3]) == int(b[3])
            fa = bm.compact_pass(lay, params, key, st, jnp.int32(L), Y, X, 1.0, Lmax, forced=st)
            fb = bm.block_pass_ref(lay, params, key, st, inb, 1.0, 20.0, Lmax, forced=st)
            assert abs(float(fa[1] - fb[1])) < 1e-2 and abs(float(fa[2] - fb[2])) < 1e-2 and int(fa[3]) == int(fb[3])
            for mode in ("uniform", "q"):
                ga = bm.compact_pass(lay, params, key, st, jnp.int32(L), Y, X, 1.0, Lmax, flip=(2.0, mode))
                gb = cf.flip_pass_ref(lay, params, key, st, inb, jnp.int32(L), 1.0, 2.0, mode, Lmax)
                assert all(bool((ga[0][k] == gb[0][k]).all()) for k in st)
                assert abs(float(ga[1] - gb[1])) < 1e-2 and abs(float(ga[4] - gb[4])) < 1e-2
