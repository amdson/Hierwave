"""cnocq (P11-1, ideas #22): reverse-KL update without tau log q for selected channels.
1. nocq_chans = []: cfast's fused update (the ckpop_hard_rkl reverse-KL step) is reproduced bit for bit.
2. nocq_chans = certificates: same draws and violations; per-world cost drops by exactly tau * sum log q(a) of
   the certificate draws; parameters change.  CPU, ground, small W."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.cfast import CFast
from castlegen.variants.cnocq import CERTS, _build_train_nocq


def _setup():
    p = problems.get("ground")
    var = CFast(p, 0, "/tmp/cnocq_test", W_train=64, B=2)
    region = Region(64)
    lay = var.layout(region)
    var.gs = {k: jnp.float32(2.0) for k in var.params}
    seeds = np.arange(5, 7)
    des = var._batch(seeds, region)
    lam = jnp.asarray(np.where(np.arange(p.T + 1) < 4, 3.0, 0.0), jnp.float32)
    args = (var.params, var.cparams, var.opt, var.copt, var.gs, jnp.asarray(seeds, jnp.int32), des,
            jnp.float32(1.0), lam, 1e-3, 1e-3, var.cfg["clip"], jax.random.PRNGKey(3))
    return var, lay, args


def test_empty_list_bit_identical_and_certs_drop_logq():
    var, lay, args = _setup()
    ref = CFast._build_train(var, lay)
    var.cfg["nocq_chans"] = []
    f0 = _build_train_nocq(var, lay)
    a, b = ref["fused"](*args), f0["fused"](*args)
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert np.array_equal(np.asarray(x), np.asarray(y)), "empty nocq list must be bit-identical"

    var.cfg["nocq_chans"] = CERTS
    f1 = _build_train_nocq(var, lay)
    c = f1["fused"](*args)
    st0, st1 = a[0], c[0]
    for k in st0:
        assert np.array_equal(np.asarray(st0[k]), np.asarray(st1[k])), "draws must not change"
    assert np.array_equal(np.asarray(a[7]), np.asarray(c[7])), "violations must not change"
    # expected drop: tau * sum of log q(a) over the certificate draws (tau = 1)
    _, recs, _, _, tot_ref, _ = ref["rec"](var.params, args[5], args[6], args[7], args[8])
    drop = 0.0
    for key, r in recs.items():
        if key[key.index("_") + 1:] in CERTS:
            lqa = jnp.take_along_axis(r["logq"], r["a"][..., None], -1)[..., 0]
            drop = drop + jnp.sum(jnp.where(r["valid"], lqa, 0.0), (1, 2))
    np.testing.assert_allclose(np.asarray(c[8]), np.asarray(a[8]) - np.asarray(drop), rtol=1e-5, atol=1e-2)
    assert float(np.abs(np.asarray(drop)).sum()) > 1.0
    diff = [float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree_util.tree_leaves(a[1]), jax.tree_util.tree_leaves(c[1]))]
    assert max(diff) > 0, "parameters must change when certificate entropy is dropped"
