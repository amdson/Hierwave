"""ccdata with its zero-change config is bit-identical to cfitpen_ml2 for one iteration (ground); the knobs change things."""
from __future__ import annotations

import jax
import numpy as np

from castlegen.bench.problems import get
from castlegen.bench.variant import load_variants


def _one_iter(name, **cfg):
    V = load_variants()
    v = V[name](get("ground"), seed=3, workdir="/tmp/ccdata_test", max_iters=1, **cfg)
    v.train(1e9, lambda d: None)
    return v


def _same(a, b):
    la, lb = jax.tree_util.tree_leaves(a.params), jax.tree_util.tree_leaves(b.params)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def test_zero_change_bit_identical():
    a, b = _one_iter("cfitpen_ml2"), _one_iter("ccdata")
    assert a.updates == b.updates and a.rkl_updates == b.rkl_updates
    assert _same(a, b)
    assert np.array_equal(np.asarray(a.lam), np.asarray(b.lam))
    for k in a.pop:
        assert np.array_equal(np.asarray(a.pop[k]), np.asarray(b.pop[k]))


def test_knobs_act():
    a = _one_iter("ccdata")
    b = _one_iter("ccdata", pop_mult=2)
    assert next(iter(b.pop.values())).shape[0] == 2 * next(iter(a.pop.values())).shape[0]
    c = _one_iter("ccdata", w3=4.0)
    assert not _same(a, c) and c.updates == a.updates
    d = _one_iter("ccdata", extra_coarse=16)
    assert d.updates == a.updates + 16
    # extra steps move the coarse levels (the penalty steps that follow also move levels < 3 through the changed L3 params)
    assert any(not np.array_equal(np.asarray(x), np.asarray(y)) for k in a.params if int(k[1:k.index("_")]) >= 3
               for x, y in zip(jax.tree_util.tree_leaves(a.params[k]), jax.tree_util.tree_leaves(d.params[k])))
