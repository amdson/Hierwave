"""cfitpen with its zero-change config is bit-identical to ckpop_hard_pen for one iteration (ground)."""
from __future__ import annotations

import jax
import numpy as np

from castlegen.bench.problems import get
from castlegen.bench.variant import load_variants


def _one_iter(name):
    V = load_variants()
    v = V[name](get("ground"), seed=3, workdir="/tmp/fitpen_test", max_iters=1)
    v.train(1e9, lambda d: None)
    return v


def test_zero_change_bit_identical():
    a, b = _one_iter("ckpop_hard_pen"), _one_iter("cfitpen")
    assert a.updates == b.updates and a.rkl_updates == b.rkl_updates
    la, lb = jax.tree_util.tree_leaves(a.params), jax.tree_util.tree_leaves(b.params)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert np.array_equal(np.asarray(a.lam), np.asarray(b.lam))
    for k in a.pop:
        assert np.array_equal(np.asarray(a.pop[k]), np.asarray(b.pop[k]))


def test_pen_every_skips():
    V = load_variants()
    v = V["cfitpen_pe2"].__new__(V["cfitpen_pe2"])
    v.cfg = {"pen_every": 2}; v._pen_calls = 0; v._pen_last = np.ones(5)
    v.problem = type("P", (), {"T": 4})()
    assert np.array_equal(v._rkl_steps(32), np.ones(5)) and v._pen_calls == 1
